//! Train a byte-level Byte Pair Encoding (BPE) merge vocabulary.
//!
//! The tokenizer starts with the 256 possible byte values. At each iteration
//! it counts adjacent token pairs, selects the most frequent pair, assigns it
//! the next vocabulary ID, and replaces every non-overlapping occurrence.
//! The resulting merge table is written as JSON for the Python tokenizer.

use fancy_regex::Regex;
use rayon::prelude::*;
use std::collections::HashMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    let racine_projet = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../");
    let chemin_fichier = racine_projet.join("data/raw/openwebtext_mini.txt");
    let chemin_vocabulaire = racine_projet.join("data/vocab/bpeVocabulary.json");
    let target_vocabulary_size: u32 = 32000;
    let merge_count = target_vocabulary_size - 256;

    println!("Reading and pre-tokenizing in bounded streaming '{}'...", chemin_fichier.display());

    let pattern = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

    let file = File::open(&chemin_fichier)?;
    let mut reader = BufReader::with_capacity(16 * 1024 * 1024, file);

    let mut global_word_counts: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut batch_lines: Vec<String> = Vec::new();
    let mut batch_bytes = 0;
    const BATCH_SIZE_LIMIT: usize = 64 * 1024 * 1024; // Traite 64 Mo à la fois

    let mut line_buffer = String::new();
    let mut total_mb_processed = 0;

    // Helper pour traiter un lot de lignes en parallèle
    let process_batch = |lines: Vec<String>, global_map: &mut HashMap<Vec<u32>, usize>| {
        let batch_map: HashMap<Vec<u32>, usize> = lines
            .into_par_iter()
            .map(|line| {
                let re = Regex::new(pattern).expect("Valid regex");
                let mut local_map: HashMap<Vec<u32>, usize> = HashMap::new();
                for match_result in re.find_iter(&line) {
                    if let Ok(m) = match_result {
                        let piece: Vec<u32> = m.as_str().bytes().map(|b| b as u32).collect();
                        *local_map.entry(piece).or_insert(0) += 1;
                    }
                }
                local_map
            })
            .reduce(
                HashMap::new,
                |mut acc, part| {
                    for (k, v) in part {
                        *acc.entry(k).or_insert(0) += v;
                    }
                    acc
                },
            );

        for (piece, count) in batch_map {
            *global_map.entry(piece).or_insert(0) += count;
        }
    };

    while reader.read_line(&mut line_buffer)? > 0 {
        batch_bytes += line_buffer.len();
        batch_lines.push(std::mem::take(&mut line_buffer));

        if batch_bytes >= BATCH_SIZE_LIMIT {
            process_batch(std::mem::take(&mut batch_lines), &mut global_word_counts);
            total_mb_processed += batch_bytes / (1024 * 1024);
            println!("Processed ~{} MB | Unique vocabulary tokens: {}", total_mb_processed, global_word_counts.len());
            batch_bytes = 0;
        }
    }

    if !batch_lines.is_empty() {
        process_batch(batch_lines, &mut global_word_counts);
    }

    println!(
        "\nPre-tokenization complete! Unique pre-tokenized pieces: {}",
        global_word_counts.len()
    );

    let mut merge_dictionary: HashMap<(u32, u32), u32> = HashMap::new();
    let initial_vocabulary_size = 256;

    println!("Learning {} merges...", merge_count);

    let mut word_counts = global_word_counts;

    for merge_index in 0..merge_count {
        let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();

        // Count every adjacent pair across every pre-tokenized piece.
        for (piece, &count) in &word_counts {
            if piece.len() < 2 {
                continue;
            }
            for window in piece.windows(2) {
                let pair = (window[0], window[1]);
                *pair_counts.entry(pair).or_insert(0) += count;
            }
        }

        if pair_counts.is_empty() {
            break;
        }

        // The most frequent pair gives the largest compression benefit for
        // this iteration.  Its new ID follows the original 0..255 byte IDs.
        let Some((&best_pair, _)) = pair_counts.iter().max_by_key(|&(_, count)| count) else {
            break;
        };
        let new_id = initial_vocabulary_size + merge_index;

        // Replace non-overlapping occurrences of the selected pair in parallel.
        let updated_entries: Vec<(Vec<u32>, usize)> = word_counts
            .into_par_iter()
            .map(|(piece, count): (Vec<u32>, usize)| {
                let mut merged_piece = Vec::with_capacity(piece.len());
                let mut position = 0;
                while position < piece.len() {
                    if position < piece.len() - 1
                        && piece[position] == best_pair.0
                        && piece[position + 1] == best_pair.1
                    {
                        merged_piece.push(new_id);
                        position += 2;
                    } else {
                        merged_piece.push(piece[position]);
                        position += 1;
                    }
                }
                (merged_piece, count)
            })
            .collect();

        let mut next_word_counts: HashMap<Vec<u32>, usize> = HashMap::with_capacity(updated_entries.len());
        for (piece, count) in updated_entries {
            *next_word_counts.entry(piece).or_insert(0) += count;
        }
        word_counts = next_word_counts;

        merge_dictionary.insert(best_pair, new_id);

        if (merge_index + 1) % 50 == 0 || merge_index == merge_count - 1 {
            println!(
                "Merge {}/{}: {:?} -> {}",
                merge_index + 1,
                merge_count,
                best_pair,
                new_id
            );
        }
    }

    // JSON object keys must be strings, so serialize (a, b) as "a,b".  The
    // Python tokenizer splits this key back into the two integer token IDs.
    let mut json_merges: HashMap<String, u32> = HashMap::new();
    for (pair, id) in merge_dictionary {
        json_merges.insert(format!("{},{}", pair.0, pair.1), id);
    }

    let json_string = serde_json::to_string_pretty(&json_merges)?;
    fs::write(&chemin_vocabulaire, json_string)?;

    println!(
        "\nTokenizer saved! Final vocabulary size: {}",
        target_vocabulary_size
    );

    Ok(())
}