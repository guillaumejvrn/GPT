use fancy_regex::Regex;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

// GPT-4 split pattern used to isolate words, contractions, numbers, and whitespaces.
const GPT4_SPLIT_PATTERN: &str = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+";

// Buffer chunk size per batch sent to Rayon worker threads (32 MB chunks).
const CHUNK_SIZE_BYTES: usize = 32 * 1024 * 1024;

/// Doubly linked-list node stored in a contiguous array to prevent dynamic heap allocations.
#[derive(Copy, Clone, Debug)]
struct Node {
    val: u32,
    prev: i32, // -1 denotes head or unlinked node
    next: i32, // -1 denotes tail or unlinked node
}

/// A merge candidate pointing to a left node index with its associated BPE merge priority.
#[derive(Copy, Clone, Eq, PartialEq)]
struct MergeCandidate {
    rank: u32,
    pos: usize,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Invert order so std::collections::BinaryHeap behaves as a min-heap on rank.
        Reverse(self.rank)
            .cmp(&Reverse(other.rank))
            .then_with(|| other.pos.cmp(&self.pos))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Encodes a slice of raw text into uint16 token IDs using an in-place priority queue.
///
/// This avoids sequential scans and eliminates intermediate Vec re-allocations
/// by updating node pointers directly inside pre-allocated scratch buffers.
fn encode_chunk_fast(
    text: &str,
    merge_ranks: &HashMap<(u32, u32), u32>,
    regex: &Regex,
    nodes: &mut Vec<Node>,
    heap: &mut BinaryHeap<MergeCandidate>,
    chunk_tokens: &mut Vec<u16>,
) {
    for piece_match in regex.find_iter(text) {
        let piece = match piece_match {
            Ok(m) => m.as_str(),
            Err(_) => continue,
        };

        let bytes = piece.as_bytes();
        let len = bytes.len();
        if len == 0 {
            continue;
        }

        // Single-byte tokens do not require merge checking.
        if len == 1 {
            chunk_tokens.push(bytes[0] as u16);
            continue;
        }

        // Build the contiguous doubly linked list from raw UTF-8 bytes.
        nodes.clear();
        for (i, &b) in bytes.iter().enumerate() {
            nodes.push(Node {
                val: b as u32,
                prev: (i as i32) - 1,
                next: if i + 1 < len { (i + 1) as i32 } else { -1 },
            });
        }

        // Seed the priority queue with all valid initial adjacent pairs.
        heap.clear();
        for i in 0..(len - 1) {
            let pair = (nodes[i].val, nodes[i + 1].val);
            if let Some(&rank) = merge_ranks.get(&pair) {
                heap.push(MergeCandidate { rank, pos: i });
            }
        }

        // Greedily apply merges in ascending rank order (highest priority first).
        while let Some(candidate) = heap.pop() {
            let pos = candidate.pos;
            let current = nodes[pos];

            // Discard stale candidates where the node has already been spliced out.
            if current.next == -1 {
                continue;
            }
            let next_node = nodes[current.next as usize];
            let pair = (current.val, next_node.val);

            let expected_rank = match merge_ranks.get(&pair) {
                Some(&r) => r,
                None => continue,
            };

            // Verify candidate validity against any prior neighbor mutations.
            if expected_rank != candidate.rank {
                continue;
            }

            // Apply merge: mutate left node with new merged token ID and splice out right node.
            let merged_id = candidate.rank;
            nodes[pos].val = merged_id;

            let after_next = next_node.next;
            nodes[pos].next = after_next;
            if after_next != -1 {
                nodes[after_next as usize].prev = pos as i32;
            }

            // Check and enqueue potential new pair formed with the left neighbor.
            if current.prev != -1 {
                let prev_pos = current.prev as usize;
                let left_pair = (nodes[prev_pos].val, merged_id);
                if let Some(&left_rank) = merge_ranks.get(&left_pair) {
                    heap.push(MergeCandidate {
                        rank: left_rank,
                        pos: prev_pos,
                    });
                }
            }

            // Check and enqueue potential new pair formed with the right neighbor.
            if after_next != -1 {
                let right_pos = after_next as usize;
                let right_pair = (merged_id, nodes[right_pos].val);
                if let Some(&right_rank) = merge_ranks.get(&right_pair) {
                    heap.push(MergeCandidate {
                        rank: right_rank,
                        pos,
                    });
                }
            }
        }

        // Traverse remaining linked list nodes sequentially to collect final token IDs.
        let mut idx = 0;
        while idx != -1 {
            chunk_tokens.push(nodes[idx as usize].val as u16);
            idx = nodes[idx as usize].next;
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    // Project root resolution relative to crate manifest directory.
    let project_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../");
    let input_path = project_root.join("data/raw/openwebtext_complet.txt");
    let vocab_path = project_root.join("data/vocab/bpeVocabulary.json");
    let output_path = project_root.join("data/processed/train.bin");

    println!("Loading BPE vocabulary from: {}", vocab_path.display());
    let vocab_data = fs::read_to_string(&vocab_path)?;
    let raw_merges: HashMap<String, u32> = serde_json::from_str(&vocab_data)?;

    // Parse "id1,id2" string keys into (u32, u32) tuple keys for fast lookups.
    let mut merge_ranks: HashMap<(u32, u32), u32> = HashMap::with_capacity(raw_merges.len());
    for (pair_str, id) in raw_merges {
        let parts: Vec<&str> = pair_str.split(',').collect();
        if parts.len() == 2 {
            let first = parts[0].parse::<u32>()?;
            let second = parts[1].parse::<u32>()?;
            merge_ranks.insert((first, second), id);
        }
    }

    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }

    println!("Reading input corpus: {}...", input_path.display());
    let file = File::open(&input_path)?;
    let mut reader = BufReader::with_capacity(32 * 1024 * 1024, file);

    let out_file = File::create(&output_path)?;
    let mut writer = BufWriter::with_capacity(32 * 1024 * 1024, out_file);

    let mut lines_batch: Vec<String> = Vec::new();
    let mut current_bytes = 0;
    let mut total_mb_processed = 0;
    let mut line_buffer = String::new();

    println!("Encoding corpus into binary train.bin with multi-threaded priority queue...");

    while reader.read_line(&mut line_buffer)? > 0 {
        current_bytes += line_buffer.len();
        lines_batch.push(std::mem::take(&mut line_buffer));

        if current_bytes >= CHUNK_SIZE_BYTES {
            let batch = std::mem::take(&mut lines_batch);

            // Rayon map_init allocates reusable scratch buffers per worker thread.
            let encoded_chunks: Vec<Vec<u16>> = batch
                .into_par_iter()
                .map_init(
                    || {
                        (
                            Regex::new(GPT4_SPLIT_PATTERN).unwrap(),
                            Vec::with_capacity(128),
                            BinaryHeap::with_capacity(128),
                            Vec::with_capacity(2048),
                        )
                    },
                    |(regex, nodes, heap, tokens), line| {
                        tokens.clear();
                        encode_chunk_fast(&line, &merge_ranks, regex, nodes, heap, tokens);
                        tokens.clone()
                    },
                )
                .collect();

            // Stream raw little-endian uint16 bytes directly to SSD via buffered writer.
            for tokens in encoded_chunks {
                for token in tokens {
                    writer.write_all(&token.to_le_bytes())?;
                }
            }

            total_mb_processed += current_bytes / (1024 * 1024);
            print!("\rProgress: ~{} MB processed", total_mb_processed);
            std::io::stdout().flush()?;
            current_bytes = 0;
        }
    }

    // Process any remaining buffered lines.
    if !lines_batch.is_empty() {
        let encoded_chunks: Vec<Vec<u16>> = lines_batch
            .into_par_iter()
            .map_init(
                || {
                    (
                        Regex::new(GPT4_SPLIT_PATTERN).unwrap(),
                        Vec::with_capacity(128),
                        BinaryHeap::with_capacity(128),
                        Vec::with_capacity(2048),
                    )
                },
                |(regex, nodes, heap, tokens), line| {
                    tokens.clear();
                    encode_chunk_fast(&line, &merge_ranks, regex, nodes, heap, tokens);
                    tokens.clone()
                },
            )
            .collect();

        for tokens in encoded_chunks {
            for token in tokens {
                writer.write_all(&token.to_le_bytes())?;
            }
        }
    }

    writer.flush()?;
    println!("\n\nEncoding completed successfully in {}!", output_path.display());

    Ok(())
}