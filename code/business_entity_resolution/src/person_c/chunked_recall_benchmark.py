"""
Person C — chunked_recall_benchmark.py

A memory-safe, chunked benchmark evaluating Person B's candidate retrieval
against the FULL S2+S3 corpus without exceeding limited RAM.

It explicitly reads the corpus in configurable chunks (default 25,000), normalizes,
indexes in a fresh MultiChannelBlocker, queries the 1,000 sampled S1 entities,
accumulates retrieved IDs, and destroys the index to prevent Out-Of-Memory (OOM).

Ground truth is strictly used ONLY for final evaluation. The blocker operates
purely on the raw sequential stream of target records.
"""

import sys
import csv
import time
import random
import gc
import psutil
import argparse
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Any, Set, Optional, Iterator

import numpy as np

# ---------------------------------------------------------------------------
# sys.path configuration
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from normalization import normalize_record
from person_b.blocking import MultiChannelBlocker
from person_b.recall_benchmark import load_and_normalize_tsv, load_ground_truth_tsv
from person_b.candidate_recall import evaluate_candidate_recall

# ---------------------------------------------------------------------------
# Constants & Memory Logging
# ---------------------------------------------------------------------------
BUFFER_SIZE = 8 * 1024 * 1024  # 8 MB explicit buffering for huge TSVs

def get_ram_mb() -> float:
    """Returns current process RSS memory in MB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def banner(title: str) -> None:
    w = 75
    print("=" * w)
    print(f"  {title}")
    print("=" * w)

def section(title: str) -> None:
    print(f"\n--- {title} ---")

# ---------------------------------------------------------------------------
# Chunking and Streaming
# ---------------------------------------------------------------------------
def stream_and_normalize_chunks(
    filepath: Path, 
    chunk_size: int, 
    max_chunks: Optional[int] = None
) -> Iterator[List[Dict[str, Any]]]:
    """
    Streams a TSV file safely, parsing and normalizing exactly `chunk_size`
    records at a time before yielding. Avoids loading the full file into memory.
    """
    if not filepath.exists():
        return
        
    chunk: List[Dict[str, Any]] = []
    chunks_yielded = 0
    
    with open(filepath, "r", encoding="utf-8", errors="replace",
              buffering=BUFFER_SIZE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entity_id = row.get("entity_id", "")
            if not entity_id:
                continue
                
            norm_dict = normalize_record(
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", "")
            )
            norm_dict["entity_id"] = entity_id
            chunk.append(norm_dict)
            
            if len(chunk) >= chunk_size:
                yield chunk
                chunks_yielded += 1
                chunk = []
                
                if max_chunks is not None and chunks_yielded >= max_chunks:
                    return
                    
    if chunk and (max_chunks is None or chunks_yielded < max_chunks):
        yield chunk

# ---------------------------------------------------------------------------
# Core Benchmark Logic
# ---------------------------------------------------------------------------
def run_chunked_benchmark(
    data_dir: Path,
    chunk_size: int = 25000,
    sample_size: int = 1000,
    random_seed: int = 42,
    max_chunks: Optional[int] = None
) -> Dict[str, Any]:
    """
    Executes the chunked recall benchmark.
    Returns evaluation metrics.
    """
    banner("CHUNKED CANDIDATE RECALL BENCHMARK")
    print(f"Data dir   : {data_dir}")
    print(f"Chunk size : {chunk_size:,} records per LSH index")
    print(f"S1 sample  : {sample_size:,} entities (seed={random_seed})")
    if max_chunks:
        print(f"Max chunks : {max_chunks} per file (DEVELOPMENT MODE)")
    print(f"Base RAM   : {get_ram_mb():.1f} MB\n")
    
    t_total = time.perf_counter()
    
    # ---------------------------------------------------------
    # 1. Load S1 and Ground Truth (uses Person B's exact loader)
    # ---------------------------------------------------------
    section("1. Loading Source 1 and Ground Truth")
    s1_all = load_and_normalize_tsv(data_dir / "train_source1.tsv")
    gt_map = load_ground_truth_tsv(data_dir / "train_ground_truth.tsv")
    
    if not s1_all:
        print("ERROR: No Source 1 records found.")
        return {}
        
    random.seed(random_seed)
    s1_sample = random.sample(s1_all, min(sample_size, len(s1_all)))
    print(f" -> Evaluated S1 entities: {len(s1_sample):,}")
    
    # Track accumulated candidates across all chunks for each S1
    global_retrieved_candidates: Dict[str, Set[str]] = defaultdict(set)
    total_records_processed = 0
    
    # ---------------------------------------------------------
    # 2. Process Target Files sequentially in chunks
    # ---------------------------------------------------------
    for file_name in ["train_source2.tsv", "train_source3.tsv"]:
        filepath = data_dir / file_name
        if not filepath.exists():
            print(f"WARNING: File {file_name} not found, skipping.")
            continue
            
        section(f"Processing {file_name}")
        file_records_processed = 0
        chunk_idx = 0
        t_file_start = time.perf_counter()
        
        for chunk in stream_and_normalize_chunks(filepath, chunk_size, max_chunks):
            chunk_idx += 1
            t_chunk_start = time.perf_counter()
            
            # Step A: Build a fresh blocker for this chunk
            blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
            blocker.index_target_records(chunk)
            
            # Step B: Query the fixed S1 sample against this chunk
            for s1_rec in s1_sample:
                s1_id = s1_rec.get("entity_id", "")
                chunk_cands = blocker.get_candidates(s1_rec)
                
                # Exclude self matches and accumulate
                if s1_id in chunk_cands:
                    chunk_cands.discard(s1_id)
                    
                if chunk_cands:
                    global_retrieved_candidates[s1_id].update(chunk_cands)
                    
            # Step C: Cleanup memory safely
            file_records_processed += len(chunk)
            total_records_processed += len(chunk)
            t_chunk_ms = (time.perf_counter() - t_chunk_start) * 1000
            ram_mb = get_ram_mb()
            
            print(f"  Chunk {chunk_idx:3d} | Recs: {file_records_processed:9,} | "
                  f"RAM: {ram_mb:6.1f} MB | Time: {t_chunk_ms:6.0f} ms")
            
            del blocker
            del chunk
            gc.collect()
            
        print(f" -> Finished {file_name} in {time.perf_counter() - t_file_start:.1f}s")
        
    # ---------------------------------------------------------
    # 3. Final Evaluation
    # ---------------------------------------------------------
    section("3. Final Recall Evaluation")
    t0 = time.perf_counter()
    
    # We must patch evaluate_candidate_recall to use our global_retrieved_candidates
    # rather than running a blocker internally. We will extract the true/retrieved logic.
    total_true_matches = 0
    true_matches_retrieved = 0
    true_matches_missed = 0
    candidate_counts = []
    
    for s1_rec in s1_sample:
        s1_id = s1_rec.get("entity_id", "")
        retrieved_set = global_retrieved_candidates.get(s1_id, set())
        
        # Determine true matches
        true_str = gt_map.get(s1_id, "")
        true_set = set(x for x in true_str.split(",") if x)
        
        total_true_matches += len(true_set)
        
        intersection = true_set.intersection(retrieved_set)
        true_matches_retrieved += len(intersection)
        true_matches_missed += (len(true_set) - len(intersection))
        
        candidate_counts.append(len(retrieved_set))
        
    candidate_recall = true_matches_retrieved / total_true_matches if total_true_matches > 0 else 1.0
    
    avg_cands = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    med_cands = float(np.median(candidate_counts)) if candidate_counts else 0.0
    max_cands = int(np.max(candidate_counts)) if candidate_counts else 0
    
    print(f"  Total target records    : {total_records_processed:,}")
    print(f"  S1 entities evaluated   : {len(s1_sample):,}")
    print(f"  Total true matches      : {total_true_matches:,}")
    print(f"  True matches retrieved  : {true_matches_retrieved:,}")
    print(f"  True matches missed     : {true_matches_missed:,}")
    print(f"  Candidate recall        : {candidate_recall:.2%}")
    print(f"  Avg candidates per S1   : {avg_cands:,.1f}")
    print(f"  Median candidates per S1: {med_cands:,.1f}")
    print(f"  Max candidates for an S1: {max_cands:,}")
    print(f"  Evaluation time         : {time.perf_counter() - t0:.2f}s")
    
    print(f"\nTotal script wall time: {time.perf_counter() - t_total:.1f}s")
    banner("DONE")
    
    return {
        "s1_evaluated": len(s1_sample),
        "total_true": total_true_matches,
        "true_retrieved": true_matches_retrieved,
        "true_missed": true_matches_missed,
        "recall": candidate_recall,
        "avg_cands": avg_cands,
        "med_cands": med_cands,
        "max_cands": max_cands,
        "total_targets_processed": total_records_processed
    }


def main():
    parser = argparse.ArgumentParser(description="Memory-safe chunked recall benchmark")
    parser.add_argument("--chunk-size", type=int, default=25000, help="Records per chunk")
    parser.add_argument("--sample-size", type=int, default=1000, help="S1 sample size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for S1 sample")
    parser.add_argument("--max-chunks", type=int, default=None, help="Max chunks per file (DEV MODE)")
    parser.add_argument("--data-dir", type=str, 
                        default=r"D:\amazom ml\6ab10eb3b23ba_student_resource\student_resource\dataset\train")
    
    args = parser.parse_args()
    
    run_chunked_benchmark(
        data_dir=Path(args.data_dir),
        chunk_size=args.chunk_size,
        sample_size=args.sample_size,
        random_seed=args.seed,
        max_chunks=args.max_chunks
    )

if __name__ == "__main__":
    main()
