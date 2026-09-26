"""
Stage 5B: Candidate Recall-vs-Cap Evaluation Utility.

Measures how candidate recall ceiling changes when retaining only the top-K
candidates after Stage 5A cheap pre-ranking on real/sample training data.
"""

import sys
import time
import argparse
import random
import csv
from pathlib import Path
from typing import Dict, List, Set, Any, Optional, Union, Tuple
from collections import defaultdict
from unittest.mock import MagicMock
import numpy as np

# Safely handle unidecode, jellyfish, metaphone imports if packages are not yet installed in local python env
for pkg in ["unidecode", "jellyfish", "metaphone"]:
    try:
        __import__(pkg)
    except ImportError:
        mock_mod = MagicMock()
        if pkg == "unidecode":
            mock_mod.unidecode = lambda s: str(s) if s is not None else ""
        elif pkg == "jellyfish":
            mock_mod.metaphone = lambda s: s[:4].upper() if s else ""
            mock_mod.double_metaphone = lambda s: (s[:4].upper(), "") if s else ("", "")
        elif pkg == "metaphone":
            mock_mod.dm = lambda s: (s[:4].upper(), "") if s else ("", "")
        sys.modules[pkg] = mock_mod

# Ensure src directory is in sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from normalization import normalize_record
from person_b.blocking import MultiChannelBlocker
from person_b.candidate_recall import parse_ground_truth_matches
from person_b.pruning import rank_and_prune_candidates, calculate_cheap_similarity_score

def evaluate_recall_for_cap(
    blocker: Any,
    s1_records: List[Dict[str, Any]],
    target_records_map: Dict[str, Dict[str, Any]],
    ground_truth: Dict[str, Union[str, Set[str], List[str]]],
    cap: Optional[int] = None,
    max_diagnostics: int = 10,
    pre_ranked_cache: Optional[List[Tuple[Dict[str, Any], List[Dict[str, Any]], Set[str]]]] = None
) -> Dict[str, Any]:
    """
    Evaluates candidate recall ceiling for a specific candidate cap.
    Reuses pre_ranked_cache if provided to eliminate redundant pairwise scoring.
    """
    total_s1 = len(s1_records)
    total_true_matches = 0
    total_found_matches = 0
    total_missed_matches = 0
    
    perfect_recall_entities = 0
    incomplete_recall_entities = 0
    zero_recall_entities = 0
    singleton_entities = 0
    
    raw_candidate_counts = []
    retained_candidate_counts = []
    
    bucket_stats: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"s1_count": 0, "total_true_matches": 0, "total_found_matches": 0}
    )
    
    missed_examples: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    
    for idx, s1_rec in enumerate(s1_records):
        s1_id = s1_rec.get("entity_id", "")
        raw_gt = ground_truth.get(s1_id, set())
        true_matches = parse_ground_truth_matches(raw_gt)
        
        if pre_ranked_cache is not None:
            _, full_ranked, raw_cand_ids = pre_ranked_cache[idx]
        else:
            raw_cand_ids = blocker.get_candidates(s1_rec)
            cand_records = [target_records_map[cid] for cid in raw_cand_ids if cid in target_records_map]
            full_ranked = rank_and_prune_candidates(s1_rec, cand_records, cap=None)
            
        raw_candidate_counts.append(len(raw_cand_ids))
        
        # Apply cap to pre-ranked candidates
        if cap is not None and cap >= 0:
            ranked_cands = full_ranked[:cap]
        else:
            ranked_cands = full_ranked
            
        retained_cand_ids = set(r["entity_id"] for r in ranked_cands)
        retained_candidate_counts.append(len(retained_cand_ids))
        
        num_true = len(true_matches)
        if num_true == 0:
            singleton_entities += 1
            continue
            
        found_matches = true_matches.intersection(retained_cand_ids)
        missed_matches = true_matches - retained_cand_ids
        
        num_found = len(found_matches)
        num_missed = len(missed_matches)
        
        total_true_matches += num_true
        total_found_matches += num_found
        total_missed_matches += num_missed
        
        if num_found == num_true:
            perfect_recall_entities += 1
        elif num_found == 0:
            zero_recall_entities += 1
        else:
            incomplete_recall_entities += 1
            
        bucket = bucket_stats[num_true]
        bucket["s1_count"] += 1
        bucket["total_true_matches"] += num_true
        bucket["total_found_matches"] += num_found
        
        if num_missed > 0 and len(missed_examples) < max_diagnostics:
            missed_scores = {}
            for m_id in missed_matches:
                if m_id in target_records_map:
                    sc = calculate_cheap_similarity_score(s1_rec, target_records_map[m_id])
                    missed_scores[m_id] = sc
                else:
                    missed_scores[m_id] = None
                    
            missed_examples.append({
                "s1_id": s1_id,
                "true_matches": sorted(list(true_matches)),
                "raw_candidate_count": len(raw_cand_ids),
                "retained_candidate_count": len(retained_cand_ids),
                "missed_matches": sorted(list(missed_matches)),
                "missed_match_cheap_scores": missed_scores
            })
            
    eval_time = time.perf_counter() - t0
    
    overall_recall = (
        total_found_matches / total_true_matches if total_true_matches > 0 else 1.0
    )
    
    avg_retained = float(np.mean(retained_candidate_counts)) if retained_candidate_counts else 0.0
    median_retained = float(np.median(retained_candidate_counts)) if retained_candidate_counts else 0.0
    max_retained = int(np.max(retained_candidate_counts)) if retained_candidate_counts else 0
    
    avg_raw = float(np.mean(raw_candidate_counts)) if raw_candidate_counts else 0.0
    reduction_pct = (
        (1.0 - (avg_retained / avg_raw)) * 100.0 if avg_raw > 0 else 0.0
    )
    
    recall_by_match_count: Dict[int, Dict[str, Any]] = {}
    for count_key in sorted(bucket_stats.keys()):
        b = bucket_stats[count_key]
        b_true = b["total_true_matches"]
        b_found = b["total_found_matches"]
        b_recall = b_found / b_true if b_true > 0 else 1.0
        
        recall_by_match_count[count_key] = {
            "s1_count": b["s1_count"],
            "total_true_matches": b_true,
            "total_found_matches": b_found,
            "recall": b_recall
        }
        
    return {
        "cap": cap,
        "total_s1": total_s1,
        "singleton_entities": singleton_entities,
        "total_true_matches": total_true_matches,
        "total_found_matches": total_found_matches,
        "total_missed_matches": total_missed_matches,
        "overall_recall": float(overall_recall),
        "perfect_recall_entities": perfect_recall_entities,
        "incomplete_recall_entities": incomplete_recall_entities,
        "zero_recall_entities": zero_recall_entities,
        "avg_retained": avg_retained,
        "median_retained": median_retained,
        "max_retained": max_retained,
        "avg_raw": avg_raw,
        "reduction_pct": reduction_pct,
        "eval_time_sec": eval_time,
        "recall_by_match_count": recall_by_match_count,
        "missed_examples": missed_examples
    }

def run_cap_evaluation_experiment(
    data_dir: Path,
    sample_size: int = 1000,
    random_seed: int = 42,
    caps: Optional[List[Optional[int]]] = None
) -> Dict[str, Any]:
    """Runs Stage 5B candidate recall-vs-cap evaluation experiment across multiple caps."""
    if caps is None:
        caps = [None, 400, 300, 200, 150, 100, 75, 50, 25]
        
    print(f"===============================================================")
    print(f"       PERSON B — STAGE 5B: RECALL-VS-CAP EVALUATION          ")
    print(f"===============================================================")
    print(f"Data Directory: {data_dir}")
    print(f"Sample Size: {sample_size} Source 1 entities")
    print(f"Random Seed: {random_seed}")
    print(f"Caps Evaluated: {caps}\n")
    
    from recall_benchmark import load_and_normalize_tsv, load_ground_truth_tsv
    
    s1_path = data_dir / "train_source1.tsv"
    s2_path = data_dir / "train_source2.tsv"
    s3_path = data_dir / "train_source3.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"
    
    t0 = time.perf_counter()
    s2_records = load_and_normalize_tsv(s2_path)
    s3_records = load_and_normalize_tsv(s3_path)
    target_records = s2_records + s3_records
    
    target_map = {rec["entity_id"]: rec for rec in target_records if "entity_id" in rec}
    t_load_target = time.perf_counter() - t0
    print(f"Loaded {len(target_records):,} target records in {t_load_target:.2f}s")
    
    t0 = time.perf_counter()
    blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
    blocker.index_target_records(target_records)
    t_index = time.perf_counter() - t0
    print(f"Indexed target records across 4 channels in {t_index:.2f}s")
    
    s1_all = load_and_normalize_tsv(s1_path)
    gt_map = load_ground_truth_tsv(gt_path)
    
    random.seed(random_seed)
    if len(s1_all) > sample_size:
        s1_sample = random.sample(s1_all, sample_size)
    else:
        s1_sample = s1_all
        
    print(f"Evaluating {len(s1_sample):,} sampled Source 1 entities...\n")
    
    print("Pre-ranking candidate lists per S1 entity...")
    t0 = time.perf_counter()
    pre_ranked_cache = []
    for s1_rec in s1_sample:
        raw_cand_ids = blocker.get_candidates(s1_rec)
        cand_records = [target_map[cid] for cid in raw_cand_ids if cid in target_map]
        full_ranked = rank_and_prune_candidates(s1_rec, cand_records, cap=None)
        pre_ranked_cache.append((s1_rec, full_ranked, raw_cand_ids))
    t_prerank = time.perf_counter() - t0
    print(f" -> Pre-ranked raw candidates for {len(s1_sample):,} S1 entities in {t_prerank:.2f}s\n")
    
    results_by_cap = []
    for cap in caps:
        res = evaluate_recall_for_cap(
            blocker=blocker,
            s1_records=s1_sample,
            target_records_map=target_map,
            ground_truth=gt_map,
            cap=cap,
            max_diagnostics=10,
            pre_ranked_cache=pre_ranked_cache
        )
        results_by_cap.append(res)
        
    baseline_res = results_by_cap[0]
    print("---------------------------------------------------------------")
    print("FAIRNESS CHECK (cap=None):")
    print(f"  Total True Matches:    {baseline_res['total_true_matches']:,}")
    print(f"  Found True Matches:    {baseline_res['total_found_matches']:,}")
    print(f"  Baseline Recall:       {baseline_res['overall_recall']*100.0:.2f}%")
    print("---------------------------------------------------------------\n")
    
    print("=========================================================================================")
    print("                        RECALL-VS-CAP EXPERIMENT RESULTS TABLE                          ")
    print("=========================================================================================")
    print("  Cap    | Retained Avg | Candidate Reduction | Found / True Matches | Missed | Recall % ")
    print("---------|--------------|---------------------|----------------------|--------|----------")
    for res in results_by_cap:
        c_str = "None" if res["cap"] is None else f"{res['cap']:4d}"
        print(
            f"  {c_str:6s} |    {res['avg_retained']:6.1f}    |       {res['reduction_pct']:6.2f}%       |    {res['total_found_matches']:4d} / {res['total_true_matches']:4d}    |  {res['total_missed_matches']:4d}  | {res['overall_recall']*100.0:6.2f}%"
        )
    print("=========================================================================================\n")
    
    for res in results_by_cap:
        if res["cap"] is not None and res["total_missed_matches"] > 0:
            c_val = res["cap"]
            print(f"---------------------------------------------------------------")
            print(f"MISSED MATCH DIAGNOSTICS FOR cap={c_val} ({res['total_missed_matches']} missed):")
            print(f"---------------------------------------------------------------")
            for idx, ex in enumerate(res["missed_examples"], 1):
                print(f"  {idx:2d}. S1 Entity: {ex['s1_id']}")
                print(f"      True Matches:    {', '.join(ex['true_matches'])}")
                print(f"      Raw Candidates:  {ex['raw_candidate_count']}")
                print(f"      Retained:        {ex['retained_candidate_count']}")
                print(f"      Missed Matches:  {', '.join(ex['missed_matches'])}")
                print(f"      Missed Scores:   {ex['missed_match_cheap_scores']}\n")
                
    return {
        "results_by_cap": results_by_cap,
        "t_index": t_index,
        "t_prerank": t_prerank
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5B Recall-vs-Cap Evaluation Experiment")
    parser.add_argument("--data-dir", type=str, default="dataset/train", help="Path to training TSV directory")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of S1 entities to evaluate")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling")
    
    args = parser.parse_args()
    data_dir = Path(args.data_dir).resolve()
    
    if not (data_dir / "train_source1.tsv").exists():
        from recall_benchmark import generate_synthetic_real_tsv_dataset
        generate_synthetic_real_tsv_dataset(data_dir, num_s1=args.sample_size)
        
    run_cap_evaluation_experiment(data_dir=data_dir, sample_size=args.sample_size, random_seed=args.random_seed)
