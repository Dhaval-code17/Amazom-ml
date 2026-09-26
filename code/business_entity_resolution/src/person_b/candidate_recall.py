"""
Stage 4: Candidate Recall Measurement Utility.

Measures the candidate recall ceiling achieved by Person B's MultiChannelBlocker
against training/validation ground truth labels before ML pairwise classification.
"""

from typing import Dict, List, Set, Any, Union, Optional
from collections import defaultdict

def parse_ground_truth_matches(matched_val: Union[str, Set[str], List[str]]) -> Set[str]:
    """
    Helper to convert ground truth matched entity IDs into a clean set.
    Handles comma-separated strings, sets, or lists.
    """
    if matched_val is None:
        return set()
    if isinstance(matched_val, str):
        return set(x.strip() for x in matched_val.split(",") if x.strip())
    if isinstance(matched_val, (set, list, tuple)):
        return set(x for x in matched_val if x)
    return set()

def evaluate_candidate_recall(
    blocker: Any,
    s1_records: List[Dict[str, Any]],
    ground_truth: Dict[str, Union[str, Set[str], List[str]]],
    max_diagnostics: int = 20
) -> Dict[str, Any]:
    """
    Evaluates candidate recall ceiling over a list of Source 1 records and ground truth.
    
    Args:
        blocker: Instance of MultiChannelBlocker with indexed target records.
        s1_records: List of normalized Source 1 record dicts.
        ground_truth: Dict mapping source1_entity_id -> comma-separated string or set of matched IDs.
        max_diagnostics: Maximum number of missed match diagnostic examples to collect.
        
    Returns:
        Dict[str, Any]: Comprehensive recall evaluation report containing metrics,
                        match count bucket breakdown, and diagnostic examples.
    """
    total_s1 = len(s1_records)
    total_true_matches = 0
    total_found_matches = 0
    total_missed_matches = 0
    
    perfect_recall_entities = 0
    incomplete_recall_entities = 0
    zero_recall_entities = 0
    singleton_entities = 0
    
    # Bucket stats: match_count -> dict
    bucket_stats: Dict[int, Dict[str, int]] = defaultdict(
        lambda: {"s1_count": 0, "total_true_matches": 0, "total_found_matches": 0}
    )
    
    missed_examples: List[Dict[str, Any]] = []
    
    for s1_record in s1_records:
        s1_id = s1_record.get("entity_id", "")
        raw_gt = ground_truth.get(s1_id, set())
        true_matches = parse_ground_truth_matches(raw_gt)
        
        # Get generated candidates from MultiChannelBlocker
        candidates = blocker.get_candidates(s1_record)
        if not isinstance(candidates, set):
            candidates = set(candidates)
            
        num_true = len(true_matches)
        
        if num_true == 0:
            singleton_entities += 1
            continue
            
        found_matches = true_matches.intersection(candidates)
        missed_matches = true_matches - candidates
        
        num_found = len(found_matches)
        num_missed = len(missed_matches)
        
        total_true_matches += num_true
        total_found_matches += num_found
        total_missed_matches += num_missed
        
        # Categorize entity recall
        if num_found == num_true:
            perfect_recall_entities += 1
        elif num_found == 0:
            zero_recall_entities += 1
        else:
            incomplete_recall_entities += 1
            
        # Update match count bucket breakdown
        bucket = bucket_stats[num_true]
        bucket["s1_count"] += 1
        bucket["total_true_matches"] += num_true
        bucket["total_found_matches"] += num_found
        
        # Collect diagnostic examples for missed matches
        if num_missed > 0 and len(missed_examples) < max_diagnostics:
            missed_examples.append({
                "s1_id": s1_id,
                "true_matches": sorted(list(true_matches)),
                "candidate_count": len(candidates),
                "missed_matches": sorted(list(missed_matches))
            })
            
    overall_recall = (
        total_found_matches / total_true_matches if total_true_matches > 0 else 1.0
    )
    
    # Format recall_by_match_count breakdown
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
        "total_s1": total_s1,
        "singleton_entities": singleton_entities,
        "total_true_matches": total_true_matches,
        "total_found_matches": total_found_matches,
        "total_missed_matches": total_missed_matches,
        "overall_recall": float(overall_recall),
        "perfect_recall_entities": perfect_recall_entities,
        "incomplete_recall_entities": incomplete_recall_entities,
        "zero_recall_entities": zero_recall_entities,
        "recall_by_match_count": recall_by_match_count,
        "missed_examples": missed_examples
    }
