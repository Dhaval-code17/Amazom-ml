from typing import Dict, Set, Iterable, Union, List
import pandas as pd
import numpy as np

def score_macro_f_beta(
    predictions: Dict[str, Union[Set[str], List[str], str]],
    ground_truth: Dict[str, Union[Set[str], List[str], str]],
    beta: float = 0.5
) -> float:
    """
    Macro-averaged F_beta score (default beta=0.5) computed per Source 1 entity.

    predictions: dict mapping source1_entity_id -> set/list of predicted matched entity IDs
                 or comma-separated string.
    ground_truth: dict mapping source1_entity_id -> set/list of ground truth matched entity IDs
                  or comma-separated string.
    """
    beta_sq = beta ** 2  # 0.25 for beta=0.5
    scale_factor = 1 + beta_sq  # 1.25 for beta=0.5

    all_s1_ids = set(ground_truth.keys()).union(set(predictions.keys()))
    if not all_s1_ids:
        return 0.0

    entity_scores = []

    for s1_id in all_s1_ids:
        # Helper to convert input values into set of IDs
        raw_pred = predictions.get(s1_id, set())
        raw_gt = ground_truth.get(s1_id, set())

        if isinstance(raw_pred, str):
            pred_set = set(x.strip() for x in raw_pred.split(",") if x.strip())
        else:
            pred_set = set(raw_pred)

        if isinstance(raw_gt, str):
            gt_set = set(x.strip() for x in raw_gt.split(",") if x.strip())
        else:
            gt_set = set(raw_gt)

        # Cases:
        # 1. actual empty AND predicted empty: F0.5 = 1.0
        if len(gt_set) == 0 and len(pred_set) == 0:
            f_score = 1.0
        # 2. actual empty AND predicted non-empty: F0.5 = 0.0
        elif len(gt_set) == 0 and len(pred_set) > 0:
            f_score = 0.0
        # 3. otherwise: compute precision and recall
        else:
            intersection = len(pred_set.intersection(gt_set))
            precision = intersection / len(pred_set) if len(pred_set) > 0 else 0.0
            recall = intersection / len(gt_set)

            denom = (beta_sq * precision) + recall
            if denom == 0:
                f_score = 0.0
            else:
                f_score = (scale_factor * precision * recall) / denom

        entity_scores.append(f_score)

    return float(np.mean(entity_scores))
