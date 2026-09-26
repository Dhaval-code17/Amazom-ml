"""
Person C — model.py

First baseline LightGBM training and GroupKFold evaluation layer.
Trains on pure numeric features from Person C.
Excludes identifiers and ground truth from the model.
Uses fixed 0.5 threshold for baseline evaluation using Person A's metric.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from scoring import score_macro_f_beta

# ---------------------------------------------------------------------------
# FEATURE CONFIGURATION
# ---------------------------------------------------------------------------

# Explicit list of exactly 45 numeric features produced by features.py.
# Using an explicit list prevents accidental inclusion of identifiers, 
# labels, or future non-feature columns.
MODEL_FEATURES = [
    # Group 1: Name (11)
    "name_exact", "aggressive_name_exact", "name_jaccard", "name_token_overlap",
    "name_containment", "name_levenshtein", "name_jaro_winkler", "name_char_ngram_similarity",
    "name_phonetic_match", "name_token_count_diff", "name_length_diff",
    # Group 2: Address (8)
    "address_exact", "address_jaccard", "address_token_overlap", "address_containment",
    "address_levenshtein", "address_jaro_winkler", "address_length_diff", "address_token_count_diff",
    # Group 3: Location (7)
    "country_match", "country_missing_either", "city_match", "city_missing_either", 
    "postal_match", "postal_missing_either", "street_number_match",
    # Group 4: Missingness (12)
    "source_name_missing", "candidate_name_missing", "source_address_missing", 
    "candidate_address_missing", "source_postal_missing", "candidate_postal_missing", 
    "source_city_missing", "candidate_city_missing", 
    "both_name_missing", "both_address_missing", "both_postal_missing", "both_city_missing",
    # Group 5: Interactions (7)
    "name_x_address", "name_x_country", "name_x_city", "name_x_postal", 
    "address_x_country", "address_x_city", "address_x_postal"
]

assert len(MODEL_FEATURES) == 45, f"Expected 45 MODEL_FEATURES, found {len(MODEL_FEATURES)}"


# ---------------------------------------------------------------------------
# EVALUATION HELPER
# ---------------------------------------------------------------------------

def _evaluate_predictions(
    df: pd.DataFrame, 
    prob_col: str, 
    true_col: str = "label",
    threshold: float = 0.5
) -> float:
    """
    Given a dataframe with source1_entity_id, candidate_entity_id, label, 
    and a probability column, evaluate macro F0.5 against the true labels.
    Uses Person A's strict scoring function.
    """
    # 1. Reconstruct ground truth dict
    gt_dict = {}
    # 2. Reconstruct predictions dict
    pred_dict = {}
    
    # We must ensure all S1 entities in this df exist in both dicts, 
    # even if they have 0 candidates matching.
    for s1_id in df["source1_entity_id"].unique():
        gt_dict[s1_id] = set()
        pred_dict[s1_id] = set()
        
    for _, row in df.iterrows():
        s1 = row["source1_entity_id"]
        cand = row["candidate_entity_id"]
        
        if row[true_col] == 1:
            gt_dict[s1].add(cand)
            
        if row[prob_col] >= threshold:
            pred_dict[s1].add(cand)
            
    return score_macro_f_beta(pred_dict, gt_dict, beta=0.5)


# ---------------------------------------------------------------------------
# CORE TRAINING PIPELINE
# ---------------------------------------------------------------------------

def run_group_kfold_baseline(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict[str, Any]:
    """
    Trains a baseline LightGBM model using GroupKFold on source1_entity_id.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'source1_entity_id', 'candidate_entity_id', 'label', 
        and all 45 MODEL_FEATURES.
    n_splits : int
        Number of CV folds.
    random_state : int
        Deterministic seed.
        
    Returns
    -------
    Dictionary containing:
        - fold_metrics: list of metric dicts per fold
        - oof_predictions: dataframe containing out-of-fold predictions
        - overall_f05: float, F0.5 computed on entire OOF set
        - model_feature_names: list of features used
    """
    # ---------------------------------------------------------
    # Pre-flight Validation
    # ---------------------------------------------------------
    missing_cols = set(MODEL_FEATURES) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required model features in DataFrame: {missing_cols}")
        
    for col in ["source1_entity_id", "candidate_entity_id", "label"]:
        if col not in df.columns:
            raise ValueError(f"Missing required identifier/label column: {col}")
            
    # Verify no NaN or Inf in features
    X_full = df[MODEL_FEATURES]
    if X_full.isnull().values.any():
        raise ValueError("NaN values found in MODEL_FEATURES.")
    if np.isinf(X_full.values).any():
        raise ValueError("Infinite values found in MODEL_FEATURES.")
        
    # Verify labels
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("label column must contain strictly 0 or 1.")

    # ---------------------------------------------------------
    # Baseline Model Config
    # ---------------------------------------------------------
    # Minimal config: no grid search, fixed threshold 0.5.
    model_params = {
        "objective": "binary",
        "random_state": random_state,
        "n_estimators": 100,
        "n_jobs": 1  # CPU friendly, deterministic
    }

    # ---------------------------------------------------------
    # Cross Validation
    # ---------------------------------------------------------
    groups = df["source1_entity_id"].values
    
    # Adjust n_splits if we have fewer unique groups than splits
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 unique source1_entity_ids for CV, found {n_groups}")
        
    gkf = GroupKFold(n_splits=actual_splits)
    
    fold_metrics = []
    oof_preds = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
        # Explicit Leakage Check
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        if not train_groups.isdisjoint(val_groups):
            raise RuntimeError(f"Data leakage in fold {fold}: Overlapping groups detected!")
            
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx].copy()
        
        # Prepare matrices
        X_train = train_df[MODEL_FEATURES]
        y_train = train_df["label"]
        X_val = val_df[MODEL_FEATURES]
        y_val = val_df["label"]
        
        # Train
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        
        # Predict
        val_df["y_prob"] = model.predict_proba(X_val)[:, 1]
        val_df["y_true"] = y_val
        val_df["fold"] = fold
        
        # OOF extraction (we only need tracking columns, not the full feature matrix)
        oof_slice = val_df[["source1_entity_id", "candidate_entity_id", "y_true", "y_prob", "fold"]]
        oof_preds.append(oof_slice)
        
        # Fold Evaluation
        fold_f05 = _evaluate_predictions(val_df, "y_prob", threshold=0.5)
        
        metrics = {
            "fold": fold,
            "val_s1_entities": len(val_groups),
            "val_pairs": len(val_df),
            "val_positives": y_val.sum(),
            "val_negatives": len(y_val) - y_val.sum(),
            "f05_score": fold_f05
        }
        fold_metrics.append(metrics)
        
    # ---------------------------------------------------------
    # Overall OOF Evaluation
    # ---------------------------------------------------------
    oof_df = pd.concat(oof_preds, ignore_index=True)
    overall_f05 = _evaluate_predictions(oof_df, "y_prob", true_col="y_true", threshold=0.5)
    
    return {
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_df,
        "overall_f05": overall_f05,
        "model_feature_names": MODEL_FEATURES,
        "model_params": model_params
    }
