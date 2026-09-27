"""
Person C — model.py

Baseline LightGBM model training, leakage-safe grouped cross-validation,
and candidate-level evaluation metrics for entity resolution.

Excludes identifiers and ground truth from feature matrix.
Exposes modular model training, prediction, and evaluation APIs for Person C.
"""

import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from scoring import score_macro_f_beta
from person_c.features import build_pair_features

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
# EVALUATION METRICS HELPER
# ---------------------------------------------------------------------------

def calculate_evaluation_metrics(
    df: pd.DataFrame,
    prob_col: str = "y_prob",
    true_col: str = "y_true",
    threshold: float = 0.5,
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Computes comprehensive entity-resolution metrics:
    - Precision, Recall, Candidate-level F_beta, F1, TP, FP, FN, TN
    - Macro F_beta (Person A's competition metric)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'source1_entity_id', 'candidate_entity_id', true_col, prob_col.
    prob_col : str
        Column name containing predicted probabilities.
    true_col : str
        Column name containing binary ground truth labels.
    threshold : float
        Decision threshold for binary classification (default 0.5).
    beta : float
        Beta parameter for F-beta score (default 0.5).

    Returns
    -------
    Dict[str, Any] containing precision, recall, f05, f1, macro_f05, tp, fp, fn, tn.
    """
    if df.empty:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "f05": 0.0,
            "f1": 0.0,
            "macro_f05": 0.0,
            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
            "total_pairs": 0,
            "total_entities": 0
        }

    for col in ["source1_entity_id", "candidate_entity_id", true_col, prob_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required metric column: {col}")

    y_true = df[true_col].values
    preds = (df[prob_col].values >= threshold).astype(int)

    tp = int(np.sum((preds == 1) & (y_true == 1)))
    fp = int(np.sum((preds == 1) & (y_true == 0)))
    fn = int(np.sum((preds == 0) & (y_true == 1)))
    tn = int(np.sum((preds == 0) & (y_true == 0)))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    beta_sq = beta ** 2
    denom = (beta_sq * precision + recall)
    f_beta = float((1 + beta_sq) * precision * recall / denom) if denom > 0 else 0.0

    f1_denom = precision + recall
    f1 = float(2 * precision * recall / f1_denom) if f1_denom > 0 else 0.0

    # Person A's Macro F0.5
    gt_dict = {}
    pred_dict = {}

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

    macro_f05 = score_macro_f_beta(pred_dict, gt_dict, beta=beta)

    return {
        "precision": precision,
        "recall": recall,
        "f05": f_beta,
        "f1": f1,
        "macro_f05": macro_f05,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "total_pairs": len(df),
        "total_entities": len(gt_dict)
    }


def _evaluate_predictions(
    df: pd.DataFrame, 
    prob_col: str, 
    true_col: str = "label",
    threshold: float = 0.5
) -> float:
    """Legacy helper for calculating macro F0.5."""
    metrics = calculate_evaluation_metrics(df, prob_col=prob_col, true_col=true_col, threshold=threshold)
    return metrics["macro_f05"]


# ---------------------------------------------------------------------------
# MODEL TRAINING & INFERENCE API
# ---------------------------------------------------------------------------

def train_baseline_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Optional[Dict[str, Any]] = None
) -> lgb.LGBMClassifier:
    """
    Trains a single LightGBM binary classifier on feature matrix X_train and labels y_train.

    Parameters
    ----------
    X_train : pd.DataFrame
        45 numeric feature columns (must match MODEL_FEATURES).
    y_train : pd.Series
        Binary integer labels (0 or 1).
    params : dict, optional
        Custom hyperparameters for LightGBM.

    Returns
    -------
    lgb.LGBMClassifier
        Trained model instance.
    """
    if X_train.empty or len(y_train) == 0:
        raise ValueError("X_train and y_train must not be empty.")

    missing = set(MODEL_FEATURES) - set(X_train.columns)
    if missing:
        raise ValueError(f"Missing required model features in X_train: {missing}")

    if not set(np.unique(y_train)).issubset({0, 1}):
        raise ValueError("y_train values must be strictly 0 or 1.")

    default_params = {
        "objective": "binary",
        "random_state": 42,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_samples": 1,
        "n_jobs": 1,
        "verbosity": -1
    }
    if params:
        default_params.update(params)

    model = lgb.LGBMClassifier(**default_params)
    model.fit(X_train[MODEL_FEATURES], y_train)
    return model


def predict_candidate_pairs(
    model: lgb.LGBMClassifier,
    X: pd.DataFrame
) -> np.ndarray:
    """
    Predicts match probabilities for candidate pairs.

    Parameters
    ----------
    model : lgb.LGBMClassifier
        Trained LightGBM model.
    X : pd.DataFrame
        DataFrame containing 45 MODEL_FEATURES.

    Returns
    -------
    np.ndarray
        Array of probabilities in range [0.0, 1.0] representing P(match=1).
    """
    if X.empty:
        return np.array([], dtype=float)

    missing = set(MODEL_FEATURES) - set(X.columns)
    if missing:
        raise ValueError(f"Missing required model features in X: {missing}")

    return model.predict_proba(X[MODEL_FEATURES])[:, 1]


# ---------------------------------------------------------------------------
# CORE GROUP-KFOLD CROSS VALIDATION PIPELINE
# ---------------------------------------------------------------------------

def run_group_kfold_baseline(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Trains a baseline LightGBM model using GroupKFold grouped by source1_entity_id.
    
    Parameters
    ----------
    df : pd.DataFrame
        Must contain 'source1_entity_id', 'candidate_entity_id', 'label', 
        and all 45 MODEL_FEATURES.
    n_splits : int
        Number of CV folds.
    random_state : int
        Deterministic seed.
    params : dict, optional
        Custom LightGBM parameters.
        
    Returns
    -------
    Dict[str, Any] containing:
        - fold_metrics: list of metric dicts per fold
        - oof_predictions: DataFrame containing out-of-fold predictions
        - overall_metrics: comprehensive metric dict over all OOF predictions
        - overall_f05: float macro F0.5 score
        - models: list of fold models
        - final_model: model trained on full dataset
        - model_feature_names: list of 45 feature names used
        - model_params: model hyperparameters used
    """
    if df.empty:
        raise ValueError("DataFrame is empty.")

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
    labels = df["label"].unique()
    if not set(labels).issubset({0, 1}):
        raise ValueError("label column must contain strictly 0 or 1.")

    model_params = {
        "objective": "binary",
        "random_state": random_state,
        "n_estimators": 100,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_samples": 1,
        "n_jobs": 1,
        "verbosity": -1
    }
    if params:
        model_params.update(params)

    groups = df["source1_entity_id"].values
    n_groups = len(np.unique(groups))
    actual_splits = min(n_splits, n_groups)
    if actual_splits < 2:
        raise ValueError(f"Need at least 2 unique source1_entity_ids for CV, found {n_groups}")
        
    gkf = GroupKFold(n_splits=actual_splits)
    
    fold_metrics = []
    oof_preds = []
    fold_models = []
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(df, groups=groups), 1):
        # Explicit Leakage Prevention Check
        train_groups = set(groups[train_idx])
        val_groups = set(groups[val_idx])
        if not train_groups.isdisjoint(val_groups):
            raise RuntimeError(f"Data leakage in fold {fold}: Overlapping groups detected!")
            
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx].copy()
        
        X_train = train_df[MODEL_FEATURES]
        y_train = train_df["label"]
        X_val = val_df[MODEL_FEATURES]
        y_val = val_df["label"]
        
        model = lgb.LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        fold_models.append(model)
        
        val_df["y_prob"] = model.predict_proba(X_val)[:, 1]
        val_df["y_true"] = y_val
        val_df["fold"] = fold
        
        oof_slice = val_df[["source1_entity_id", "candidate_entity_id", "y_true", "y_prob", "fold"]]
        oof_preds.append(oof_slice)
        
        f_metrics = calculate_evaluation_metrics(val_df, prob_col="y_prob", true_col="y_true", threshold=0.5)
        f_metrics["fold"] = fold
        f_metrics["val_s1_entities"] = len(val_groups)
        f_metrics["val_pairs"] = len(val_df)
        f_metrics["val_positives"] = int(y_val.sum())
        f_metrics["val_negatives"] = int(len(y_val) - y_val.sum())
        f_metrics["f05_score"] = f_metrics["macro_f05"]
        
        fold_metrics.append(f_metrics)
        
    oof_df = pd.concat(oof_preds, ignore_index=True)
    overall_metrics = calculate_evaluation_metrics(oof_df, prob_col="y_prob", true_col="y_true", threshold=0.5)
    
    # Train final model on entire dataset
    final_model = train_baseline_model(df[MODEL_FEATURES], df["label"], params=model_params)

    return {
        "fold_metrics": fold_metrics,
        "oof_predictions": oof_df,
        "overall_metrics": overall_metrics,
        "overall_f05": overall_metrics["macro_f05"],
        "models": fold_models,
        "final_model": final_model,
        "model_feature_names": MODEL_FEATURES,
        "model_params": model_params
    }


# ---------------------------------------------------------------------------
# HARD NEGATIVE MINING API
# ---------------------------------------------------------------------------

def select_hard_negatives(
    df: pd.DataFrame,
    prob_col: str = "y_prob",
    true_col: str = "label",
    negative_ratio: float = 3.0,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Selects informative hard negative candidate pairs for model training.

    Guarantees:
    - Never creates synthetic positives.
    - Never alters ground-truth labels.
    - Never drops a true positive candidate (label == 1).
    - Selects negative candidates with high predicted difficulty (prob_col).
    - Preserves source1_entity_id group structures.
    - Deterministic with random_state tie-breaking.

    Parameters
    ----------
    df : pd.DataFrame
        Training candidates dataset containing 'source1_entity_id', true_col, prob_col,
        and features.
    prob_col : str
        Column name used for difficulty ranking (higher values = harder negative).
    true_col : str
        Column name for ground-truth label (1 = positive, 0 = negative).
    negative_ratio : float
        Target ratio of negative candidates per positive candidate for each S1 entity.
        For S1 entities with 0 positive matches, selects max(1, int(negative_ratio)) negatives.
    random_state : int
        Seed for deterministic sorting and tie-breaking.

    Returns
    -------
    pd.DataFrame
        Filtered dataset containing ALL positive rows plus the selected hard negative rows.
    """
    if df.empty:
        return df.copy()

    for col in ["source1_entity_id", true_col, prob_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column for hard negative selection: {col}")

    # 1. Always keep all positive rows (label == 1)
    positives_df = df[df[true_col] == 1].copy()
    negatives_df = df[df[true_col] == 0].copy()

    if negatives_df.empty:
        return positives_df

    # 2. Add deterministic random tie-breaker column
    rng = np.random.RandomState(random_state)
    negatives_df = negatives_df.copy()
    negatives_df["_tie_break"] = rng.rand(len(negatives_df))

    # 3. Sort negatives descending by difficulty score (prob_col) and tie-breaker
    negatives_df = negatives_df.sort_values(
        by=[prob_col, "_tie_break"], ascending=[False, True]
    )

    selected_negatives = []

    # Group by source1_entity_id to select hard negatives per S1 entity
    s1_pos_counts = positives_df.groupby("source1_entity_id").size().to_dict()

    for s1_id, s1_negs in negatives_df.groupby("source1_entity_id", sort=False):
        n_pos = s1_pos_counts.get(s1_id, 0)
        if n_pos > 0:
            k = max(1, int(np.ceil(n_pos * negative_ratio)))
        else:
            k = max(1, int(negative_ratio))

        selected_negs = s1_negs.head(k)
        selected_negatives.append(selected_negs)

    if selected_negatives:
        selected_neg_df = pd.concat(selected_negatives, ignore_index=True)
        selected_neg_df = selected_neg_df.drop(columns=["_tie_break"])
    else:
        selected_neg_df = pd.DataFrame(columns=[c for c in negatives_df.columns if c != "_tie_break"])

    # Combine positives and selected hard negatives
    result_df = pd.concat([positives_df, selected_neg_df], ignore_index=True)
    return result_df


# ---------------------------------------------------------------------------
# THRESHOLD TUNING API
# ---------------------------------------------------------------------------

def tune_decision_threshold(
    oof_df: pd.DataFrame,
    prob_col: str = "y_prob",
    true_col: str = "y_true",
    metric: str = "macro_f05",
    threshold_range: Optional[np.ndarray] = None,
    beta: float = 0.5
) -> Dict[str, Any]:
    """
    Tunes the classification decision threshold using out-of-fold (OOF) predictions.

    Parameters
    ----------
    oof_df : pd.DataFrame
        DataFrame containing out-of-fold predictions. Must contain 'source1_entity_id',
        'candidate_entity_id', true_col, and prob_col.
    prob_col : str
        Column name containing predicted probabilities.
    true_col : str
        Column name containing ground-truth binary labels.
    metric : str
        Metric to optimize ('macro_f05', 'f05', 'f1', 'precision', 'recall').
    threshold_range : np.ndarray, optional
        Array of candidate threshold values. Defaults to np.arange(0.05, 0.96, 0.01).
    beta : float
        Beta parameter for F-beta score calculations.

    Returns
    -------
    Dict[str, Any] containing:
        - best_threshold: float, threshold maximizing target metric
        - best_score: float, maximum achieved metric value
        - selected_metric: str, target metric name
        - threshold_results: pd.DataFrame, table of all evaluated thresholds and metrics
        - default_05_metrics: dict, metrics achieved at default threshold 0.5
    """
    if oof_df.empty:
        return {
            "best_threshold": 0.5,
            "best_score": 0.0,
            "selected_metric": metric,
            "threshold_results": pd.DataFrame(),
            "default_05_metrics": calculate_evaluation_metrics(oof_df, prob_col=prob_col, true_col=true_col, threshold=0.5, beta=beta)
        }

    for col in ["source1_entity_id", "candidate_entity_id", true_col, prob_col]:
        if col not in oof_df.columns:
            raise ValueError(f"Missing required column for threshold tuning: {col}")

    if threshold_range is None:
        threshold_range = np.arange(0.05, 0.96, 0.01)

    results = []

    for t in threshold_range:
        t_val = float(t)
        eval_res = calculate_evaluation_metrics(
            oof_df, prob_col=prob_col, true_col=true_col, threshold=t_val, beta=beta
        )
        eval_res["threshold"] = t_val
        results.append(eval_res)

    res_df = pd.DataFrame(results)

    if metric not in res_df.columns:
        valid_metrics = [c for c in res_df.columns if c not in ("threshold", "tp", "fp", "fn", "tn", "total_pairs", "total_entities")]
        raise ValueError(f"Invalid optimization metric {metric!r}. Choose from: {valid_metrics}")

    max_idx = res_df[metric].idxmax()
    best_row = res_df.loc[max_idx]

    best_threshold = float(best_row["threshold"])
    best_score = float(best_row[metric])

    default_05_metrics = calculate_evaluation_metrics(
        oof_df, prob_col=prob_col, true_col=true_col, threshold=0.5, beta=beta
    )

    return {
        "best_threshold": best_threshold,
        "best_score": best_score,
        "selected_metric": metric,
        "threshold_results": res_df,
        "default_05_metrics": default_05_metrics
    }


# ---------------------------------------------------------------------------
# PROBABILITY CALIBRATION API
# ---------------------------------------------------------------------------

class ProbabilityCalibrator:
    """
    Probability calibrator for post-processing model predictions.
    Supports 'sigmoid' (Platt scaling via LogisticRegression) and 'isotonic' regression.
    """
    def __init__(self, method: str = "sigmoid"):
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"Unsupported calibration method {method!r}. Choose 'sigmoid' or 'isotonic'.")
        self.method = method
        self._calibrator = None
        self.is_fitted = False

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        """
        Fits the calibrator using out-of-fold predicted probabilities and true binary labels.
        """
        if len(y_prob) == 0 or len(y_true) == 0:
            self.is_fitted = False
            return self

        y_prob_arr = np.asarray(y_prob, dtype=float)
        y_true_arr = np.asarray(y_true, dtype=int)

        unique_labels = np.unique(y_true_arr)
        if len(unique_labels) < 2:
            self.is_fitted = False
            return self

        if self.method == "sigmoid":
            clf = LogisticRegression(C=1.0, solver="lbfgs")
            clf.fit(y_prob_arr.reshape(-1, 1), y_true_arr)
            self._calibrator = clf
        elif self.method == "isotonic":
            clf = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
            clf.fit(y_prob_arr, y_true_arr)
            self._calibrator = clf

        self.is_fitted = True
        return self

    def calibrate(self, y_prob: np.ndarray) -> np.ndarray:
        """
        Transforms raw predicted probabilities into calibrated probabilities in range [0.0, 1.0].
        """
        if len(y_prob) == 0:
            return np.array([], dtype=float)

        y_prob_arr = np.asarray(y_prob, dtype=float)

        if not self.is_fitted or self._calibrator is None:
            return np.clip(y_prob_arr.copy(), 0.0, 1.0)

        if self.method == "sigmoid":
            calibrated = self._calibrator.predict_proba(y_prob_arr.reshape(-1, 1))[:, 1]
        elif self.method == "isotonic":
            calibrated = self._calibrator.predict(y_prob_arr)
        else:
            calibrated = y_prob_arr.copy()

        return np.clip(calibrated, 0.0, 1.0)


def fit_oof_calibrator(
    oof_df: pd.DataFrame,
    method: str = "sigmoid",
    prob_col: str = "y_prob",
    true_col: str = "y_true"
) -> ProbabilityCalibrator:
    """
    Fits a ProbabilityCalibrator on Out-Of-Fold (OOF) prediction DataFrame.
    """
    calibrator = ProbabilityCalibrator(method=method)
    if not oof_df.empty and prob_col in oof_df.columns and true_col in oof_df.columns:
        calibrator.fit(oof_df[prob_col].values, oof_df[true_col].values)
    return calibrator


def compare_oof_calibration(
    oof_df: pd.DataFrame,
    calibrator: Optional[ProbabilityCalibrator] = None,
    prob_col: str = "y_prob",
    true_col: str = "y_true",
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Compares raw vs. calibrated metrics on OOF predictions.
    """
    raw_metrics = calculate_evaluation_metrics(
        oof_df, prob_col=prob_col, true_col=true_col, threshold=threshold
    )

    if calibrator is not None and calibrator.is_fitted:
        cal_df = oof_df.copy()
        cal_df["calibrated_prob"] = calibrator.calibrate(oof_df[prob_col].values)
        cal_metrics = calculate_evaluation_metrics(
            cal_df, prob_col="calibrated_prob", true_col=true_col, threshold=threshold
        )
    else:
        cal_metrics = dict(raw_metrics)

    return {
        "raw_metrics": raw_metrics,
        "calibrated_metrics": cal_metrics,
        "threshold": threshold,
        "is_calibrated": calibrator is not None and calibrator.is_fitted
    }


# ---------------------------------------------------------------------------
# PRODUCTION MATCH INFERENCE API
# ---------------------------------------------------------------------------

def predict_matches(
    source1_records: List[Dict[str, Any]],
    target_records: List[Dict[str, Any]],
    candidate_pairs: pd.DataFrame,
    model: Any,
    calibrator: Optional[ProbabilityCalibrator] = None,
    threshold: float = 0.5,
    top_k: Optional[int] = None
) -> pd.DataFrame:
    """
    Production match inference pipeline for Person C.

    Pipeline Steps:
    1. Parses candidate pairs and validates record lookups.
    2. Computes the 45 MODEL_FEATURES for each candidate pair (shared with training).
    3. Runs model prediction (raw probabilities).
    4. Applies optional ProbabilityCalibrator.
    5. Applies decision thresholding (uses calibrated probability if calibrator enabled, else raw).
    6. Formats output DataFrame with candidate ranks and match decisions.

    Parameters
    ----------
    source1_records : List of normalized S1 record dicts.
    target_records : List of normalized target (S2/S3) record dicts.
    candidate_pairs : pd.DataFrame with 'source1_entity_id' and 'candidate_entity_ids'.
    model : Trained LightGBM binary classifier model object.
    calibrator : Optional ProbabilityCalibrator instance.
    threshold : float decision threshold.
    top_k : Optional int maximum candidates retained per S1 entity.

    Returns
    -------
    pd.DataFrame with columns:
        - source1_entity_id (str)
        - candidate_entity_id (str)
        - raw_probability (float)
        - calibrated_probability (float) [if calibrator present]
        - match_decision (int, 0 or 1)
        - rank (int, 1-based rank within S1 query)
    """
    if "source1_entity_id" not in candidate_pairs.columns or "candidate_entity_ids" not in candidate_pairs.columns:
        raise ValueError("candidate_pairs DataFrame must have 'source1_entity_id' and 'candidate_entity_ids' columns.")

    output_cols = ["source1_entity_id", "candidate_entity_id", "raw_probability"]
    if calibrator is not None:
        output_cols.append("calibrated_probability")
    output_cols.extend(["match_decision", "rank"])

    if candidate_pairs.empty:
        return pd.DataFrame(columns=output_cols)

    s1_map = {r["entity_id"]: r for r in source1_records if isinstance(r, dict) and "entity_id" in r}
    target_map = {r["entity_id"]: r for r in target_records if isinstance(r, dict) and "entity_id" in r}

    features_list = []
    pair_metadata = []

    for _, row in candidate_pairs.iterrows():
        s1_id = row["source1_entity_id"]
        cand_ids_val = row["candidate_entity_ids"]

        if cand_ids_val is None or (isinstance(cand_ids_val, str) and not cand_ids_val.strip()):
            continue

        if isinstance(cand_ids_val, list):
            cand_ids = [str(c).strip() for c in cand_ids_val if c]
        elif isinstance(cand_ids_val, str):
            cand_ids = [c.strip() for c in cand_ids_val.split(",") if c.strip()]
        else:
            cand_ids = []

        if not cand_ids:
            continue

        if s1_id not in s1_map:
            raise ValueError(f"S1 ID {s1_id!r} found in candidate_pairs but missing from source1_records.")
        s1_rec = s1_map[s1_id]

        for cand_id in cand_ids:
            if cand_id not in target_map:
                raise ValueError(f"Candidate ID {cand_id!r} found in candidate_pairs but missing from target_records.")
            cand_rec = target_map[cand_id]

            # 1. Feature Generation (strictly label-blind, shared with training)
            f_dict = build_pair_features(s1_rec, cand_rec)
            features_list.append(f_dict)
            pair_metadata.append((s1_id, cand_id))

    if not features_list:
        return pd.DataFrame(columns=output_cols)

    features_df = pd.DataFrame(features_list)

    # 2. Extract feature matrix X (45 MODEL_FEATURES strictly)
    missing = set(MODEL_FEATURES) - set(features_df.columns)
    if missing:
        raise ValueError(f"Missing required model features for prediction: {missing}")

    X = features_df[MODEL_FEATURES]

    # 3. Model Prediction
    raw_probs = predict_candidate_pairs(model, X)

    # 4. Calibration
    if calibrator is not None:
        cal_probs = calibrator.calibrate(raw_probs)
        eval_probs = cal_probs
    else:
        cal_probs = None
        eval_probs = raw_probs

    # 5. Build results DataFrame
    results = []
    for idx, (s1_id, cand_id) in enumerate(pair_metadata):
        rp = float(raw_probs[idx])
        ep = float(eval_probs[idx])
        dec = int(ep >= threshold)

        res_row = {
            "source1_entity_id": s1_id,
            "candidate_entity_id": cand_id,
            "raw_probability": rp,
            "eval_prob": ep,
            "match_decision": dec
        }
        if calibrator is not None:
            res_row["calibrated_probability"] = float(cal_probs[idx])
        results.append(res_row)

    res_df = pd.DataFrame(results)

    # 6. Global ranking instead of sorting each S1 group separately
    final_df = res_df.sort_values(
        by=["source1_entity_id", "eval_prob", "candidate_entity_id"],
        ascending=[True, False, True]
    ).reset_index(drop=True)

    # Rank within each S1 entity
    final_df["rank"] = (
        final_df.groupby("source1_entity_id", sort=False)
        .cumcount()
        + 1
    )

    # Apply top_k after ranking
    if top_k is not None and top_k > 0:
        final_df = final_df[final_df["rank"] <= top_k].copy()

    # Remove internal probability column
    final_df = final_df.drop(columns=["eval_prob"])

    return final_df[output_cols]



