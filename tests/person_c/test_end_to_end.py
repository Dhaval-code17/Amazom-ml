"""
Person C — test_end_to_end.py

End-to-End Integration Test Suite for Business Entity Resolution.
Validates full pipeline flow:
  Raw Records -> Person A Normalization -> Person B Blocking & Candidate Generation
  -> Person C Feature Engineering -> GroupKFold Baseline Model -> Hard Negative Mining
  -> OOF Threshold Tuning -> Probability Calibration -> Production Match Inference.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from normalization import normalize_record
from person_c.candidate_generation import generate_candidate_pairs
from person_c.features import build_pair_features
from person_c.training_data import build_training_dataset, extract_X_y_metadata
from person_c.model import (
    MODEL_FEATURES,
    run_group_kfold_baseline,
    train_baseline_model,
    predict_candidate_pairs,
    select_hard_negatives,
    tune_decision_threshold,
    ProbabilityCalibrator,
    fit_oof_calibrator,
    compare_oof_calibration,
    predict_matches,
)


def _build_synthetic_dataset():
    """
    Constructs normalized S1 and Target (S2/S3) records covering:
    1. Exact/near-exact matches (Acme Corp)
    2. Hard negative candidates (Acme Logistics)
    3. Multi-match S1 (Global Tech Solutions -> S2-M1 and S3-M2)
    4. Transliterated / cross-script match (Shree Foods vs Tamil transliterated S3)
    5. Singleton S1 with no true match (Unique Enterprise Delta)
    6. Rejection candidate with location/address overlap
    """
    # Raw S1 Records
    raw_s1 = [
        ("S1-EXACT", "Acme Corporation Inc", "100 Main Street, New York, NY 10001", "US"),
        ("S1-MULTI", "Global Tech Solutions Limited", "450 Technology Parkway, San Jose, CA 95110", "US"),
        ("S1-TRANSLIT", "Shree Foods Private Limited", "St Andrews Church Campus Parish Counsel Office, Manakudy, Kanyakumari, Tamil Nadu", "India"),
        ("S1-SINGLETON", "Unique Enterprise Delta", "999 Nowhere Lane, Austin, TX 78701", "US"),
    ]

    # Raw Target (S2 / S3) Records
    raw_target = [
        ("S2-EXACT-MATCH", "Acme Corporation Inc", "100 Main Street, New York, NY 10001", "US"),
        ("S2-HARD-NEG", "Acme Logistics Services", "500 Logistics Boulevard, Chicago, IL 60601", "US"),
        ("S2-M1", "Global Tech Solutions", "450 Technology Pkwy, San Jose, CA 95110", "US"),
        ("S3-M2", "Global Tech Soln Ltd", "450 Technology Parkway, San Jose, CA 95110", "US"),
        ("S3-TRANSLIT-MATCH", "Srii Hputts Piraiveett Limittett", "St Andrews Church Campus Parish Counsel Office, Manakudy, Kanyakumari, Tmilllnaattu", "India"),
        ("S2-UNRELATED-NEG", "Random Bakery & Cafe", "999 Nowhere Lane, Austin, TX 78701", "US"),
    ]

    s1_records = []
    for entity_id, name, addr, country in raw_s1:
        rec = normalize_record(name, addr, country)
        rec["entity_id"] = entity_id
        s1_records.append(rec)

    target_records = []
    for entity_id, name, addr, country in raw_target:
        rec = normalize_record(name, addr, country)
        rec["entity_id"] = entity_id
        target_records.append(rec)

    gt_dict = {
        "S1-EXACT": {"S2-EXACT-MATCH"},
        "S1-MULTI": {"S2-M1", "S3-M2"},
        "S1-TRANSLIT": {"S3-TRANSLIT-MATCH"},
        "S1-SINGLETON": set(),
    }

    return s1_records, target_records, gt_dict


def test_end_to_end_full_pipeline_verification():
    """
    Executes and asserts the complete end-to-end entity resolution pipeline:
    Raw Records -> Normalization -> Person B Blocker -> Person C Features
    -> GroupKFold LightGBM -> Hard Negative Mining -> Threshold Tuning
    -> Calibration -> Match Inference.
    """
    s1_records, target_records, gt_dict = _build_synthetic_dataset()

    # ------------------------------------------------------------------
    # STEP 5C: Person B Candidate Generation
    # ------------------------------------------------------------------
    cand_pairs = generate_candidate_pairs(s1_records, target_records, cap=50)
    assert isinstance(cand_pairs, pd.DataFrame)
    assert list(cand_pairs.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert len(cand_pairs) == len(s1_records)

    # Verify expected candidates generated for S1 entities
    s1_cand_map = dict(zip(cand_pairs["source1_entity_id"], cand_pairs["candidate_entity_ids"]))

    # All true matches in target set must be present in candidate sets
    assert "S2-EXACT-MATCH" in s1_cand_map["S1-EXACT"]
    assert "S2-M1" in s1_cand_map["S1-MULTI"]
    assert "S3-M2" in s1_cand_map["S1-MULTI"]
    assert "S3-TRANSLIT-MATCH" in s1_cand_map["S1-TRANSLIT"]

    # ------------------------------------------------------------------
    # STEP 5D: Build Person C Features & Feature Matrix
    # ------------------------------------------------------------------
    training_df = build_training_dataset(s1_records, target_records, cand_pairs, gt_dict)
    assert not training_df.empty

    X, y, metadata = extract_X_y_metadata(training_df)

    # Strict feature assertions
    assert len(MODEL_FEATURES) == 45
    assert list(X.columns) == MODEL_FEATURES
    assert "source1_entity_id" not in X.columns
    assert "candidate_entity_id" not in X.columns
    assert "label" not in X.columns
    assert not X.isnull().values.any()
    assert not np.isinf(X.values).any()

    # ------------------------------------------------------------------
    # STEP 5E: Baseline Model Training with GroupKFold
    # ------------------------------------------------------------------
    cv_results = run_group_kfold_baseline(training_df, n_splits=3, random_state=42)
    assert "oof_predictions" in cv_results
    assert "final_model" in cv_results
    assert len(cv_results["oof_predictions"]) == len(training_df)

    oof_df = cv_results["oof_predictions"]
    # Confirm no group leakage across folds
    fold_groups = oof_df.groupby("source1_entity_id")["fold"].nunique()
    assert (fold_groups == 1).all()

    # ------------------------------------------------------------------
    # STEP 5F: Hard Negative Mining
    # ------------------------------------------------------------------
    df_with_oof = training_df.merge(
        oof_df[["source1_entity_id", "candidate_entity_id", "y_prob"]],
        on=["source1_entity_id", "candidate_entity_id"]
    )
    mined_df = select_hard_negatives(df_with_oof, prob_col="y_prob", negative_ratio=2.0, random_state=42)

    # Positive count is unchanged
    assert (mined_df["label"] == 1).sum() == (training_df["label"] == 1).sum()

    # ------------------------------------------------------------------
    # STEP 5G: Decision Threshold Tuning
    # ------------------------------------------------------------------
    tuning_res = tune_decision_threshold(oof_df, prob_col="y_prob", true_col="y_true", metric="macro_f05")
    best_threshold = tuning_res["best_threshold"]
    assert 0.05 <= best_threshold <= 0.95
    assert tuning_res["best_score"] >= 0.0

    # ------------------------------------------------------------------
    # STEP 5H: Probability Calibration
    # ------------------------------------------------------------------
    calibrator = fit_oof_calibrator(oof_df, method="sigmoid")
    cal_comp = compare_oof_calibration(oof_df, calibrator=calibrator, threshold=best_threshold)
    assert "raw_metrics" in cal_comp
    assert "calibrated_metrics" in cal_comp

    # ------------------------------------------------------------------
    # STEP 5I & 5J: Final Match Inference & Pipeline Assertions
    # ------------------------------------------------------------------
    final_model = cv_results["final_model"]

    # Inference without calibration
    pred_uncal = predict_matches(
        s1_records, target_records, cand_pairs, final_model, calibrator=None, threshold=best_threshold
    )
    assert list(pred_uncal.columns) == ["source1_entity_id", "candidate_entity_id", "raw_probability", "match_decision", "rank"]

    # Inference with calibration
    pred_cal = predict_matches(
        s1_records, target_records, cand_pairs, final_model, calibrator=calibrator, threshold=best_threshold
    )
    assert "calibrated_probability" in pred_cal.columns
    assert pred_cal["calibrated_probability"].min() >= 0.0
    assert pred_cal["calibrated_probability"].max() <= 1.0

    # 1. Exact match correctly identified in uncalibrated and calibrated predictions
    exact_match_uncal = pred_uncal[(pred_uncal["source1_entity_id"] == "S1-EXACT") & (pred_uncal["candidate_entity_id"] == "S2-EXACT-MATCH")].iloc[0]
    assert exact_match_uncal["match_decision"] == 1
    assert exact_match_uncal["rank"] == 1

    # 2. Hard negative correctly rejected (lower raw probability)
    hard_neg_uncal = pred_uncal[(pred_uncal["source1_entity_id"] == "S1-EXACT") & (pred_uncal["candidate_entity_id"] == "S2-HARD-NEG")].iloc[0]
    assert exact_match_uncal["raw_probability"] > hard_neg_uncal["raw_probability"]
    assert hard_neg_uncal["match_decision"] == 0

    # 3. Multi-match S1 retains BOTH true matches (S2-M1 and S3-M2)
    multi_matches = pred_uncal[(pred_uncal["source1_entity_id"] == "S1-MULTI") & (pred_uncal["match_decision"] == 1)]
    assert len(multi_matches) >= 2
    assert "S2-M1" in multi_matches["candidate_entity_id"].values
    assert "S3-M2" in multi_matches["candidate_entity_id"].values

    # 4. Transliterated case successfully matched
    translit_match = pred_uncal[(pred_uncal["source1_entity_id"] == "S1-TRANSLIT") & (pred_uncal["candidate_entity_id"] == "S3-TRANSLIT-MATCH")].iloc[0]
    assert translit_match["match_decision"] == 1

    # 5. Singleton S1 with no candidates or rejected candidate produces 0 positive matches
    singleton_matches = pred_uncal[(pred_uncal["source1_entity_id"] == "S1-SINGLETON") & (pred_uncal["match_decision"] == 1)]
    assert len(singleton_matches) == 0

    # 6. Pipeline determinism check
    pred_uncal2 = predict_matches(
        s1_records, target_records, cand_pairs, final_model, calibrator=None, threshold=best_threshold
    )
    pd.testing.assert_frame_equal(pred_uncal, pred_uncal2)
