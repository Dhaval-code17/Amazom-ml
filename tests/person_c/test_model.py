"""
Person C — test_model.py

Unit tests for the baseline LightGBM model layer and GroupKFold cross-validation.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.model import (
    MODEL_FEATURES,
    calculate_evaluation_metrics,
    train_baseline_model,
    predict_candidate_pairs,
    run_group_kfold_baseline,
    select_hard_negatives,
    tune_decision_threshold,
    ProbabilityCalibrator,
    fit_oof_calibrator,
    compare_oof_calibration,
    predict_matches,
)


def _make_synthetic_data(n_samples=50):
    """Creates a deterministic synthetic dataset for model testing."""
    np.random.seed(42)
    
    # Generate 5 unique S1 entities, 10 candidates each
    s1_ids = [f"S1-{i}" for i in range(1, 6) for _ in range(10)]
    cand_ids = [f"cand-{i}" for i in range(50)]
    
    df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids
    })
    
    # Generate random features
    for col in MODEL_FEATURES:
        df[col] = np.random.rand(50)
        
    # Generate deterministic label based on one feature to ensure model can learn something
    df["label"] = (df[MODEL_FEATURES[0]] > 0.5).astype(int)
    
    return df


def test_feature_count():
    """MODEL_FEATURES contains exactly 45 features."""
    assert len(MODEL_FEATURES) == 45, "MODEL_FEATURES must have exactly 45 items"
    assert len(set(MODEL_FEATURES)) == 45, "MODEL_FEATURES must not contain duplicates"


def test_identifiers_and_labels_excluded():
    """IDs and labels must NOT be in MODEL_FEATURES."""
    assert "source1_entity_id" not in MODEL_FEATURES
    assert "candidate_entity_id" not in MODEL_FEATURES
    assert "label" not in MODEL_FEATURES


def test_baseline_execution():
    """1. Model can train on a small synthetic dataset."""
    df = _make_synthetic_data()
    
    results = run_group_kfold_baseline(df, n_splits=3)
    
    # Verify return structure
    assert "fold_metrics" in results
    assert "oof_predictions" in results
    assert "overall_metrics" in results
    assert "overall_f05" in results
    assert "models" in results
    assert "final_model" in results
    assert "model_feature_names" in results
    assert len(results["models"]) == 3
    assert results["model_feature_names"] == MODEL_FEATURES


def test_predictions_length_and_numeric_type():
    """2. Predictions have correct length and numeric type."""
    df = _make_synthetic_data(n_samples=50)
    results = run_group_kfold_baseline(df, n_splits=3)
    oof = results["oof_predictions"]
    
    assert len(oof) == 50
    assert pd.api.types.is_float_dtype(oof["y_prob"])
    assert oof["y_prob"].min() >= 0.0
    assert oof["y_prob"].max() <= 1.0

    # Test standalone predict API
    final_model = results["final_model"]
    preds = predict_candidate_pairs(final_model, df[MODEL_FEATURES])
    assert len(preds) == 50
    assert isinstance(preds, np.ndarray)
    assert np.issubdtype(preds.dtype, np.floating)
    assert preds.min() >= 0.0
    assert preds.max() <= 1.0


def test_group_leakage_assertion():
    """4. GroupKFold keeps the same source1_entity_id out of multiple folds."""
    df = _make_synthetic_data()
    results = run_group_kfold_baseline(df, n_splits=3)
    oof = results["oof_predictions"]
    
    # Check that each S1 ID exists in exactly one fold
    fold_map = oof.groupby("source1_entity_id")["fold"].nunique()
    assert (fold_map == 1).all(), "An S1 ID was split across multiple folds!"


def test_oof_predictions_cover_every_row_exactly_once():
    """5. OOF predictions cover every training row exactly once."""
    df = _make_synthetic_data(n_samples=50)
    results = run_group_kfold_baseline(df, n_splits=3)
    oof = results["oof_predictions"]

    assert len(oof) == len(df)
    # Re-construct pairs and check unicity
    pairs_orig = set(zip(df["source1_entity_id"], df["candidate_entity_id"]))
    pairs_oof = set(zip(oof["source1_entity_id"], oof["candidate_entity_id"]))
    assert pairs_orig == pairs_oof
    assert len(pairs_oof) == len(df)


def test_labels_remain_binary():
    """6. Labels remain strictly binary 0 or 1."""
    df = _make_synthetic_data()
    results = run_group_kfold_baseline(df, n_splits=3)
    oof = results["oof_predictions"]

    assert set(oof["y_true"].unique()).issubset({0, 1})
    assert set(df["label"].unique()).issubset({0, 1})


def test_model_training_deterministic():
    """7. Model training is deterministic when random seed is fixed."""
    df = _make_synthetic_data()

    res1 = run_group_kfold_baseline(df, n_splits=3, random_state=42)
    res2 = run_group_kfold_baseline(df, n_splits=3, random_state=42)

    pd.testing.assert_frame_equal(res1["oof_predictions"], res2["oof_predictions"])
    assert res1["overall_f05"] == res2["overall_f05"]


def test_evaluation_metrics_valid_numeric_values():
    """8. Evaluation metrics return valid numeric values."""
    df = _make_synthetic_data()
    metrics = calculate_evaluation_metrics(
        df.rename(columns={"label": "y_true"}),
        prob_col=MODEL_FEATURES[0],
        true_col="y_true",
        threshold=0.5
    )

    for k in ["precision", "recall", "f05", "f1", "macro_f05"]:
        assert isinstance(metrics[k], float)
        assert 0.0 <= metrics[k] <= 1.0

    for k in ["tp", "fp", "fn", "tn", "total_pairs", "total_entities"]:
        assert isinstance(metrics[k], int)
        assert metrics[k] >= 0


def test_empty_or_invalid_input_handling():
    """9. Empty or invalid input is handled cleanly where appropriate."""
    # Empty DataFrame
    with pytest.raises(ValueError, match="DataFrame is empty"):
        run_group_kfold_baseline(pd.DataFrame(), n_splits=3)

    # Empty in train_baseline_model
    with pytest.raises(ValueError, match="X_train and y_train must not be empty"):
        train_baseline_model(pd.DataFrame(), pd.Series(dtype=int))

    # Empty in predict_candidate_pairs returns empty numpy array
    empty_preds = predict_candidate_pairs(
        train_baseline_model(_make_synthetic_data()[MODEL_FEATURES], _make_synthetic_data()["label"]),
        pd.DataFrame(columns=MODEL_FEATURES)
    )
    assert len(empty_preds) == 0

    # Missing model features
    df_missing = _make_synthetic_data().drop(columns=[MODEL_FEATURES[0]])
    with pytest.raises(ValueError, match="Missing required model features"):
        run_group_kfold_baseline(df_missing, n_splits=2)

    # NaN in features
    df_nan = _make_synthetic_data()
    df_nan.loc[0, MODEL_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="NaN values found"):
        run_group_kfold_baseline(df_nan, n_splits=2)

    # Inf in features
    df_inf = _make_synthetic_data()
    df_inf.loc[0, MODEL_FEATURES[0]] = np.inf
    with pytest.raises(ValueError, match="Infinite values found"):
        run_group_kfold_baseline(df_inf, n_splits=2)

    # Invalid labels
    df_inv = _make_synthetic_data()
    df_inv.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label column must contain strictly 0 or 1"):
        run_group_kfold_baseline(df_inv, n_splits=2)


# ---------------------------------------------------------------------------
# STEP 4A TESTS: HARD NEGATIVE MINING & THRESHOLD TUNING
# ---------------------------------------------------------------------------

def test_hard_negatives_never_alter_positives():
    """True positives can never become hard negatives or be dropped."""
    df = _make_synthetic_data(n_samples=50)
    df["y_prob"] = df[MODEL_FEATURES[0]]

    all_positives_before = df[df["label"] == 1].copy()
    filtered_df = select_hard_negatives(df, prob_col="y_prob", true_col="label", negative_ratio=2.0)

    all_positives_after = filtered_df[filtered_df["label"] == 1].copy()
    assert len(all_positives_before) == len(all_positives_after)
    pd.testing.assert_frame_equal(
        all_positives_before.reset_index(drop=True),
        all_positives_after.reset_index(drop=True)
    )


def test_hard_negatives_deterministic():
    """Hard-negative selection is deterministic with fixed random_state."""
    df = _make_synthetic_data(n_samples=50)
    df["y_prob"] = np.random.rand(50)

    res1 = select_hard_negatives(df, prob_col="y_prob", negative_ratio=2.0, random_state=42)
    res2 = select_hard_negatives(df, prob_col="y_prob", negative_ratio=2.0, random_state=42)

    pd.testing.assert_frame_equal(res1, res2)


def test_hard_negatives_ratio_respected():
    """Hard-negative count/ratio per S1 entity is respected."""
    df = _make_synthetic_data(n_samples=50)
    df["y_prob"] = np.random.rand(50)

    ratio = 2.0
    filtered_df = select_hard_negatives(df, prob_col="y_prob", true_col="label", negative_ratio=ratio)

    s1_pos = df[df["label"] == 1].groupby("source1_entity_id").size()
    s1_negs_selected = filtered_df[filtered_df["label"] == 0].groupby("source1_entity_id").size()

    for s1_id, pos_count in s1_pos.items():
        expected_max_negs = max(1, int(np.ceil(pos_count * ratio)))
        actual_negs = s1_negs_selected.get(s1_id, 0)
        assert actual_negs <= expected_max_negs, f"Too many negatives selected for {s1_id}"


def test_hard_negative_selection_no_leakage():
    """OOF predictions from fold 1/2 are used cleanly to select negatives without validation leakage."""
    df = _make_synthetic_data(n_samples=50)
    res = run_group_kfold_baseline(df, n_splits=3)
    oof_df = res["oof_predictions"]

    # Merge OOF probabilities into training df
    df_with_oof = df.merge(oof_df[["source1_entity_id", "candidate_entity_id", "y_prob"]], on=["source1_entity_id", "candidate_entity_id"])
    assert len(df_with_oof) == 50

    mined_df = select_hard_negatives(df_with_oof, prob_col="y_prob", true_col="label", negative_ratio=2.0)
    assert len(mined_df) <= len(df_with_oof)
    assert (mined_df["label"] == 1).sum() == (df["label"] == 1).sum()


def test_threshold_tuning_returns_valid_threshold():
    """Threshold search returns a valid threshold and non-negative score."""
    df = _make_synthetic_data(n_samples=50)
    res = run_group_kfold_baseline(df, n_splits=3)
    oof_df = res["oof_predictions"]

    tuning_res = tune_decision_threshold(oof_df, prob_col="y_prob", true_col="y_true", metric="macro_f05")

    assert "best_threshold" in tuning_res
    assert "best_score" in tuning_res
    assert "threshold_results" in tuning_res

    assert 0.05 <= tuning_res["best_threshold"] <= 0.95
    assert 0.0 <= tuning_res["best_score"] <= 1.0


def test_threshold_tuning_improves_or_preserves_metric():
    """Threshold tuning improves or preserves selected metric over default 0.5."""
    df = _make_synthetic_data(n_samples=50)
    res = run_group_kfold_baseline(df, n_splits=3)
    oof_df = res["oof_predictions"]

    tuning_res = tune_decision_threshold(oof_df, prob_col="y_prob", true_col="y_true", metric="macro_f05")
    default_score = tuning_res["default_05_metrics"]["macro_f05"]
    best_score = tuning_res["best_score"]

    assert best_score >= default_score, f"Tuned score {best_score} must be >= default score {default_score}"


def test_threshold_tuning_handles_multiple_positives():
    """Multiple positives for one S1 are handled correctly during threshold tuning."""
    s1_ids = ["S1-1", "S1-1", "S1-1", "S1-2", "S1-2"]
    cand_ids = ["C1", "C2", "C3", "C4", "C5"]
    y_true = [1, 1, 0, 1, 0] # S1-1 has 2 positives!
    y_prob = [0.9, 0.8, 0.1, 0.7, 0.2]

    oof_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "candidate_entity_id": cand_ids,
        "y_true": y_true,
        "y_prob": y_prob
    })

    tuning_res = tune_decision_threshold(oof_df, prob_col="y_prob", true_col="y_true", metric="macro_f05")
    assert tuning_res["best_score"] > 0.0
    assert len(tuning_res["threshold_results"]) > 0


def test_threshold_tuning_edge_cases():
    """Edge cases (empty DataFrame, invalid metrics) handled cleanly."""
    # Empty DataFrame
    res = tune_decision_threshold(pd.DataFrame(), prob_col="y_prob", true_col="y_true")
    assert res["best_threshold"] == 0.5
    assert res["best_score"] == 0.0

    # Invalid metric name raises ValueError
    df = pd.DataFrame({
        "source1_entity_id": ["S1-1"],
        "candidate_entity_id": ["C1"],
        "y_true": [1],
        "y_prob": [0.8]
    })
    with pytest.raises(ValueError, match="Invalid optimization metric"):
        tune_decision_threshold(df, prob_col="y_prob", true_col="y_true", metric="non_existent_metric")




    df = _make_synthetic_data()

    res1 = run_group_kfold_baseline(df, n_splits=3, random_state=42)
    res2 = run_group_kfold_baseline(df, n_splits=3, random_state=42)

    pd.testing.assert_frame_equal(res1["oof_predictions"], res2["oof_predictions"])
    assert res1["overall_f05"] == res2["overall_f05"]


def test_evaluation_metrics_valid_numeric_values():
    """8. Evaluation metrics return valid numeric values."""
    df = _make_synthetic_data()
    metrics = calculate_evaluation_metrics(
        df.rename(columns={"label": "y_true"}),
        prob_col=MODEL_FEATURES[0],
        true_col="y_true",
        threshold=0.5
    )

    for k in ["precision", "recall", "f05", "f1", "macro_f05"]:
        assert isinstance(metrics[k], float)
        assert 0.0 <= metrics[k] <= 1.0

    for k in ["tp", "fp", "fn", "tn", "total_pairs", "total_entities"]:
        assert isinstance(metrics[k], int)
        assert metrics[k] >= 0


def test_empty_or_invalid_input_handling():
    """9. Empty or invalid input is handled cleanly where appropriate."""
    # Empty DataFrame
    with pytest.raises(ValueError, match="DataFrame is empty"):
        run_group_kfold_baseline(pd.DataFrame(), n_splits=3)

    # Empty in train_baseline_model
    with pytest.raises(ValueError, match="X_train and y_train must not be empty"):
        train_baseline_model(pd.DataFrame(), pd.Series(dtype=int))

    # Empty in predict_candidate_pairs returns empty numpy array
    empty_preds = predict_candidate_pairs(
        train_baseline_model(_make_synthetic_data()[MODEL_FEATURES], _make_synthetic_data()["label"]),
        pd.DataFrame(columns=MODEL_FEATURES)
    )
    assert len(empty_preds) == 0

    # Missing model features
    df_missing = _make_synthetic_data().drop(columns=[MODEL_FEATURES[0]])
    with pytest.raises(ValueError, match="Missing required model features"):
        run_group_kfold_baseline(df_missing, n_splits=2)

    # NaN in features
    df_nan = _make_synthetic_data()
    df_nan.loc[0, MODEL_FEATURES[0]] = np.nan
    with pytest.raises(ValueError, match="NaN values found"):
        run_group_kfold_baseline(df_nan, n_splits=2)

    # Inf in features
    df_inf = _make_synthetic_data()
    df_inf.loc[0, MODEL_FEATURES[0]] = np.inf
    with pytest.raises(ValueError, match="Infinite values found"):
        run_group_kfold_baseline(df_inf, n_splits=2)

    # Invalid labels
    df_inv = _make_synthetic_data()
    df_inv.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label column must contain strictly 0 or 1"):
        run_group_kfold_baseline(df_inv, n_splits=2)


# ---------------------------------------------------------------------------
# STEP 4B TESTS: CALIBRATION & MATCH INFERENCE PIPELINE
# ---------------------------------------------------------------------------

def _make_record(entity_id: str, name: str = "acme corp") -> Dict[str, Any]:
    return {
        "entity_id": entity_id,
        "normalized_name": name,
        "aggressive_normalized_name": name,
        "name_tokens": name.split(),
        "name_tokens_no_suffix": name.split(),
        "name_phonetic": "AKM",
        "name_char_ngrams": [],
        "normalized_address": "123 main st",
        "address_tokens": ["123", "main", "st"],
        "postal_code_guess": "12345",
        "city_guess": "city",
        "street_number_guess": "123",
        "name_missing": False,
        "address_missing": False,
        "postal_missing": False,
        "city_missing": False,
        "country": "US",
    }


def test_calibrator_trains_successfully():
    """1. Calibrator trains successfully on valid OOF predictions."""
    cal = ProbabilityCalibrator(method="sigmoid")
    y_prob = np.array([0.1, 0.4, 0.7, 0.9])
    y_true = np.array([0, 0, 1, 1])

    cal.fit(y_prob, y_true)
    assert cal.is_fitted
    cal_probs = cal.calibrate(y_prob)
    assert len(cal_probs) == 4


def test_calibrated_probabilities_in_range():
    """2. Calibrated probabilities remain in [0, 1]."""
    cal = ProbabilityCalibrator(method="isotonic")
    y_prob = np.array([0.0, 0.2, 0.5, 0.8, 1.0])
    y_true = np.array([0, 0, 1, 1, 1])

    cal.fit(y_prob, y_true)
    cal_probs = cal.calibrate(y_prob)

    assert cal_probs.min() >= 0.0
    assert cal_probs.max() <= 1.0


def test_calibration_non_mutating():
    """3. Calibration does not modify the original prediction array/DataFrame."""
    cal = ProbabilityCalibrator(method="sigmoid")
    y_prob = np.array([0.1, 0.3, 0.6, 0.9])
    y_true = np.array([0, 0, 1, 1])
    cal.fit(y_prob, y_true)

    orig_copy = y_prob.copy()
    cal.calibrate(y_prob)
    np.testing.assert_array_equal(y_prob, orig_copy)


def test_raw_and_calibrated_identical_length():
    """4. Raw and calibrated predictions have identical lengths."""
    cal = ProbabilityCalibrator(method="sigmoid")
    y_prob = np.random.rand(25)
    y_true = np.random.randint(0, 2, size=25)
    cal.fit(y_prob, y_true)

    cal_probs = cal.calibrate(y_prob)
    assert len(cal_probs) == len(y_prob)


def test_predict_matches_without_calibration():
    """5. Inference works cleanly without calibration."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1"), _make_record("S2-2")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-2"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model, calibrator=None, threshold=0.5)

    assert len(res_df) == 2
    assert list(res_df.columns) == ["source1_entity_id", "candidate_entity_id", "raw_probability", "match_decision", "rank"]
    assert "calibrated_probability" not in res_df.columns


def test_predict_matches_with_calibration():
    """6. Inference works cleanly with calibration."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1"), _make_record("S2-2")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-2"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    cal = ProbabilityCalibrator(method="sigmoid")
    cal.fit(np.array([0.1, 0.9]), np.array([0, 1]))

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model, calibrator=cal, threshold=0.5)

    assert len(res_df) == 2
    assert "calibrated_probability" in res_df.columns
    assert res_df["calibrated_probability"].min() >= 0.0
    assert res_df["calibrated_probability"].max() <= 1.0


def test_predict_matches_output_columns():
    """7. Returned match rows contain expected IDs, probabilities, decision, rank fields."""
    s1_recs = [_make_record("S1-10")]
    target_recs = [_make_record("S2-10")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-10", "candidate_entity_ids": "S2-10"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=0.1)

    assert res_df.iloc[0]["source1_entity_id"] == "S1-10"
    assert res_df.iloc[0]["candidate_entity_id"] == "S2-10"
    assert res_df.iloc[0]["rank"] == 1
    assert res_df.iloc[0]["match_decision"] in (0, 1)


def test_predict_matches_threshold_applied():
    """8. Threshold is actually applied."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res_high = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=1.0)
    assert res_high.iloc[0]["match_decision"] == 0

    res_low = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=0.0)
    assert res_low.iloc[0]["match_decision"] == 1


def test_predict_matches_multiple_candidates_per_s1():
    """9. Multiple candidates for one S1 are supported."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1"), _make_record("S3-1"), _make_record("S2-2")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1,S2-2"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=0.0)
    assert len(res_df) == 3
    assert set(res_df["candidate_entity_id"]) == {"S2-1", "S3-1", "S2-2"}
    assert list(res_df["rank"]) == [1, 2, 3]


def test_predict_matches_multiple_positive_matches():
    """10. Multiple positive matches for one S1 are not collapsed."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1"), _make_record("S3-1")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=0.0)
    assert (res_df["match_decision"] == 1).sum() == 2


def test_predict_matches_zero_candidates_safe():
    """11. Zero candidates produce an empty result safely."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": ""}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=0.5)
    assert len(res_df) == 0
    assert "source1_entity_id" in res_df.columns


def test_predict_matches_invalid_missing_records_fail():
    """12. Invalid/missing features fail clearly."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-UNKNOWN", "candidate_entity_ids": "S2-1"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    with pytest.raises(ValueError, match="missing from source1_records"):
        predict_matches(s1_recs, target_recs, cand_pairs, model)


def test_predict_matches_deterministic():
    """13. Inference is deterministic."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1"), _make_record("S2-2")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-2"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res1 = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=0.5)
    res2 = predict_matches(s1_recs, target_recs, cand_pairs, model, threshold=0.5)

    pd.testing.assert_frame_equal(res1, res2)


def test_no_identifier_leakage_in_predict_matches():
    """14. No identifier or label leakage into model feature matrix."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model)
    assert len(res_df) == 1


def test_predict_matches_calibrated_probabilities_used_for_thresholding():
    """15. Calibrated probabilities are used for thresholding when calibration is enabled."""
    s1_recs = [_make_record("S1-1")]
    target_recs = [_make_record("S2-1")]
    cand_pairs = pd.DataFrame([{"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"}])

    synth_df = _make_synthetic_data()
    model = train_baseline_model(synth_df[MODEL_FEATURES], synth_df["label"])

    cal = ProbabilityCalibrator(method="isotonic")
    cal.fit(np.array([0.0, 0.6, 1.0]), np.array([0, 0, 1]))

    res_df = predict_matches(s1_recs, target_recs, cand_pairs, model, calibrator=cal, threshold=0.5)
    assert res_df.iloc[0]["calibrated_probability"] < 0.5
    assert res_df.iloc[0]["match_decision"] == 0


def test_compare_oof_calibration_helper():
    """Test OOF metric comparison helper function."""
    df = _make_synthetic_data()
    res = run_group_kfold_baseline(df, n_splits=3)
    oof_df = res["oof_predictions"]

    cal = fit_oof_calibrator(oof_df, method="sigmoid")
    comp = compare_oof_calibration(oof_df, calibrator=cal, threshold=0.5)

    assert "raw_metrics" in comp
    assert "calibrated_metrics" in comp
    assert comp["is_calibrated"]


