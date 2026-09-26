"""
Person C — test_model.py

Unit tests for the baseline LightGBM model layer.
"""

import sys
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.model import MODEL_FEATURES, run_group_kfold_baseline


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
    # and we get positive & negative labels
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
    """LightGBM can train on a synthetic dataset and produce valid output."""
    df = _make_synthetic_data()
    
    results = run_group_kfold_baseline(df, n_splits=3)
    
    # Verify return structure
    assert "fold_metrics" in results
    assert "oof_predictions" in results
    assert "overall_f05" in results
    assert "model_feature_names" in results
    
    # Verify predictions dataframe
    oof = results["oof_predictions"]
    assert len(oof) == 50
    assert "y_true" in oof.columns
    assert "y_prob" in oof.columns
    assert "fold" in oof.columns
    
    # Probabilities in [0, 1]
    assert oof["y_prob"].min() >= 0.0
    assert oof["y_prob"].max() <= 1.0


def test_group_leakage_assertion():
    """GroupKFold behaves correctly and does not leak groups."""
    df = _make_synthetic_data()
    
    # Deliberately cause leakage by duplicating an S1 ID into a different fold context?
    # Actually, GroupKFold guarantees no leakage. We can just verify the folds created
    # by checking the fold_metrics or the OOF.
    results = run_group_kfold_baseline(df, n_splits=3)
    
    oof = results["oof_predictions"]
    
    # Check that each S1 ID exists in exactly one fold
    fold_map = oof.groupby("source1_entity_id")["fold"].nunique()
    assert (fold_map == 1).all(), "An S1 ID was split across multiple folds!"


def test_missing_features_raise_error():
    """Missing model features raise ValueError."""
    df = _make_synthetic_data()
    df = df.drop(columns=[MODEL_FEATURES[0]])
    
    with pytest.raises(ValueError, match="Missing required model features"):
        run_group_kfold_baseline(df, n_splits=2)


def test_nan_values_raise_error():
    """NaN values in features raise ValueError."""
    df = _make_synthetic_data()
    df.loc[0, MODEL_FEATURES[0]] = np.nan
    
    with pytest.raises(ValueError, match="NaN values found"):
        run_group_kfold_baseline(df, n_splits=2)


def test_inf_values_raise_error():
    """Infinite values in features raise ValueError."""
    df = _make_synthetic_data()
    df.loc[0, MODEL_FEATURES[0]] = np.inf
    
    with pytest.raises(ValueError, match="Infinite values found"):
        run_group_kfold_baseline(df, n_splits=2)


def test_invalid_labels_raise_error():
    """Labels other than 0 or 1 raise ValueError."""
    df = _make_synthetic_data()
    df.loc[0, "label"] = 2
    
    with pytest.raises(ValueError, match="label column must contain strictly 0 or 1"):
        run_group_kfold_baseline(df, n_splits=2)
