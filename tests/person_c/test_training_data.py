"""
Person C — test_training_data.py

Unit tests for build_training_dataset().
"""

import sys
import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.training_data import build_training_dataset, extract_X_y_metadata


def _make_record(entity_id: str) -> Dict[str, Any]:
    return {
        "entity_id": entity_id,
        "normalized_name": "acme corp",
        "aggressive_normalized_name": "acme corp",
        "name_tokens": ["acme", "corp"],
        "name_tokens_no_suffix": ["acme"],
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


@pytest.fixture
def temp_gt_file():
    """Creates a temporary ground truth file."""
    fd, path = tempfile.mkstemp(suffix=".tsv")
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        f.write("S1-1\tS2-1,S3-1\n")
        f.write("S1-2\tS2-2\n")
        f.write("S1-3\t\n")  # No match
    yield path
    os.remove(path)


def test_positive_and_negative_candidates(temp_gt_file):
    """Test standard case with both 1s and 0s and one-to-many matches."""
    s1_records = [_make_record("S1-1"), _make_record("S1-2")]
    target_records = [
        _make_record("S2-1"), 
        _make_record("S3-1"), 
        _make_record("S2-X"), # Negative
        _make_record("S2-2")
    ]
    
    # S1-1 candidates: S2-1 (pos), S3-1 (pos), S2-X (neg)
    # S1-2 candidates: S2-2 (pos)
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S3-1,S2-X"},
        {"source1_entity_id": "S1-2", "candidate_entity_ids": "S2-2"},
    ])
    
    df = build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)
    
    assert len(df) == 4
    assert list(df.columns) == ["source1_entity_id", "candidate_entity_id"] + [c for c in df.columns if c not in ["source1_entity_id", "candidate_entity_id", "label"]] + ["label"]
    
    # Check labels
    row1 = df[(df["source1_entity_id"] == "S1-1") & (df["candidate_entity_id"] == "S2-1")].iloc[0]
    assert row1["label"] == 1
    
    row2 = df[(df["source1_entity_id"] == "S1-1") & (df["candidate_entity_id"] == "S3-1")].iloc[0]
    assert row2["label"] == 1
    
    row3 = df[(df["source1_entity_id"] == "S1-1") & (df["candidate_entity_id"] == "S2-X")].iloc[0]
    assert row3["label"] == 0
    
    row4 = df[(df["source1_entity_id"] == "S1-2") & (df["candidate_entity_id"] == "S2-2")].iloc[0]
    assert row4["label"] == 1


def test_duplicate_candidate_rejection(temp_gt_file):
    """Raise ValueError if the candidate pairs contain duplicates."""
    s1_records = [_make_record("S1-1")]
    target_records = [_make_record("S2-1"), _make_record("S2-X")]
    
    # S2-1 appears twice
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-1,S2-X"}
    ])
    
    with pytest.raises(ValueError, match="Duplicate candidate pair"):
        build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)


def test_missing_s1_id_rejection(temp_gt_file):
    """Raise ValueError if S1 ID is missing from source1_records."""
    target_records = [_make_record("S2-1"), _make_record("S2-X")]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-X"}
    ])
    
    with pytest.raises(ValueError, match="not in source1_records"):
        build_training_dataset([], target_records, cand_pairs, temp_gt_file)


def test_missing_candidate_id_rejection(temp_gt_file):
    """Raise ValueError if candidate ID is missing from target_records."""
    s1_records = [_make_record("S1-1")]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"}
    ])
    
    with pytest.raises(ValueError, match="not in target_records"):
        build_training_dataset(s1_records, [], cand_pairs, temp_gt_file)


def test_s1_id_as_candidate_rejection(temp_gt_file):
    """Raise ValueError if S1 ID appears as a candidate."""
    s1_records = [_make_record("S1-1")]
    target_records = [_make_record("S1-1")]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S1-1"}
    ])
    
    with pytest.raises(ValueError, match="is listed as its own candidate"):
        build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)


def test_no_positive_label_rejection(temp_gt_file):
    """Raise ValueError if dataset has only negative labels."""
    s1_records = [_make_record("S1-3")] # S1-3 has no match in GT
    target_records = [_make_record("S2-X")]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-3", "candidate_entity_ids": "S2-X"}
    ])
    
    with pytest.raises(ValueError, match="positive label must exist"):
        build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)


def test_no_negative_label_rejection(temp_gt_file):
    """Raise ValueError if dataset has only positive labels."""
    s1_records = [_make_record("S1-1")]
    target_records = [_make_record("S2-1")]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1"}
    ])
    
    with pytest.raises(ValueError, match="negative label must exist"):
        build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)


def test_dict_ground_truth_and_extract_x_y_metadata():
    """Test with in-memory dict ground truth and extract_X_y_metadata helper."""
    gt_dict = {
        "S1-100": {"S2-100", "S3-100"},
        "S1-200": set()  # Empty matched_entity_ids
    }
    s1_records = [_make_record("S1-100"), _make_record("S1-200")]
    target_records = [
        _make_record("S2-100"),
        _make_record("S3-100"),
        _make_record("S2-999"),  # Negative candidate for S1-100
        _make_record("S2-888"),  # Negative candidate for S1-200
    ]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-100", "candidate_entity_ids": "S2-100,S3-100,S2-999"},
        {"source1_entity_id": "S1-200", "candidate_entity_ids": "S2-888"},
    ])

    df = build_training_dataset(s1_records, target_records, cand_pairs, gt_dict)
    assert len(df) == 4

    X, y, metadata = extract_X_y_metadata(df)

    # 1. Exact 45-feature X schema
    from person_c.model import MODEL_FEATURES
    assert list(X.columns) == MODEL_FEATURES
    assert len(X.columns) == 45

    # 2. Identifiers and labels excluded from X
    assert "source1_entity_id" not in X.columns
    assert "candidate_entity_id" not in X.columns
    assert "label" not in X.columns

    # 3. Label correctness
    assert len(y) == 4
    # S1-100 vs S2-100 -> 1
    # S1-100 vs S3-100 -> 1
    # S1-100 vs S2-999 -> 0
    # S1-200 vs S2-888 -> 0 (empty matched_entity_ids in GT)
    assert df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S2-100")]["label"].values[0] == 1
    assert df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S3-100")]["label"].values[0] == 1
    assert df[(df["source1_entity_id"] == "S1-100") & (df["candidate_entity_id"] == "S2-999")]["label"].values[0] == 0
    assert df[(df["source1_entity_id"] == "S1-200") & (df["candidate_entity_id"] == "S2-888")]["label"].values[0] == 0

    # 4. Metadata columns
    assert list(metadata.columns) == ["source1_entity_id", "candidate_entity_id"]
    assert len(metadata) == 4


def test_deterministic_output(temp_gt_file):
    """Calling build_training_dataset twice produces identical DataFrames."""
    s1_records = [_make_record("S1-1")]
    target_records = [_make_record("S2-1"), _make_record("S2-X")]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-X"}
    ])

    df1 = build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)
    df2 = build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)

    pd.testing.assert_frame_equal(df1, df2)


def test_no_accidental_mutation_of_input_data(temp_gt_file):
    """Input data structures are not mutated during dataset construction."""
    s1_rec = _make_record("S1-1")
    t1_rec = _make_record("S2-1")
    t2_rec = _make_record("S2-X")
    
    s1_rec_copy = dict(s1_rec)
    t1_rec_copy = dict(t1_rec)
    t2_rec_copy = dict(t2_rec)

    s1_records = [s1_rec]
    target_records = [t1_rec, t2_rec]
    cand_pairs = pd.DataFrame([
        {"source1_entity_id": "S1-1", "candidate_entity_ids": "S2-1,S2-X"}
    ])
    cand_pairs_copy = cand_pairs.copy()

    build_training_dataset(s1_records, target_records, cand_pairs, temp_gt_file)

    assert s1_rec == s1_rec_copy
    assert t1_rec == t1_rec_copy
    assert t2_rec == t2_rec_copy
    pd.testing.assert_frame_equal(cand_pairs, cand_pairs_copy)

