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

from person_c.training_data import build_training_dataset


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
