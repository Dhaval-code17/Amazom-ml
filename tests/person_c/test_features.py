"""
Person C — test_features.py

Unit tests for build_pair_features() and build_feature_matrix().
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# sys.path: make bare imports work exactly as the rest of the project does.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.features import build_pair_features, build_feature_matrix


def _make_record(
    entity_id: str,
    normalized_name: str = "acme restaurant",
    aggressive_normalized_name: str = "acme restaurant",
    name_tokens=None,
    name_tokens_no_suffix=None,
    name_phonetic: str = "AKMS",
    name_char_ngrams=None,
    normalized_address: str = "123 main street pune",
    address_tokens=None,
    postal_code_guess: str = "411001",
    city_guess: str = "pune",
    street_number_guess: str = "123",
    name_missing=None,
    address_missing=None,
    postal_missing=None,
    city_missing=None,
    country: str = "India",
) -> Dict[str, Any]:
    """
    Return a minimal but fully valid normalized record dict matching
    Person A's normalize_record() output schema + entity_id.
    """
    if name_tokens is None:
        name_tokens = normalized_name.split() if normalized_name else []
    if name_tokens_no_suffix is None:
        name_tokens_no_suffix = aggressive_normalized_name.split() if aggressive_normalized_name else []
    if name_char_ngrams is None:
        text = f" {normalized_name} " if normalized_name else "  "
        name_char_ngrams = [text[i:i+3] for i in range(max(0, len(text) - 2))] if normalized_name else []
    if address_tokens is None:
        address_tokens = normalized_address.split() if normalized_address else []

    if name_missing is None: name_missing = not normalized_name
    if address_missing is None: address_missing = not normalized_address
    if postal_missing is None: postal_missing = not postal_code_guess
    if city_missing is None: city_missing = not city_guess

    return {
        "entity_id": entity_id,
        "normalized_name": normalized_name,
        "aggressive_normalized_name": aggressive_normalized_name,
        "name_tokens": name_tokens,
        "name_tokens_no_suffix": name_tokens_no_suffix,
        "name_phonetic": name_phonetic,
        "name_char_ngrams": name_char_ngrams,
        "normalized_address": normalized_address,
        "address_tokens": address_tokens,
        "postal_code_guess": postal_code_guess,
        "city_guess": city_guess,
        "street_number_guess": street_number_guess,
        "name_missing": name_missing,
        "address_missing": address_missing,
        "postal_missing": postal_missing,
        "city_missing": city_missing,
        "country": country,
    }


def test_identical_records():
    """Identical records produce high/maximum similarity features."""
    rec1 = _make_record("S1-1")
    rec2 = _make_record("S2-1")
    f = build_pair_features(rec1, rec2)
    
    assert f["name_exact"] == 1
    assert f["name_jaccard"] == 1.0
    assert f["name_levenshtein"] == 1.0
    assert f["name_jaro_winkler"] == 1.0
    assert f["country_match"] == 1
    assert f["city_match"] == 1
    assert f["postal_match"] == 1
    assert f["street_number_match"] == 1
    assert f["name_x_address"] == 1.0

def test_different_names():
    """Completely different names produce low name similarity."""
    rec1 = _make_record("S1-1", normalized_name="acme corp", name_phonetic="AKM")
    rec2 = _make_record("S2-1", normalized_name="global tech", name_phonetic="GLB")
    f = build_pair_features(rec1, rec2)
    
    assert f["name_exact"] == 0
    assert f["name_jaccard"] == 0.0
    assert f["name_token_overlap"] == 0.0
    assert f["name_containment"] == 0
    assert f["name_levenshtein"] < 0.5
    assert f["name_phonetic_match"] == 0

def test_country_match():
    """Identical country produces country_match=1, different produces 0."""
    rec1 = _make_record("S1-1", country="India")
    rec2 = _make_record("S2-1", country="India")
    rec3 = _make_record("S2-2", country="US")
    
    assert build_pair_features(rec1, rec2)["country_match"] == 1
    assert build_pair_features(rec1, rec3)["country_match"] == 0

def test_missing_values_do_not_match():
    """None/empty vs None/empty produces 0 for equality."""
    rec1 = _make_record("S1-1", postal_code_guess="", city_guess="", country="")
    rec2 = _make_record("S2-1", postal_code_guess="", city_guess="", country="")
    
    f = build_pair_features(rec1, rec2)
    assert f["postal_match"] == 0
    assert f["city_match"] == 0
    assert f["country_match"] == 0
    assert f["postal_missing_either"] == 1
    assert f["city_missing_either"] == 1
    assert f["country_missing_either"] == 1
    
    # Missingness flags
    assert f["both_postal_missing"] == 1
    assert f["both_city_missing"] == 1

def test_missingness_indicators():
    """Missingness indicators are correct."""
    rec1 = _make_record("S1-1", normalized_name="")
    rec2 = _make_record("S2-1")
    f = build_pair_features(rec1, rec2)
    
    assert f["source_name_missing"] == 1
    assert f["candidate_name_missing"] == 0
    assert f["both_name_missing"] == 0

def test_street_number_matching():
    """Street number matching works."""
    rec1 = _make_record("S1-1", street_number_guess="42")
    rec2 = _make_record("S2-1", street_number_guess="42")
    rec3 = _make_record("S2-2", street_number_guess="99")
    
    assert build_pair_features(rec1, rec2)["street_number_match"] == 1
    assert build_pair_features(rec1, rec3)["street_number_match"] == 0

def test_output_is_deterministic():
    """Output is deterministic."""
    rec1 = _make_record("S1-1")
    rec2 = _make_record("S2-1", normalized_name="variant name")
    
    f1 = build_pair_features(rec1, rec2)
    f2 = build_pair_features(rec1, rec2)
    
    assert f1 == f2

def test_similarities_in_range():
    """All similarity features are within [0,1]."""
    rec1 = _make_record("S1-1", normalized_name="abc def")
    rec2 = _make_record("S2-1", normalized_name="def ghi")
    f = build_pair_features(rec1, rec2)
    
    sim_keys = [k for k in f if 'jaccard' in k or 'levenshtein' in k or 'jaro' in k or 'overlap' in k]
    for k in sim_keys:
        val = f[k]
        assert 0.0 <= val <= 1.0, f"{k} value {val} out of range [0, 1]"

def test_feature_names_stable():
    """Feature names are stable."""
    f = build_pair_features(_make_record("S1-1"), _make_record("S2-1"))
    expected_keys = [
        "name_exact", "aggressive_name_exact", "name_jaccard", "name_token_overlap",
        "name_containment", "name_levenshtein", "name_jaro_winkler", "name_char_ngram_similarity",
        "name_phonetic_match", "name_token_count_diff", "name_length_diff",
        "address_exact", "address_jaccard", "address_token_overlap", "address_containment",
        "address_levenshtein", "address_jaro_winkler", "address_length_diff",
        "address_token_count_diff", "street_number_match",
        "country_match", "city_match", "postal_match"
    ]
    for k in expected_keys:
        assert k in f, f"Missing expected key {k}"

def test_feature_matrix():
    """Verify matrix properties: numeric, no nan/inf."""
    f1 = build_pair_features(_make_record("S1-1"), _make_record("S2-1"))
    f2 = build_pair_features(_make_record("S1-1", normalized_name=""), _make_record("S2-2", normalized_name="a"))
    
    df = build_feature_matrix([f1, f2])
    
    # Identifiers present
    assert "source1_entity_id" in df.columns
    assert "candidate_entity_id" in df.columns
    
    # Check no NaN
    assert not df.isnull().values.any()
    
    # Check no Inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    assert not np.isinf(df[numeric_cols].values).any()
    
    # All columns except identifiers should be numeric
    non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
    assert set(non_numeric_cols) == {"source1_entity_id", "candidate_entity_id"}
