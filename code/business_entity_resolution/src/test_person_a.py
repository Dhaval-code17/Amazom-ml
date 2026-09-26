import pytest
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

# Add src to path
src_dir = Path(__file__).resolve().parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from normalization import normalize_record
from golden_match import golden_match
from scoring import score_macro_f_beta
from validation import get_group_kfold_splits, verify_no_group_leakage

# =====================================================================
# 1. NORMALIZATION TESTS
# =====================================================================

def test_normalization_lowercase_and_unidecode():
    res = normalize_record("Café Résumé INC.", "123 Main St.", "US")
    assert res["raw_name"] == "Café Résumé INC."
    assert res["normalized_name"] == "cafe resume inc"
    assert res["aggressive_normalized_name"] == "cafe resume"
    assert res["country"] == "US"

def test_normalization_ampersand_and_punctuation():
    res = normalize_record("A & B -- Corp.", "456 5th Ave., Suite #10", "India")
    assert res["normalized_name"] == "a and b corp"
    assert res["aggressive_normalized_name"] == "a and b"
    assert "road" not in res["normalized_address"]
    assert "avenue" in res["normalized_address"]

def test_normalization_legal_suffixes():
    suffixes = ["Corp", "Corporation", "Inc", "Incorporated", "Ltd", "Limited", "Pvt", "Private", "LLC", "LLP", "Sarl", "Sas", "Co"]
    for suf in suffixes:
        res = normalize_record(f"Acme {suf}", "101 High St", "US")
        assert res["aggressive_normalized_name"] == "acme", f"Failed suffix removal for {suf}"
        assert res["name_tokens_no_suffix"] == ["acme"]

def test_normalization_postal_and_street_extraction():
    # US Postal
    res1 = normalize_record("Acme", "123 Main St, New York, NY 10001-1234", "US")
    assert res1["postal_code_guess"] == "10001"
    assert res1["street_number_guess"] == "123"

    # Indian Postal
    res2 = normalize_record("Tata", "MG Road, Bangalore 560001", "India")
    assert res2["postal_code_guess"] == "560001"
    assert res2["street_number_guess"] == ""

    # French Postal
    res3 = normalize_record("Total", "10 Rue de la Paix, Paris 75002", "France")
    assert res3["postal_code_guess"] == "75002"
    assert res3["street_number_guess"] == "10"
    assert res3["country"] == "France"

def test_normalization_missing_flags():
    res = normalize_record("", None, "US")
    assert res["name_missing"] is True
    assert res["address_missing"] is True
    assert res["postal_missing"] is True
    assert res["city_missing"] is True

def test_normalization_ngrams_and_phonetic():
    res = normalize_record("Google", "1600 Amphitheatre Pkwy", "US")
    assert len(res["name_char_ngrams"]) > 0
    assert isinstance(res["name_phonetic"], str)
    assert len(res["name_phonetic"]) > 0

# =====================================================================
# 2. GOLDEN MATCH TESTS
# =====================================================================

def test_golden_match_rule1_exact_name_address_country():
    rec1 = normalize_record("Acme Corp", "100 Main St", "US")
    rec2 = normalize_record("Acme Inc", "100 Main Street", "US")
    assert golden_match(rec1, rec2) is True

def test_golden_match_rule2_exact_name_postal():
    rec1 = normalize_record("Acme Corp", "100 Main St, 10001", "US")
    rec2 = normalize_record("Acme Inc", "200 Broadway, 10001", "US")
    assert golden_match(rec1, rec2) is True

def test_golden_match_name_only_fails():
    rec1 = normalize_record("Acme Corp", "100 Main St", "US")
    rec2 = normalize_record("Acme Inc", "200 Broadway", "India")
    assert golden_match(rec1, rec2) is False

def test_golden_match_rule1_different_country_fails():
    rec1 = normalize_record("Acme Corp", "100 Main Street", "US")
    rec2 = normalize_record("Acme Inc", "100 Main Street", "France")
    assert golden_match(rec1, rec2) is False

def test_golden_match_missing_postal_cannot_trigger_rule2():
    rec1 = normalize_record("Acme Corp", "Main Street", "US")
    rec2 = normalize_record("Acme Inc", "Second Street", "US")
    assert golden_match(rec1, rec2) is False

def test_golden_match_different_name_fails():
    rec1 = normalize_record("Acme Corp", "100 Main St, 10001", "US")
    rec2 = normalize_record("Beta Corp", "100 Main St, 10001", "US")
    assert golden_match(rec1, rec2) is False

# =====================================================================
# 3. SCORER TESTS
# =====================================================================

def test_scorer_perfect_match():
    gt = {"S1-1": {"S2-100"}, "S1-2": {"S3-200"}}
    pred = {"S1-1": {"S2-100"}, "S1-2": {"S3-200"}}
    assert score_macro_f_beta(pred, gt) == 1.0

def test_scorer_actual_empty_pred_empty():
    gt = {"S1-1": set()}
    pred = {"S1-1": set()}
    assert score_macro_f_beta(pred, gt) == 1.0

def test_scorer_actual_empty_pred_nonempty():
    gt = {"S1-1": set()}
    pred = {"S1-1": {"S2-100"}}
    assert score_macro_f_beta(pred, gt) == 0.0

def test_scorer_partial_match():
    gt = {"S1-1": {"S2-1", "S2-2"}}
    pred = {"S1-1": {"S2-1"}}
    score = score_macro_f_beta(pred, gt)
    assert abs(score - (0.625 / 0.75)) < 1e-6

def test_scorer_wrong_prediction():
    gt = {"S1-1": {"S2-100"}}
    pred = {"S1-1": {"S3-999"}}
    assert score_macro_f_beta(pred, gt) == 0.0

def test_scorer_macro_averaging():
    gt = {
        "S1-1": set(),
        "S1-2": set(),
        "S1-3": {"S2-100"}
    }
    pred = {
        "S1-1": set(),
        "S1-2": {"S2-50"},
        "S1-3": {"S2-100"}
    }
    score = score_macro_f_beta(pred, gt)
    assert abs(score - (2.0 / 3.0)) < 1e-6

# =====================================================================
# 4. GROUPKFOLD LEAKAGE TESTS
# =====================================================================

def test_group_kfold_no_leakage():
    df = pd.DataFrame({
        "source1_entity_id": ["S1-1", "S1-1", "S1-2", "S1-2", "S1-3", "S1-3", "S1-4", "S1-5"],
        "val": range(8)
    })
    splits = list(get_group_kfold_splits(df, group_col="source1_entity_id", n_splits=3))
    assert len(splits) == 3

    for train_idx, val_idx in splits:
        assert verify_no_group_leakage(df, train_idx, val_idx, group_col="source1_entity_id") is True

# =====================================================================
# 5. REGRESSION & SPECIFICATION VERIFICATION TESTS (A - G)
# =====================================================================

def test_regression_case_a():
    # Same aggressive name + same address + same country -> MATCH
    r1 = normalize_record("Acme Corp", "100 Main St", "US")
    r2 = normalize_record("Acme Inc", "100 Main St", "US")
    assert golden_match(r1, r2) is True

def test_regression_case_b():
    # Same aggressive name + same postal code + same country -> MATCH
    r1 = normalize_record("Acme Corp", "100 Main St, 10001", "US")
    r2 = normalize_record("Acme Inc", "200 Broadway, 10001", "US")
    assert golden_match(r1, r2) is True

def test_regression_case_c():
    # Same name but different address/country -> NOT MATCH
    r1 = normalize_record("Acme Corp", "100 Main St", "US")
    r2 = normalize_record("Acme Inc", "200 Broadway", "India")
    assert golden_match(r1, r2) is False

def test_regression_case_d():
    # Same name only -> NOT MATCH
    r1 = normalize_record("Acme Corp", "100 Main St", "US")
    r2 = normalize_record("Acme Inc", "200 Broadway", "US")
    assert golden_match(r1, r2) is False

def test_regression_case_e():
    # Same postal code only -> NOT MATCH
    r1 = normalize_record("Alpha Corp", "100 Main St, 10001", "US")
    r2 = normalize_record("Beta Corp", "200 Broadway, 10001", "US")
    assert golden_match(r1, r2) is False

def test_regression_case_f():
    # Golden match Rule 2: same aggressive name + same postal code but DIFFERENT country -> FALSE
    r1 = normalize_record("Acme Corp", "100 Main St, 10001", "US")
    r2 = normalize_record("Acme Inc", "200 Broadway, 10001", "India")
    assert golden_match(r1, r2) is False

def test_regression_case_g_city_guess():
    # Refined city_guess extraction
    r1 = normalize_record("Test", "1795 Westchester Drive, High Point, NC", "US")
    assert r1["city_guess"] == "high point"

    r2 = normalize_record("Test", "2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD", "US")
    assert r2["city_guess"] == "dundalk"

    r3 = normalize_record("Test", "2505, Tower 1, Oakwood, Runwal Greens, Mulund Goreagon Link Road, Near Fortis Hospital, Bhandup West, Mumbai, Maharashtra", "India")
    assert r3["city_guess"] == "mumbai"

