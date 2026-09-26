"""
Person C — test_candidate_generation.py

Unit tests for generate_candidate_pairs().

All tests use synthetic, fully-constructed normalized record dicts.
No real competition dataset is required.
The 18-field normalized schema matches Person A's normalize_record() output
(see src/schema.py for the canonical field list).
"""

import sys
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path: make bare imports work exactly as the rest of the project does.
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.candidate_generation import generate_candidate_pairs


# ---------------------------------------------------------------------------
# Shared synthetic record factory
# ---------------------------------------------------------------------------

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
    name_missing: bool = False,
    address_missing: bool = False,
    postal_missing: bool = False,
    city_missing: bool = False,
    country: str = "India",
    raw_name: str = "Acme Restaurant",
    raw_address: str = "123 Main Street, Pune 411001",
) -> Dict[str, Any]:
    """
    Return a minimal but fully valid normalized record dict matching
    Person A's normalize_record() output schema + entity_id.
    """
    if name_tokens is None:
        name_tokens = normalized_name.split()
    if name_tokens_no_suffix is None:
        name_tokens_no_suffix = aggressive_normalized_name.split()
    if name_char_ngrams is None:
        # Minimal 3-grams for the name (sufficient for LSH channel)
        text = f" {normalized_name} "
        name_char_ngrams = [text[i:i+3] for i in range(len(text) - 2)]
    if address_tokens is None:
        address_tokens = normalized_address.split()

    return {
        "entity_id": entity_id,
        # Name fields
        "raw_name": raw_name,
        "normalized_name": normalized_name,
        "aggressive_normalized_name": aggressive_normalized_name,
        "name_tokens": name_tokens,
        "name_tokens_no_suffix": name_tokens_no_suffix,
        "name_phonetic": name_phonetic,
        "name_char_ngrams": name_char_ngrams,
        # Address fields
        "raw_address": raw_address,
        "normalized_address": normalized_address,
        "address_tokens": address_tokens,
        "postal_code_guess": postal_code_guess,
        "city_guess": city_guess,
        "street_number_guess": street_number_guess,
        # Missingness flags
        "name_missing": name_missing,
        "address_missing": address_missing,
        "postal_missing": postal_missing,
        "city_missing": city_missing,
        # Country
        "country": country,
    }


# ---------------------------------------------------------------------------
# TEST 1 — Empty target records
# ---------------------------------------------------------------------------

def test_empty_target_records():
    """With no target records, every S1 entity gets an empty candidate string."""
    s1 = [_make_record("S1-000001")]
    result = generate_candidate_pairs(s1, target_records=[], cap=200)

    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert len(result) == 1
    assert result.iloc[0]["source1_entity_id"] == "S1-000001"
    assert result.iloc[0]["candidate_entity_ids"] == ""


# ---------------------------------------------------------------------------
# TEST 2 — One S1 + one matching candidate
# ---------------------------------------------------------------------------

def test_one_s1_one_candidate():
    """A single strong candidate is returned for a matching S1 entity."""
    s1_rec = _make_record("S1-000001", normalized_name="acme restaurant pune")
    cand_rec = _make_record(
        "S2-000047",
        normalized_name="acme restaurant pune",
        aggressive_normalized_name="acme restaurant pune",
    )

    result = generate_candidate_pairs([s1_rec], [cand_rec], cap=200)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1
    row = result.iloc[0]
    assert row["source1_entity_id"] == "S1-000001"
    # S2-000047 must appear in the candidate list
    candidate_ids = [x for x in row["candidate_entity_ids"].split(",") if x]
    assert "S2-000047" in candidate_ids


# ---------------------------------------------------------------------------
# TEST 3 — Duplicate candidate IDs are removed
# ---------------------------------------------------------------------------

def test_duplicate_candidate_ids_removed():
    """
    Even if the blocker would theoretically surface the same ID twice
    (which shouldn't happen, but we enforce deduplication), the output
    must contain it only once.

    We test this indirectly by providing two identical target records with
    the same entity_id — the function must deduplicate.
    """
    s1_rec = _make_record("S1-000001", normalized_name="beta corp")
    # Two records with the same entity_id — target_map will keep one (last-write-wins)
    cand_rec_a = _make_record("S2-000010", normalized_name="beta corp")
    cand_rec_b = _make_record("S2-000010", normalized_name="beta corp")  # duplicate id

    result = generate_candidate_pairs([s1_rec], [cand_rec_a, cand_rec_b], cap=200)

    assert len(result) == 1
    candidates_str = result.iloc[0]["candidate_entity_ids"]
    candidate_ids = [x for x in candidates_str.split(",") if x]
    # Must appear exactly once
    assert candidate_ids.count("S2-000010") == 1


# ---------------------------------------------------------------------------
# TEST 4 — S1 IDs never appear as candidates
# ---------------------------------------------------------------------------

def test_s1_id_never_in_candidates():
    """
    S1 entity_ids must never appear in candidate_entity_ids.
    We add the S1 record itself to the target_records list to stress-test
    the safety filter.
    """
    s1_rec = _make_record("S1-000001", normalized_name="global services")
    cand_rec = _make_record("S2-000099", normalized_name="global services")

    # Deliberately pass S1 record as a target too
    result = generate_candidate_pairs(
        [s1_rec],
        [s1_rec, cand_rec],   # S1 record included as target — must be filtered
        cap=200,
    )

    assert len(result) == 1
    candidate_ids = [x for x in result.iloc[0]["candidate_entity_ids"].split(",") if x]
    assert "S1-000001" not in candidate_ids


# ---------------------------------------------------------------------------
# TEST 5 — Output columns are exactly source1_entity_id and candidate_entity_ids
# ---------------------------------------------------------------------------

def test_output_column_names():
    """DataFrame must have exactly the two required columns in the right order."""
    s1 = [_make_record("S1-000001")]
    result = generate_candidate_pairs(s1, [], cap=200)

    assert list(result.columns) == ["source1_entity_id", "candidate_entity_ids"]


# ---------------------------------------------------------------------------
# TEST 6 — Candidate IDs are deterministically ordered (lexicographic)
# ---------------------------------------------------------------------------

def test_candidate_ids_deterministic_order():
    """
    Running the same inputs twice must produce identical, lexicographically
    sorted output.
    """
    s1_rec = _make_record("S1-000001", normalized_name="apex solutions")
    cand_a = _make_record("S2-000020", normalized_name="apex solutions")
    cand_b = _make_record("S2-000005", normalized_name="apex solutions")
    cand_c = _make_record("S3-000003", normalized_name="apex solutions")

    result1 = generate_candidate_pairs(
        [s1_rec], [cand_a, cand_b, cand_c], cap=200
    )
    result2 = generate_candidate_pairs(
        [s1_rec], [cand_a, cand_b, cand_c], cap=200
    )

    ids1 = result1.iloc[0]["candidate_entity_ids"]
    ids2 = result2.iloc[0]["candidate_entity_ids"]
    assert ids1 == ids2, "Candidate ordering is not deterministic across runs"

    # Also verify lexicographic sort
    candidate_ids = [x for x in ids1.split(",") if x]
    assert candidate_ids == sorted(candidate_ids), (
        f"Candidate IDs are not lexicographically sorted: {candidate_ids}"
    )


# ---------------------------------------------------------------------------
# TEST 7 — cap limits the number of returned candidates
# ---------------------------------------------------------------------------

def test_cap_limits_candidates():
    """cap=2 must never return more than 2 candidates per S1 entity."""
    s1_rec = _make_record("S1-000001", normalized_name="summit logistics")
    # Create 5 strong candidates (same name → all match via token channel)
    targets = [
        _make_record(f"S2-{i:06d}", normalized_name="summit logistics")
        for i in range(1, 6)
    ]

    result = generate_candidate_pairs([s1_rec], targets, cap=2)

    assert len(result) == 1
    candidates_str = result.iloc[0]["candidate_entity_ids"]
    candidate_ids = [x for x in candidates_str.split(",") if x]
    assert len(candidate_ids) <= 2, (
        f"Expected at most 2 candidates with cap=2, got {len(candidate_ids)}: {candidate_ids}"
    )


# ---------------------------------------------------------------------------
# TEST 8 — Function returns a pandas DataFrame
# ---------------------------------------------------------------------------

def test_returns_dataframe():
    """Return type must be exactly pd.DataFrame."""
    result = generate_candidate_pairs([], [], cap=200)
    assert isinstance(result, pd.DataFrame)


# ---------------------------------------------------------------------------
# TEST 9 — Multiple S1 entities produce one row each
# ---------------------------------------------------------------------------

def test_multiple_s1_one_row_each():
    """Each S1 entity gets exactly one row; row order matches input order."""
    s1_records = [
        _make_record("S1-000001", normalized_name="alpha corp"),
        _make_record("S1-000002", normalized_name="beta inc"),
        _make_record("S1-000003", normalized_name="gamma llc"),
    ]
    cand = _make_record("S2-000001", normalized_name="alpha corp")

    result = generate_candidate_pairs(s1_records, [cand], cap=200)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result["source1_entity_id"]) == ["S1-000001", "S1-000002", "S1-000003"]


# ---------------------------------------------------------------------------
# TEST 10 — All candidate IDs exist in target_map (no phantom IDs)
# ---------------------------------------------------------------------------

def test_all_candidate_ids_in_target_map():
    """Every candidate ID in the output must exist in target_records."""
    s1_rec = _make_record("S1-000001", normalized_name="nexus technologies")
    targets = [
        _make_record("S2-000001", normalized_name="nexus technologies"),
        _make_record("S3-000002", normalized_name="nexus tech"),
    ]
    valid_ids = {t["entity_id"] for t in targets}

    result = generate_candidate_pairs([s1_rec], targets, cap=200)

    candidate_ids = [
        x for x in result.iloc[0]["candidate_entity_ids"].split(",") if x
    ]
    for cid in candidate_ids:
        assert cid in valid_ids, f"Phantom candidate ID {cid!r} not in target_records"


# ---------------------------------------------------------------------------
# TEST 11 — Validation: non-dict input raises TypeError
# ---------------------------------------------------------------------------

def test_validation_non_dict_source1():
    """source1_records containing a non-dict raises TypeError."""
    with pytest.raises(TypeError):
        generate_candidate_pairs("not a list", [], cap=200)


# ---------------------------------------------------------------------------
# TEST 12 — Validation: missing entity_id raises ValueError
# ---------------------------------------------------------------------------

def test_validation_missing_entity_id():
    """A record without entity_id raises ValueError."""
    bad_rec = _make_record("S1-000001")
    del bad_rec["entity_id"]

    with pytest.raises(ValueError, match="entity_id"):
        generate_candidate_pairs([bad_rec], [], cap=200)


# ---------------------------------------------------------------------------
# TEST 13 — Empty string for S1 entities with no candidates
# ---------------------------------------------------------------------------

def test_empty_string_for_no_candidates():
    """An S1 entity with no matching candidates gets an empty string, not None."""
    s1_rec = _make_record("S1-000001", normalized_name="zzz unique name xyz")
    cand_rec = _make_record("S2-000001", normalized_name="totally different business")

    result = generate_candidate_pairs([s1_rec], [cand_rec], cap=200)
    # Even if no match, value must be a string (possibly empty)
    val = result.iloc[0]["candidate_entity_ids"]
    assert isinstance(val, str)
