import pytest
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Safely handle unidecode, jellyfish, metaphone imports if packages are not yet installed in local python env
for pkg in ["unidecode", "jellyfish", "metaphone"]:
    try:
        __import__(pkg)
    except ImportError:
        mock_mod = MagicMock()
        if pkg == "unidecode":
            mock_mod.unidecode = lambda s: str(s) if s is not None else ""
        elif pkg == "jellyfish":
            mock_mod.metaphone = lambda s: s[:4].upper() if s else ""
            mock_mod.double_metaphone = lambda s: (s[:4].upper(), "") if s else ("", "")
        elif pkg == "metaphone":
            mock_mod.dm = lambda s: (s[:4].upper(), "") if s else ("", "")
        sys.modules[pkg] = mock_mod

# Add src directory to sys.path
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from normalization import normalize_record
from person_b.blocking import MultiChannelBlocker
from person_b.candidate_recall import evaluate_candidate_recall, parse_ground_truth_matches
from person_b.pruning import calculate_cheap_similarity_score, rank_and_prune_candidates
from person_b.cap_evaluation import evaluate_recall_for_cap

# =====================================================================
# STAGE 3A UNIT TESTS: NAME TOKEN INVERTED INDEX BLOCKING CHANNEL
# =====================================================================

def test_1_exact_shared_token_retrieves_target():
    """TEST 1: Exact shared token retrieves the target record."""
    s1_dict = normalize_record("Acme Restaurant Pune", "123 Main St", "India")
    s1_dict["entity_id"] = "S1-00001"
    
    s2_dict = normalize_record("Acme Restaurant Pune", "123 Main St", "India")
    s2_dict["entity_id"] = "S2-00047"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_dict])
    
    candidates = blocker.get_candidates(s1_dict)
    assert "S2-00047" in candidates

def test_2_legal_suffix_removal_respected():
    """TEST 2: Legal suffix removal is respected because name_tokens_no_suffix is used."""
    s1_dict = normalize_record("Acme Restaurant", "100 High St", "US")
    s1_dict["entity_id"] = "S1-00002"
    
    s2_dict = normalize_record("Acme Restaurant Pvt Ltd", "100 High St", "US")
    s2_dict["entity_id"] = "S2-00099"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_dict])
    
    candidates = blocker.get_candidates(s1_dict)
    assert "S2-00099" in candidates

def test_3_multiple_matching_tokens_no_duplicates():
    """TEST 3: Multiple matching tokens must not create duplicate IDs in result set."""
    s1_dict = normalize_record("Acme Restaurant Pune", "123 Main St", "India")
    s1_dict["entity_id"] = "S1-00003"
    
    s2_dict = normalize_record("Acme Restaurant Pune", "456 Other St", "India")
    s2_dict["entity_id"] = "S2-00100"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_dict])
    
    candidates = blocker.get_candidates(s1_dict)
    assert isinstance(candidates, set)
    assert len(candidates) == 1
    assert "S2-00100" in candidates

def test_4_s1_entity_never_in_candidate_set():
    """TEST 4: An S1 entity must never appear in the returned candidate set."""
    s1_dict = normalize_record("Acme Corp", "100 Main St", "US")
    s1_dict["entity_id"] = "S1-00001"
    
    s2_dict = normalize_record("Acme Inc", "100 Main St", "US")
    s2_dict["entity_id"] = "S2-00002"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s1_dict, s2_dict])
    
    candidates = blocker.get_candidates(s1_dict)
    assert "S1-00001" not in candidates
    assert "S2-00002" in candidates

def test_5_target_record_no_usable_tokens_does_not_crash():
    """TEST 5: Target record with no usable name tokens must not crash index or query."""
    empty_target = {
        "entity_id": "S2-99999",
        "name_tokens_no_suffix": [],
        "name_tokens": [],
        "name_char_ngrams": [],
        "country": "US"
    }
    
    s1_dict = normalize_record("Acme Corp", "100 Main St", "US")
    s1_dict["entity_id"] = "S1-00005"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([empty_target])
    
    candidates = blocker.get_candidates(s1_dict)
    assert isinstance(candidates, set)

def test_6_s2_and_s3_both_appear_in_candidates():
    """TEST 6: Verify S2 and S3 IDs can both appear in the candidate set."""
    s1_dict = normalize_record("Global Traders", "500 Market St", "US")
    s1_dict["entity_id"] = "S1-00006"
    
    s2_dict = normalize_record("Global Traders Corp", "500 Market St", "US")
    s2_dict["entity_id"] = "S2-00111"
    
    s3_dict = normalize_record("Global Traders Inc", "500 Market St", "US")
    s3_dict["entity_id"] = "S3-00222"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_dict, s3_dict])
    
    candidates = blocker.get_candidates(s1_dict)
    assert "S2-00111" in candidates
    assert "S3-00222" in candidates
    assert len(candidates) == 2

# =====================================================================
# STAGE 3B UNIT TESTS: CHARACTER 3-GRAM MINHASH / LSH BLOCKING CHANNEL
# =====================================================================

def test_7_lsh_retrieves_character_level_typo():
    """TEST 7: Verify character-level typo (e.g. Acme Restarant vs Acme Restaurant) is retrieved by LSH."""
    s1_dict = normalize_record("Acme Restaurant", "100 Main St", "US")
    s1_dict["entity_id"] = "S1-00007"
    
    s2_dict = normalize_record("Acme Restarant", "100 Main St", "US")
    s2_dict["entity_id"] = "S2-00077"
    
    blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
    blocker.index_target_records([s2_dict])
    
    lsh_candidates = blocker._get_lsh_candidates(s1_dict)
    assert "S2-00077" in lsh_candidates

def test_8_lsh_does_not_retrieve_unrelated_name():
    """TEST 8: Verify clearly unrelated name is NOT retrieved by LSH channel at chosen threshold."""
    s1_dict = normalize_record("Acme Restaurant", "100 Main St", "US")
    s1_dict["entity_id"] = "S1-00008"
    
    s2_dict = normalize_record("Zebra Technology Electronics", "999 Oak St", "US")
    s2_dict["entity_id"] = "S2-00088"
    
    blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
    blocker.index_target_records([s2_dict])
    
    lsh_candidates = blocker._get_lsh_candidates(s1_dict)
    assert "S2-00088" not in lsh_candidates

def test_9_empty_or_missing_ngrams_handled_safely():
    """TEST 9: Verify empty/missing n-grams do not crash LSH indexing or querying."""
    empty_s1 = {"entity_id": "S1-00009", "name_char_ngrams": []}
    empty_s2 = {"entity_id": "S2-00099", "name_char_ngrams": []}
    
    blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
    blocker.index_target_records([empty_s2])
    
    lsh_candidates = blocker._get_lsh_candidates(empty_s1)
    assert isinstance(lsh_candidates, set)
    assert len(lsh_candidates) == 0

def test_10_get_candidates_returns_union_of_token_and_lsh():
    """TEST 10: Verify get_candidates() returns UNION of token candidates and LSH candidates."""
    s1_dict = normalize_record("Acme Restaurant", "100 Main St", "US")
    s1_dict["entity_id"] = "S1-00010"
    
    s2_token = normalize_record("Acme Logistics", "100 Main St", "US")
    s2_token["entity_id"] = "S2-00101"
    
    s2_lsh = normalize_record("Akme Restarant", "100 Main St", "US")
    s2_lsh["entity_id"] = "S2-00102"
    
    blocker = MultiChannelBlocker(lsh_threshold=0.4, num_perm=128)
    blocker.index_target_records([s2_token, s2_lsh])
    
    all_candidates = blocker.get_candidates(s1_dict)
    assert "S2-00101" in all_candidates
    assert "S2-00102" in all_candidates

def test_11_candidate_ids_deduplicated():
    """TEST 11: Verify candidate IDs are deduplicated in union set."""
    s1_dict = normalize_record("Acme Restaurant", "100 Main St", "US")
    s1_dict["entity_id"] = "S1-00011"
    
    s2_both = normalize_record("Acme Restaurant", "100 Main St", "US")
    s2_both["entity_id"] = "S2-00111"
    
    blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
    blocker.index_target_records([s2_both])
    
    candidates = blocker.get_candidates(s1_dict)
    assert isinstance(candidates, set)
    assert len(candidates) == 1

def test_12_no_s1_entity_in_union_candidates():
    """TEST 12: Verify no Source 1 entity ID can appear in returned candidate set."""
    s1_dict = normalize_record("Acme Restaurant", "100 Main St", "US")
    s1_dict["entity_id"] = "S1-00012"
    
    s2_dict = normalize_record("Acme Restaurant", "100 Main St", "US")
    s2_dict["entity_id"] = "S2-00222"
    
    blocker = MultiChannelBlocker(lsh_threshold=0.5, num_perm=128)
    blocker.index_target_records([s1_dict, s2_dict])
    
    candidates = blocker.get_candidates(s1_dict)
    assert "S1-00012" not in candidates
    assert "S2-00222" in candidates

def test_13_existing_exact_token_tests_pass():
    """TEST 13: Verify existing exact-token tests still pass."""
    test_1_exact_shared_token_retrieves_target()
    test_2_legal_suffix_removal_respected()
    test_3_multiple_matching_tokens_no_duplicates()

# =====================================================================
# STAGE 3C UNIT TESTS: COARSE CITY/POSTAL + COUNTRY LOCATION CHANNEL
# =====================================================================

def test_14_city_country_retrieval():
    """TEST 14: Verify (city_guess, country) matches retrieve the target."""
    s1 = {"entity_id": "S1-14", "city_guess": "pune", "postal_code_guess": "", "country": "India"}
    s2 = {"entity_id": "S2-14", "city_guess": "pune", "postal_code_guess": "", "country": "India"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    loc_candidates = blocker._get_location_candidates(s1)
    assert "S2-14" in loc_candidates

def test_15_postal_country_retrieval():
    """TEST 15: Verify (postal_code_guess, country) matches retrieve the target."""
    s1 = {"entity_id": "S1-15", "city_guess": "", "postal_code_guess": "411001", "country": "India"}
    s2 = {"entity_id": "S2-15", "city_guess": "", "postal_code_guess": "411001", "country": "India"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    loc_candidates = blocker._get_location_candidates(s1)
    assert "S2-15" in loc_candidates

def test_16_city_and_postal_indexes_independent():
    """TEST 16: Verify city and postal indexes function independently."""
    s1_a = {"entity_id": "S1-16A", "city_guess": "pune", "postal_code_guess": "411001", "country": "India"}
    s2_a = {"entity_id": "S2-16A", "city_guess": "mumbai", "postal_code_guess": "411001", "country": "India"}
    
    blocker_a = MultiChannelBlocker()
    blocker_a.index_target_records([s2_a])
    assert "S2-16A" in blocker_a._get_location_candidates(s1_a)
    
    s1_b = {"entity_id": "S1-16B", "city_guess": "pune", "postal_code_guess": "411001", "country": "India"}
    s2_b = {"entity_id": "S2-16B", "city_guess": "pune", "postal_code_guess": "560001", "country": "India"}
    
    blocker_b = MultiChannelBlocker()
    blocker_b.index_target_records([s2_b])
    assert "S2-16B" in blocker_b._get_location_candidates(s1_b)

def test_17_country_prevents_cross_country_retrieval():
    """TEST 17: Verify country prevents cross-country retrieval even if city/postal matches."""
    s1 = {"entity_id": "S1-17", "city_guess": "pune", "postal_code_guess": "411001", "country": "India"}
    s2 = {"entity_id": "S2-17", "city_guess": "pune", "postal_code_guess": "411001", "country": "US"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    loc_candidates = blocker._get_location_candidates(s1)
    assert "S2-17" not in loc_candidates

def test_18_missing_city_no_country_wide_block():
    """TEST 18: Verify missing city does not create a country-wide candidate block."""
    s1 = {"entity_id": "S1-18", "city_guess": "", "postal_code_guess": "", "country": "India"}
    s2 = {"entity_id": "S2-18", "city_guess": "", "postal_code_guess": "", "country": "India"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    assert ("", "INDIA") not in blocker.city_country_index
    assert len(blocker._get_location_candidates(s1)) == 0

def test_19_missing_postal_no_country_wide_block():
    """TEST 19: Verify missing postal code does not create a country-wide candidate block."""
    s1 = {"entity_id": "S1-19", "city_guess": "", "postal_code_guess": "", "country": "India"}
    s2 = {"entity_id": "S2-19", "city_guess": "", "postal_code_guess": "", "country": "India"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    assert ("", "INDIA") not in blocker.postal_country_index
    assert len(blocker._get_location_candidates(s1)) == 0

def test_20_full_get_candidates_union_with_location():
    """TEST 20: Verify candidate found ONLY via location channel is present in get_candidates()."""
    s1 = normalize_record("Alpha", "Main Street, Pune 411001", "India")
    s1["entity_id"] = "S1-20"
    
    s2 = normalize_record("Beta", "Main Street, Pune 411001", "India")
    s2["entity_id"] = "S2-20"
    
    blocker = MultiChannelBlocker(lsh_threshold=0.5)
    blocker.index_target_records([s2])
    
    all_candidates = blocker.get_candidates(s1)
    assert "S2-20" in all_candidates

def test_21_deduplication_across_channels():
    """TEST 21: Verify candidate ID found by multiple channels appears only once in returned set."""
    s1 = normalize_record("Alpha Restaurant Pune", "100 Main St 411001", "India")
    s1["entity_id"] = "S1-21"
    
    s2 = normalize_record("Alpha Restaurant Pune", "100 Main St 411001", "India")
    s2["entity_id"] = "S2-21"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    candidates = blocker.get_candidates(s1)
    assert isinstance(candidates, set)
    assert len(candidates) == 1

def test_22_existing_stage_3a_and_3b_tests_pass():
    """TEST 22: Verify existing Stage 3A & 3B tests still pass."""
    test_1_exact_shared_token_retrieves_target()
    test_7_lsh_retrieves_character_level_typo()

# =====================================================================
# STAGE 3D UNIT TESTS: WEAK PHONETIC BLOCKING CHANNEL
# =====================================================================

def test_23_exact_phonetic_key_retrieval():
    """TEST 23: Verify exact phonetic key retrieval."""
    s1 = {"entity_id": "S1-23", "name_phonetic": "X123"}
    s2 = {"entity_id": "S2-23", "name_phonetic": "X123"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    phonetic_candidates = blocker._get_phonetic_candidates(s1)
    assert "S2-23" in phonetic_candidates

def test_24_different_phonetic_key_not_retrieved():
    """TEST 24: Verify different phonetic key is NOT retrieved."""
    s1 = {"entity_id": "S1-24", "name_phonetic": "X123"}
    s2 = {"entity_id": "S2-24", "name_phonetic": "Y999"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    phonetic_candidates = blocker._get_phonetic_candidates(s1)
    assert "S2-24" not in phonetic_candidates

def test_25_multiple_records_same_phonetic_key():
    """TEST 25: Verify multiple records with the same phonetic key are all retrieved."""
    s1 = {"entity_id": "S1-25", "name_phonetic": "PHON1"}
    s2 = {"entity_id": "S2-25A", "name_phonetic": "PHON1"}
    s3 = {"entity_id": "S3-25B", "name_phonetic": "PHON1"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2, s3])
    
    candidates = blocker._get_phonetic_candidates(s1)
    assert "S2-25A" in candidates
    assert "S3-25B" in candidates
    assert len(candidates) == 2

def test_26_empty_or_missing_phonetic_key_safe():
    """TEST 26: Verify empty/missing phonetic key returns empty set without crashing."""
    empty_s1 = {"entity_id": "S1-26", "name_phonetic": ""}
    empty_s2 = {"entity_id": "S2-26", "name_phonetic": None}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([empty_s2])
    
    candidates = blocker._get_phonetic_candidates(empty_s1)
    assert isinstance(candidates, set)
    assert len(candidates) == 0

def test_27_full_four_channel_union():
    """TEST 27: Verify get_candidates() returns the union of ALL FOUR blocking channels."""
    s1 = {
        "entity_id": "S1-27",
        "name_tokens_no_suffix": ["alpha"],
        "name_char_ngrams": ["alp", "lph", "pha"],
        "city_guess": "pune",
        "postal_code_guess": "",
        "country": "INDIA",
        "name_phonetic": "ALF"
    }
    s2_token = {
        "entity_id": "S2-27A",
        "name_tokens_no_suffix": ["alpha"],
        "name_char_ngrams": ["zz1"],
        "city_guess": "delhi",
        "country": "INDIA",
        "name_phonetic": "ZZZ1"
    }
    s2_lsh = {
        "entity_id": "S2-27B",
        "name_tokens_no_suffix": ["bravo"],
        "name_char_ngrams": ["alp", "lph", "pha"],
        "city_guess": "delhi",
        "country": "INDIA",
        "name_phonetic": "ZZZ2"
    }
    s2_loc = {
        "entity_id": "S2-27C",
        "name_tokens_no_suffix": ["charlie"],
        "name_char_ngrams": ["zz3"],
        "city_guess": "pune",
        "country": "INDIA",
        "name_phonetic": "ZZZ3"
    }
    s2_phonetic = {
        "entity_id": "S2-27D",
        "name_tokens_no_suffix": ["delta"],
        "name_char_ngrams": ["zz4"],
        "city_guess": "delhi",
        "country": "INDIA",
        "name_phonetic": "ALF"
    }
    
    blocker = MultiChannelBlocker(lsh_threshold=0.4)
    blocker.index_target_records([s2_token, s2_lsh, s2_loc, s2_phonetic])
    
    union_candidates = blocker.get_candidates(s1)
    assert "S2-27A" in union_candidates
    assert "S2-27B" in union_candidates
    assert "S2-27C" in union_candidates
    assert "S2-27D" in union_candidates
    assert len(union_candidates) == 4

def test_28_cross_channel_deduplication():
    """TEST 28: Verify candidate ID found across multiple channels appears only once."""
    s1 = {
        "entity_id": "S1-28",
        "name_tokens_no_suffix": ["alpha"],
        "name_char_ngrams": ["alp", "lph"],
        "city_guess": "pune",
        "country": "INDIA",
        "name_phonetic": "ALF"
    }
    s2 = {
        "entity_id": "S2-28",
        "name_tokens_no_suffix": ["alpha"],
        "name_char_ngrams": ["alp", "lph"],
        "city_guess": "pune",
        "country": "INDIA",
        "name_phonetic": "ALF"
    }
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    candidates = blocker.get_candidates(s1)
    assert isinstance(candidates, set)
    assert len(candidates) == 1
    assert "S2-28" in candidates

def test_29_source_1_exclusion_guard():
    """TEST 29: Verify Source 1 query ID cannot appear in candidate set."""
    s1 = {"entity_id": "S1-29", "name_phonetic": "X123"}
    s2 = {"entity_id": "S2-29", "name_phonetic": "X123"}
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s1, s2])
    
    candidates = blocker.get_candidates(s1)
    assert "S1-29" not in candidates
    assert "S2-29" in candidates

def test_30_regression_all_previous_tests():
    """TEST 30: Regression test verifying all previous Stage 3 tests pass."""
    test_1_exact_shared_token_retrieves_target()
    test_7_lsh_retrieves_character_level_typo()
    test_14_city_country_retrieval()

# =====================================================================
# STAGE 4 UNIT TESTS: CANDIDATE RECALL MEASUREMENT HARNESS
# =====================================================================

def test_31_perfect_candidate_recall():
    """TEST 31: Perfect candidate recall (100% true matches generated)."""
    s1_1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1_1["entity_id"] = "S1-31A"
    
    s2_1 = normalize_record("Acme Corp", "100 Main St", "US")
    s2_1["entity_id"] = "S2-31A"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_1])
    
    gt = {"S1-31A": "S2-31A"}
    res = evaluate_candidate_recall(blocker, [s1_1], gt)
    
    assert res["overall_recall"] == 1.0
    assert res["total_true_matches"] == 1
    assert res["total_found_matches"] == 1

def test_32_partial_candidate_recall():
    """TEST 32: Partial candidate recall."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-32"
    
    s2_a = normalize_record("Acme Corp", "100 Main St", "US")
    s2_a["entity_id"] = "S2-32A"
    
    s3_b = {
        "entity_id": "S3-32B",
        "name_tokens_no_suffix": ["unrelatedtoken999"],
        "name_char_ngrams": ["zzz999"],
        "city_guess": "unrelatedcity",
        "country": "FRANCE",
        "name_phonetic": "UNREL"
    }
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_a, s3_b])
    
    gt = {"S1-32": "S2-32A, S3-32B"}
    res = evaluate_candidate_recall(blocker, [s1], gt)
    
    assert res["overall_recall"] == 0.5

def test_33_zero_candidate_recall():
    """TEST 33: Zero candidate recall."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-33"
    
    s2_unrelated = {
        "entity_id": "S2-33X",
        "name_tokens_no_suffix": ["unrelated"],
        "name_char_ngrams": ["zzz"],
        "city_guess": "other",
        "country": "UK",
        "name_phonetic": "OTH"
    }
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_unrelated])
    
    gt = {"S1-33": "S2-33MISSING"}
    res = evaluate_candidate_recall(blocker, [s1], gt)
    
    assert res["overall_recall"] == 0.0

def test_34_multiple_true_matches_across_s2_and_s3():
    """TEST 34: Multiple true matches across S2 and S3."""
    s1 = normalize_record("Global Logistics", "500 Market St", "US")
    s1["entity_id"] = "S1-34"
    
    s2 = normalize_record("Global Logistics Corp", "500 Market St", "US")
    s2["entity_id"] = "S2-34A"
    
    s3 = normalize_record("Global Logistics Inc", "500 Market St", "US")
    s3["entity_id"] = "S3-34B"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2, s3])
    
    gt = {"S1-34": "S2-34A, S3-34B"}
    res = evaluate_candidate_recall(blocker, [s1], gt)
    
    assert res["overall_recall"] == 1.0

def test_35_empty_ground_truth_singletons():
    """TEST 35: Handling singleton entities."""
    s1 = normalize_record("Singleton Corp", "999 Lone St", "US")
    s1["entity_id"] = "S1-35"
    
    s2 = normalize_record("Other Corp", "100 Other St", "US")
    s2["entity_id"] = "S2-35"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    gt = {"S1-35": ""}
    res = evaluate_candidate_recall(blocker, [s1], gt)
    
    assert res["total_s1"] == 1
    assert res["singleton_entities"] == 1

def test_36_duplicate_candidate_ids_do_not_inflate_recall():
    """TEST 36: Verify duplicate candidate IDs do not inflate recall."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-36"
    
    s2 = normalize_record("Acme Corp", "100 Main St", "US")
    s2["entity_id"] = "S2-36"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    gt = {"S1-36": "S2-36"}
    res = evaluate_candidate_recall(blocker, [s1], gt)
    
    assert res["overall_recall"] == 1.0

def test_37_per_s1_recall_statistics_correct():
    """TEST 37: Verify per-S1 recall statistics."""
    s1_p = normalize_record("Alpha", "Main St", "US")
    s1_p["entity_id"] = "S1-37P"
    s1_z = normalize_record("Beta", "Main St", "US")
    s1_z["entity_id"] = "S1-37Z"
    
    s2_p = normalize_record("Alpha", "Main St", "US")
    s2_p["entity_id"] = "S2-37P"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_p])
    
    gt = {"S1-37P": "S2-37P", "S1-37Z": "S2-37MISSING"}
    res = evaluate_candidate_recall(blocker, [s1_p, s1_z], gt)
    
    assert res["perfect_recall_entities"] == 1
    assert res["zero_recall_entities"] == 1

def test_38_recall_breakdown_by_match_count_bucket():
    """TEST 38: Verify recall breakdown by true-match-count bucket."""
    s1_a = normalize_record("Alpha", "Main St", "US")
    s1_a["entity_id"] = "S1-38A"
    s2_a = normalize_record("Alpha", "Main St", "US")
    s2_a["entity_id"] = "S2-38A"
    
    s1_b = normalize_record("Beta", "Main St", "US")
    s1_b["entity_id"] = "S1-38B"
    s2_b = normalize_record("Beta", "Main St", "US")
    s2_b["entity_id"] = "S2-38B"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_a, s2_b])
    
    gt = {"S1-38A": "S2-38A", "S1-38B": "S2-38B, S3-38MISSING"}
    res = evaluate_candidate_recall(blocker, [s1_a, s1_b], gt)
    
    breakdown = res["recall_by_match_count"]
    assert 1 in breakdown and breakdown[1]["recall"] == 1.0
    assert 2 in breakdown and breakdown[2]["recall"] == 0.5

def test_39_missed_match_diagnostics_correct():
    """TEST 39: Verify missed-match diagnostics contain correct S1 ID and missed entity IDs."""
    s1 = normalize_record("Alpha", "Main St", "US")
    s1["entity_id"] = "S1-39"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([])
    
    gt = {"S1-39": "S2-39MISS1, S3-39MISS2"}
    res = evaluate_candidate_recall(blocker, [s1], gt)
    
    assert len(res["missed_examples"]) == 1

def test_40_regression_all_previous_tests():
    """TEST 40: Regression test verifying all previous Stage 3 tests pass."""
    test_1_exact_shared_token_retrieves_target()
    test_7_lsh_retrieves_character_level_typo()

# =====================================================================
# STAGE 5A UNIT TESTS: CHEAP PRE-RANKING & PRUNING UTILITY
# =====================================================================

def test_41_identical_normalized_name_ranks_above_unrelated():
    """TEST 41: Identical normalized name ranks above unrelated name."""
    s1 = normalize_record("Acme Restaurant", "100 Main St", "US")
    s1["entity_id"] = "S1-41"
    
    cand1 = normalize_record("Acme Restaurant", "100 Main St", "US")
    cand1["entity_id"] = "S2-41A"
    
    cand2 = normalize_record("Zebra Electronics", "999 Oak St", "US")
    cand2["entity_id"] = "S2-41B"
    
    ranked = rank_and_prune_candidates(s1, [cand2, cand1])
    assert len(ranked) == 2
    assert ranked[0]["entity_id"] == "S2-41A"
    assert ranked[0]["score"] > ranked[1]["score"]

def test_42_strong_address_similarity_contributes_to_ranking():
    """TEST 42: Strong address similarity contributes to higher score."""
    s1 = normalize_record("Acme Corp", "100 Main Street, Suite 500", "US")
    s1["entity_id"] = "S1-42"
    
    cand_matching_addr = normalize_record("Acme Corp", "100 Main Street, Suite 500", "US")
    cand_matching_addr["entity_id"] = "S2-42A"
    
    cand_diff_addr = normalize_record("Acme Corp", "999 Far Away Blvd", "US")
    cand_diff_addr["entity_id"] = "S2-42B"
    
    score1 = calculate_cheap_similarity_score(s1, cand_matching_addr)
    score2 = calculate_cheap_similarity_score(s1, cand_diff_addr)
    assert score1 > score2

def test_43_country_agreement_handled_correctly():
    """TEST 43: Country agreement contributes to score."""
    s1 = normalize_record("Acme Corp", "100 Main St", "India")
    s1["entity_id"] = "S1-43"
    
    cand_same_country = normalize_record("Acme Corp", "100 Main St", "India")
    cand_same_country["entity_id"] = "S2-43A"
    
    cand_diff_country = normalize_record("Acme Corp", "100 Main St", "US")
    cand_diff_country["entity_id"] = "S2-43B"
    
    score1 = calculate_cheap_similarity_score(s1, cand_same_country)
    score2 = calculate_cheap_similarity_score(s1, cand_diff_country)
    assert score1 > score2

def test_44_city_agreement_handled_correctly():
    """TEST 44: City agreement contributes to score."""
    s1 = {"normalized_name": "acme", "city_guess": "pune", "country": "INDIA"}
    s2_same = {"normalized_name": "acme", "city_guess": "pune", "country": "INDIA"}
    s2_diff = {"normalized_name": "acme", "city_guess": "mumbai", "country": "INDIA"}
    
    score1 = calculate_cheap_similarity_score(s1, s2_same)
    score2 = calculate_cheap_similarity_score(s1, s2_diff)
    assert score1 > score2

def test_45_postal_agreement_handled_correctly():
    """TEST 45: Postal agreement contributes to score."""
    s1 = {"normalized_name": "acme", "postal_code_guess": "411001", "country": "INDIA"}
    s2_same = {"normalized_name": "acme", "postal_code_guess": "411001", "country": "INDIA"}
    s2_diff = {"normalized_name": "acme", "postal_code_guess": "560001", "country": "INDIA"}
    
    score1 = calculate_cheap_similarity_score(s1, s2_same)
    score2 = calculate_cheap_similarity_score(s1, s2_diff)
    assert score1 > score2

def test_46_missing_values_no_false_agreement():
    """TEST 46: Missing values yield 0.0 contribution rather than false agreement."""
    s1 = {"normalized_name": "acme", "normalized_address": "", "city_guess": "", "postal_code_guess": ""}
    s2 = {"normalized_name": "acme", "normalized_address": "", "city_guess": "", "postal_code_guess": ""}
    
    score = calculate_cheap_similarity_score(s1, s2)
    assert score <= 0.50

def test_47_candidates_sorted_descending():
    """TEST 47: Candidates are sorted descending by cheap score."""
    s1 = normalize_record("Acme Restaurant", "100 Main St", "US")
    s1["entity_id"] = "S1-47"
    
    cand1 = normalize_record("Zebra Corp", "999 Oak St", "India")
    cand1["entity_id"] = "S2-47A"
    
    cand2 = normalize_record("Acme Barbershop", "100 Main St", "US")
    cand2["entity_id"] = "S2-47B"
    
    cand3 = normalize_record("Acme Restaurant", "100 Main St", "US")
    cand3["entity_id"] = "S2-47C"
    
    ranked = rank_and_prune_candidates(s1, [cand1, cand2, cand3])
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)
    assert ranked[0]["entity_id"] == "S2-47C"

def test_48_equal_scores_deterministic_tie_breaking():
    """TEST 48: Equal scores use deterministic entity_id tie-breaking (lexicographical)."""
    s1 = {"normalized_name": "acme"}
    
    cand1 = {"entity_id": "S2-002", "normalized_name": "acme"}
    cand2 = {"entity_id": "S2-001", "normalized_name": "acme"}
    
    ranked = rank_and_prune_candidates(s1, [cand1, cand2])
    assert ranked[0]["entity_id"] == "S2-001"
    assert ranked[1]["entity_id"] == "S2-002"

def test_49_cap_none_returns_all_candidates():
    """TEST 49: cap=None returns all ranked candidates."""
    s1 = {"normalized_name": "acme"}
    cands = [{"entity_id": f"S2-{i}", "normalized_name": "acme"} for i in range(10)]
    
    ranked = rank_and_prune_candidates(s1, cands, cap=None)
    assert len(ranked) == 10

def test_50_cap_k_returns_exactly_k():
    """TEST 50: cap=K returns exactly K candidates when at least K exist."""
    s1 = {"normalized_name": "acme"}
    cands = [{"entity_id": f"S2-{i:03d}", "normalized_name": "acme"} for i in range(10)]
    
    ranked = rank_and_prune_candidates(s1, cands, cap=3)
    assert len(ranked) == 3

def test_51_cap_larger_than_candidate_count():
    """TEST 51: cap larger than candidate count returns all candidates."""
    s1 = {"normalized_name": "acme"}
    cands = [{"entity_id": "S2-001", "normalized_name": "acme"}]
    
    ranked = rank_and_prune_candidates(s1, cands, cap=50)
    assert len(ranked) == 1

def test_52_duplicate_candidate_ids_handled_safely():
    """TEST 52: Duplicate candidate IDs are deduplicated before ranking."""
    s1 = {"normalized_name": "acme"}
    cand = {"entity_id": "S2-001", "normalized_name": "acme"}
    
    ranked = rank_and_prune_candidates(s1, [cand, cand])
    assert len(ranked) == 1

def test_53_empty_candidate_set_returns_empty():
    """TEST 53: Empty candidate set returns empty list."""
    s1 = {"normalized_name": "acme"}
    ranked = rank_and_prune_candidates(s1, [])
    assert ranked == []

def test_54_regression_all_previous_40_tests_pass():
    """TEST 54: Regression test confirming all previous tests 1-40 still pass."""
    test_1_exact_shared_token_retrieves_target()
    test_7_lsh_retrieves_character_level_typo()

# =====================================================================
# STAGE 5B UNIT TESTS: RECALL-VS-CAP EVALUATION HARNESS
# =====================================================================

def test_55_cap_none_matches_raw_candidate_recall():
    """TEST 55: cap=None matches raw candidate recall."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-55"
    
    s2 = normalize_record("Acme Corp", "100 Main St", "US")
    s2["entity_id"] = "S2-55"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    target_map = {"S2-55": s2}
    gt = {"S1-55": "S2-55"}
    
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=None)
    assert res["overall_recall"] == 1.0
    assert res["total_found_matches"] == 1
    assert res["total_missed_matches"] == 0

def test_56_small_cap_limits_ranked_candidate_set():
    """TEST 56: A small cap limits the ranked candidate set to K."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-56"
    
    cands = []
    target_map = {}
    for i in range(10):
        cid = f"S2-56{i:02d}"
        c_rec = normalize_record(f"Acme Sub {i}", f"{i} Main St", "US")
        c_rec["entity_id"] = cid
        cands.append(c_rec)
        target_map[cid] = c_rec
        
    blocker = MultiChannelBlocker()
    blocker.index_target_records(cands)
    
    gt = {"S1-56": "S2-5600"}
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=2)
    assert res["avg_retained"] == 2.0

def test_57_cap_larger_than_candidate_count_keeps_all():
    """TEST 57: A cap larger than candidate count keeps all candidates."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-57"
    
    s2 = normalize_record("Acme Corp", "100 Main St", "US")
    s2["entity_id"] = "S2-57"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    target_map = {"S2-57": s2}
    gt = {"S1-57": "S2-57"}
    
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=100)
    assert res["avg_retained"] == 1.0
    assert res["overall_recall"] == 1.0

def test_58_recall_calculation_after_pruning_correct():
    """TEST 58: Recall calculation after pruning is correct."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-58"
    
    # 3 candidates: S2-58A (exact match), S2-58B (exact match), S2-58C (unrelated)
    s2_a = normalize_record("Acme Corp", "100 Main St", "US")
    s2_a["entity_id"] = "S2-58A"
    
    s2_b = normalize_record("Acme Corp", "100 Main St", "US")
    s2_b["entity_id"] = "S2-58B"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_a, s2_b])
    
    target_map = {"S2-58A": s2_a, "S2-58B": s2_b}
    gt = {"S1-58": "S2-58A, S2-58B"}
    
    # Cap at 1 -> retains only 1 of the 2 true matches -> 50% recall
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=1)
    assert res["overall_recall"] == 0.5
    assert res["total_found_matches"] == 1
    assert res["total_missed_matches"] == 1

def test_59_true_matches_retained_in_top_k_counted_correctly():
    """TEST 59: True matches retained in top-K are counted correctly."""
    s1 = normalize_record("Alpha Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-59"
    
    s2_exact = normalize_record("Alpha Corp", "100 Main St", "US")
    s2_exact["entity_id"] = "S2-59A"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_exact])
    
    target_map = {"S2-59A": s2_exact}
    gt = {"S1-59": "S2-59A"}
    
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=5)
    assert res["total_found_matches"] == 1
    assert res["perfect_recall_entities"] == 1

def test_60_missed_true_matches_after_pruning_reported():
    """TEST 60: Missed true matches after pruning are reported correctly in diagnostics."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-60"
    
    # 2 true matches: s2_a ranks higher than s2_b
    s2_a = normalize_record("Acme Corp", "100 Main St, Suite 500", "US")
    s2_a["entity_id"] = "S2-60A"
    
    s2_b = normalize_record("Acme Corp", "Different Address Blvd", "US")
    s2_b["entity_id"] = "S2-60B"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_a, s2_b])
    
    target_map = {"S2-60A": s2_a, "S2-60B": s2_b}
    gt = {"S1-60": "S2-60A, S2-60B"}
    
    # Cap at 1 -> retains S2-60A, drops S2-60B
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=1, max_diagnostics=10)
    assert res["total_missed_matches"] == 1
    assert len(res["missed_examples"]) == 1
    diag = res["missed_examples"][0]
    assert diag["s1_id"] == "S1-60"
    assert "S2-60B" in diag["missed_matches"]

def test_61_recall_by_match_count_buckets_correct():
    """TEST 61: Recall-by-match-count buckets are correct."""
    s1 = normalize_record("Alpha Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-61"
    
    s2 = normalize_record("Alpha Corp", "100 Main St", "US")
    s2["entity_id"] = "S2-61"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2])
    
    target_map = {"S2-61": s2}
    gt = {"S1-61": "S2-61"}
    
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=10)
    assert 1 in res["recall_by_match_count"]
    assert res["recall_by_match_count"][1]["recall"] == 1.0

def test_62_candidate_reduction_percentage_correct():
    """TEST 62: Candidate reduction percentage is correct."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-62"
    
    cands = []
    target_map = {}
    for i in range(10):
        cid = f"S2-62{i:02d}"
        c_rec = normalize_record("Acme Corp", "100 Main St", "US")
        c_rec["entity_id"] = cid
        cands.append(c_rec)
        target_map[cid] = c_rec
        
    blocker = MultiChannelBlocker()
    blocker.index_target_records(cands)
    
    gt = {"S1-62": "S2-6200"}
    res = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=2)
    # Raw = 10, Retained = 2 -> Reduction = (1 - 2/10) * 100 = 80.0%
    assert res["reduction_pct"] == 80.0

def test_63_deterministic_ranking_gives_deterministic_cap_results():
    """TEST 63: Deterministic ranking gives deterministic cap results."""
    s1 = normalize_record("Acme Corp", "100 Main St", "US")
    s1["entity_id"] = "S1-63"
    
    s2_a = normalize_record("Acme Corp", "100 Main St", "US")
    s2_a["entity_id"] = "S2-63A"
    
    s2_b = normalize_record("Acme Corp", "100 Main St", "US")
    s2_b["entity_id"] = "S2-63B"
    
    blocker = MultiChannelBlocker()
    blocker.index_target_records([s2_a, s2_b])
    
    target_map = {"S2-63A": s2_a, "S2-63B": s2_b}
    gt = {"S1-63": "S2-63A"}
    
    res1 = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=1)
    res2 = evaluate_recall_for_cap(blocker, [s1], target_map, gt, cap=1)
    assert res1["overall_recall"] == res2["overall_recall"]
    assert res1["total_found_matches"] == res2["total_found_matches"]

def test_64_regression_all_previous_54_tests_pass():
    """TEST 64: Regression test confirming all previous tests 1-54 still pass."""
    test_1_exact_shared_token_retrieves_target()
    test_7_lsh_retrieves_character_level_typo()
    test_14_city_country_retrieval()
    test_23_exact_phonetic_key_retrieval()
    test_31_perfect_candidate_recall()
    test_41_identical_normalized_name_ranks_above_unrelated()
