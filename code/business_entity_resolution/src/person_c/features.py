"""
Person C — features.py
Deterministic feature engineering for entity resolution pairs.

Implements pairwise feature computation comparing ONE normalized S1 record
against ONE normalized S2/S3 candidate record. No ML, no TF-IDF, no ground truth.
All features are purely numeric and safely handle missing data without NaN/inf.
"""

from typing import Dict, Any, List
import pandas as pd
from rapidfuzz import fuzz, distance

def _safe_jaccard(list1: List[str], list2: List[str]) -> float:
    if not list1 or not list2:
        return 0.0
    set1, set2 = set(list1), set(list2)
    union_len = len(set1 | set2)
    if union_len == 0:
        return 0.0
    return float(len(set1 & set2)) / union_len

def _safe_overlap(list1: List[str], list2: List[str]) -> float:
    if not list1 or not list2:
        return 0.0
    set1, set2 = set(list1), set(list2)
    min_len = min(len(set1), len(set2))
    if min_len == 0:
        return 0.0
    return float(len(set1 & set2)) / min_len

def _safe_containment(str1: str, str2: str) -> int:
    if not str1 or not str2:
        return 0
    return 1 if (str1 in str2 or str2 in str1) else 0

def _safe_ratio(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    return fuzz.ratio(str1, str2) / 100.0

def _safe_jaro_winkler(str1: str, str2: str) -> float:
    if not str1 or not str2:
        return 0.0
    return distance.JaroWinkler.normalized_similarity(str1, str2)

def _exact_match(str1: str, str2: str) -> int:
    if not str1 or not str2:
        return 0
    return 1 if str1 == str2 else 0

def build_pair_features(source_record: Dict[str, Any], candidate_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes a deterministic feature vector for a pair of normalized records.
    Returns a dictionary of numeric features plus identifiers.
    """
    f = {}
    
    # Identifiers
    f["source1_entity_id"] = source_record.get("entity_id", "")
    f["candidate_entity_id"] = candidate_record.get("entity_id", "")
    
    # Extract fields safely
    s_name = source_record.get("normalized_name", "")
    c_name = candidate_record.get("normalized_name", "")
    
    s_agg_name = source_record.get("aggressive_normalized_name", "")
    c_agg_name = candidate_record.get("aggressive_normalized_name", "")
    
    s_name_toks = source_record.get("name_tokens", [])
    c_name_toks = candidate_record.get("name_tokens", [])
    
    s_char_ngrams = source_record.get("name_char_ngrams", [])
    c_char_ngrams = candidate_record.get("name_char_ngrams", [])
    
    s_phonetic = source_record.get("name_phonetic", "")
    c_phonetic = candidate_record.get("name_phonetic", "")
    
    s_addr = source_record.get("normalized_address", "")
    c_addr = candidate_record.get("normalized_address", "")
    
    s_addr_toks = source_record.get("address_tokens", [])
    c_addr_toks = candidate_record.get("address_tokens", [])
    
    s_country = source_record.get("country", "")
    c_country = candidate_record.get("country", "")
    
    s_city = source_record.get("city_guess", "")
    c_city = candidate_record.get("city_guess", "")
    
    s_postal = source_record.get("postal_code_guess", "")
    c_postal = candidate_record.get("postal_code_guess", "")
    
    s_street = source_record.get("street_number_guess", "")
    c_street = candidate_record.get("street_number_guess", "")
    
    # Group 1: NAME
    f["name_exact"] = _exact_match(s_name, c_name)
    f["aggressive_name_exact"] = _exact_match(s_agg_name, c_agg_name)
    f["name_jaccard"] = _safe_jaccard(s_name_toks, c_name_toks)
    f["name_token_overlap"] = _safe_overlap(s_name_toks, c_name_toks)
    f["name_containment"] = _safe_containment(s_name, c_name)
    f["name_levenshtein"] = _safe_ratio(s_name, c_name)
    f["name_jaro_winkler"] = _safe_jaro_winkler(s_name, c_name)
    f["name_char_ngram_similarity"] = _safe_jaccard(s_char_ngrams, c_char_ngrams)
    f["name_phonetic_match"] = _exact_match(s_phonetic, c_phonetic)
    f["name_token_count_diff"] = abs(len(s_name_toks) - len(c_name_toks))
    f["name_length_diff"] = abs(len(s_name) - len(c_name))
    
    # Group 2: ADDRESS
    f["address_exact"] = _exact_match(s_addr, c_addr)
    f["address_jaccard"] = _safe_jaccard(s_addr_toks, c_addr_toks)
    f["address_token_overlap"] = _safe_overlap(s_addr_toks, c_addr_toks)
    f["address_containment"] = _safe_containment(s_addr, c_addr)
    f["address_levenshtein"] = _safe_ratio(s_addr, c_addr)
    f["address_jaro_winkler"] = _safe_jaro_winkler(s_addr, c_addr)
    f["address_length_diff"] = abs(len(s_addr) - len(c_addr))
    f["address_token_count_diff"] = abs(len(s_addr_toks) - len(c_addr_toks))
    
    # Group 3: LOCATION
    f["country_match"] = _exact_match(s_country, c_country)
    f["country_missing_either"] = 1 if not s_country or not c_country else 0
    f["city_match"] = _exact_match(s_city, c_city)
    f["city_missing_either"] = 1 if not s_city or not c_city else 0
    f["postal_match"] = _exact_match(s_postal, c_postal)
    f["postal_missing_either"] = 1 if not s_postal or not c_postal else 0
    f["street_number_match"] = _exact_match(s_street, c_street)
    
    # Group 4: MISSINGNESS
    s_name_miss = bool(source_record.get("name_missing", not s_name))
    c_name_miss = bool(candidate_record.get("name_missing", not c_name))
    s_addr_miss = bool(source_record.get("address_missing", not s_addr))
    c_addr_miss = bool(candidate_record.get("address_missing", not c_addr))
    s_post_miss = bool(source_record.get("postal_missing", not s_postal))
    c_post_miss = bool(candidate_record.get("postal_missing", not c_postal))
    s_city_miss = bool(source_record.get("city_missing", not s_city))
    c_city_miss = bool(candidate_record.get("city_missing", not c_city))
    
    f["source_name_missing"] = int(s_name_miss)
    f["candidate_name_missing"] = int(c_name_miss)
    f["source_address_missing"] = int(s_addr_miss)
    f["candidate_address_missing"] = int(c_addr_miss)
    f["source_postal_missing"] = int(s_post_miss)
    f["candidate_postal_missing"] = int(c_post_miss)
    f["source_city_missing"] = int(s_city_miss)
    f["candidate_city_missing"] = int(c_city_miss)
    
    f["both_name_missing"] = int(s_name_miss and c_name_miss)
    f["both_address_missing"] = int(s_addr_miss and c_addr_miss)
    f["both_postal_missing"] = int(s_post_miss and c_post_miss)
    f["both_city_missing"] = int(s_city_miss and c_city_miss)
    
    # Group 5: INTERACTIONS
    name_lev = f["name_levenshtein"]
    addr_lev = f["address_levenshtein"]
    
    f["name_x_address"] = name_lev * addr_lev
    f["name_x_country"] = name_lev * f["country_match"]
    f["name_x_city"] = name_lev * f["city_match"]
    f["name_x_postal"] = name_lev * f["postal_match"]
    
    f["address_x_country"] = addr_lev * f["country_match"]
    f["address_x_city"] = addr_lev * f["city_match"]
    f["address_x_postal"] = addr_lev * f["postal_match"]
    
    return f


def build_feature_matrix(pairs_features: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts a list of pairwise feature dictionaries into a pandas DataFrame.
    Returns the dataframe with numeric features ready for ML, plus identifiers.
    """
    df = pd.DataFrame(pairs_features)
    # The identifiers 'source1_entity_id' and 'candidate_entity_id' are retained
    # as string columns. All other columns should be numeric.
    return df
