"""
Stage 5A: Cheap Pre-Ranking and Pruning Utility.

Calculates a deterministic cheap similarity score for each candidate record against a Source 1 entity,
sorts candidates in descending order (with deterministic entity_id tie-breaking), and caps at `cap`.
"""

from typing import Dict, List, Any, Optional
from rapidfuzz import fuzz

def calculate_cheap_similarity_score(s1_rec: Dict[str, Any], cand_rec: Dict[str, Any]) -> float:
    """
    Computes a fast deterministic similarity score between a Source 1 record and a candidate record.
    
    Formula:
      Score = 0.50 * NameScore + 0.30 * AddressScore + 0.10 * CountryScore + 0.05 * CityScore + 0.05 * PostalScore
      
    Rules:
      - NameScore: Combined rapidfuzz token_sort_ratio and token_set_ratio on normalized/aggressive names.
      - AddressScore: rapidfuzz token_sort_ratio on normalized_address (0.0 if either address missing).
      - CountryScore: 1.0 if exact match on non-empty country, else 0.0.
      - CityScore: 1.0 if exact match on non-empty city_guess, else 0.0.
      - PostalScore: 1.0 if exact match on non-empty postal_code_guess, else 0.0.
    """
    # 1. Name Similarity (Weight 0.50)
    s1_name = s1_rec.get("aggressive_normalized_name") or s1_rec.get("normalized_name") or ""
    cand_name = cand_rec.get("aggressive_normalized_name") or cand_rec.get("normalized_name") or ""
    
    if s1_name and cand_name:
        sort_ratio = fuzz.token_sort_ratio(s1_name, cand_name) / 100.0
        set_ratio = fuzz.token_set_ratio(s1_name, cand_name) / 100.0
        name_score = 0.70 * sort_ratio + 0.30 * set_ratio
    else:
        name_score = 0.0
        
    # 2. Address Similarity (Weight 0.30) - Missing values yield 0.0
    s1_addr = (s1_rec.get("normalized_address") or "").strip()
    cand_addr = (cand_rec.get("normalized_address") or "").strip()
    
    if s1_addr and cand_addr:
        address_score = fuzz.token_sort_ratio(s1_addr, cand_addr) / 100.0
    else:
        address_score = 0.0
        
    # 3. Country Agreement (Weight 0.10) - Missing values yield 0.0
    s1_country = (s1_rec.get("country") or "").strip().upper()
    cand_country = (cand_rec.get("country") or "").strip().upper()
    
    if s1_country and cand_country:
        country_score = 1.0 if s1_country == cand_country else 0.0
    else:
        country_score = 0.0
        
    # 4. City Agreement (Weight 0.05) - Missing values yield 0.0
    s1_city = (s1_rec.get("city_guess") or "").strip().lower()
    cand_city = (cand_rec.get("city_guess") or "").strip().lower()
    
    if s1_city and cand_city:
        city_score = 1.0 if s1_city == cand_city else 0.0
    else:
        city_score = 0.0
        
    # 5. Postal Code Agreement (Weight 0.05) - Missing values yield 0.0
    s1_postal = (s1_rec.get("postal_code_guess") or "").strip().lower()
    cand_postal = (cand_rec.get("postal_code_guess") or "").strip().lower()
    
    if s1_postal and cand_postal:
        postal_score = 1.0 if s1_postal == cand_postal else 0.0
    else:
        postal_score = 0.0
        
    total_score = (
        0.50 * name_score
        + 0.30 * address_score
        + 0.10 * country_score
        + 0.05 * city_score
        + 0.05 * postal_score
    )
    
    return float(round(total_score, 6))

def rank_and_prune_candidates(
    s1_rec: Dict[str, Any],
    candidate_records: List[Dict[str, Any]],
    cap: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Ranks candidate records by cheap similarity score in descending order and caps at `cap`.
    
    Args:
        s1_rec: Normalized Source 1 record dictionary.
        candidate_records: List of candidate target record dictionaries.
        cap: Optional integer cap. If None, returns all ranked candidates.
        
    Returns:
        List[Dict[str, Any]]: Bounded list of ranked dicts: [{'entity_id': str, 'score': float}, ...]
    """
    if not candidate_records:
        return []
        
    # Deduplicate candidate records by entity_id to ensure safety
    seen_ids = set()
    unique_candidates = []
    for cand in candidate_records:
        cand_id = cand.get("entity_id", "")
        if cand_id and cand_id not in seen_ids:
            seen_ids.add(cand_id)
            unique_candidates.append(cand)
            
    scored_candidates = []
    for cand in unique_candidates:
        cand_id = cand.get("entity_id", "")
        score = calculate_cheap_similarity_score(s1_rec, cand)
        scored_candidates.append({"entity_id": cand_id, "score": score})
        
    # Deterministic sorting: Descending by score, then ascending by entity_id string
    scored_candidates.sort(key=lambda x: (-x["score"], x["entity_id"]))
    
    if cap is not None and cap >= 0:
        return scored_candidates[:cap]
        
    return scored_candidates
