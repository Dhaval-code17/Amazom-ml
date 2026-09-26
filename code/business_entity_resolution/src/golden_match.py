from typing import Dict, Any

def golden_match(s1_record: Dict[str, Any], candidate_record: Dict[str, Any]) -> bool:
    """
    Deterministic Golden-Match Layer (Stage 2).
    
    Rule (conjunctive, not name-alone):
    TRUE if:
      (
        aggressive_normalized_name == aggressive_normalized_name
        AND normalized_address == normalized_address
        AND country matches
      )
      OR
      (
        aggressive_normalized_name == aggressive_normalized_name
        AND postal_code_guess is equal
        AND postal_code_guess is non-null/non-empty
      )
    """
    s1_agg_name = s1_record.get("aggressive_normalized_name", "")
    cand_agg_name = candidate_record.get("aggressive_normalized_name", "")

    # Both aggressive names must be non-empty and equal
    if not s1_agg_name or s1_agg_name != cand_agg_name:
        return False

    # Rule 1: Aggressive name match AND Normalized address match AND Country match
    s1_addr = s1_record.get("normalized_address", "")
    cand_addr = candidate_record.get("normalized_address", "")
    s1_country = s1_record.get("country", "")
    cand_country = candidate_record.get("country", "")

    rule1 = (
        bool(s1_addr) and (s1_addr == cand_addr) and
        bool(s1_country) and (s1_country == cand_country)
    )
    if rule1:
        return True

    # Rule 2: Aggressive name match AND Postal code equal AND non-null/non-empty AND Country match
    s1_postal = s1_record.get("postal_code_guess", "")
    cand_postal = candidate_record.get("postal_code_guess", "")

    rule2 = (
        bool(s1_postal) and (s1_postal == cand_postal) and
        bool(s1_country) and (s1_country == cand_country)
    )
    if rule2:
        return True

    return False
