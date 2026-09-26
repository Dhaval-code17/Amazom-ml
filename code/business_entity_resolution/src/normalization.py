import re
import string
from typing import Dict, List, Any
from unidecode import unidecode
import jellyfish
try:
    import metaphone
except ImportError:
    metaphone = None
from schema import NormalizedRecord

# Legal Suffixes to remove for aggressive_normalized_name
# Lookup set (case-insensitive token comparison)
LEGAL_SUFFIXES = {
    "corp", "corporation", "inc", "incorporated", "ltd", "limited",
    "pvt", "private", "llc", "llp", "sarl", "sas", "co"
}

# Street Abbreviation Lookup Table
STREET_ABBREVIATIONS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "blvd": "boulevard",
    "dr": "drive",
    "ln": "lane",
    "ct": "court",
    "pkwy": "parkway",
    "sq": "square",
    "pl": "place",
    "hwy": "highway",
}

# Regex patterns for postal code extraction
POSTAL_PATTERNS = [
    re.compile(r'\b\d{5}(?:-\d{4})?\b'),  # US 5-digit ZIP (or ZIP+4, takes 5 digits)
    re.compile(r'\b\d{6}\b'),              # Indian 6-digit PIN
    re.compile(r'\b\d{5}\b'),              # French 5-digit postal code
]

def _get_char_ngrams(text: str, n: int = 3) -> List[str]:
    """Generate character n-grams from text."""
    if not text:
        return []
    text_padded = f" {text.strip()} "
    if len(text_padded) < n:
        return [text_padded]
    return [text_padded[i:i+n] for i in range(len(text_padded) - n + 1)]

def _extract_postal_code(address_str: str) -> str:
    """Extract postal code using regex heuristics for US, India, and France."""
    if not address_str:
        return ""
    for pattern in POSTAL_PATTERNS:
        match = pattern.search(address_str)
        if match:
            # Return standard 5-digit or 6-digit match
            full_match = match.group(0)
            if "-" in full_match:
                return full_match.split("-")[0]
            return full_match
    return ""

def _extract_city(address_str: str, tokens: List[str], postal_code: str) -> str:
    """
    Heuristic/structure/token-based city extraction.
    Extracts the chunk/token preceding a detected postal code or secondary location token,
    using comma-separated address structures when available.
    """
    if address_str:
        parts = [p.strip() for p in address_str.split(',') if p.strip()]
        if len(parts) >= 2:
            cand = ""
            if postal_code:
                for i, part in enumerate(parts):
                    if postal_code in part and i > 0:
                        cand = parts[i - 1]
                        break
            
            if not cand:
                cand = parts[-2]
                
            cand = cand.lower().translate(str.maketrans('', '', string.punctuation)).strip()
            
            if cand and not any(c.isdigit() for c in cand):
                if not any(cand.startswith(x) for x in ["unit ", "suite ", "apt ", "apartment ", "block "]):
                    return cand

    if not tokens:
        return ""
    if postal_code and postal_code in tokens:
        idx = tokens.index(postal_code)
        if idx > 0:
            return tokens[idx - 1]
    for token in reversed(tokens):
        if not token.isdigit() and len(token) > 2:
            return token
    return ""


def _extract_street_number(tokens: List[str]) -> str:
    """Extract leading numeric token as street number guess."""
    if not tokens:
        return ""
    if tokens[0].isdigit():
        return tokens[0]
    # Check if first token contains leading digits like "123a"
    match = re.match(r'^(\d+)', tokens[0])
    if match:
        return match.group(1)
    return ""

def normalize_record(name: str, address: str, country: str) -> dict:
    """
    Pure function to normalize a single business record.
    Identically applied to Source 1, Source 2, and Source 3.
    """
    # 1. Handle missingness
    raw_name = "" if name is None else str(name)
    raw_address = "" if address is None else str(address)
    raw_country = "" if country is None else str(country)

    name_missing = bool(not raw_name.strip())
    address_missing = bool(not raw_address.strip())

    # 2. Name Normalization
    # Transliterate Unicode -> ASCII
    norm_name = unidecode(raw_name).lower()
    # Replace '&' with 'and'
    norm_name = norm_name.replace('&', ' and ')
    # Remove leading noise-prefix tokens such as '--' or '<<'
    norm_name = re.sub(r'^[^\w]+', '', norm_name)
    # Strip punctuation
    norm_name_clean = norm_name.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    # Standardize whitespace
    norm_name_clean = re.sub(r'\s+', ' ', norm_name_clean).strip()

    name_tokens = norm_name_clean.split() if norm_name_clean else []

    # Aggressive name normalization: strip legal suffixes and noise tokens
    tokens_no_suffix = [tok for tok in name_tokens if tok not in LEGAL_SUFFIXES]
    aggressive_normalized_name = " ".join(tokens_no_suffix)
    name_tokens_no_suffix = tokens_no_suffix

    # Name Phonetic (Double Metaphone)
    if norm_name_clean:
        try:
            if metaphone and hasattr(metaphone, "dm"):
                dm_res = metaphone.dm(norm_name_clean)
                name_phonetic = dm_res[0] if isinstance(dm_res, (tuple, list)) else str(dm_res)
            elif hasattr(jellyfish, "double_metaphone"):
                dmeta = jellyfish.double_metaphone(norm_name_clean)
                name_phonetic = dmeta[0] if isinstance(dmeta, (tuple, list)) else dmeta
            else:
                name_phonetic = jellyfish.metaphone(norm_name_clean)
        except Exception:
            name_phonetic = ""
    else:
        name_phonetic = ""

    # Name Character N-grams (3-grams)
    name_char_ngrams = _get_char_ngrams(norm_name_clean, n=3)

    # 3. Address Normalization
    norm_addr = unidecode(raw_address).lower()
    norm_addr = norm_addr.replace('&', ' and ')
    norm_addr_clean = norm_addr.translate(str.maketrans(string.punctuation, ' ' * len(string.punctuation)))
    norm_addr_clean = re.sub(r'\s+', ' ', norm_addr_clean).strip()

    raw_addr_tokens = norm_addr_clean.split() if norm_addr_clean else []
    # Abbreviation expansion
    expanded_addr_tokens = [STREET_ABBREVIATIONS.get(tok, tok) for tok in raw_addr_tokens]
    normalized_address = " ".join(expanded_addr_tokens)
    address_tokens = expanded_addr_tokens

    # Postal code extraction
    postal_code_guess = _extract_postal_code(raw_address)
    postal_missing = bool(not postal_code_guess)

    # City extraction
    city_guess = _extract_city(raw_address, address_tokens, postal_code_guess)
    city_missing = bool(not city_guess)

    # Street number extraction
    street_number_guess = _extract_street_number(address_tokens)

    # 4. Country
    country_out = raw_country.strip()

    record = NormalizedRecord(
        raw_name=raw_name,
        normalized_name=norm_name_clean,
        aggressive_normalized_name=aggressive_normalized_name,
        name_tokens=name_tokens,
        name_tokens_no_suffix=name_tokens_no_suffix,
        name_phonetic=name_phonetic,
        name_char_ngrams=name_char_ngrams,
        raw_address=raw_address,
        normalized_address=normalized_address,
        address_tokens=address_tokens,
        postal_code_guess=postal_code_guess,
        city_guess=city_guess,
        street_number_guess=street_number_guess,
        name_missing=name_missing,
        address_missing=address_missing,
        postal_missing=postal_missing,
        city_missing=city_missing,
        country=country_out
    )

    return record.to_dict()
