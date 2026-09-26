from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass(frozen=True)
class NormalizedRecord:
    # Name fields
    raw_name: str
    normalized_name: str
    aggressive_normalized_name: str
    name_tokens: List[str]
    name_tokens_no_suffix: List[str]
    name_phonetic: str
    name_char_ngrams: List[str]

    # Address fields
    raw_address: str
    normalized_address: str
    address_tokens: List[str]
    postal_code_guess: str
    city_guess: str
    street_number_guess: str

    # Missingness flags
    name_missing: bool
    address_missing: bool
    postal_missing: bool
    city_missing: bool

    # Country
    country: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_name": self.raw_name,
            "normalized_name": self.normalized_name,
            "aggressive_normalized_name": self.aggressive_normalized_name,
            "name_tokens": self.name_tokens,
            "name_tokens_no_suffix": self.name_tokens_no_suffix,
            "name_phonetic": self.name_phonetic,
            "name_char_ngrams": self.name_char_ngrams,
            "raw_address": self.raw_address,
            "normalized_address": self.normalized_address,
            "address_tokens": self.address_tokens,
            "postal_code_guess": self.postal_code_guess,
            "city_guess": self.city_guess,
            "street_number_guess": self.street_number_guess,
            "name_missing": self.name_missing,
            "address_missing": self.address_missing,
            "postal_missing": self.postal_missing,
            "city_missing": self.city_missing,
            "country": self.country,
        }
