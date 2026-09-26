"""
Multi-Channel Blocking Engine — Stage 3 Candidate Generation.

This module implements candidate blocking to drastically cut down the search space
from O(N * M) all-pairs comparisons down to O(candidates per entity) by indexing
Source 2 and Source 3 target records and querying them with Source 1 reference records.

Stage 3A: Name Token Inverted Index Channel
Stage 3B: Character 3-Gram MinHash / LSH Channel
Stage 3C: Coarse City/Postal + Country Location Channel
Stage 3D: Weak Phonetic Blocking Channel
- Uses `name_phonetic` (Double Metaphone key) from Person A's normalized record.
- Inverted index lookup mapping exact phonetic keys to candidate target entity IDs.
- Supplementary weak channel to capture phonetic spelling variations.
"""

from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple
from datasketch import MinHash, MinHashLSH

class MultiChannelBlocker:
    """
    Multi-Channel Candidate Generation Engine for Business Entity Resolution.
    
    Target records (Source 2 and Source 3) are indexed across multiple retrieval channels.
    Source 1 entities query the indexes to generate candidate matching entity IDs.
    """
    
    def __init__(
        self,
        lsh_threshold: float = 0.5,
        num_perm: int = 128,
        max_posting_size: Optional[int] = 50000
    ):
        """
        Initialize the MultiChannelBlocker.
        
        Args:
            lsh_threshold: Jaccard similarity threshold for MinHash LSH (Stage 3B).
            num_perm: Number of permutations for MinHash signatures (Stage 3B).
            max_posting_size: Optional threshold to skip hyper-frequent mega-block tokens.
        """
        self.lsh_threshold = lsh_threshold
        self.num_perm = num_perm
        self.max_posting_size = max_posting_size
        
        # Channel A Index: dict[token_str -> set[entity_id]]
        self.token_index: Dict[str, Set[str]] = defaultdict(set)
        
        # Channel B Index: MinHashLSH for character n-gram Jaccard retrieval
        self.lsh_index = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        self._lsh_indexed_ids: Set[str] = set()
        
        # Channel C Indexes: Composite Location Keys -> set[entity_id]
        self.city_country_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        self.postal_country_index: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        
        # Channel D Index: Weak Phonetic Index -> set[entity_id]
        self.phonetic_index: Dict[str, Set[str]] = defaultdict(set)

    def index_target_records(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Indexes Source 2 and Source 3 target records across all four blocking channels.
        
        Args:
            target_records: List of normalized record dictionaries (produced by normalize_record)
                            or dicts containing 'entity_id' and normalized fields.
        """
        self._build_token_index(target_records)
        self._build_lsh_index(target_records)
        self._build_location_indexes(target_records)
        self._build_phonetic_index(target_records)

    def _build_token_index(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel A: Name Token Inverted Index.
        Indexes target (S2 / S3) records by their `name_tokens_no_suffix`.
        """
        self.token_index.clear()
        
        for record in target_records:
            entity_id = record.get("entity_id", "")
            if not entity_id:
                continue
                
            tokens = record.get("name_tokens_no_suffix")
            if tokens is None:
                tokens = record.get("name_tokens", [])
                
            if not tokens:
                continue
                
            # Deduplicate tokens per record to avoid repeated entries
            unique_tokens = set(t.strip() for t in tokens if t and t.strip())
            
            for token in unique_tokens:
                self.token_index[token].add(entity_id)

    def _build_lsh_index(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel B: Character 3-Gram MinHash / LSH Index.
        Indexes target (S2 / S3) records by their `name_char_ngrams`.
        """
        # Re-initialize LSH index on new indexing call to prevent duplicate key errors
        self.lsh_index = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        self._lsh_indexed_ids.clear()
        
        for record in target_records:
            entity_id = record.get("entity_id", "")
            if not entity_id or entity_id in self._lsh_indexed_ids:
                continue
                
            ngrams = record.get("name_char_ngrams", [])
            if not ngrams:
                continue
                
            m = MinHash(num_perm=self.num_perm)
            for ngram in ngrams:
                if ngram:
                    m.update(ngram.encode("utf-8"))
                    
            try:
                self.lsh_index.insert(entity_id, m)
                self._lsh_indexed_ids.add(entity_id)
            except Exception:
                # Safely ignore duplicate or invalid key insertions
                pass

    def _build_location_indexes(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel C: Coarse Location Blocking Indexes.
        Builds (city_guess, country) and (postal_code_guess, country) indexes.
        """
        self.city_country_index.clear()
        self.postal_country_index.clear()
        
        for record in target_records:
            entity_id = record.get("entity_id", "")
            if not entity_id:
                continue
                
            city = str(record.get("city_guess", "") or "").strip().lower()
            postal = str(record.get("postal_code_guess", "") or "").strip().lower()
            country = str(record.get("country", "") or "").strip().upper()
            
            # City + Country Composite Key (Requires non-empty city and country)
            if city and country:
                self.city_country_index[(city, country)].add(entity_id)
                
            # Postal Code + Country Composite Key (Requires non-empty postal code and country)
            if postal and country:
                self.postal_country_index[(postal, country)].add(entity_id)

    def _build_phonetic_index(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel D: Weak Phonetic Blocking Index.
        Indexes target (S2 / S3) records by their `name_phonetic` key.
        """
        self.phonetic_index.clear()
        
        for record in target_records:
            entity_id = record.get("entity_id", "")
            if not entity_id:
                continue
                
            phonetic = str(record.get("name_phonetic", "") or "").strip()
            if phonetic:
                self.phonetic_index[phonetic].add(entity_id)

    def get_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """
        Retrieves union of candidate entity IDs (S2-*, S3-*) for a Source 1 entity across all 4 channels.
        
        Args:
            s1_record: Normalized record dictionary for a Source 1 entity.
            
        Returns:
            Set[str]: Deduplicated set of candidate entity IDs (never includes S1 entity IDs).
        """
        candidates: Set[str] = set()
        
        # Channel A: Name Token Inverted Index lookup
        token_candidates = self._get_token_candidates(s1_record)
        
        # Channel B: Character 3-Gram MinHash LSH lookup
        lsh_candidates = self._get_lsh_candidates(s1_record)
        
        # Channel C: Coarse Location Blocker lookup
        location_candidates = self._get_location_candidates(s1_record)
        
        # Channel D: Weak Phonetic Blocker lookup
        phonetic_candidates = self._get_phonetic_candidates(s1_record)
        
        # Union all 4 candidate channels
        candidates = (
            token_candidates
            | lsh_candidates
            | location_candidates
            | phonetic_candidates
        )
        
        # Safety filter: ensure no S1 IDs slip into candidate set
        s1_id = s1_record.get("entity_id", "")
        if s1_id in candidates:
            candidates.remove(s1_id)
            
        return candidates

    def _get_token_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """
        Internal retrieval for Channel A: Name Token Inverted Index lookup.
        Queries token_index using `name_tokens_no_suffix` of the S1 query record.
        """
        candidates: Set[str] = set()
        
        tokens = s1_record.get("name_tokens_no_suffix")
        if tokens is None:
            tokens = s1_record.get("name_tokens", [])
            
        if not tokens:
            return candidates
            
        unique_tokens = set(t.strip() for t in tokens if t and t.strip())
        
        for token in unique_tokens:
            posting_list = self.token_index.get(token)
            if posting_list:
                # Safeguard for mega-blocks: skip tokens whose posting list exceeds max_posting_size
                if self.max_posting_size is not None and len(posting_list) > self.max_posting_size:
                    continue
                candidates.update(posting_list)
                
        return candidates

    def _get_lsh_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """
        Internal retrieval for Channel B: Character 3-Gram MinHash LSH lookup.
        Queries lsh_index using `name_char_ngrams` of the S1 query record.
        """
        ngrams = s1_record.get("name_char_ngrams", [])
        if not ngrams or self.lsh_index is None:
            return set()
            
        m = MinHash(num_perm=self.num_perm)
        for ngram in ngrams:
            if ngram:
                m.update(ngram.encode("utf-8"))
                
        result_keys = self.lsh_index.query(m)
        return set(result_keys)

    def _get_location_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """
        Internal retrieval for Channel C: Coarse Location Blocker lookup.
        Queries city_country_index and postal_country_index of the S1 record.
        """
        candidates: Set[str] = set()
        
        city = str(s1_record.get("city_guess", "") or "").strip().lower()
        postal = str(s1_record.get("postal_code_guess", "") or "").strip().lower()
        country = str(s1_record.get("country", "") or "").strip().upper()
        
        if city and country:
            city_matches = self.city_country_index.get((city, country))
            if city_matches:
                candidates.update(city_matches)
                
        if postal and country:
            postal_matches = self.postal_country_index.get((postal, country))
            if postal_matches:
                candidates.update(postal_matches)
                
        return candidates

    def _get_phonetic_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """
        Internal retrieval for Channel D: Weak Phonetic Blocker lookup.
        Queries phonetic_index using `name_phonetic` of the S1 record.
        """
        phonetic = str(s1_record.get("name_phonetic", "") or "").strip()
        if not phonetic:
            return set()
            
        matches = self.phonetic_index.get(phonetic)
        return set(matches) if matches else set()
