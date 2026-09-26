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

Memory Optimization (Phase 1):
- Internal indexes store compact integer IDs instead of full string entity IDs.
- String IDs are mapped via entity_id_to_int / int_to_entity_id on ingestion.
- Token posting lists are capped at max_posting_size during indexing, not just at query time,
  preventing unnecessary memory accumulation for hyper-frequent tokens.
- The external get_candidates() API is unchanged: it still returns Set[str] of string entity IDs.
- datasketch MinHashLSH stores integer keys internally (converted to strings for LSH insertion
  because MinHashLSH requires string-compatible keys; the integer values are used for all
  other in-memory structures).
"""

from collections import defaultdict
from typing import Dict, List, Set, Any, Optional, Tuple
from datasketch import MinHash, MinHashLSH


class MultiChannelBlocker:
    """
    Multi-Channel Candidate Generation Engine for Business Entity Resolution.

    Target records (Source 2 and Source 3) are indexed across multiple retrieval channels.
    Source 1 entities query the indexes to generate candidate matching entity IDs.

    Memory Optimization:
        Internal posting lists store compact integer IDs (int) rather than
        full string entity IDs, reducing per-element overhead significantly.
        String <-> int translation is handled transparently at index/query boundaries.
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
            max_posting_size: Optional threshold; once a token posting list reaches this
                size, additional entity IDs are NOT stored (saving memory). The retrieval
                logic also skips mega-block tokens during querying for the same reason.
                Defaults to 50,000.
        """
        self.lsh_threshold = lsh_threshold
        self.num_perm = num_perm
        self.max_posting_size = max_posting_size

        # --- Internal String <-> Integer ID Mappings ---
        # Only populated during index_target_records(); query-side S1 IDs are NOT mapped.
        self._entity_id_to_int: Dict[str, int] = {}
        self._int_to_entity_id: Dict[int, str] = {}
        self._next_int_id: int = 0

        # Channel A Index: dict[token_str -> set[int_id]]
        # Posting sets contain compact integer IDs to reduce memory vs. Set[str].
        self.token_index: Dict[str, Set[int]] = defaultdict(set)

        # Channel B Index: MinHashLSH for character n-gram Jaccard retrieval.
        # Keys inserted into LSH are str(int_id) to satisfy datasketch's string requirement
        # while still using compact integer values rather than full entity ID strings.
        self.lsh_index = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        self._lsh_indexed_int_ids: Set[int] = set()

        # Channel C Indexes: Composite Location Keys -> set[int_id]
        self.city_country_index: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        self.postal_country_index: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

        # Channel D Index: Weak Phonetic Index -> set[int_id]
        self.phonetic_index: Dict[str, Set[int]] = defaultdict(set)

    # ------------------------------------------------------------------
    # Internal ID mapping helpers
    # ------------------------------------------------------------------

    def _intern_entity_id(self, entity_id: str) -> int:
        """
        Maps a string entity ID to a compact integer ID, creating a new mapping
        if the entity ID has not been seen before.
        """
        if entity_id not in self._entity_id_to_int:
            int_id = self._next_int_id
            self._entity_id_to_int[entity_id] = int_id
            self._int_to_entity_id[int_id] = entity_id
            self._next_int_id += 1
        return self._entity_id_to_int[entity_id]

    def _int_ids_to_strings(self, int_ids: Set[int]) -> Set[str]:
        """Converts a set of internal integer IDs to their original string entity IDs."""
        return {self._int_to_entity_id[i] for i in int_ids if i in self._int_to_entity_id}

    # ------------------------------------------------------------------
    # Public indexing API
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Channel A: Name Token Inverted Index
    # ------------------------------------------------------------------

    def _build_token_index(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel A: Name Token Inverted Index.
        Indexes target (S2 / S3) records by their `name_tokens_no_suffix`.

        Memory optimization: posting lists store integer IDs and are capped at
        max_posting_size during *indexing* to prevent accumulation of millions of
        redundant entries for hyper-frequent tokens (e.g. "the", "international").
        The query-side max_posting_size check is preserved as an additional safeguard.
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

            int_id = self._intern_entity_id(entity_id)

            # Deduplicate tokens per record to avoid repeated entries
            unique_tokens = set(t.strip() for t in tokens if t and t.strip())

            for token in unique_tokens:
                posting = self.token_index[token]
                # Cap posting list during indexing: once the list reaches max_posting_size,
                # do not store further IDs. These mega-block tokens will be skipped at
                # query time anyway (see _get_token_candidates), so storing additional
                # entries only wastes memory.
                if self.max_posting_size is None or len(posting) < self.max_posting_size:
                    posting.add(int_id)

    # ------------------------------------------------------------------
    # Channel B: MinHash / LSH
    # ------------------------------------------------------------------

    def _build_lsh_index(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel B: Character 3-Gram MinHash / LSH Index.
        Indexes target (S2 / S3) records by their `name_char_ngrams`.

        Integer IDs are converted to strings for insertion into datasketch MinHashLSH
        (which requires hashable string-compatible keys). The mapping back to original
        entity IDs is performed in _get_lsh_candidates().
        """
        # Re-initialize LSH index on new indexing call to prevent duplicate key errors
        self.lsh_index = MinHashLSH(threshold=self.lsh_threshold, num_perm=self.num_perm)
        self._lsh_indexed_int_ids.clear()

        for record in target_records:
            entity_id = record.get("entity_id", "")
            if not entity_id:
                continue

            int_id = self._intern_entity_id(entity_id)

            if int_id in self._lsh_indexed_int_ids:
                continue

            ngrams = record.get("name_char_ngrams", [])
            if not ngrams:
                continue

            m = MinHash(num_perm=self.num_perm)
            for ngram in ngrams:
                if ngram:
                    m.update(ngram.encode("utf-8"))

            try:
                # Store as str(int_id) to satisfy datasketch's key type requirement
                # while keeping the key value compact (short numeric string vs full entity ID).
                self.lsh_index.insert(str(int_id), m)
                self._lsh_indexed_int_ids.add(int_id)
            except Exception:
                # Safely ignore duplicate or invalid key insertions
                pass

    # ------------------------------------------------------------------
    # Channel C: Location Indexes
    # ------------------------------------------------------------------

    def _build_location_indexes(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel C: Coarse Location Blocking Indexes.
        Builds (city_guess, country) and (postal_code_guess, country) indexes.
        Posting sets store integer IDs.
        """
        self.city_country_index.clear()
        self.postal_country_index.clear()

        for record in target_records:
            entity_id = record.get("entity_id", "")
            if not entity_id:
                continue

            int_id = self._intern_entity_id(entity_id)

            city = str(record.get("city_guess", "") or "").strip().lower()
            postal = str(record.get("postal_code_guess", "") or "").strip().lower()
            country = str(record.get("country", "") or "").strip().upper()

            # City + Country Composite Key (Requires non-empty city and country)
            if city and country:
                self.city_country_index[(city, country)].add(int_id)

            # Postal Code + Country Composite Key (Requires non-empty postal code and country)
            if postal and country:
                self.postal_country_index[(postal, country)].add(int_id)

    # ------------------------------------------------------------------
    # Channel D: Phonetic Index
    # ------------------------------------------------------------------

    def _build_phonetic_index(self, target_records: List[Dict[str, Any]]) -> None:
        """
        Internal implementation of Channel D: Weak Phonetic Blocking Index.
        Indexes target (S2 / S3) records by their `name_phonetic` key.
        Posting sets store integer IDs.
        """
        self.phonetic_index.clear()

        for record in target_records:
            entity_id = record.get("entity_id", "")
            if not entity_id:
                continue

            int_id = self._intern_entity_id(entity_id)

            phonetic = str(record.get("name_phonetic", "") or "").strip()
            if phonetic:
                self.phonetic_index[phonetic].add(int_id)

    # ------------------------------------------------------------------
    # Public query API — returns Set[str] of original string entity IDs
    # ------------------------------------------------------------------

    def get_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """
        Retrieves union of candidate entity IDs (S2-*, S3-*) for a Source 1 entity across all 4 channels.

        Args:
            s1_record: Normalized record dictionary for a Source 1 entity.

        Returns:
            Set[str]: Deduplicated set of candidate entity IDs (never includes S1 entity IDs).
                      Always returns original string entity IDs, never internal integer IDs.
        """
        # Gather internal integer-ID candidates from each channel
        int_candidates: Set[int] = set()

        int_candidates.update(self._get_token_candidates_int(s1_record))
        int_candidates.update(self._get_lsh_candidates_int(s1_record))
        int_candidates.update(self._get_location_candidates_int(s1_record))
        int_candidates.update(self._get_phonetic_candidates_int(s1_record))

        # Translate back to string entity IDs before returning
        candidates = self._int_ids_to_strings(int_candidates)

        # Safety filter: ensure no S1 IDs slip into candidate set
        s1_id = s1_record.get("entity_id", "")
        if s1_id in candidates:
            candidates.remove(s1_id)

        return candidates

    # ------------------------------------------------------------------
    # Channel A retrieval
    # ------------------------------------------------------------------

    def _get_token_candidates_int(self, s1_record: Dict[str, Any]) -> Set[int]:
        """
        Internal retrieval for Channel A: Name Token Inverted Index lookup.
        Queries token_index using `name_tokens_no_suffix` of the S1 query record.
        Returns a set of internal integer IDs.
        """
        candidates: Set[int] = set()

        tokens = s1_record.get("name_tokens_no_suffix")
        if tokens is None:
            tokens = s1_record.get("name_tokens", [])

        if not tokens:
            return candidates

        unique_tokens = set(t.strip() for t in tokens if t and t.strip())

        for token in unique_tokens:
            posting_list = self.token_index.get(token)
            if posting_list:
                # Safeguard for mega-blocks: skip tokens whose posting list
                # reached max_posting_size (these were capped during indexing too).
                if self.max_posting_size is not None and len(posting_list) >= self.max_posting_size:
                    continue
                candidates.update(posting_list)

        return candidates

    # ------------------------------------------------------------------
    # Channel B retrieval
    # ------------------------------------------------------------------

    def _get_lsh_candidates_int(self, s1_record: Dict[str, Any]) -> Set[int]:
        """
        Internal retrieval for Channel B: Character 3-Gram MinHash LSH lookup.
        Queries lsh_index using `name_char_ngrams` of the S1 query record.
        Returns a set of internal integer IDs (converted from str(int_id) LSH keys).
        """
        ngrams = s1_record.get("name_char_ngrams", [])
        if not ngrams or self.lsh_index is None:
            return set()

        m = MinHash(num_perm=self.num_perm)
        for ngram in ngrams:
            if ngram:
                m.update(ngram.encode("utf-8"))

        result_keys = self.lsh_index.query(m)
        # Convert str(int_id) keys back to integer IDs
        int_ids: Set[int] = set()
        for key in result_keys:
            try:
                int_ids.add(int(key))
            except (ValueError, TypeError):
                pass
        return int_ids

    # ------------------------------------------------------------------
    # Channel C retrieval
    # ------------------------------------------------------------------

    def _get_location_candidates_int(self, s1_record: Dict[str, Any]) -> Set[int]:
        """
        Internal retrieval for Channel C: Coarse Location Blocker lookup.
        Queries city_country_index and postal_country_index of the S1 record.
        Returns a set of internal integer IDs.
        """
        candidates: Set[int] = set()

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

    # ------------------------------------------------------------------
    # Channel D retrieval
    # ------------------------------------------------------------------

    def _get_phonetic_candidates_int(self, s1_record: Dict[str, Any]) -> Set[int]:
        """
        Internal retrieval for Channel D: Weak Phonetic Blocker lookup.
        Queries phonetic_index using `name_phonetic` of the S1 record.
        Returns a set of internal integer IDs.
        """
        phonetic = str(s1_record.get("name_phonetic", "") or "").strip()
        if not phonetic:
            return set()

        matches = self.phonetic_index.get(phonetic)
        return set(matches) if matches else set()

    # ------------------------------------------------------------------
    # Legacy private method aliases
    # Legacy: kept for backward compatibility with any code that calls
    # the old private methods directly (e.g. test_person_b.py tests that
    # inspect individual channels). These translate the integer results
    # back to strings so callers are unaffected.
    # ------------------------------------------------------------------

    def _get_token_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """Legacy string-returning wrapper for Channel A (backward compatibility)."""
        return self._int_ids_to_strings(self._get_token_candidates_int(s1_record))

    def _get_lsh_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """Legacy string-returning wrapper for Channel B (backward compatibility)."""
        return self._int_ids_to_strings(self._get_lsh_candidates_int(s1_record))

    def _get_location_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """Legacy string-returning wrapper for Channel C (backward compatibility)."""
        return self._int_ids_to_strings(self._get_location_candidates_int(s1_record))

    def _get_phonetic_candidates(self, s1_record: Dict[str, Any]) -> Set[str]:
        """Legacy string-returning wrapper for Channel D (backward compatibility)."""
        return self._int_ids_to_strings(self._get_phonetic_candidates_int(s1_record))
