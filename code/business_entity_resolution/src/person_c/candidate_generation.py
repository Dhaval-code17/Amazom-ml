"""
Person C — candidate_generation.py
Stage 3/4/5 integration shim: wraps Person B's blocking + pruning pipeline
into a single deterministic function consumed by Person C's feature engineering
and classifier stages.

Public API
----------
generate_candidate_pairs(source1_records, target_records, cap=200) -> pd.DataFrame

Columns returned
----------------
source1_entity_id    str   — entity_id of the S1 query record
candidate_entity_ids str   — comma-separated S2/S3 candidate IDs, sorted
                             deterministically; empty string if no candidates

Design constraints enforced here
---------------------------------
- No ML, no feature engineering, no ground-truth usage.
- No TF-IDF, no calibration, no thresholding.
- Candidate IDs are deduplicated, sorted (deterministic), and validated to
  exist in target_map and never equal the querying S1 entity_id.
- Relies exclusively on Person A's normalize_record output schema and
  Person B's MultiChannelBlocker / rank_and_prune_candidates interfaces.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

# ---------------------------------------------------------------------------
# sys.path: ensure the src/ directory is importable as a package root.
# All modules in this project use bare imports (e.g. `from normalization import …`).
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_b.blocking import MultiChannelBlocker
from person_b.pruning import rank_and_prune_candidates


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_records(records: List[Dict], label: str) -> None:
    """
    Raise ValueError if any record in *records* is not a dict or lacks entity_id.

    Parameters
    ----------
    records : list of dicts — the record list to validate.
    label   : str — human-readable name used in error messages ("source1" / "target").
    """
    for i, rec in enumerate(records):
        if not isinstance(rec, dict):
            raise ValueError(
                f"{label}[{i}] is not a dict (got {type(rec).__name__!r}). "
                "All records must be plain dicts produced by normalize_record() "
                "with 'entity_id' added manually."
            )
        if "entity_id" not in rec or not rec["entity_id"]:
            raise ValueError(
                f"{label}[{i}] is missing a non-empty 'entity_id' key. "
                "Add it with: norm_dict['entity_id'] = entity_id"
            )


def _build_target_map(target_records: List[Dict]) -> Dict[str, Dict]:
    """
    Build entity_id → full normalized record dict for fast O(1) lookup.

    Duplicates are silently kept as last-write-wins (same ID appearing twice
    is a data issue upstream, not this module's concern).

    Parameters
    ----------
    target_records : list of normalized dicts (S2 + S3) with entity_id.

    Returns
    -------
    Dict[str, dict] mapping entity_id -> full normalized record dict.
    """
    return {rec["entity_id"]: rec for rec in target_records}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_candidate_pairs(
    source1_records: List[Dict],
    target_records: List[Dict],
    cap: Optional[int] = 200,
    lsh_threshold: float = 0.5,
    num_perm: int = 128,
    max_posting_size: Optional[int] = 50000,
) -> pd.DataFrame:
    """
    Generate a DataFrame of blocked + pruned candidate pairs for Person C's
    feature-engineering and classifier stages.

    Uses Person B's MultiChannelBlocker (4-channel blocking) followed by
    rank_and_prune_candidates (cheap pre-ranking + cap).  No ML, no ground
    truth, no feature computation happens here.

    Parameters
    ----------
    source1_records : List[Dict]
        Normalized Source-1 record dicts.  Each must contain 'entity_id' and
        the 18 fields produced by Person A's normalize_record().
    target_records : List[Dict]
        Normalized Source-2 + Source-3 record dicts.  Same schema + entity_id.
    cap : int or None, default 200
        Maximum number of candidates retained per S1 entity after pre-ranking.
        None means no cap (all ranked candidates are kept).
    lsh_threshold : float, default 0.5
        Jaccard similarity threshold passed to MinHashLSH (Channel B).
    num_perm : int, default 128
        Number of MinHash permutations (Channel B).
    max_posting_size : int or None, default 50_000
        Maximum token posting-list size before a token is skipped as a
        mega-block suppressor (Channel A).

    Returns
    -------
    pd.DataFrame with columns:
        source1_entity_id    (str)  — S1 entity_id
        candidate_entity_ids (str)  — comma-separated S2/S3 IDs, sorted
                                      lexicographically; "" if no candidates.

    Guarantees
    ----------
    - One row per S1 entity (same order as source1_records).
    - Candidate IDs are unique per row (deduplication applied).
    - Candidate IDs are sorted lexicographically (deterministic).
    - S1 entity_ids never appear in candidate_entity_ids.
    - All returned candidate IDs exist in target_map (i.e. in target_records).
    - No candidate IDs from outside target_records can appear in output.
    """
    # ------------------------------------------------------------------
    # 1. Input validation
    # ------------------------------------------------------------------
    if not isinstance(source1_records, list):
        raise TypeError(
            f"source1_records must be a list, got {type(source1_records).__name__!r}"
        )
    if not isinstance(target_records, list):
        raise TypeError(
            f"target_records must be a list, got {type(target_records).__name__!r}"
        )

    _validate_records(source1_records, "source1_records")
    _validate_records(target_records, "target_records")

    # ------------------------------------------------------------------
    # 2. Build target_map for O(1) full-record retrieval
    # ------------------------------------------------------------------
    target_map: Dict[str, Dict] = _build_target_map(target_records)

    # ------------------------------------------------------------------
    # 3. Build and index MultiChannelBlocker (once, over all target records)
    # ------------------------------------------------------------------
    blocker = MultiChannelBlocker(
        lsh_threshold=lsh_threshold,
        num_perm=num_perm,
        max_posting_size=max_posting_size,
    )
    blocker.index_target_records(target_records)

    # ------------------------------------------------------------------
    # 4. Per S1 entity: block → prune → deduplicate → sort
    # ------------------------------------------------------------------
    rows: List[Dict] = []

    for s1_rec in source1_records:
        s1_id: str = s1_rec["entity_id"]

        # 4a. Blocking: get raw candidate IDs from all 4 channels
        raw_candidate_ids = blocker.get_candidates(s1_rec)
        # get_candidates() already filters out s1_id; we enforce it anyway.
        raw_candidate_ids.discard(s1_id)

        # 4b. Restrict to IDs that actually exist in target_map.
        #     IDs returned by the blocker that are absent from target_map would
        #     indicate an indexing bug — we silently drop them here so the rest
        #     of the pipeline is never handed a phantom ID.
        valid_candidate_ids = {
            cid for cid in raw_candidate_ids if cid in target_map
        }

        # 4c. Retrieve full normalized records for cheap pre-ranking
        candidate_records: List[Dict] = [
            target_map[cid] for cid in valid_candidate_ids
        ]

        # 4d. Cheap pre-ranking + cap (Person B's pruning)
        #     rank_and_prune_candidates returns:
        #         [{"entity_id": str, "score": float}, ...]
        #     sorted descending by score, then ascending entity_id (tie-break).
        pruned = rank_and_prune_candidates(s1_rec, candidate_records, cap=cap)

        # 4e. Extract entity_ids, deduplicate (should be clean already), sort
        pruned_ids_seen: set = set()
        pruned_ids_ordered: List[str] = []
        for item in pruned:
            cid = item["entity_id"]
            if cid not in pruned_ids_seen and cid != s1_id:
                pruned_ids_seen.add(cid)
                pruned_ids_ordered.append(cid)

        # 4f. Deterministic lexicographic sort
        #     rank_and_prune_candidates already sorts by (-score, entity_id).
        #     We re-sort lexicographically here to guarantee determinism
        #     independent of scoring ties or future pruning changes.
        final_ids = sorted(pruned_ids_ordered)

        # 4g. Serialise to comma-separated string (empty string = no candidates)
        candidate_entity_ids_str = ",".join(final_ids)

        rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": candidate_entity_ids_str,
            }
        )

    # ------------------------------------------------------------------
    # 5. Assemble DataFrame with guaranteed column order
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    return df
