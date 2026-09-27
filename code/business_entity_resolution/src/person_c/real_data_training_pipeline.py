"""
Person C -- real_data_training_pipeline.py

Step 6: Real-Data Development Training & Validation Pipeline.

Runs the COMPLETE Person A -> Person B -> Person C pipeline on a controlled,
deterministic sample of the real competition training data.

Design constraints
------------------
- NEVER loads the entire ~10.3 M S2+S3 corpus into RAM.
- Does NOT redesign Person A or Person B.
- Calls Person B via the existing generate_candidate_pairs() API.
- Calls Person C via build_training_dataset(), run_group_kfold_baseline(),
  select_hard_negatives(), tune_decision_threshold(), fit_oof_calibrator(),
  compare_oof_calibration().
- All training/threshold/calibration decisions are based ONLY on training data
  and OOF predictions -- never on a held-out test set.
- Produces a reproducible JSON report plus a saved model configuration.
- Deterministic with the supplied --seed.

Memory-safe target loading strategy
-------------------------------------
The full S2+S3 corpus is ~10.3 M records and cannot be loaded at once.
Instead we build a "development micro-corpus":
  1. Collect the true target IDs for every sampled S1 entity (guaranteed present).
  2. Stream each target TSV once, normalising ONLY records whose entity_id is
     in the true-ID set  OR  whose entity_id passes a deterministic random
     keep-with-probability filter (noise_fraction).  This keeps the target
     corpus size manageable while ensuring a realistic negative pool.
  3. Target-record limit (--target-sample-size) hard-caps the total retained
     records so the experiment always finishes quickly.

Typical invocation
------------------
  cd code/business_entity_resolution/src
  python person_c/real_data_training_pipeline.py \
      --data-dir "D:/amazom ml/6ab10eb3b23ba_student_resource/student_resource/dataset/train" \
      --sample-size 500 \
      --target-sample-size 50000 \
      --noise-fraction 0.005 \
      --seed 42 \
      --n-splits 5 \
      --negative-ratio 3.0 \
      --cap 200 \
      --output-dir person_c/pipeline_run_output
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path -- bare-module imports as per project convention
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from normalization import normalize_record
from person_c.candidate_generation import generate_candidate_pairs
from person_c.training_data import build_training_dataset, extract_X_y_metadata
from person_c.model import (
    MODEL_FEATURES,
    run_group_kfold_baseline,
    select_hard_negatives,
    tune_decision_threshold,
    fit_oof_calibrator,
    compare_oof_calibration,
    calculate_evaluation_metrics,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
BUFFER_SIZE = 8 * 1024 * 1024   # 8 MB -- avoids OSError 22 on Windows large files
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
DATE_FORMAT = "%H:%M:%S"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path) -> logging.Logger:
    """Creates a logger that writes to both stdout and a log file."""
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "pipeline.log", encoding="utf-8", mode="w")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def banner(logger: logging.Logger, title: str) -> None:
    bar = "=" * 72
    logger.info(bar)
    logger.info("  %s", title)
    logger.info(bar)


def section(logger: logging.Logger, title: str) -> None:
    logger.info("\n--- %s ---", title)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_sampled_s1_records(
    filepath: Path,
    sample_size: int,
    random_seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Efficiently loads a deterministic sample of S1 records.

    Reads ALL raw rows first (minimal I/O), samples without normalisation,
    then normalises ONLY the sampled rows.  Avoids paying the normalisation
    cost for every S1 record in the full ~2.2 M S1 corpus.

    Parameters
    ----------
    filepath    : Path to train_source1.tsv
    sample_size : Number of S1 entities to sample
    random_seed : Seed for reproducibility

    Returns
    -------
    List of normalised dicts, each with an 'entity_id' key added.
    """
    raw_rows: List[Dict[str, str]] = []
    if not filepath.exists():
        return []

    with open(filepath, "r", encoding="utf-8", errors="replace",
              buffering=BUFFER_SIZE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("entity_id", ""):
                raw_rows.append(dict(row))

    rng = random.Random(random_seed)
    sampled_raw = rng.sample(raw_rows, min(sample_size, len(raw_rows)))

    records: List[Dict[str, Any]] = []
    for row in sampled_raw:
        entity_id = row["entity_id"]
        norm = normalize_record(
            row.get("business_name", ""),
            row.get("business_address", ""),
            row.get("country", ""),
        )
        norm["entity_id"] = entity_id
        records.append(norm)

    return records


def load_ground_truth_raw(filepath: Path) -> Dict[str, str]:
    """
    Reads train_ground_truth.tsv into a raw dict  s1_id -> comma_separated_ids.

    Kept as raw strings to avoid parsing overhead -- the true IDs are only
    extracted for the sampled S1 entities.

    Parameters
    ----------
    filepath : Path to train_ground_truth.tsv

    Returns
    -------
    Dict mapping source1_entity_id -> matched_entity_ids (comma-separated str).
    """
    gt_map: Dict[str, str] = {}
    if not filepath.exists():
        return gt_map

    with open(filepath, "r", encoding="utf-8", errors="replace",
              buffering=BUFFER_SIZE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            s1_id = row.get("source1_entity_id", "")
            if s1_id:
                gt_map[s1_id] = row.get("matched_entity_ids", "")

    return gt_map


def extract_true_target_ids(
    s1_records: List[Dict[str, Any]],
    gt_map_raw: Dict[str, str],
) -> Set[str]:
    """
    Returns the union of all true target entity_ids for the given S1 sample.

    Parameters
    ----------
    s1_records  : Normalised S1 records with 'entity_id'.
    gt_map_raw  : Raw ground-truth dict (s1_id -> comma-separated ids).

    Returns
    -------
    Set[str] -- all target entity_ids that are true matches for at least
               one entity in s1_records.
    """
    true_ids: Set[str] = set()
    for rec in s1_records:
        matched_str = gt_map_raw.get(rec["entity_id"], "")
        if matched_str:
            true_ids.update(x.strip() for x in matched_str.split(",") if x.strip())
    return true_ids


def stream_target_micro_corpus(
    filepath: Path,
    true_ids: Set[str],
    noise_fraction: float,
    random_seed: int,
    target_sample_size: Optional[int],
    logger: logging.Logger,
) -> List[Dict[str, Any]]:
    """
    Memory-safe: streams one target TSV, keeping only:
      - records whose entity_id is in true_ids (guaranteed),
      - records that pass a probabilistic noise filter.

    Hard-stops at target_sample_size total retained records.
    Normalises ONLY the retained rows.

    Parameters
    ----------
    filepath           : Path to train_source2.tsv or train_source3.tsv.
    true_ids           : Set of entity_ids that must be retained.
    noise_fraction     : Probability [0,1] that a non-true record is kept.
    random_seed        : RNG seed for reproducibility.
    target_sample_size : Hard cap on total retained records from this file
                         (None = no cap).
    logger             : Logger instance.

    Returns
    -------
    List of normalised dicts with 'entity_id' key.
    """
    records: List[Dict[str, Any]] = []
    if not filepath.exists():
        logger.warning("  File not found, skipping: %s", filepath)
        return records

    rng = random.Random(random_seed)
    kept_true = 0
    kept_noise = 0

    with open(filepath, "r", encoding="utf-8", errors="replace",
              buffering=BUFFER_SIZE, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entity_id = row.get("entity_id", "")
            if not entity_id:
                continue

            is_true = entity_id in true_ids
            keep_noise = (not is_true) and (rng.random() < noise_fraction)

            if not (is_true or keep_noise):
                continue

            norm = normalize_record(
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", ""),
            )
            norm["entity_id"] = entity_id
            records.append(norm)

            if is_true:
                kept_true += 1
            else:
                kept_noise += 1

            if target_sample_size is not None and len(records) >= target_sample_size:
                logger.info(
                    "    [hard cap] reached %d records, stopping stream of %s",
                    target_sample_size, filepath.name,
                )
                break

    logger.info(
        "    %s: %d true + %d noise = %d total retained",
        filepath.name, kept_true, kept_noise, len(records),
    )
    return records


def build_gt_dict_for_sample(
    s1_records: List[Dict[str, Any]],
    gt_map_raw: Dict[str, str],
) -> Dict[str, Set[str]]:
    """
    Converts the raw ground-truth strings for the sampled S1 entities into
    the dict[str, set[str]] format consumed by build_training_dataset().

    Parameters
    ----------
    s1_records  : Sampled S1 normalised records with 'entity_id'.
    gt_map_raw  : Full raw ground-truth dict.

    Returns
    -------
    Dict mapping s1_id -> set of matched target entity_ids.
    """
    gt_dict: Dict[str, Set[str]] = {}
    for rec in s1_records:
        s1_id = rec["entity_id"]
        matched_str = gt_map_raw.get(s1_id, "")
        if matched_str:
            gt_dict[s1_id] = {x.strip() for x in matched_str.split(",") if x.strip()}
        else:
            gt_dict[s1_id] = set()
    return gt_dict


# ---------------------------------------------------------------------------
# Validation helpers (used independently so they can be unit-tested)
# ---------------------------------------------------------------------------

def validate_feature_matrix(
    X: pd.DataFrame,
    y: pd.Series,
) -> None:
    """
    Asserts strict correctness of the feature matrix X and label vector y.

    Checks
    ------
    1. Exactly 45 feature columns.
    2. Column names match MODEL_FEATURES exactly and in order.
    3. No identifier columns leaked into X.
    4. No 'label' column leaked into X.
    5. No NaN values in X.
    6. No Inf values in X.
    7. y is binary (only 0 and 1).

    Parameters
    ----------
    X : Feature DataFrame.
    y : Label Series.

    Raises
    ------
    AssertionError if any check fails.
    """
    assert list(X.columns) == MODEL_FEATURES, (
        "Feature columns mismatch.\n"
        f"  Expected: {MODEL_FEATURES}\n"
        f"  Got: {list(X.columns)}"
    )
    assert len(X.columns) == 45, f"Expected 45 features, got {len(X.columns)}"
    assert "source1_entity_id" not in X.columns, "source1_entity_id leaked into feature matrix"
    assert "candidate_entity_id" not in X.columns, "candidate_entity_id leaked into feature matrix"
    assert "label" not in X.columns, "label column leaked into feature matrix X"
    assert not X.isnull().values.any(), "NaN values found in feature matrix X"
    assert not np.isinf(X.values).any(), "Infinite values found in feature matrix X"
    unique_labels = set(y.unique())
    assert unique_labels.issubset({0, 1}), f"y contains non-binary values: {unique_labels}"


def validate_candidates_contain_true_matches(
    candidate_pairs: pd.DataFrame,
    gt_dict: Dict[str, Set[str]],
    target_ids_in_corpus: Set[str],
) -> Dict[str, Any]:
    """
    Checks that for each S1 entity, all true target IDs present in the indexed
    micro-corpus are also present in the candidate set.

    Parameters
    ----------
    candidate_pairs     : Output of generate_candidate_pairs().
    gt_dict             : Dict mapping s1_id -> set of true target ids.
    target_ids_in_corpus: Set of all entity_ids in the target micro-corpus.

    Returns
    -------
    Dict containing:
        - total_true_available : int
        - total_true_retrieved : int
        - micro_corpus_recall  : float
        - missed_pairs         : list of (s1_id, missed_target_id) tuples
    """
    cand_map: Dict[str, Set[str]] = {}
    for _, row in candidate_pairs.iterrows():
        s1_id = row["source1_entity_id"]
        cids_str = row["candidate_entity_ids"]
        if cids_str and isinstance(cids_str, str) and cids_str.strip():
            cand_map[s1_id] = {c.strip() for c in cids_str.split(",") if c.strip()}
        else:
            cand_map[s1_id] = set()

    total_true_available = 0
    total_true_retrieved = 0
    missed_pairs: List[Tuple[str, str]] = []

    for s1_id, true_ids in gt_dict.items():
        true_in_corpus = true_ids & target_ids_in_corpus
        total_true_available += len(true_in_corpus)
        candidates = cand_map.get(s1_id, set())
        retrieved = true_in_corpus & candidates
        total_true_retrieved += len(retrieved)
        for tid in true_in_corpus - candidates:
            missed_pairs.append((s1_id, tid))

    recall = (
        total_true_retrieved / total_true_available
        if total_true_available > 0
        else 0.0
    )
    return {
        "total_true_available": total_true_available,
        "total_true_retrieved": total_true_retrieved,
        "micro_corpus_recall": recall,
        "missed_pairs": missed_pairs,
    }


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------

def _fmt(v: Any) -> str:
    """Format a scalar value for display (handles float, int, other)."""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


# ---------------------------------------------------------------------------
# Artifact saving
# ---------------------------------------------------------------------------

def save_artifacts(
    output_dir: Path,
    report: Dict[str, Any],
    final_model: Any,
    best_threshold: float,
    calibration_method: str,
    logger: logging.Logger,
) -> None:
    """
    Saves reproducibility artifacts:
      - pipeline_report.json   : Full JSON report
      - model_config.json      : Model parameters and feature list
      - threshold.txt          : Selected decision threshold
      - calibration.txt        : Selected calibration method
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / "pipeline_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("  Saved report        -> %s", report_path)

    model_config = {
        "feature_names": MODEL_FEATURES,
        "num_features": len(MODEL_FEATURES),
        "model_params": getattr(final_model, "get_params", lambda: {})(),
        "best_threshold": best_threshold,
        "calibration_method": calibration_method,
    }
    config_path = output_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, default=str)
    logger.info("  Saved model config  -> %s", config_path)

    thresh_path = output_dir / "threshold.txt"
    thresh_path.write_text(str(best_threshold), encoding="utf-8")
    logger.info("  Saved threshold     -> %s", thresh_path)

    cal_path = output_dir / "calibration.txt"
    cal_path.write_text(calibration_method, encoding="utf-8")
    logger.info("  Saved calibration   -> %s", cal_path)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_dir: Path,
    sample_size: int = 500,
    target_sample_size: int = 50_000,
    noise_fraction: float = 0.005,
    seed: int = 42,
    n_splits: int = 5,
    negative_ratio: float = 3.0,
    cap: int = 200,
    output_dir: Path = Path("person_c/pipeline_run_output"),
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Full real-data development training pipeline.

    Returns
    -------
    Dict containing all key metrics and statistics for the run.
    """
    if logger is None:
        logger = logging.getLogger("pipeline")

    t_wall = time.perf_counter()
    report: Dict[str, Any] = {
        "config": {
            "data_dir": str(data_dir),
            "sample_size": sample_size,
            "target_sample_size": target_sample_size,
            "noise_fraction": noise_fraction,
            "seed": seed,
            "n_splits": n_splits,
            "negative_ratio": negative_ratio,
            "cap": cap,
        }
    }

    # ------------------------------------------------------------------
    # STEP 1 -- Load S1 + ground truth
    # ------------------------------------------------------------------
    section(logger, "STEP 1: Load S1 records + ground truth")
    t0 = time.perf_counter()

    s1_path = data_dir / "train_source1.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    if not s1_path.exists():
        raise FileNotFoundError(f"S1 file not found: {s1_path}")
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    s1_records = load_sampled_s1_records(s1_path, sample_size, random_seed=seed)
    gt_map_raw = load_ground_truth_raw(gt_path)
    s1_records = [r for r in s1_records if r["entity_id"] in gt_map_raw]

    logger.info(
        "  Loaded %d S1 records (after GT filter) | GT entries: %d | %.1fs",
        len(s1_records), len(gt_map_raw), time.perf_counter() - t0,
    )
    report["step1"] = {"s1_loaded": len(s1_records), "gt_entries": len(gt_map_raw)}

    if not s1_records:
        raise RuntimeError("No S1 records loaded. Check data_dir and file paths.")

    # ------------------------------------------------------------------
    # STEP 2 -- Extract true target IDs for the sample
    # ------------------------------------------------------------------
    section(logger, "STEP 2: Identify required true target IDs")
    true_target_ids = extract_true_target_ids(s1_records, gt_map_raw)
    logger.info(
        "  True target IDs needed: %d (across %d S1 entities)",
        len(true_target_ids), len(s1_records),
    )
    report["step2"] = {
        "true_target_ids_required": len(true_target_ids),
        "s1_entities": len(s1_records),
    }

    # ------------------------------------------------------------------
    # STEP 3 -- Build target micro-corpus (memory-safe streaming)
    # ------------------------------------------------------------------
    section(logger, "STEP 3: Stream + build target micro-corpus")
    t0 = time.perf_counter()
    per_file_cap = target_sample_size // 2 if target_sample_size else None

    s2_records = stream_target_micro_corpus(
        data_dir / "train_source2.tsv",
        true_ids=true_target_ids,
        noise_fraction=noise_fraction,
        random_seed=seed,
        target_sample_size=per_file_cap,
        logger=logger,
    )
    gc.collect()

    s3_records = stream_target_micro_corpus(
        data_dir / "train_source3.tsv",
        true_ids=true_target_ids,
        noise_fraction=noise_fraction,
        random_seed=seed + 1,
        target_sample_size=per_file_cap,
        logger=logger,
    )
    gc.collect()

    target_records = s2_records + s3_records
    target_ids_in_corpus: Set[str] = {r["entity_id"] for r in target_records}
    true_in_corpus = true_target_ids & target_ids_in_corpus

    logger.info(
        "  Target micro-corpus: S2=%d  S3=%d  total=%d | "
        "true IDs in corpus: %d/%d | %.1fs",
        len(s2_records), len(s3_records), len(target_records),
        len(true_in_corpus), len(true_target_ids),
        time.perf_counter() - t0,
    )
    report["step3"] = {
        "s2_records": len(s2_records),
        "s3_records": len(s3_records),
        "total_target_records": len(target_records),
        "true_ids_in_corpus": len(true_in_corpus),
        "true_ids_required": len(true_target_ids),
        "coverage_pct": (
            round(100 * len(true_in_corpus) / len(true_target_ids), 2)
            if true_target_ids else 0.0
        ),
    }

    # ------------------------------------------------------------------
    # STEP 4 -- Person B candidate generation
    # ------------------------------------------------------------------
    section(logger, "STEP 4: Person B candidate generation")
    t0 = time.perf_counter()

    candidate_pairs = generate_candidate_pairs(
        source1_records=s1_records,
        target_records=target_records,
        cap=cap,
    )

    total_cand_pairs = int(candidate_pairs["candidate_entity_ids"].apply(
        lambda s: len(s.split(",")) if isinstance(s, str) and s.strip() else 0
    ).sum())
    s1_with_cands = int(candidate_pairs["candidate_entity_ids"].apply(
        lambda s: bool(isinstance(s, str) and s.strip())
    ).sum())
    avg_cands = total_cand_pairs / len(s1_records) if s1_records else 0.0

    logger.info(
        "  S1 entities: %d | with candidates: %d | "
        "total pairs: %d | avg per S1: %.1f | %.1fs",
        len(s1_records), s1_with_cands, total_cand_pairs, avg_cands,
        time.perf_counter() - t0,
    )

    gt_dict_sample = build_gt_dict_for_sample(s1_records, gt_map_raw)
    recall_check = validate_candidates_contain_true_matches(
        candidate_pairs, gt_dict_sample, target_ids_in_corpus
    )
    logger.info(
        "  Micro-corpus candidate recall: %d/%d (%.2f%%)",
        recall_check["total_true_retrieved"],
        recall_check["total_true_available"],
        100 * recall_check["micro_corpus_recall"],
    )
    if recall_check["missed_pairs"]:
        logger.warning(
            "  Missed %d true pairs in candidates (see JSON report)",
            len(recall_check["missed_pairs"]),
        )

    report["step4"] = {
        "s1_entities": len(s1_records),
        "s1_with_candidates": s1_with_cands,
        "total_candidate_pairs": total_cand_pairs,
        "avg_candidates_per_s1": round(avg_cands, 2),
        "true_available_in_corpus": recall_check["total_true_available"],
        "true_retrieved": recall_check["total_true_retrieved"],
        "micro_corpus_recall_pct": round(100 * recall_check["micro_corpus_recall"], 2),
        "missed_pair_count": len(recall_check["missed_pairs"]),
    }

    # ------------------------------------------------------------------
    # STEP 5 -- Build Person C training dataset
    # ------------------------------------------------------------------
    section(logger, "STEP 5: Build labeled training dataset")
    t0 = time.perf_counter()

    training_df = build_training_dataset(
        source1_records=s1_records,
        target_records=target_records,
        candidate_pairs=candidate_pairs,
        ground_truth=gt_dict_sample,
    )

    n_pos = int((training_df["label"] == 1).sum())
    n_neg = int((training_df["label"] == 0).sum())
    pos_rate = n_pos / len(training_df) if len(training_df) else 0.0

    logger.info(
        "  Training pairs: %d | pos: %d (%.2f%%) | neg: %d | %.1fs",
        len(training_df), n_pos, 100 * pos_rate, n_neg,
        time.perf_counter() - t0,
    )

    if training_df.empty or n_pos == 0:
        raise RuntimeError(
            "Training dataset is empty or has no positive labels. "
            "Increase --sample-size or check ground truth coverage."
        )

    # ------------------------------------------------------------------
    # STEP 6 -- Validate feature matrix (strict)
    # ------------------------------------------------------------------
    section(logger, "STEP 6: Validate feature matrix")

    X, y, metadata = extract_X_y_metadata(training_df)
    validate_feature_matrix(X, y)
    logger.info(
        "  Features: %d | pairs: %d | pos: %d | neg: %d  ALL VALID",
        len(X.columns), len(X), int(y.sum()), int((y == 0).sum()),
    )

    report["step5_6"] = {
        "training_pairs": len(training_df),
        "n_positive": n_pos,
        "n_negative": n_neg,
        "positive_rate_pct": round(100 * pos_rate, 2),
        "feature_columns": len(MODEL_FEATURES),
        "validation_passed": True,
    }

    # ------------------------------------------------------------------
    # STEP 7 -- GroupKFold LightGBM baseline
    # ------------------------------------------------------------------
    section(logger, f"STEP 7: GroupKFold baseline (n_splits={n_splits}, seed={seed})")
    t0 = time.perf_counter()

    cv_results = run_group_kfold_baseline(
        training_df,
        n_splits=n_splits,
        random_state=seed,
    )

    oof_df = cv_results["oof_predictions"]
    overall = cv_results["overall_metrics"]
    final_model = cv_results["final_model"]

    logger.info("\n  Fold breakdown:")
    logger.info(
        "  %4s  %12s  %8s  %6s  %8s  %8s",
        "Fold", "S1 Entities", "Pairs", "Pos", "Neg", "F0.5",
    )
    logger.info("  %s", "-" * 54)
    for m in cv_results["fold_metrics"]:
        logger.info(
            "  %4d  %12d  %8d  %6d  %8d  %8.4f",
            m["fold"], m["val_s1_entities"], m["val_pairs"],
            m["val_positives"], m["val_negatives"], m["f05_score"],
        )

    logger.info("\n  Overall OOF metrics (threshold=0.5):")
    for k in ["precision", "recall", "f05", "f1", "macro_f05", "tp", "fp", "fn", "tn"]:
        logger.info("    %-20s %s", k, _fmt(overall[k]))

    logger.info("  Training time: %.1fs", time.perf_counter() - t0)
    report["step7"] = {
        "oof_overall_metrics": {
            k: overall[k]
            for k in ["precision", "recall", "f05", "f1", "macro_f05", "tp", "fp", "fn", "tn"]
        },
        "fold_metrics": cv_results["fold_metrics"],
    }

    # ------------------------------------------------------------------
    # STEP 8 -- Hard negative mining (statistics)
    # ------------------------------------------------------------------
    section(logger, "STEP 8: Hard negative mining (statistics)")

    df_with_oof = training_df.merge(
        oof_df[["source1_entity_id", "candidate_entity_id", "y_prob"]],
        on=["source1_entity_id", "candidate_entity_id"],
        how="inner",
    )
    mined_df = select_hard_negatives(
        df_with_oof,
        prob_col="y_prob",
        negative_ratio=negative_ratio,
        random_state=seed,
    )
    n_pos_mined = int((mined_df["label"] == 1).sum())
    n_neg_mined = int((mined_df["label"] == 0).sum())
    logger.info(
        "  After mining: total=%d  pos=%d (unchanged)  neg=%d  (ratio=%.1f)",
        len(mined_df), n_pos_mined, n_neg_mined,
        n_neg_mined / n_pos_mined if n_pos_mined else float("nan"),
    )
    report["step8"] = {
        "mined_total": len(mined_df),
        "mined_positive": n_pos_mined,
        "mined_negative": n_neg_mined,
        "effective_ratio": round(n_neg_mined / n_pos_mined, 2) if n_pos_mined else None,
    }

    # ------------------------------------------------------------------
    # STEP 9 -- Threshold tuning (OOF only)
    # ------------------------------------------------------------------
    section(logger, "STEP 9: Decision threshold tuning (OOF, metric=macro_f05)")

    tuning = tune_decision_threshold(
        oof_df,
        prob_col="y_prob",
        true_col="y_true",
        metric="macro_f05",
    )
    best_threshold = tuning["best_threshold"]
    best_score = tuning["best_score"]

    logger.info(
        "  Best threshold: %.4f | Best macro F0.5: %.4f (vs default 0.5: %.4f)",
        best_threshold, best_score,
        tuning["default_05_metrics"]["macro_f05"],
    )
    report["step9"] = {
        "best_threshold": best_threshold,
        "best_macro_f05": best_score,
        "default_05_macro_f05": tuning["default_05_metrics"]["macro_f05"],
    }

    # ------------------------------------------------------------------
    # STEP 10 -- Calibration comparison
    # ------------------------------------------------------------------
    section(logger, "STEP 10: Calibration comparison (sigmoid vs isotonic)")

    calibrator_sigmoid = fit_oof_calibrator(oof_df, method="sigmoid")
    comp_sigmoid = compare_oof_calibration(
        oof_df, calibrator=calibrator_sigmoid, threshold=best_threshold
    )

    calibration_method = "sigmoid"
    comp_isotonic = None

    if len(oof_df) >= 50 and n_pos >= 5:
        calibrator_isotonic = fit_oof_calibrator(oof_df, method="isotonic")
        comp_isotonic = compare_oof_calibration(
            oof_df, calibrator=calibrator_isotonic, threshold=best_threshold
        )
        if (
            comp_isotonic["calibrated_metrics"]["macro_f05"]
            > comp_sigmoid["calibrated_metrics"]["macro_f05"]
        ):
            calibration_method = "isotonic"
            best_cal_comp = comp_isotonic
        else:
            best_cal_comp = comp_sigmoid
    else:
        logger.info(
            "  Skipping isotonic (too few samples: %d pairs, %d positives)",
            len(oof_df), n_pos,
        )
        best_cal_comp = comp_sigmoid

    raw_f05 = best_cal_comp["raw_metrics"]["macro_f05"]
    cal_f05 = best_cal_comp["calibrated_metrics"]["macro_f05"]

    logger.info("  Raw OOF macro F0.5    : %.4f", raw_f05)
    logger.info("  Sigmoid calibrated    : %.4f", comp_sigmoid["calibrated_metrics"]["macro_f05"])
    if comp_isotonic:
        logger.info("  Isotonic calibrated   : %.4f", comp_isotonic["calibrated_metrics"]["macro_f05"])
    logger.info("  Selected method       : %s (macro F0.5=%.4f)", calibration_method, cal_f05)

    cal_report_dict: Dict[str, Any] = {
        "selected_method": calibration_method,
        "raw_macro_f05": raw_f05,
        "sigmoid_macro_f05": comp_sigmoid["calibrated_metrics"]["macro_f05"],
    }
    if comp_isotonic:
        cal_report_dict["isotonic_macro_f05"] = comp_isotonic["calibrated_metrics"]["macro_f05"]
    report["step10_calibration"] = cal_report_dict

    # ------------------------------------------------------------------
    # STEP 11 -- Final metrics at best threshold
    # ------------------------------------------------------------------
    section(logger, "STEP 11: Final OOF metrics at best threshold")

    final_metrics = calculate_evaluation_metrics(
        oof_df, prob_col="y_prob", true_col="y_true", threshold=best_threshold
    )
    logger.info("  At threshold=%.4f:", best_threshold)
    for k in ["precision", "recall", "f05", "f1", "macro_f05", "tp", "fp", "fn", "tn"]:
        logger.info("    %-20s %s", k, _fmt(final_metrics[k]))

    report["step11_final_metrics"] = {
        k: final_metrics[k]
        for k in ["precision", "recall", "f05", "f1", "macro_f05", "tp", "fp", "fn", "tn"]
    }
    report["step11_final_metrics"]["threshold"] = best_threshold

    # ------------------------------------------------------------------
    # STEP 12 -- Save artifacts
    # ------------------------------------------------------------------
    section(logger, "STEP 12: Save reproducibility artifacts")

    wall_time = time.perf_counter() - t_wall
    report["wall_time_seconds"] = round(wall_time, 1)
    report["summary"] = {
        "s1_entities": len(s1_records),
        "target_records": len(target_records),
        "total_candidate_pairs": total_cand_pairs,
        "positive_pairs": n_pos,
        "negative_pairs": n_neg,
        "avg_candidates_per_s1": round(avg_cands, 2),
        "micro_corpus_recall_pct": round(100 * recall_check["micro_corpus_recall"], 2),
        "oof_macro_f05_at_05": overall["macro_f05"],
        "oof_macro_f05_at_best_threshold": best_score,
        "best_threshold": best_threshold,
        "calibration_method": calibration_method,
        "calibrated_macro_f05": cal_f05,
    }

    save_artifacts(
        output_dir=output_dir,
        report=report,
        final_model=final_model,
        best_threshold=best_threshold,
        calibration_method=calibration_method,
        logger=logger,
    )

    banner(logger, "PIPELINE COMPLETE")
    logger.info("  Wall time            : %.1fs", wall_time)
    logger.info("  S1 entities          : %d", len(s1_records))
    logger.info("  Target records       : %d", len(target_records))
    logger.info("  Candidate pairs      : %d", total_cand_pairs)
    logger.info("  Positives / Negatives: %d / %d", n_pos, n_neg)
    logger.info("  Micro-corpus recall  : %.2f%%", 100 * recall_check["micro_corpus_recall"])
    logger.info("  OOF macro F0.5 at 0.5: %.4f", overall["macro_f05"])
    logger.info("  Best threshold       : %.4f", best_threshold)
    logger.info("  Best OOF macro F0.5  : %.4f", best_score)
    logger.info("  Calibration method   : %s", calibration_method)
    logger.info("  Calibrated F0.5      : %.4f", cal_f05)
    logger.info("  Output dir           : %s", output_dir)

    return report


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Person C Step 6 -- Real-Data Development Training Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=Path(r"D:\amazom ml\6ab10eb3b23ba_student_resource\student_resource\dataset\train"),
        help="Directory containing the four training TSV files.",
    )
    p.add_argument("--sample-size", type=int, default=500,
                   help="Number of S1 entities to sample.")
    p.add_argument("--target-sample-size", type=int, default=50_000,
                   help="Hard cap on total retained target records (split 50/50 S2/S3).")
    p.add_argument("--noise-fraction", type=float, default=0.005,
                   help="Per-record probability of retaining a non-true target record.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for full reproducibility.")
    p.add_argument("--n-splits", type=int, default=5,
                   help="Number of GroupKFold CV folds.")
    p.add_argument("--negative-ratio", type=float, default=3.0,
                   help="Hard-negative mining: negatives per positive per S1 entity.")
    p.add_argument("--cap", type=int, default=200,
                   help="Max candidates per S1 entity after Person B blocking.")
    p.add_argument("--output-dir", type=Path, default=Path("person_c/pipeline_run_output"),
                   help="Directory for output artifacts.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir
    logger = setup_logging(output_dir)
    banner(logger, "PERSON C STEP 6: REAL-DATA DEVELOPMENT TRAINING PIPELINE")
    logger.info("  data_dir           : %s", args.data_dir)
    logger.info("  sample_size        : %d", args.sample_size)
    logger.info("  target_sample_size : %d", args.target_sample_size)
    logger.info("  noise_fraction     : %.4f", args.noise_fraction)
    logger.info("  seed               : %d", args.seed)
    logger.info("  n_splits           : %d", args.n_splits)
    logger.info("  negative_ratio     : %.1f", args.negative_ratio)
    logger.info("  cap                : %d", args.cap)
    logger.info("  output_dir         : %s", output_dir)
    run_pipeline(
        data_dir=args.data_dir,
        sample_size=args.sample_size,
        target_sample_size=args.target_sample_size,
        noise_fraction=args.noise_fraction,
        seed=args.seed,
        n_splits=args.n_splits,
        negative_ratio=args.negative_ratio,
        cap=args.cap,
        output_dir=output_dir,
        logger=logger,
    )


if __name__ == "__main__":
    main()
