"""
Person C — production_pipeline.py

Step 7/8: Production Training & Competition Inference Pipeline.

End-to-end production training, artifact persistence, memory-safe batched test inference,
and competition submission generation for Amazon ML Business Entity Resolution.

Design Principles:
- Modular, deterministic, and fully reproducible with --seed.
- Does NOT load the entire 10M+ target corpus or all candidate pairs into RAM simultaneously;
  uses streaming and chunked indexing with batching (10,000 S1 queries per batch).
- Preserves all 45 MODEL_FEATURES and GroupKFold leak-free structure.
- Generates exact UTF-8 tab-separated submission files:
    - output/matching_results.tsv  (scored leaderboard file)
    - output/candidate_pairs.tsv   (blocking candidate archive)
- Invokes competition submission validator to guarantee format compliance.

Public API
----------
train_production_model(...)
save_production_artifacts(...)
load_production_artifacts(...)
run_competition_inference(...)
build_submission(...)
validate_submission_output(...)
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import logging
import pickle
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# sys.path -- bare-module imports as per project convention
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# Root of the workspace
_WORKSPACE_ROOT = _SRC_DIR.parent.parent.parent

from normalization import normalize_record
from scoring import score_macro_f_beta
from person_b.blocking import MultiChannelBlocker
from person_b.pruning import rank_and_prune_candidates
from person_c.candidate_generation import generate_candidate_pairs
from person_c.training_data import build_training_dataset, extract_X_y_metadata
from person_c.features import build_pair_features
from person_c.model import (
    MODEL_FEATURES,
    run_group_kfold_baseline,
    select_hard_negatives,
    tune_decision_threshold,
    fit_oof_calibrator,
    compare_oof_calibration,
    calculate_evaluation_metrics,
    train_baseline_model,
    predict_candidate_pairs,
    predict_matches,
    ProbabilityCalibrator,
)

# ---------------------------------------------------------------------------
# Constants & Default Paths
# ---------------------------------------------------------------------------
BUFFER_SIZE = 8 * 1024 * 1024  # 8 MB buffer for large file I/O on Windows
LOG_FORMAT = "%(asctime)s  %(levelname)-7s  %(message)s"
DATE_FORMAT = "%H:%M:%S"

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]

DEFAULT_TRAIN_DIR = _WORKSPACE_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "train"
DEFAULT_TEST_DIR = _WORKSPACE_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "dataset" / "test"
DEFAULT_ARTIFACTS_DIR = _SRC_DIR / "person_c" / "production_artifacts"
DEFAULT_OUTPUT_DIR = _WORKSPACE_ROOT / "output"


# ---------------------------------------------------------------------------
# Logging & Setup
# ---------------------------------------------------------------------------

def setup_logging(output_dir: Path, name: str = "production_pipeline") -> logging.Logger:
    """Creates a logger writing to stdout and file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / f"{name}.log", encoding="utf-8", mode="w")
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
# Data I/O Helpers
# ---------------------------------------------------------------------------

def stream_normalized_source_records(
    filepath: Path,
    limit: Optional[int] = None,
):
    """
    Yields normalized record dicts one by one from a Source TSV file.
    Memory-safe streaming generator using standard text-mode buffering
    to prevent Windows CRT OSError 22 (Invalid argument).

    Parameters
    ----------
    filepath : Path to TSV file (e.g. test_source1.tsv)
    limit    : Optional max records to yield (for testing/debug)

    Yields
    ------
    Normalized record dict with 'entity_id'.
    """
    if not filepath.exists():
        return

    count = 0
    with open(filepath, "r", encoding="utf-8", errors="replace", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            entity_id = row.get("entity_id", "")
            if not entity_id:
                continue

            norm = normalize_record(
                row.get("business_name", ""),
                row.get("business_address", ""),
                row.get("country", ""),
            )
            norm["entity_id"] = entity_id
            yield norm

            count += 1
            if limit is not None and count >= limit:
                break


def load_normalized_source_records(
    filepath: Path,
    limit: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Reads a Source TSV file and normalizes each record.

    Parameters
    ----------
    filepath : Path to TSV file (e.g. test_source1.tsv)
    limit    : Optional max records to load (for testing/debug)

    Returns
    -------
    List of normalized dicts with 'entity_id'.
    """
    return list(stream_normalized_source_records(filepath, limit=limit))


# ---------------------------------------------------------------------------
# Artifact Persistence
# ---------------------------------------------------------------------------

def save_production_artifacts(
    artifacts_dir: Path,
    final_model: Any,
    best_threshold: float,
    calibrator: Optional[ProbabilityCalibrator],
    calibration_method: str,
    feature_schema: List[str] = MODEL_FEATURES,
    metadata: Optional[Dict[str, Any]] = None,
    logger: Optional[logging.Logger] = None,
) -> None:
    """
    Saves production model artifacts needed for offline/competition inference.

    Saved files:
    - model.pkl             : Trained LightGBM model object
    - calibrator.pkl        : Fitted ProbabilityCalibrator object
    - threshold.txt         : Floating point threshold
    - model_config.json     : Model metadata & feature schema
    """
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifacts_dir / "model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)

    cal_path = artifacts_dir / "calibrator.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)

    thresh_path = artifacts_dir / "threshold.txt"
    thresh_path.write_text(str(best_threshold), encoding="utf-8")

    config_data = {
        "feature_names": feature_schema,
        "num_features": len(feature_schema),
        "best_threshold": best_threshold,
        "calibration_method": calibration_method,
        "is_calibrated": calibrator is not None and getattr(calibrator, "is_fitted", False),
        "metadata": metadata or {},
    }
    config_path = artifacts_dir / "model_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2, default=str)

    if logger:
        logger.info("Saved production artifacts to: %s", artifacts_dir)


def load_production_artifacts(
    artifacts_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Any, float, Optional[ProbabilityCalibrator], Dict[str, Any]]:
    """
    Loads saved production model artifacts.

    Returns
    -------
    (model, threshold, calibrator, config_dict)
    """
    if not artifacts_dir.exists():
        raise FileNotFoundError(f"Artifacts directory not found: {artifacts_dir}")

    model_path = artifacts_dir / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    thresh_path = artifacts_dir / "threshold.txt"
    threshold = float(thresh_path.read_text(encoding="utf-8").strip())

    cal_path = artifacts_dir / "calibrator.pkl"
    calibrator = None
    if cal_path.exists():
        with open(cal_path, "rb") as f:
            calibrator = pickle.load(f)

    config_path = artifacts_dir / "model_config.json"
    config_dict = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)

    if logger:
        logger.info("Loaded production artifacts from: %s", artifacts_dir)
        logger.info("  Threshold: %.4f | Calibrated: %s", threshold, calibrator is not None and getattr(calibrator, "is_fitted", False))

    return model, threshold, calibrator, config_dict


# ---------------------------------------------------------------------------
# Submission Output Building & Validation
# ---------------------------------------------------------------------------

def build_submission(
    matching_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
    output_dir: Path,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, Path]:
    """
    Writes matching_results.tsv and candidate_pairs.tsv as UTF-8 tab-separated files.

    Parameters
    ----------
    matching_df  : DataFrame with columns 'source1_entity_id', 'matched_entity_ids'
    candidate_df : DataFrame with columns 'source1_entity_id', 'candidate_entity_ids'
    output_dir   : Output directory (e.g. output/)

    Returns
    -------
    (matching_file_path, candidate_file_path)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_path = output_dir / "matching_results.tsv"
    matching_df.to_csv(matching_path, sep="\t", index=False, encoding="utf-8")

    candidate_path = output_dir / "candidate_pairs.tsv"
    candidate_df.to_csv(candidate_path, sep="\t", index=False, encoding="utf-8")

    if logger:
        logger.info("  Written matching_results.tsv -> %s (%d rows)", matching_path, len(matching_df))
        logger.info("  Written candidate_pairs.tsv  -> %s (%d rows)", candidate_path, len(candidate_df))

    return matching_path, candidate_path


def validate_submission_output(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
    check_ids: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[List[str], List[str]]:
    """
    Invokes the official validate_submission.py module programmatically.

    Returns
    -------
    (errors, warnings)
    """
    validator_script = _WORKSPACE_ROOT / "6ab10eb3b23ba_student_resource" / "student_resource" / "utils" / "validate_submission.py"

    if validator_script.exists():
        import importlib.util
        spec = importlib.util.spec_from_file_location("validate_submission", validator_script)
        val_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(val_mod)
        errors, warnings = val_mod.validate(
            matching_path=str(matching_path),
            candidate_path=str(candidate_path),
            test_dir=str(test_dir),
            check_ids=check_ids,
        )
    else:
        errors = [f"Validator script not found at {validator_script}"]
        warnings = []

    if logger:
        if errors:
            logger.error("Submission Validation FAILED with %d issues:", len(errors))
            for err in errors:
                logger.error("  - %s", err)
        else:
            logger.info("Submission Validation PASSED cleanly!")
            for warn in warnings:
                logger.warning("  Warning: %s", warn)

    return errors, warnings


# ---------------------------------------------------------------------------
# Full Production Training Function
# ---------------------------------------------------------------------------

def train_production_model(
    data_dir: Path = DEFAULT_TRAIN_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    sample_size: int = 2000,
    target_sample_size: int = 100_000,
    noise_fraction: float = 0.01,
    seed: int = 42,
    n_splits: int = 5,
    negative_ratio: float = 3.0,
    cap: int = 200,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Trains the production LightGBM model on real competition training data.
    Uses GroupKFold for leak-free threshold tuning and calibration, then
    fits the final model on all sampled training data and saves reproducible artifacts.
    """
    from person_c.real_data_training_pipeline import (
        load_sampled_s1_records,
        load_ground_truth_raw,
        extract_true_target_ids,
        stream_target_micro_corpus,
        build_gt_dict_for_sample,
        validate_feature_matrix,
    )

    if logger is None:
        logger = setup_logging(artifacts_dir)

    banner(logger, "STARTING PRODUCTION MODEL TRAINING")
    t0_start = time.perf_counter()

    # 1. Load S1 records + ground truth
    section(logger, "1. Loading training S1 records + Ground Truth")
    s1_path = data_dir / "train_source1.tsv"
    gt_path = data_dir / "train_ground_truth.tsv"

    s1_records = load_sampled_s1_records(s1_path, sample_size, random_seed=seed)
    gt_map_raw = load_ground_truth_raw(gt_path)
    s1_records = [r for r in s1_records if r["entity_id"] in gt_map_raw]
    logger.info("Loaded %d valid training S1 records", len(s1_records))

    # 2. Extract target IDs
    true_target_ids = extract_true_target_ids(s1_records, gt_map_raw)
    logger.info("Required true target IDs: %d", len(true_target_ids))

    # 3. Stream target corpus
    section(logger, "2. Streaming target records (S2 + S3)")
    per_file_cap = target_sample_size // 2 if target_sample_size else None
    s2_records = stream_target_micro_corpus(
        data_dir / "train_source2.tsv", true_ids=true_target_ids,
        noise_fraction=noise_fraction, random_seed=seed, target_sample_size=per_file_cap, logger=logger
    )
    s3_records = stream_target_micro_corpus(
        data_dir / "train_source3.tsv", true_ids=true_target_ids,
        noise_fraction=noise_fraction, random_seed=seed + 1, target_sample_size=per_file_cap, logger=logger
    )
    target_records = s2_records + s3_records
    logger.info("Total target micro-corpus: %d records", len(target_records))

    # 4. Generate candidate pairs (Person B)
    section(logger, "3. Candidate Generation (Person B Blocker)")
    candidate_pairs = generate_candidate_pairs(s1_records, target_records, cap=cap)
    logger.info("Generated candidate pairs for %d S1 entities", len(candidate_pairs))

    # 5. Build labeled training dataset (Person C)
    section(logger, "4. Feature Engineering & Dataset Construction")
    gt_dict_sample = build_gt_dict_for_sample(s1_records, gt_map_raw)
    training_df = build_training_dataset(s1_records, target_records, candidate_pairs, gt_dict_sample)

    X, y, metadata = extract_X_y_metadata(training_df)
    validate_feature_matrix(X, y)
    logger.info("Dataset shape: %d pairs, %d features, pos rate: %.2f%%",
                len(training_df), len(X.columns), 100 * (y == 1).mean())

    # 6. GroupKFold baseline & OOF predictions
    section(logger, "5. GroupKFold CV & OOF Prediction")
    cv_results = run_group_kfold_baseline(training_df, n_splits=n_splits, random_state=seed)
    oof_df = cv_results["oof_predictions"]
    final_model = cv_results["final_model"]
    logger.info("OOF Macro F0.5 at default 0.5: %.4f", cv_results["overall_metrics"]["macro_f05"])

    # 7. Threshold tuning
    section(logger, "6. Threshold Tuning (Macro F0.5)")
    tuning = tune_decision_threshold(oof_df, metric="macro_f05")
    best_threshold = tuning["best_threshold"]
    best_macro_f05 = tuning["best_score"]
    logger.info("Optimized Decision Threshold: %.4f (Macro F0.5 = %.4f)", best_threshold, best_macro_f05)

    # 8. Calibration
    section(logger, "7. Calibration Evaluation")
    calibrator_sigmoid = fit_oof_calibrator(oof_df, method="sigmoid")
    comp_sig = compare_oof_calibration(oof_df, calibrator=calibrator_sigmoid, threshold=best_threshold)

    calibration_method = "sigmoid"
    selected_calibrator = calibrator_sigmoid

    if len(oof_df) >= 100 and (y == 1).sum() >= 10:
        calibrator_iso = fit_oof_calibrator(oof_df, method="isotonic")
        comp_iso = compare_oof_calibration(oof_df, calibrator=calibrator_iso, threshold=best_threshold)
        if comp_iso["calibrated_metrics"]["macro_f05"] > comp_sig["calibrated_metrics"]["macro_f05"]:
            calibration_method = "isotonic"
            selected_calibrator = calibrator_iso

    logger.info("Selected Calibration Method: %s", calibration_method)

    # 9. Save artifacts
    section(logger, "8. Saving Production Artifacts")
    meta_info = {
        "train_sample_s1": len(s1_records),
        "target_sample_size": len(target_records),
        "oof_macro_f05": best_macro_f05,
        "seed": seed,
        "training_time_sec": round(time.perf_counter() - t0_start, 1),
    }
    save_production_artifacts(
        artifacts_dir=artifacts_dir,
        final_model=final_model,
        best_threshold=best_threshold,
        calibrator=selected_calibrator,
        calibration_method=calibration_method,
        feature_schema=MODEL_FEATURES,
        metadata=meta_info,
        logger=logger,
    )

    banner(logger, "PRODUCTION TRAINING COMPLETE")
    return {
        "best_threshold": best_threshold,
        "best_macro_f05": best_macro_f05,
        "calibration_method": calibration_method,
        "artifacts_dir": str(artifacts_dir),
    }


# ---------------------------------------------------------------------------
# Memory-Safe Target-Chunked Competition Inference Functions
# ---------------------------------------------------------------------------

def _evaluate_batch_against_target_chunk(
    batch_records: List[Dict[str, Any]],
    target_chunk: List[Dict[str, Any]],
    cap: Optional[int],
    model: Any,
    calibrator: Optional[ProbabilityCalibrator],
    threshold: float,
    batch_cand_map: Dict[str, Dict[str, float]],
    batch_match_map: Dict[str, List[str]],
) -> None:
    if not batch_records or not target_chunk:
        return

    cand_df = generate_candidate_pairs(batch_records, target_chunk, cap=cap)
    if cand_df.empty:
        return

    for _, row in cand_df.iterrows():
        s1_id = row["source1_entity_id"]
        cands = row["candidate_entity_ids"]
        if isinstance(cands, str) and cands.strip():
            for cid in cands.split(","):
                cid_s = cid.strip()
                if cid_s:
                    batch_cand_map[s1_id][cid_s] = 1.0

    pred_df = predict_matches(
        source1_records=batch_records,
        target_records=target_chunk,
        candidate_pairs=cand_df,
        model=model,
        calibrator=calibrator,
        threshold=threshold,
    )

    if not pred_df.empty:
        positives = pred_df[pred_df["match_decision"] == 1]
        for s1_id, grp in positives.groupby("source1_entity_id"):
            batch_match_map[s1_id].extend(list(grp["candidate_entity_id"]))


def run_competition_inference(
    test_dir: Path = DEFAULT_TEST_DIR,
    artifacts_dir: Path = DEFAULT_ARTIFACTS_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    cap: int = 200,
    batch_size: int = 10_000,
    target_chunk_size: int = 50_000,
    s1_limit: Optional[int] = None,
    target_limit: Optional[int] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Path, Path]:
    """
    Runs memory-safe production test inference on competition test data.
    Uses a target-chunked streaming architecture (RAM < 400 MB).
    """
    if logger is None:
        logger = setup_logging(output_dir, name="competition_inference")

    banner(logger, "STARTING MEMORY-SAFE COMPETITION TEST INFERENCE")
    t0_start = time.perf_counter()

    # 1. Load artifacts
    model, threshold, calibrator, config_dict = load_production_artifacts(artifacts_dir, logger=logger)

    # 2. Check test records
    section(logger, "1. Checking Test File Paths")
    s1_file = test_dir / "test_source1.tsv"
    s2_file = test_dir / "test_source2.tsv"
    s3_file = test_dir / "test_source3.tsv"

    if not s1_file.exists():
        raise FileNotFoundError(f"Test S1 file not found: {s1_file}")

    target_files = [f for f in [s2_file, s3_file] if f.exists()]

    # 3. Stream & Batch Query S1 Entities
    section(logger, f"2. Streaming S1 Queries & Target Chunk Inference (batch_size={batch_size:,}, target_chunk_size={target_chunk_size:,})")

    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    # Initialize output files with headers
    with open(matching_path, "w", encoding="utf-8", newline="") as fm, \
         open(candidate_path, "w", encoding="utf-8", newline="") as fc:
        fm.write("\t".join(MATCHING_HEADER) + "\n")
        fc.write("\t".join(CANDIDATE_HEADER) + "\n")

    total_s1_processed = 0
    total_matches_found = 0
    total_candidates_generated = 0

    s1_batch: List[Dict[str, Any]] = []

    def _process_s1_batch(batch_records: List[Dict[str, Any]]) -> None:
        nonlocal total_s1_processed, total_matches_found, total_candidates_generated
        if not batch_records:
            return

        batch_cand_map: Dict[str, Dict[str, float]] = {r["entity_id"]: {} for r in batch_records}
        batch_match_map: Dict[str, List[str]] = {r["entity_id"]: [] for r in batch_records}

        per_file_target_limit = target_limit // len(target_files) if (target_limit and target_files) else None

        for t_file in target_files:
            recs_read_file = 0
            target_chunk: List[Dict[str, Any]] = []

            for norm_target in stream_normalized_source_records(t_file, limit=per_file_target_limit):
                target_chunk.append(norm_target)
                recs_read_file += 1

                if len(target_chunk) >= target_chunk_size:
                    _evaluate_batch_against_target_chunk(
                        batch_records=batch_records,
                        target_chunk=target_chunk,
                        cap=cap,
                        model=model,
                        calibrator=calibrator,
                        threshold=threshold,
                        batch_cand_map=batch_cand_map,
                        batch_match_map=batch_match_map,
                    )
                    target_chunk.clear()

            if target_chunk:
                _evaluate_batch_against_target_chunk(
                    batch_records=batch_records,
                    target_chunk=target_chunk,
                    cap=cap,
                    model=model,
                    calibrator=calibrator,
                    threshold=threshold,
                    batch_cand_map=batch_cand_map,
                    batch_match_map=batch_match_map,
                )
                target_chunk.clear()

        # Format and append batch results directly to disk
        with open(matching_path, "a", encoding="utf-8", newline="") as fm, \
             open(candidate_path, "a", encoding="utf-8", newline="") as fc:
            for r in batch_records:
                s1_id = r["entity_id"]
                m_ids = batch_match_map.get(s1_id, [])
                c_map = batch_cand_map.get(s1_id, {})

                top_cand_ids = sorted(c_map.keys())[:cap] if cap else sorted(c_map.keys())
                c_str = ",".join(top_cand_ids) if top_cand_ids else ""

                if m_ids:
                    unique_m = sorted(set(m_ids))
                    total_matches_found += len(unique_m)
                    m_str = ",".join(unique_m)
                else:
                    m_str = ""

                total_candidates_generated += len(top_cand_ids)
                fm.write(f"{s1_id}\t{m_str}\n")
                fc.write(f"{s1_id}\t{c_str}\n")

        total_s1_processed += len(batch_records)

    # Stream test_source1.tsv and process in query batches
    for norm in stream_normalized_source_records(s1_file, limit=s1_limit):
        s1_batch.append(norm)

        if len(s1_batch) >= batch_size:
            _process_s1_batch(s1_batch)
            logger.info("  Processed %d test S1 records (matches so far: %d)...", total_s1_processed, total_matches_found)
            s1_batch.clear()
            gc.collect()

    if s1_batch:
        _process_s1_batch(s1_batch)
        s1_batch.clear()
        gc.collect()

    logger.info("Processed ALL %d Test S1 entities | Total Matches: %d | Total Candidates: %d",
                total_s1_processed, total_matches_found, total_candidates_generated)

    # 4. Validate submission output format
    section(logger, "3. Submission Format Validation Check")
    errors, warnings = validate_submission_output(matching_path, candidate_path, test_dir=test_dir, check_ids=False, logger=logger)

    banner(logger, "COMPETITION INFERENCE COMPLETE")
    logger.info("Total inference time: %.1f seconds", time.perf_counter() - t0_start)
    return matching_path, candidate_path


# ---------------------------------------------------------------------------
# CLI Entry-Point
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Person C Step 7/8 -- Production Training & Competition Inference Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mode", type=str, choices=["train", "predict", "all"], default="all",
                   help="Pipeline mode: 'train', 'predict', or 'all'")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_TRAIN_DIR,
                   help="Directory containing training TSV files")
    p.add_argument("--test-dir", type=Path, default=DEFAULT_TEST_DIR,
                   help="Directory containing test TSV files")
    p.add_argument("--artifacts-dir", type=Path, default=DEFAULT_ARTIFACTS_DIR,
                   help="Directory for saving/loading production model artifacts")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                   help="Directory for final submission files (matching_results.tsv, candidate_pairs.tsv)")
    p.add_argument("--sample-size", type=int, default=2000, help="Training S1 sample size")
    p.add_argument("--target-sample-size", type=int, default=100_000, help="Training target sample size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--cap", type=int, default=200, help="Max candidates per S1 entity")
    p.add_argument("--batch-size", type=int, default=10_000, help="S1 batch size for memory-safe inference")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path.cwd() / output_dir

    logger = setup_logging(output_dir)

    if args.mode in ("train", "all"):
        train_production_model(
            data_dir=args.data_dir,
            artifacts_dir=args.artifacts_dir,
            sample_size=args.sample_size,
            target_sample_size=args.target_sample_size,
            seed=args.seed,
            cap=args.cap,
            logger=logger,
        )

    if args.mode in ("predict", "all"):
        run_competition_inference(
            test_dir=args.test_dir,
            artifacts_dir=args.artifacts_dir,
            output_dir=output_dir,
            cap=args.cap,
            batch_size=args.batch_size,
            logger=logger,
        )


if __name__ == "__main__":
    main()
