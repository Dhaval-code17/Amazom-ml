"""
Person C -- test_real_data_training_pipeline.py

Focused unit tests for the helper functions in real_data_training_pipeline.py.

Design constraints:
- Does NOT require the full 10 M dataset.
- Does NOT run the full pipeline (no I/O to real TSV files).
- Uses only synthetic in-memory data.
- Does NOT test load_sampled_s1_records / load_ground_truth_raw /
  stream_target_micro_corpus at real-file level (those are I/O tests
  that need the actual data directory).
- Covers all testable pure-Python/numpy/pandas helper functions.
"""

from __future__ import annotations

import sys
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Set

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# sys.path
# ---------------------------------------------------------------------------
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from normalization import normalize_record
from person_c.model import MODEL_FEATURES
from person_c.real_data_training_pipeline import (
    extract_true_target_ids,
    build_gt_dict_for_sample,
    validate_feature_matrix,
    validate_candidates_contain_true_matches,
    save_artifacts,
    _fmt,
    _parse_args,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_norm_rec(entity_id: str, name: str = "Acme Corp") -> Dict[str, Any]:
    """Creates a minimal normalized record dict."""
    rec = normalize_record(name, "123 Main St, New York, NY 10001", "US")
    rec["entity_id"] = entity_id
    return rec


def _make_feature_df(n_rows: int = 10, n_pos: int = 2) -> pd.DataFrame:
    """Creates a syntheitc training DataFrame with exactly MODEL_FEATURES columns."""
    rows = []
    for i in range(n_rows):
        row = {feat: float(np.random.rand()) for feat in MODEL_FEATURES}
        row["source1_entity_id"] = f"S1-{i % 3}"
        row["candidate_entity_id"] = f"CAND-{i}"
        row["label"] = 1 if i < n_pos else 0
        rows.append(row)
    return pd.DataFrame(rows)


def _make_X_y(n_rows: int = 10, n_pos: int = 2):
    """Returns X (feature DataFrame) and y (label Series)."""
    df = _make_feature_df(n_rows, n_pos)
    X = df[MODEL_FEATURES].copy()
    y = df["label"].copy()
    return X, y


# ---------------------------------------------------------------------------
# Tests: extract_true_target_ids
# ---------------------------------------------------------------------------

class TestExtractTrueTargetIds:
    def test_basic_extraction(self):
        s1_records = [
            _make_norm_rec("S1-1"),
            _make_norm_rec("S1-2"),
        ]
        gt_map_raw = {
            "S1-1": "T1,T2,T3",
            "S1-2": "T2,T4",
        }
        result = extract_true_target_ids(s1_records, gt_map_raw)
        assert result == {"T1", "T2", "T3", "T4"}

    def test_s1_not_in_gt_gives_empty(self):
        s1_records = [_make_norm_rec("S1-ORPHAN")]
        gt_map_raw = {"S1-OTHER": "T1,T2"}
        result = extract_true_target_ids(s1_records, gt_map_raw)
        assert result == set()

    def test_empty_s1_records(self):
        result = extract_true_target_ids([], {"S1-1": "T1"})
        assert result == set()

    def test_empty_gt_map(self):
        s1_records = [_make_norm_rec("S1-1")]
        result = extract_true_target_ids(s1_records, {})
        assert result == set()

    def test_strips_whitespace(self):
        s1_records = [_make_norm_rec("S1-1")]
        gt_map_raw = {"S1-1": " T1 , T2 , T3 "}
        result = extract_true_target_ids(s1_records, gt_map_raw)
        assert "T1" in result
        assert "T2" in result
        assert "T3" in result

    def test_deduplication(self):
        s1_records = [
            _make_norm_rec("S1-1"),
            _make_norm_rec("S1-2"),
        ]
        gt_map_raw = {
            "S1-1": "T1,T2",
            "S1-2": "T1,T3",  # T1 appears twice across S1 entities
        }
        result = extract_true_target_ids(s1_records, gt_map_raw)
        assert result == {"T1", "T2", "T3"}  # deduplicated

    def test_empty_matched_string_gives_no_ids(self):
        s1_records = [_make_norm_rec("S1-1")]
        gt_map_raw = {"S1-1": ""}
        result = extract_true_target_ids(s1_records, gt_map_raw)
        assert result == set()


# ---------------------------------------------------------------------------
# Tests: build_gt_dict_for_sample
# ---------------------------------------------------------------------------

class TestBuildGtDictForSample:
    def test_basic_conversion(self):
        s1_records = [_make_norm_rec("S1-1"), _make_norm_rec("S1-2")]
        gt_map_raw = {"S1-1": "T1,T2", "S1-2": "T3"}
        result = build_gt_dict_for_sample(s1_records, gt_map_raw)
        assert result["S1-1"] == {"T1", "T2"}
        assert result["S1-2"] == {"T3"}

    def test_missing_s1_gets_empty_set(self):
        s1_records = [_make_norm_rec("S1-ORPHAN")]
        gt_map_raw = {}
        result = build_gt_dict_for_sample(s1_records, gt_map_raw)
        assert result["S1-ORPHAN"] == set()

    def test_empty_matched_ids_gives_empty_set(self):
        s1_records = [_make_norm_rec("S1-1")]
        gt_map_raw = {"S1-1": ""}
        result = build_gt_dict_for_sample(s1_records, gt_map_raw)
        assert result["S1-1"] == set()

    def test_only_sampled_s1_entities_included(self):
        s1_records = [_make_norm_rec("S1-1")]
        gt_map_raw = {"S1-1": "T1", "S1-999": "T999"}
        result = build_gt_dict_for_sample(s1_records, gt_map_raw)
        assert "S1-999" not in result
        assert "S1-1" in result

    def test_strips_whitespace_in_ids(self):
        s1_records = [_make_norm_rec("S1-1")]
        gt_map_raw = {"S1-1": " T1 , T2 "}
        result = build_gt_dict_for_sample(s1_records, gt_map_raw)
        assert result["S1-1"] == {"T1", "T2"}

    def test_deterministic_output(self):
        s1_records = [_make_norm_rec("S1-A"), _make_norm_rec("S1-B")]
        gt_map_raw = {"S1-A": "T1,T2,T3", "S1-B": "T4"}
        r1 = build_gt_dict_for_sample(s1_records, gt_map_raw)
        r2 = build_gt_dict_for_sample(s1_records, gt_map_raw)
        assert r1 == r2


# ---------------------------------------------------------------------------
# Tests: validate_feature_matrix
# ---------------------------------------------------------------------------

class TestValidateFeatureMatrix:
    def test_valid_matrix_passes(self):
        X, y = _make_X_y(10, 2)
        validate_feature_matrix(X, y)  # should not raise

    def test_wrong_feature_order_fails(self):
        X, y = _make_X_y(10, 2)
        cols_shuffled = list(reversed(MODEL_FEATURES))
        X_bad = X[cols_shuffled]
        with pytest.raises(AssertionError, match="Feature columns mismatch"):
            validate_feature_matrix(X_bad, y)

    def test_missing_feature_column_fails(self):
        X, y = _make_X_y(10, 2)
        X_bad = X.drop(columns=[MODEL_FEATURES[0]])
        with pytest.raises(AssertionError):
            validate_feature_matrix(X_bad, y)

    def test_identifier_in_X_fails_source1(self):
        X, y = _make_X_y(10, 2)
        X_bad = X.copy()
        X_bad["source1_entity_id"] = "S1-1"
        with pytest.raises(AssertionError, match="source1_entity_id"):
            validate_feature_matrix(X_bad, y)

    def test_identifier_in_X_fails_candidate(self):
        X, y = _make_X_y(10, 2)
        X_bad = X.copy()
        X_bad["candidate_entity_id"] = "CAND-1"
        with pytest.raises(AssertionError, match="candidate_entity_id"):
            validate_feature_matrix(X_bad, y)

    def test_label_in_X_fails(self):
        X, y = _make_X_y(10, 2)
        X_bad = X.copy()
        X_bad["label"] = 0
        with pytest.raises(AssertionError, match="label"):
            validate_feature_matrix(X_bad, y)

    def test_nan_in_X_fails(self):
        X, y = _make_X_y(10, 2)
        X_bad = X.copy()
        X_bad.iloc[0, 0] = float("nan")
        with pytest.raises(AssertionError, match="NaN"):
            validate_feature_matrix(X_bad, y)

    def test_inf_in_X_fails(self):
        X, y = _make_X_y(10, 2)
        X_bad = X.copy()
        X_bad.iloc[0, 0] = float("inf")
        with pytest.raises(AssertionError, match="Infinite"):
            validate_feature_matrix(X_bad, y)

    def test_non_binary_y_fails(self):
        X, y = _make_X_y(10, 2)
        y_bad = y.copy()
        y_bad.iloc[0] = 2
        with pytest.raises(AssertionError, match="non-binary"):
            validate_feature_matrix(X, y_bad)

    def test_exactly_45_features_enforced(self):
        X, y = _make_X_y(10, 2)
        assert len(X.columns) == 45


# ---------------------------------------------------------------------------
# Tests: validate_candidates_contain_true_matches
# ---------------------------------------------------------------------------

class TestValidateCandidatesContainTrueMatches:
    def _make_cand_pairs(self, mapping: Dict[str, str]) -> pd.DataFrame:
        rows = [
            {"source1_entity_id": s1, "candidate_entity_ids": cids}
            for s1, cids in mapping.items()
        ]
        return pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])

    def test_perfect_recall(self):
        gt_dict = {"S1-1": {"T1", "T2"}}
        corpus = {"T1", "T2", "NOISE-1"}
        cand_pairs = self._make_cand_pairs({"S1-1": "T1,T2,NOISE-1"})
        result = validate_candidates_contain_true_matches(cand_pairs, gt_dict, corpus)
        assert result["total_true_available"] == 2
        assert result["total_true_retrieved"] == 2
        assert result["micro_corpus_recall"] == 1.0
        assert result["missed_pairs"] == []

    def test_missed_true_id(self):
        gt_dict = {"S1-1": {"T1", "T2"}}
        corpus = {"T1", "T2"}
        cand_pairs = self._make_cand_pairs({"S1-1": "T1"})   # T2 missed
        result = validate_candidates_contain_true_matches(cand_pairs, gt_dict, corpus)
        assert result["total_true_available"] == 2
        assert result["total_true_retrieved"] == 1
        assert result["micro_corpus_recall"] == 0.5
        assert ("S1-1", "T2") in result["missed_pairs"]

    def test_true_id_not_in_corpus_not_counted(self):
        # T2 is true but not in corpus -- should not count as missed
        gt_dict = {"S1-1": {"T1", "T2"}}
        corpus = {"T1"}          # T2 not indexed
        cand_pairs = self._make_cand_pairs({"S1-1": "T1"})
        result = validate_candidates_contain_true_matches(cand_pairs, gt_dict, corpus)
        assert result["total_true_available"] == 1  # only T1 counts
        assert result["total_true_retrieved"] == 1
        assert result["micro_corpus_recall"] == 1.0
        assert result["missed_pairs"] == []

    def test_empty_candidate_list(self):
        gt_dict = {"S1-1": {"T1"}}
        corpus = {"T1"}
        cand_pairs = self._make_cand_pairs({"S1-1": ""})
        result = validate_candidates_contain_true_matches(cand_pairs, gt_dict, corpus)
        assert result["total_true_available"] == 1
        assert result["total_true_retrieved"] == 0
        assert result["micro_corpus_recall"] == 0.0

    def test_empty_gt_dict(self):
        gt_dict: Dict[str, Set[str]] = {}
        corpus = {"T1", "T2"}
        cand_pairs = self._make_cand_pairs({"S1-1": "T1,T2"})
        result = validate_candidates_contain_true_matches(cand_pairs, gt_dict, corpus)
        assert result["total_true_available"] == 0
        assert result["micro_corpus_recall"] == 0.0

    def test_multiple_s1_entities(self):
        gt_dict = {"S1-1": {"T1"}, "S1-2": {"T2", "T3"}}
        corpus = {"T1", "T2", "T3"}
        cand_pairs = self._make_cand_pairs({
            "S1-1": "T1",
            "S1-2": "T2",   # T3 missed
        })
        result = validate_candidates_contain_true_matches(cand_pairs, gt_dict, corpus)
        assert result["total_true_available"] == 3
        assert result["total_true_retrieved"] == 2
        assert ("S1-2", "T3") in result["missed_pairs"]


# ---------------------------------------------------------------------------
# Tests: _fmt
# ---------------------------------------------------------------------------

class TestFmt:
    def test_float_formatting(self):
        assert _fmt(0.12345678) == "0.1235"

    def test_int_formatting(self):
        assert _fmt(42) == "42"

    def test_string_passthrough(self):
        assert _fmt("sigmoid") == "sigmoid"

    def test_zero_float(self):
        assert _fmt(0.0) == "0.0000"


# ---------------------------------------------------------------------------
# Tests: save_artifacts
# ---------------------------------------------------------------------------

class TestSaveArtifacts:
    def _make_mock_model(self):
        """Minimal stub with get_params()."""
        class _MockModel:
            def get_params(self):
                return {"n_estimators": 100, "objective": "binary"}
        return _MockModel()

    def test_all_artifacts_created(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            report = {"summary": {"s1_entities": 10}}
            logger = logging.getLogger("test_save")
            logger.setLevel(logging.CRITICAL)  # suppress output during test

            save_artifacts(
                output_dir=out,
                report=report,
                final_model=self._make_mock_model(),
                best_threshold=0.35,
                calibration_method="sigmoid",
                logger=logger,
            )

            assert (out / "pipeline_report.json").exists()
            assert (out / "model_config.json").exists()
            assert (out / "threshold.txt").exists()
            assert (out / "calibration.txt").exists()

    def test_report_json_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            report = {"summary": {"s1_entities": 42}, "config": {"seed": 99}}
            logger = logging.getLogger("test_json")
            logger.setLevel(logging.CRITICAL)

            save_artifacts(out, report, self._make_mock_model(), 0.4, "isotonic", logger)

            loaded = json.loads((out / "pipeline_report.json").read_text(encoding="utf-8"))
            assert loaded["summary"]["s1_entities"] == 42
            assert loaded["config"]["seed"] == 99

    def test_threshold_txt_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            logger = logging.getLogger("test_thresh")
            logger.setLevel(logging.CRITICAL)

            save_artifacts(out, {}, self._make_mock_model(), 0.273, "sigmoid", logger)
            assert (out / "threshold.txt").read_text(encoding="utf-8") == "0.273"

    def test_calibration_txt_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            logger = logging.getLogger("test_cal")
            logger.setLevel(logging.CRITICAL)

            save_artifacts(out, {}, self._make_mock_model(), 0.5, "isotonic", logger)
            assert (out / "calibration.txt").read_text(encoding="utf-8") == "isotonic"

    def test_model_config_feature_count(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir)
            logger = logging.getLogger("test_cfg")
            logger.setLevel(logging.CRITICAL)

            save_artifacts(out, {}, self._make_mock_model(), 0.5, "sigmoid", logger)
            cfg = json.loads((out / "model_config.json").read_text(encoding="utf-8"))
            assert cfg["num_features"] == 45
            assert len(cfg["feature_names"]) == 45

    def test_output_dir_created_if_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "new_nested_dir"
            assert not out.exists()
            logger = logging.getLogger("test_mkdir")
            logger.setLevel(logging.CRITICAL)

            save_artifacts(out, {}, self._make_mock_model(), 0.5, "sigmoid", logger)
            assert out.exists()


# ---------------------------------------------------------------------------
# Tests: _parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_defaults(self):
        args = _parse_args([])
        assert args.sample_size == 500
        assert args.target_sample_size == 50_000
        assert args.noise_fraction == 0.005
        assert args.seed == 42
        assert args.n_splits == 5
        assert args.negative_ratio == 3.0
        assert args.cap == 200

    def test_override_sample_size(self):
        args = _parse_args(["--sample-size", "100"])
        assert args.sample_size == 100

    def test_override_seed(self):
        args = _parse_args(["--seed", "99"])
        assert args.seed == 99

    def test_override_n_splits(self):
        args = _parse_args(["--n-splits", "3"])
        assert args.n_splits == 3

    def test_override_negative_ratio(self):
        args = _parse_args(["--negative-ratio", "5.0"])
        assert args.negative_ratio == 5.0

    def test_override_noise_fraction(self):
        args = _parse_args(["--noise-fraction", "0.01"])
        assert pytest.approx(args.noise_fraction) == 0.01

    def test_override_cap(self):
        args = _parse_args(["--cap", "300"])
        assert args.cap == 300

    def test_output_dir_is_path(self):
        args = _parse_args(["--output-dir", "some/dir"])
        assert isinstance(args.output_dir, Path)
