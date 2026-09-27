"""
Person C — test_production_pipeline.py

Unit and integration tests for production_pipeline.py.

Verifies:
1. Save and load of production artifacts (model, calibrator, threshold, config).
2. Submission building (matching_results.tsv and candidate_pairs.tsv formatting).
3. Integration with the official submission validator script.
4. Lightweight end-to-end synthetic training and test inference pipeline.
"""

import sys
import tempfile
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.model import train_baseline_model, MODEL_FEATURES, ProbabilityCalibrator
from person_c.production_pipeline import (
    save_production_artifacts,
    load_production_artifacts,
    build_submission,
    validate_submission_output,
    train_production_model,
    run_competition_inference,
    stream_normalized_source_records,
    load_normalized_source_records,
)


def _create_synthetic_model():
    """Helper to create a small fitted LightGBM model for testing."""
    X = pd.DataFrame(np.random.randn(20, 45), columns=MODEL_FEATURES)
    y = pd.Series([1] * 10 + [0] * 10)
    return train_baseline_model(X, y)


def test_save_and_load_production_artifacts(tmp_path):
    """Test saving and re-loading production artifacts."""
    model = _create_synthetic_model()
    threshold = 0.65
    calibrator = ProbabilityCalibrator(method="sigmoid")
    calibrator.fit(np.array([0.1, 0.8, 0.9]), np.array([0, 1, 1]))

    artifacts_dir = tmp_path / "artifacts"
    save_production_artifacts(
        artifacts_dir=artifacts_dir,
        final_model=model,
        best_threshold=threshold,
        calibrator=calibrator,
        calibration_method="sigmoid",
        feature_schema=MODEL_FEATURES,
        metadata={"test": "unit"},
    )

    assert (artifacts_dir / "model.pkl").exists()
    assert (artifacts_dir / "calibrator.pkl").exists()
    assert (artifacts_dir / "threshold.txt").exists()
    assert (artifacts_dir / "model_config.json").exists()

    loaded_model, loaded_thresh, loaded_cal, loaded_config = load_production_artifacts(artifacts_dir)

    assert loaded_thresh == pytest.approx(0.65)
    assert loaded_cal is not None
    assert loaded_cal.is_fitted is True
    assert loaded_config["calibration_method"] == "sigmoid"
    assert loaded_config["metadata"]["test"] == "unit"


def test_build_submission(tmp_path):
    """Test building matching_results.tsv and candidate_pairs.tsv."""
    matching_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-101,S3-201"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": ""},
    ])
    candidate_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_ids": "S2-101,S3-201"},
        {"source1_entity_id": "S1-002", "candidate_entity_ids": "S2-102"},
    ])

    out_dir = tmp_path / "output"
    m_path, c_path = build_submission(matching_df, candidate_df, output_dir=out_dir)

    assert m_path.exists()
    assert c_path.exists()

    m_lines = m_path.read_text(encoding="utf-8").splitlines()
    assert m_lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert m_lines[1] == "S1-001\tS2-101,S3-201"
    assert m_lines[2] == "S1-002\t"

    c_lines = c_path.read_text(encoding="utf-8").splitlines()
    assert c_lines[0] == "source1_entity_id\tcandidate_entity_ids"
    assert c_lines[1] == "S1-001\tS2-101,S3-201"
    assert c_lines[2] == "S1-002\tS2-102"


def test_validate_submission_output_with_synthetic_files(tmp_path):
    """Test running validate_submission_output on valid submission TSVs."""
    test_dir = tmp_path / "dataset" / "test"
    test_dir.mkdir(parents=True)

    # Create dummy test_source1.tsv
    s1_file = test_dir / "test_source1.tsv"
    s1_file.write_text("entity_id\tbusiness_name\tbusiness_address\tcountry\nS1-001\tAcme\tMain St\tUS\nS1-002\tBeta\tSecond St\tUS\n", encoding="utf-8")

    matching_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "matched_entity_ids": "S2-101"},
        {"source1_entity_id": "S1-002", "matched_entity_ids": ""},
    ])
    candidate_df = pd.DataFrame([
        {"source1_entity_id": "S1-001", "candidate_entity_ids": "S2-101"},
        {"source1_entity_id": "S1-002", "candidate_entity_ids": "S2-102"},
    ])

    out_dir = tmp_path / "output"
    m_path, c_path = build_submission(matching_df, candidate_df, output_dir=out_dir)

    errors, warnings = validate_submission_output(m_path, c_path, test_dir=test_dir, check_ids=False)
    assert errors == []


def test_synthetic_end_to_end_pipeline(tmp_path):
    """Test full synthetic end-to-end training and inference execution."""
    train_dir = tmp_path / "train"
    train_dir.mkdir(parents=True)

    # 1. Write synthetic train TSV files
    (train_dir / "train_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAcme Foods\t123 Main St\tUSA\n"
        "S1-002\tGlobal Tech\t456 Park Ave\tUSA\n",
        encoding="utf-8"
    )
    (train_dir / "train_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-101\tAcme Foods Inc\t123 Main Street\tUSA\n"
        "S2-999\tUnrelated Enterprise\t999 Faraway St\tUSA\n",
        encoding="utf-8"
    )
    (train_dir / "train_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-201\tGlobal Technologies\t456 Park Avenue\tUSA\n",
        encoding="utf-8"
    )
    (train_dir / "train_ground_truth.tsv").write_text(
        "source1_entity_id\tmatched_entity_ids\n"
        "S1-001\tS2-101\n"
        "S1-002\tS3-201\n",
        encoding="utf-8"
    )

    artifacts_dir = tmp_path / "artifacts"
    train_res = train_production_model(
        data_dir=train_dir,
        artifacts_dir=artifacts_dir,
        sample_size=2,
        target_sample_size=10,
        noise_fraction=1.0,
        seed=42,
        n_splits=2,
        cap=10,
    )

    assert (artifacts_dir / "model.pkl").exists()

    # 2. Write synthetic test TSV files
    test_dir = tmp_path / "test"
    test_dir.mkdir(parents=True)
    (test_dir / "test_source1.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S1-001\tAcme Foods\t123 Main St\tUSA\n",
        encoding="utf-8"
    )
    (test_dir / "test_source2.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S2-101\tAcme Foods Inc\t123 Main Street\tUSA\n",
        encoding="utf-8"
    )
    (test_dir / "test_source3.tsv").write_text(
        "entity_id\tbusiness_name\tbusiness_address\tcountry\n"
        "S3-201\tGlobal Technologies\t456 Park Avenue\tUSA\n",
        encoding="utf-8"
    )

    output_dir = tmp_path / "output"
    m_path, c_path = run_competition_inference(
        test_dir=test_dir,
        artifacts_dir=artifacts_dir,
        output_dir=output_dir,
        cap=10,
    )

    assert m_path.exists()
    assert c_path.exists()
    lines = m_path.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "source1_entity_id\tmatched_entity_ids"
    assert len(lines) == 2  # Header + 1 S1 entity


def test_reader_handles_large_records_and_utf8_without_oserror(tmp_path):
    """
    Regression test for Windows Python OSError: [Errno 22] Invalid argument.
    Ensures stream_normalized_source_records and load_normalized_source_records
    read multi-line/UTF-8 TSV data cleanly without buffer invalidation errors.
    """
    tsv_file = tmp_path / "test_reader.tsv"
    lines = ["entity_id\tbusiness_name\tbusiness_address\tcountry\n"]
    for i in range(500):
        name = f"Company {i} — Café & Restaurant \u2605"
        addr = f"{i} Main St, Suite {i * 10} " + "X" * 200
        lines.append(f"ENT-{i:05d}\t{name}\t{addr}\tUSA\n")

    tsv_file.write_text("".join(lines), encoding="utf-8")

    streamed = list(stream_normalized_source_records(tsv_file))
    assert len(streamed) == 500
    assert streamed[0]["entity_id"] == "ENT-00000"
    assert "cafe" in streamed[0]["normalized_name"] or "restaurant" in streamed[0]["normalized_name"]

    loaded = load_normalized_source_records(tsv_file, limit=100)
    assert len(loaded) == 100

