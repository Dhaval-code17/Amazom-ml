"""
Person C — test_chunked_recall_benchmark.py

Lightweight tests for the chunked recall benchmark.
Uses synthetic files. Does NOT load real dataset.
"""

import sys
import os
import tempfile
import csv
from pathlib import Path
from typing import Dict, Any

import pytest

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.chunked_recall_benchmark import stream_and_normalize_chunks, run_chunked_benchmark


@pytest.fixture
def synthetic_data_dir():
    """Creates a temporary directory with synthetic S1, S2, S3, and ground truth TSVs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # 1. Source 1
        with open(temp_path / "train_source1.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerow(["S1-A", "Acme Corp", "123 Main St", "US"])
            writer.writerow(["S1-B", "Globex", "456 Oak Ave", "CA"])
            
        # 2. Source 2 (3 rows)
        with open(temp_path / "train_source2.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerow(["S2-A1", "Acme Corp", "123 Main St", "US"])  # True match for A
            writer.writerow(["S2-B1", "Globex Inc", "456 Oak Avenue", "CA"]) # True match for B
            writer.writerow(["S2-X", "Unrelated", "789 Pine", "US"])      # Noise
            
        # 3. Source 3 (2 rows)
        with open(temp_path / "train_source3.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["entity_id", "business_name", "business_address", "country"])
            writer.writerow(["S3-A1", "Acme Corporation", "123 Main Street", "US"]) # True match for A
            writer.writerow(["S3-Y", "Noise Co", "Nowhere", "UK"])                  # Noise
            
        # 4. Ground Truth
        with open(temp_path / "train_ground_truth.tsv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            writer.writerow(["source1_entity_id", "matched_entity_ids"])
            writer.writerow(["S1-A", "S2-A1,S3-A1"])
            writer.writerow(["S1-B", "S2-B1"])
            
        yield temp_path


def test_stream_and_normalize_chunks(synthetic_data_dir):
    """Verifies that chunks are yielded correctly and normalized."""
    s2_path = synthetic_data_dir / "train_source2.tsv"
    
    # Chunk size 2 -> should yield 2 chunks (one of size 2, one of size 1)
    chunks = list(stream_and_normalize_chunks(s2_path, chunk_size=2))
    
    assert len(chunks) == 2
    assert len(chunks[0]) == 2
    assert len(chunks[1]) == 1
    
    # Check normalization occurred (keys like name_tokens should exist)
    assert "name_tokens" in chunks[0][0]
    assert chunks[0][0]["entity_id"] == "S2-A1"


def test_chunked_benchmark_end_to_end(synthetic_data_dir):
    """
    Verifies that run_chunked_benchmark retrieves candidates correctly 
    across chunks and computes true recall without leaking ground truth.
    """
    # Run with chunk size 2 to force cross-chunk and cross-file aggregation
    results = run_chunked_benchmark(
        data_dir=synthetic_data_dir,
        chunk_size=2,
        sample_size=2,
        random_seed=42,
        max_chunks=None
    )
    
    # Metrics assertions
    assert results["s1_evaluated"] == 2
    assert results["total_targets_processed"] == 5 # 3 from S2, 2 from S3
    
    # Ground truth total matches = 3 (A has 2, B has 1)
    assert results["total_true"] == 3
    
    # We expect 100% recall on this tiny synthetic set since exact token overlap exists
    assert results["recall"] == 1.0
    assert results["true_retrieved"] == 3
    assert results["true_missed"] == 0
    
    # S1-A should retrieve S2-A1 and S3-A1 (2)
    # S1-B should retrieve S2-B1 (1)
    # Avg cands = (2 + 1) / 2 = 1.5
    assert results["avg_cands"] == 1.5


def test_max_chunks_limit(synthetic_data_dir):
    """Verifies that --max-chunks development mode limits processing."""
    # S2 has 3 rows. Chunk size 1 -> 3 chunks normally. 
    # Max chunks 1 -> should only process 1 row from S2 and 1 row from S3.
    results = run_chunked_benchmark(
        data_dir=synthetic_data_dir,
        chunk_size=1,
        sample_size=2,
        random_seed=42,
        max_chunks=1
    )
    
    # Only 1 chunk of size 1 per file -> 2 targets processed total
    assert results["total_targets_processed"] == 2
