"""
Person C — training_data.py

Converts generated candidate pairs + normalized records + ground truth
into a labeled feature dataset for training.

Enforces Candidate Restriction: Only candidates produced by the retrieval
pipeline (Person B) are included. No arbitrary negative sampling is performed.
Handles one-to-many ground truth safely without leaking information to the
feature generator.
"""

import sys
import csv
from pathlib import Path
from typing import Dict, List, Set, Union, Any, Tuple, Optional

import pandas as pd
import numpy as np

_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from person_c.features import build_pair_features
from person_c.model import MODEL_FEATURES


def _load_ground_truth(
    ground_truth: Union[str, Path, Dict[str, Any]]
) -> Dict[str, Set[str]]:
    """
    Reads train_ground_truth.tsv or accepts a dict into a Dict mapping S1 ID to a Set of true target IDs.

    Parameters
    ----------
    ground_truth : str, Path, or Dict
        Filepath to train_ground_truth.tsv or in-memory dict mapping s1_id to matched candidate IDs.
    """
    if isinstance(ground_truth, dict):
        gt_map: Dict[str, Set[str]] = {}
        for s1_id, val in ground_truth.items():
            if not s1_id:
                continue
            if isinstance(val, set):
                gt_map[str(s1_id)] = {str(x).strip() for x in val if x}
            elif isinstance(val, list):
                gt_map[str(s1_id)] = {str(x).strip() for x in val if x}
            elif isinstance(val, str):
                gt_map[str(s1_id)] = {x.strip() for x in val.split(",") if x.strip()}
            elif val is None:
                gt_map[str(s1_id)] = set()
            else:
                gt_map[str(s1_id)] = {str(val).strip()}
        return gt_map

    filepath = Path(ground_truth)
    if not filepath.exists():
        raise ValueError(f"Ground truth file not found: {filepath}")

    gt_map: Dict[str, Set[str]] = {}
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if not reader.fieldnames or "source1_entity_id" not in reader.fieldnames or "matched_entity_ids" not in reader.fieldnames:
            raise ValueError("Ground truth file must contain 'source1_entity_id' and 'matched_entity_ids' columns.")
        
        for row in reader:
            s1_id = row.get("source1_entity_id", "")
            matched_str = row.get("matched_entity_ids", "")
            if s1_id:
                if matched_str:
                    gt_map[s1_id] = {x.strip() for x in matched_str.split(",") if x.strip()}
                else:
                    gt_map[s1_id] = set()
                    
    return gt_map


def build_training_dataset(
    source1_records: List[Dict[str, Any]],
    target_records: List[Dict[str, Any]],
    candidate_pairs: pd.DataFrame,
    ground_truth: Union[str, Path, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Construct a labeled dataset from candidate pairs.
    
    Parameters
    ----------
    source1_records : List of normalized S1 dicts.
    target_records : List of normalized target (S2/S3) dicts.
    candidate_pairs : DataFrame with columns 'source1_entity_id' and 'candidate_entity_ids'.
    ground_truth : Path to train_ground_truth.tsv or in-memory dict mapping s1_id to ground truth.
    
    Returns
    -------
    pd.DataFrame containing source1_entity_id, candidate_entity_id, 45 numeric
    features, and the integer label (0 or 1).
    """
    if "source1_entity_id" not in candidate_pairs.columns or "candidate_entity_ids" not in candidate_pairs.columns:
        raise ValueError("candidate_pairs DataFrame must have 'source1_entity_id' and 'candidate_entity_ids' columns.")

    # Build O(1) lookups for records
    s1_map = {r["entity_id"]: r for r in source1_records if isinstance(r, dict) and "entity_id" in r}
    target_map = {r["entity_id"]: r for r in target_records if isinstance(r, dict) and "entity_id" in r}
    
    # Load ground truth map: s1_id -> set of true candidate ids
    gt_map = _load_ground_truth(ground_truth)
    
    features_list = []
    
    # Track pairs to prevent duplicates
    seen_pairs = set()

    for _, row in candidate_pairs.iterrows():
        s1_id = row["source1_entity_id"]
        cand_ids_val = row["candidate_entity_ids"]
        
        if cand_ids_val is None or (isinstance(cand_ids_val, str) and not cand_ids_val.strip()):
            continue
            
        if isinstance(cand_ids_val, list):
            cand_ids = [str(c).strip() for c in cand_ids_val if c]
        elif isinstance(cand_ids_val, str):
            cand_ids = [c.strip() for c in cand_ids_val.split(",") if c.strip()]
        else:
            cand_ids = []
            
        if not cand_ids:
            continue
        
        # Validation 1: S1 ID must exist
        if s1_id not in s1_map:
            raise ValueError(f"S1 ID {s1_id!r} found in candidate_pairs but not in source1_records.")
            
        s1_rec = s1_map[s1_id]
        true_cands = gt_map.get(s1_id, set())
        
        for cand_id in cand_ids:
            # Validation 3: No S1 ID as candidate
            if s1_id == cand_id:
                raise ValueError(f"S1 ID {s1_id!r} is listed as its own candidate.")
                
            # Validation 2: Candidate ID must exist
            if cand_id not in target_map:
                raise ValueError(f"Candidate ID {cand_id!r} found in candidate_pairs but not in target_records.")
                
            # Validation 4: No duplicate pairs
            pair = (s1_id, cand_id)
            if pair in seen_pairs:
                raise ValueError(f"Duplicate candidate pair found: {pair}")
            seen_pairs.add(pair)
            
            cand_rec = target_map[cand_id]
            
            # 1. Feature Generation (label blind)
            f_dict = build_pair_features(s1_rec, cand_rec)
            
            # 2. Label Assignment (One-to-many aware)
            # 1 if this specific candidate is in the true candidate set for this S1 entity
            label = 1 if cand_id in true_cands else 0
            f_dict["label"] = label
            
            features_list.append(f_dict)
            
    df = pd.DataFrame(features_list)
    
    if df.empty:
        # Edge case: No candidates to train on
        return df
        
    # Validation 5: labels are strictly 0 or 1
    if not set(df["label"].unique()).issubset({0, 1}):
        raise ValueError("Labels contain values other than 0 and 1.")
        
    # Validation 9 & 10: At least one positive and one negative label
    if 1 not in df["label"].values:
        raise ValueError("At least one positive label must exist in the dataset.")
    if 0 not in df["label"].values:
        raise ValueError("At least one negative label must exist in the dataset.")
        
    # Validations 6, 7, 8: Numeric, no NaN, no Inf
    feature_cols = [c for c in df.columns if c not in ("source1_entity_id", "candidate_entity_id", "label")]
    
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Feature column '{col}' is not numeric (type: {df[col].dtype}).")
            
    if df[feature_cols].isnull().values.any():
        raise ValueError("NaN values found in the feature matrix.")
        
    if np.isinf(df[feature_cols].values).any():
        raise ValueError("Infinite values found in the feature matrix.")
        
    return df


def extract_X_y_metadata(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extracts X (45 numeric features), y (binary labels), and metadata (identifiers).

    Parameters
    ----------
    df : pd.DataFrame
        Dataset produced by build_training_dataset().

    Returns
    -------
    X : pd.DataFrame
        Exactly the 45 numeric model features (excluding identifiers and labels).
    y : pd.Series
        Binary label vector (0 or 1).
    metadata : pd.DataFrame
        DataFrame with 'source1_entity_id' and 'candidate_entity_id'.
    """
    if df.empty:
        X = pd.DataFrame(columns=MODEL_FEATURES)
        y = pd.Series(dtype=int, name="label")
        metadata = pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id"])
        return X, y, metadata

    for col in MODEL_FEATURES:
        if col not in df.columns:
            raise ValueError(f"Required model feature column '{col}' missing from DataFrame.")

    X = df[MODEL_FEATURES].copy()
    y = df["label"].copy() if "label" in df.columns else pd.Series(dtype=int)
    metadata = df[["source1_entity_id", "candidate_entity_id"]].copy()

    return X, y, metadata

