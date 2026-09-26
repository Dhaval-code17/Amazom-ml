from typing import Iterator, Tuple, List, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

def get_group_kfold_splits(
    df: pd.DataFrame,
    group_col: str = "source1_entity_id",
    n_splits: int = 5
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate GroupKFold splits grouped by source1_entity_id to ensure no leakage.
    
    df: DataFrame containing the data to split
    group_col: Column name to group on (default: 'source1_entity_id')
    n_splits: Number of folds (default: 5)
    
    Yields (train_indices, val_indices) tuples.
    """
    gkf = GroupKFold(n_splits=n_splits)
    groups = df[group_col].values
    X = np.zeros(len(df))  # Dummy X array for sklearn generator interface
    
    for train_idx, val_idx in gkf.split(X, groups=groups):
        yield train_idx, val_idx

def verify_no_group_leakage(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    group_col: str = "source1_entity_id"
) -> bool:
    """
    Check if any group from train_idx appears in val_idx. Returns True if zero leakage.
    """
    train_groups = set(df.iloc[train_idx][group_col].unique())
    val_groups = set(df.iloc[val_idx][group_col].unique())
    overlap = train_groups.intersection(val_groups)
    return len(overlap) == 0
