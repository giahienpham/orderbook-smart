from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd


def time_split(df: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n = len(df)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train = df.iloc[:n_train]
    val = df.iloc[n_train : n_train + n_val]
    test = df.iloc[n_train + n_val :]
    return train, val, test


def build_xy(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> Tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=float)
    y = df[label_col].to_numpy()
    return X, y

