from __future__ import annotations

import numpy as np
import pandas as pd


def label_next_direction(df: pd.DataFrame, horizon: int = 5, neutral_eps: float = 0.0005) -> pd.Series:
    """Labels: -1, 0, +1 based on next-horizon return of close.
    horizon measured in rows (e.g., 5 for ~5 minutes if 1m bars).
    neutral_eps: threshold for flat.
    """
    future = df["close"].shift(-horizon)
    ret = (future - df["close"]) / df["close"]
    y = np.where(ret > neutral_eps, 1, np.where(ret < -neutral_eps, -1, 0))
    return pd.Series(y.flatten(), index=df.index)

