from __future__ import annotations

import pandas as pd


def _returns(s: pd.Series, periods: int) -> pd.Series:
    return s.pct_change(periods)


def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    gain = up.ewm(alpha=1 / period, adjust=False).mean()
    loss = down.ewm(alpha=1 / period, adjust=False).mean()
    rs = gain / (loss + 1e-12)
    return 100 - (100 / (1 + rs))


def make_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    """Input columns: ts, open, high, low, close, volume
    Output: adds feature columns and drops rows with NA.
    """
    out = df.copy()
    out["ret_1"] = _returns(out["close"], 1)
    out["ret_5"] = _returns(out["close"], 5)
    out["ema_10"] = _ema(out["close"], 10)
    out["ema_50"] = _ema(out["close"], 50)
    out["rsi_14"] = _rsi(out["close"], 14)
    out["vol_roll_20"] = out["ret_1"].rolling(20).std()
    out["tod_min"] = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    return out.dropna().reset_index(drop=True)

