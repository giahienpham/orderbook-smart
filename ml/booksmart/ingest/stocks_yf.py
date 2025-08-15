from __future__ import annotations

import datetime as dt
from typing import Iterable

import pandas as pd
import yfinance as yf


def fetch_ohlcv(symbol: str, interval: str = "1m", lookback_days: int = 5) -> pd.DataFrame:
    end = dt.datetime.utcnow()
    start = end - dt.timedelta(days=lookback_days)
    df = yf.download(symbol, start=start, end=end, interval=interval, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {symbol} interval={interval}")
    df = (
        df.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })
        .reset_index()
        .rename(columns={"Datetime": "ts"})
    )
    return df[["ts", "open", "high", "low", "close", "volume"]]


def fetch_many(symbols: Iterable[str], interval: str = "1m", lookback_days: int = 5) -> dict[str, pd.DataFrame]:
    return {sym: fetch_ohlcv(sym, interval=interval, lookback_days=lookback_days) for sym in symbols}

