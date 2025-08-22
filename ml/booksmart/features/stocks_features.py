from __future__ import annotations

import numpy as np
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


def _garch_volatility(returns: pd.Series, alpha: float = 0.1, beta: float = 0.85, omega: float = 0.00001) -> pd.Series:
    variance = pd.Series(index=returns.index, dtype=float)
    variance.iloc[0] = returns.var()
    
    for i in range(1, len(returns)):
        if pd.notna(returns.iloc[i-1]):
            variance.iloc[i] = omega + alpha * (returns.iloc[i-1] ** 2) + beta * variance.iloc[i-1]
        else:
            variance.iloc[i] = variance.iloc[i-1]
    
    return np.sqrt(variance * 252)


def _orderbook_imbalance(bid_vol: pd.Series, ask_vol: pd.Series) -> pd.Series:
    total_vol = bid_vol + ask_vol
    return (bid_vol - ask_vol) / (total_vol + 1e-12)


def _bid_ask_spread_bps(bid: pd.Series, ask: pd.Series) -> pd.Series:
    mid = (bid + ask) / 2
    return ((ask - bid) / (mid + 1e-12)) * 10000


def _volume_weighted_price(price: pd.Series, volume: pd.Series, window: int = 10) -> pd.Series:
    return (price * volume).rolling(window).sum() / volume.rolling(window).sum()


def _price_momentum(price: pd.Series, short_window: int = 5, long_window: int = 20) -> pd.Series:
    short_ma = price.rolling(short_window).mean()
    long_ma = price.rolling(long_window).mean()
    return (short_ma / long_ma) - 1


def _volatility_clustering(returns: pd.Series, window: int = 20) -> pd.Series:
    vol = returns.rolling(window).std() * np.sqrt(252)
    vol_ma = vol.rolling(window).mean()
    return (vol / vol_ma) - 1


def _volume_price_correlation(price: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    price_change = price.pct_change()
    return price_change.rolling(window).corr(volume)


def _abnormal_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    vol_ma = volume.rolling(window).mean()
    return volume / (vol_ma + 1e-12)


def make_stock_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    
    out["ret_1"] = _returns(out["close"], 1)
    out["ret_5"] = _returns(out["close"], 5)
    out["ret_20"] = _returns(out["close"], 20)
    
    # Moving averages and trend
    out["ema_10"] = _ema(out["close"], 10)
    out["ema_50"] = _ema(out["close"], 50)
    out["price_to_ema10"] = out["close"] / out["ema_10"] - 1
    out["price_to_ema50"] = out["close"] / out["ema_50"] - 1
    
    # Technical indicators
    out["rsi_14"] = _rsi(out["close"], 14)
    out["rsi_5"] = _rsi(out["close"], 5)
    
    out["vol_roll_20"] = out["ret_1"].rolling(20).std() * np.sqrt(252)
    out["vol_roll_5"] = out["ret_1"].rolling(5).std() * np.sqrt(252)
    out["garch_vol"] = _garch_volatility(out["ret_1"])
    out["vol_clustering"] = _volatility_clustering(out["ret_1"], 20)
    out["vol_ratio"] = out["vol_roll_5"] / (out["vol_roll_20"] + 1e-12)
    
    out["momentum_5_20"] = _price_momentum(out["close"], 5, 20)
    out["momentum_2_10"] = _price_momentum(out["close"], 2, 10)
    
    out["volume_ma_10"] = out["volume"].rolling(10).mean()
    out["volume_ma_50"] = out["volume"].rolling(50).mean()
    out["volume_ratio"] = out["volume"] / (out["volume_ma_10"] + 1e-12)
    out["abnormal_volume"] = _abnormal_volume(out["volume"], 20)
    out["volume_momentum"] = _returns(out["volume"], 5)
    out["vol_price_corr"] = _volume_price_correlation(out["close"], out["volume"], 20)
    
    out["range_pct"] = (out["high"] - out["low"]) / (out["close"] + 1e-12)
    out["upper_shadow"] = (out["high"] - np.maximum(out["open"], out["close"])) / (out["close"] + 1e-12)
    out["lower_shadow"] = (np.minimum(out["open"], out["close"]) - out["low"]) / (out["close"] + 1e-12)
    out["body_pct"] = np.abs(out["close"] - out["open"]) / (out["close"] + 1e-12)
    
    spread_est = out["vol_roll_20"] * out["close"] * 0.0001  
    out["bid_est"] = out["close"] - spread_est / 2
    out["ask_est"] = out["close"] + spread_est / 2
    out["bid_vol_est"] = out["volume"] * 0.5  
    out["ask_vol_est"] = out["volume"] * 0.5
    
    out["spread_bps"] = _bid_ask_spread_bps(out["bid_est"], out["ask_est"])
    out["orderbook_imbalance"] = _orderbook_imbalance(out["bid_vol_est"], out["ask_vol_est"])
    out["vwap_10"] = _volume_weighted_price(out["close"], out["volume"], 10)
    out["price_to_vwap"] = out["close"] / (out["vwap_10"] + 1e-12) - 1
    
    out["tod_min"] = out["ts"].dt.hour * 60 + out["ts"].dt.minute
    out["dow"] = out["ts"].dt.dayofweek
    out["is_market_open"] = ((out["tod_min"] >= 570) & (out["tod_min"] <= 960)).astype(int)  # 9:30-16:00
    
    out["realized_vol_30"] = out["ret_1"].rolling(30).std() * np.sqrt(252)
    out["skewness_20"] = out["ret_1"].rolling(20).skew()
    out["kurtosis_20"] = out["ret_1"].rolling(20).kurt()
    
    return out.dropna().reset_index(drop=True)

