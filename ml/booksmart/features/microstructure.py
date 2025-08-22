from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def orderbook_flow_imbalance(
    bids: np.ndarray, 
    asks: np.ndarray, 
    bid_sizes: np.ndarray, 
    ask_sizes: np.ndarray,
    depth: int = 5
) -> float:
    """Calculate orderbook flow imbalance across multiple levels.
    
    Args:
        bids: Array of bid prices (sorted descending)
        asks: Array of ask prices (sorted ascending)  
        bid_sizes: Corresponding bid sizes
        ask_sizes: Corresponding ask sizes
        depth: Number of levels to consider
        
    Returns:
        Flow imbalance metric [-1, 1]
    """
    if len(bids) == 0 or len(asks) == 0:
        return 0.0
        
    # Take top N levels
    depth = min(depth, len(bids), len(asks))
    
    # Weight by inverse of distance from mid
    mid = (bids[0] + asks[0]) / 2
    
    bid_weights = 1.0 / (1.0 + np.abs(bids[:depth] - mid))
    ask_weights = 1.0 / (1.0 + np.abs(asks[:depth] - mid))
    
    weighted_bid_vol = np.sum(bid_sizes[:depth] * bid_weights)
    weighted_ask_vol = np.sum(ask_sizes[:depth] * ask_weights)
    
    total_vol = weighted_bid_vol + weighted_ask_vol
    if total_vol == 0:
        return 0.0
        
    return (weighted_bid_vol - weighted_ask_vol) / total_vol


def microprice(bid: float, ask: float, bid_size: float, ask_size: float) -> float:
    """Calculate microprice - size-weighted mid price.
    
    More accurate than simple mid when sizes are imbalanced.
    """
    total_size = bid_size + ask_size
    if total_size == 0:
        return (bid + ask) / 2
    
    return (bid * ask_size + ask * bid_size) / total_size


def effective_spread(trade_price: float, mid_price: float) -> float:
    """Effective spread in basis points."""
    if mid_price == 0:
        return 0.0
    return 2 * abs(trade_price - mid_price) / mid_price * 10000


def price_impact_indicator(
    prices: np.ndarray,
    volumes: np.ndarray, 
    window: int = 10
) -> np.ndarray:
    """Estimate price impact using volume-weighted price changes."""
    if len(prices) < window:
        return np.zeros_like(prices)
    
    impacts = np.zeros_like(prices)
    
    for i in range(window, len(prices)):
        recent_prices = prices[i-window:i]
        recent_volumes = volumes[i-window:i]
        if np.sum(recent_volumes) > 0:
            vwap = np.sum(recent_prices * recent_volumes) / np.sum(recent_volumes)
            impacts[i] = (prices[i] - vwap) / vwap
        
    return impacts


def order_flow_toxicity(
    trade_prices: np.ndarray,
    trade_sizes: np.ndarray,
    trade_directions: np.ndarray,  # 1 for buy, -1 for sell
    window: int = 50
) -> np.ndarray:
    """VPIN (Volume-Synchronized Probability of Informed Trading) approximation."""
    if len(trade_prices) < window:
        return np.zeros_like(trade_prices)
    
    vpin = np.zeros_like(trade_prices)
    
    for i in range(window, len(trade_prices)):
        recent_sizes = trade_sizes[i-window:i]
        recent_directions = trade_directions[i-window:i]
        
        buy_volume = np.sum(recent_sizes[recent_directions > 0])
        sell_volume = np.sum(recent_sizes[recent_directions < 0])
        total_volume = buy_volume + sell_volume
        
        if total_volume > 0:
            vpin[i] = abs(buy_volume - sell_volume) / total_volume
            
    return vpin


def volatility_signature_plot_adjustment(
    returns: np.ndarray,
    sampling_freq: int = 60  
) -> float:
    """Adjust returns for microstructure noise using signature plot method."""
    if len(returns) < 100:
        return np.std(returns)
    frequencies = [1, 5, 10, 30, 60, 300]  
    variances = []
    
    for freq in frequencies:
        if freq <= sampling_freq and len(returns) > freq:
            subsampled = returns[::freq]
            if len(subsampled) > 10:
                variances.append(np.var(subsampled) * freq)
    
    if len(variances) < 2:
        return np.std(returns)
    min_var = min(variances)
    return np.sqrt(min_var)


def trade_classification_lr(
    trade_price: float,
    bid: float,
    ask: float,
    prev_trade_price: Optional[float] = None
) -> int:
    """Lee-Ready trade classification algorithm.
    
    Returns:
        1 for buyer-initiated (aggressive buy)
        -1 for seller-initiated (aggressive sell)
        0 for uncertain
    """
    mid = (bid + ask) / 2
    
    if trade_price > mid:
        return 1
    elif trade_price < mid:
        return -1
    else:
        if prev_trade_price is None:
            return 0
        elif trade_price > prev_trade_price:
            return 1
        elif trade_price < prev_trade_price:
            return -1
        else:
            return 0


def realized_kernel_volatility(
    prices: np.ndarray,
    kernel_type: str = "bartlett",
    bandwidth: int = 10
) -> float:
    """Realized kernel volatility estimator for handling microstructure noise."""
    if len(prices) < bandwidth * 2:
        return 0.0
    
    log_prices = np.log(prices)
    returns = np.diff(log_prices)
    
    if kernel_type == "bartlett":
        weights = np.array([1 - abs(h) / (bandwidth + 1) for h in range(-bandwidth, bandwidth + 1)])
    else:
        weights = np.ones(2 * bandwidth + 1) / (2 * bandwidth + 1)
    
    gamma = np.zeros(2 * bandwidth + 1)
    for h in range(-bandwidth, bandwidth + 1):
        if h == 0:
            gamma[h + bandwidth] = np.var(returns)
        else:
            if len(returns) > abs(h):
                gamma[h + bandwidth] = np.cov(returns[:-abs(h)], returns[abs(h):])[0, 1]
    
    realized_var = np.sum(weights * gamma)
    return np.sqrt(max(realized_var, 0.0))


def market_quality_metrics(
    bid_ask_spreads: np.ndarray,
    depths: np.ndarray,
    trade_sizes: np.ndarray,
    window: int = 100
) -> dict:
    """Calculate market quality metrics over rolling window."""
    if len(bid_ask_spreads) < window:
        return {
            "avg_spread": 0.0,
            "spread_volatility": 0.0,
            "avg_depth": 0.0,
            "depth_volatility": 0.0,
            "resilience": 0.0
        }
    
    recent_spreads = bid_ask_spreads[-window:]
    recent_depths = depths[-window:]
    recent_sizes = trade_sizes[-window:]
    
    spread_changes = np.diff(recent_spreads)
    size_impact = np.corrcoef(recent_sizes[:-1], np.abs(spread_changes))[0, 1] if len(recent_sizes) > 1 else 0.0
    
    return {
        "avg_spread": np.mean(recent_spreads),
        "spread_volatility": np.std(recent_spreads),
        "avg_depth": np.mean(recent_depths),
        "depth_volatility": np.std(recent_depths),
        "resilience": -size_impact if not np.isnan(size_impact) else 0.0  # Lower correlation = better resilience
    }


def high_frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add high-frequency microstructure features to dataframe.
    
    Expected columns: timestamp, bid, ask, bid_size, ask_size, last_price, volume
    """
    out = df.copy()
    
    out["mid_price"] = (out["bid"] + out["ask"]) / 2
    out["spread"] = out["ask"] - out["bid"]
    out["spread_bps"] = (out["spread"] / out["mid_price"]) * 10000
    out["microprice"] = out.apply(
        lambda x: microprice(x["bid"], x["ask"], x["bid_size"], x["ask_size"]), axis=1
    )
    
    out["imbalance"] = (out["bid_size"] - out["ask_size"]) / (out["bid_size"] + out["ask_size"])
    
    out["price_to_mid"] = (out["last_price"] - out["mid_price"]) / out["mid_price"]
    out["price_to_micro"] = (out["last_price"] - out["microprice"]) / out["microprice"]
    
    # Rolling features
    for window in [10, 30, 100]:
        out[f"spread_ma_{window}"] = out["spread_bps"].rolling(window).mean()
        out[f"imbalance_ma_{window}"] = out["imbalance"].rolling(window).mean()
        out[f"volume_ma_{window}"] = out["volume"].rolling(window).mean()
        
        returns = out["mid_price"].pct_change()
        out[f"realized_vol_{window}"] = returns.rolling(window).std() * np.sqrt(252 * 390)  # Assuming 1-min bars
        
        out[f"spread_vol_{window}"] = out["spread_bps"].rolling(window).std()
    
    if len(out) > 50:
        # Trade direction classification (simplified)
        out["trade_direction"] = np.where(
            out["last_price"] > out["mid_price"], 1,
            np.where(out["last_price"] < out["mid_price"], -1, 0)
        )
        
        # Order flow toxicity approximation
        out["order_flow_toxic"] = out["trade_direction"].rolling(50).apply(
            lambda x: np.abs(np.sum(x)) / len(x) if len(x) > 0 else 0
        )
    
    return out.dropna()
