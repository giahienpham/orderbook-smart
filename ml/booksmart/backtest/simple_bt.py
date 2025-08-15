from __future__ import annotations

import numpy as np
import pandas as pd


def simple_direction_backtest(df: pd.DataFrame, predictions: np.ndarray, 
                            transaction_cost: float = 0.0005) -> dict:
    """Simple backtest for direction predictions.
    
    Args:
        df: DataFrame with 'close' and 'ret_1' columns
        predictions: Array of -1, 0, 1 predictions
        transaction_cost: Cost per trade (fraction of notional)
    
    Returns:
        Dict with performance metrics
    """
    if len(df) != len(predictions):
        raise ValueError("Length mismatch between df and predictions")
    
    # Simple strategy: take position based on prediction
    positions = predictions.copy()
    
    # Calculate returns: position[t] * return[t+1]
    forward_returns = df['ret_1'].shift(-1).fillna(0)
    strategy_returns = positions[:-1] * forward_returns.iloc[:-1]
    
    # Apply transaction costs when position changes
    position_changes = np.diff(np.concatenate([[0], positions]))
    transaction_costs = np.abs(position_changes) * transaction_cost
    
    # Net returns after costs
    net_returns = strategy_returns - transaction_costs[1:]
    
    # Performance metrics
    total_return = (1 + net_returns).prod() - 1
    annualized_return = (1 + total_return) ** (252 * 390 / len(net_returns)) - 1  # Assuming 1min bars
    volatility = net_returns.std() * np.sqrt(252 * 390)
    sharpe = annualized_return / volatility if volatility > 0 else 0
    max_dd = (net_returns.cumsum() - net_returns.cumsum().cummax()).min()
    
    # Win rate
    win_rate = (net_returns > 0).mean()
    
    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "volatility": volatility,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "win_rate": win_rate,
        "num_trades": len(net_returns),
        "avg_return_per_trade": net_returns.mean()
    }