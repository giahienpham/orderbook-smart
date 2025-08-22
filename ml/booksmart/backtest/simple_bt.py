from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple


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


def enhanced_direction_backtest(predictions: np.ndarray, 
                              actual: np.ndarray,
                              returns: Optional[np.ndarray] = None,
                              confidence: Optional[np.ndarray] = None,
                              timestamps: Optional[pd.DatetimeIndex] = None) -> Dict:
    """
    Enhanced backtest for direction prediction models with comprehensive metrics
    
    Args:
        predictions: Model predictions (0=down, 1=flat, 2=up)
        actual: Actual directions (0=down, 1=flat, 2=up) 
        returns: Optional actual returns for each period
        confidence: Optional confidence scores for each prediction
        timestamps: Optional timestamps for temporal analysis
    
    Returns:
        Dictionary containing comprehensive backtest results
    """
    
    if len(predictions) != len(actual):
        raise ValueError("Predictions and actual must have same length")
    
    # Initialize confidence if not provided
    if confidence is None:
        confidence = np.ones(len(predictions))
    
    # Calculate basic accuracy
    correct_predictions = (predictions == actual)
    accuracy = np.mean(correct_predictions)
    
    # Generate trading returns
    trading_returns = []
    positions = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        act = actual[i]
        conf = confidence[i]
        
        # Determine position based on prediction
        if pred == 2:  # Up prediction
            position = 1.0 * conf  # Long position weighted by confidence
        elif pred == 0:  # Down prediction
            position = -1.0 * conf  # Short position weighted by confidence
        else:  # Flat prediction
            position = 0.0
        
        positions.append(position)
        
        # Calculate return
        if returns is not None and len(returns) > i:
            # Use actual market returns
            period_return = position * returns[i]
        else:
            # Simulate returns based on actual direction
            if act == 2:  # Actually went up
                market_return = 0.01  # 1% up move
            elif act == 0:  # Actually went down
                market_return = -0.01  # 1% down move
            else:  # Actually flat
                market_return = 0.0
            
            period_return = position * market_return
        
        trading_returns.append(period_return)
    
    trading_returns = np.array(trading_returns)
    positions = np.array(positions)
    
    # Calculate performance metrics
    total_return = np.sum(trading_returns)
    num_trades = np.sum(np.abs(positions) > 0.1)  # Count significant positions
    
    # Risk metrics
    if len(trading_returns) > 1:
        volatility = np.std(trading_returns)
        sharpe_ratio = np.mean(trading_returns) / volatility if volatility > 0 else 0
        
        # Annualized metrics (assuming daily data)
        annualized_return = np.mean(trading_returns) * 252
        annualized_vol = volatility * np.sqrt(252)
        annualized_sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0
    else:
        volatility = 0
        sharpe_ratio = 0
        annualized_return = 0
        annualized_vol = 0
        annualized_sharpe = 0
    
    # Drawdown analysis
    cumulative_returns = np.cumsum(trading_returns)
    running_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = cumulative_returns - running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
    
    # Win rate and trade analysis
    profitable_trades = trading_returns[trading_returns > 0]
    losing_trades = trading_returns[trading_returns < 0]
    
    win_rate = len(profitable_trades) / num_trades if num_trades > 0 else 0
    avg_win = np.mean(profitable_trades) if len(profitable_trades) > 0 else 0
    avg_loss = np.mean(losing_trades) if len(losing_trades) > 0 else 0
    profit_factor = abs(np.sum(profitable_trades) / np.sum(losing_trades)) if len(losing_trades) > 0 else np.inf
    
    # Additional metrics
    hit_rate = accuracy  # Same as accuracy for direction prediction
    
    # Information ratio (if we have benchmark returns)
    information_ratio = 0  # Placeholder - would need benchmark
    
    # Calmar ratio (return/max_drawdown)
    calmar_ratio = abs(annualized_return / max_drawdown) if max_drawdown < 0 else 0
    
    # Compile results
    results = {
        # Basic metrics
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': annualized_vol,
        'sharpe_ratio': annualized_sharpe,
        'information_ratio': information_ratio,
        'calmar_ratio': calmar_ratio,
        
        # Risk metrics
        'max_drawdown': max_drawdown,
        'var_95': np.percentile(trading_returns, 5) if len(trading_returns) > 0 else 0,
        'var_99': np.percentile(trading_returns, 1) if len(trading_returns) > 0 else 0,
        
        # Trading metrics
        'num_trades': num_trades,
        'win_rate': win_rate,
        'hit_rate': hit_rate,
        'avg_return_per_trade': total_return / num_trades if num_trades > 0 else 0,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        
        # Additional details
        'accuracy': accuracy,
        'total_periods': len(predictions),
        'active_periods': num_trades,
        'activity_rate': num_trades / len(predictions) if len(predictions) > 0 else 0
    }
    
    # Add temporal analysis if timestamps provided
    if timestamps is not None:
        temporal_results = _analyze_temporal_performance(
            trading_returns, timestamps, positions
        )
        results['temporal_analysis'] = temporal_results
    
    # Add confidence analysis if provided
    if confidence is not None:
        confidence_results = _analyze_confidence_performance(
            trading_returns, confidence, correct_predictions
        )
        results['confidence_analysis'] = confidence_results
    
    return results


def _analyze_temporal_performance(returns: np.ndarray, 
                                timestamps: pd.DatetimeIndex,
                                positions: np.ndarray) -> Dict:
    """Analyze performance over time"""
    
    df = pd.DataFrame({
        'returns': returns,
        'positions': positions,
        'timestamp': timestamps
    })
    
    # Monthly performance
    df['month'] = df['timestamp'].dt.to_period('M')
    monthly_stats = df.groupby('month').agg({
        'returns': ['sum', 'std', 'count'],
        'positions': 'mean'
    }).round(4)
    
    # Rolling performance
    df['cumulative_return'] = df['returns'].cumsum()
    df['rolling_sharpe_30'] = df['returns'].rolling(30).mean() / df['returns'].rolling(30).std()
    
    return {
        'monthly_returns': monthly_stats['returns']['sum'].to_dict(),
        'monthly_volatility': monthly_stats['returns']['std'].to_dict(),
        'monthly_trade_count': monthly_stats['returns']['count'].to_dict(),
        'performance_trend': _calculate_performance_trend(df['cumulative_return'].values),
        'stability_score': 1.0 / (1.0 + df['returns'].std()) if df['returns'].std() > 0 else 1.0
    }


def _analyze_confidence_performance(returns: np.ndarray,
                                  confidence: np.ndarray, 
                                  correct_predictions: np.ndarray) -> Dict:
    """Analyze performance by confidence levels"""
    
    # Confidence buckets
    conf_buckets = pd.qcut(confidence, q=5, labels=['Low', 'Med-Low', 'Med', 'Med-High', 'High'])
    
    bucket_stats = {}
    for bucket in conf_buckets.cat.categories:
        mask = conf_buckets == bucket
        if np.sum(mask) > 0:
            bucket_stats[bucket] = {
                'count': np.sum(mask),
                'avg_return': np.mean(returns[mask]),
                'accuracy': np.mean(correct_predictions[mask]),
                'sharpe': np.mean(returns[mask]) / np.std(returns[mask]) if np.std(returns[mask]) > 0 else 0
            }
    
    # Confidence-return correlation
    conf_return_corr = np.corrcoef(confidence, returns)[0, 1] if len(confidence) > 1 else 0
    conf_accuracy_corr = np.corrcoef(confidence, correct_predictions.astype(float))[0, 1] if len(confidence) > 1 else 0
    
    return {
        'bucket_performance': bucket_stats,
        'confidence_return_correlation': conf_return_corr,
        'confidence_accuracy_correlation': conf_accuracy_corr,
        'high_confidence_accuracy': np.mean(correct_predictions[confidence > 0.8]) if np.sum(confidence > 0.8) > 0 else 0
    }


def _calculate_performance_trend(cumulative_returns: np.ndarray) -> float:
    """Calculate performance trend using linear regression"""
    if len(cumulative_returns) < 2:
        return 0.0
    
    x = np.arange(len(cumulative_returns))
    try:
        slope = np.polyfit(x, cumulative_returns, 1)[0]
        return slope
    except:
        return 0.0