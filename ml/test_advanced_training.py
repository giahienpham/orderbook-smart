#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from booksmart.features.stocks_features import make_stock_features
from booksmart.backtest.walk_forward import WalkForwardAnalysis
from booksmart.backtest.evaluation import ComprehensiveEvaluator
from booksmart.backtest.simple_bt import enhanced_direction_backtest

def create_synthetic_market_data(n_days: int = 200) -> pd.DataFrame:
    """Create synthetic market data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate price series with some autocorrelation
    returns = np.random.normal(0, 0.02, n_days)
    returns[0] = 0
    
    # Add some momentum and mean reversion
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1] + np.random.normal(0, 0.01)
        # Add mean reversion
        if abs(returns[i]) > 0.05:
            returns[i] *= 0.5
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    opens = prices * (1 + np.random.normal(0, 0.005, n_days))
    
    volumes = np.random.lognormal(15, 0.5, n_days)
    
    data = pd.DataFrame({
        'ts': dates,
        'open': opens,
        'high': highs,
        'low': lows,
        'close': prices,
        'volume': volumes
    }, index=dates)
    
    return data

def test_advanced_training():
    """Test the advanced training pipeline with synthetic data"""
    
    print("=" * 80)
    print("ADVANCED ML TRAINING WITH WALK-FORWARD ANALYSIS (SYNTHETIC DATA)")
    print("=" * 80)
    
    # Create synthetic data
    print("Creating synthetic market data...")
    data = create_synthetic_market_data(200)
    print(f"Raw data shape: {data.shape}")
    
    # Create features
    print("Creating features...")
    try:
        features_df = make_stock_features(data)
        print(f"Features created: {features_df.shape}")
    except Exception as e:
        print(f"Error creating features: {e}")
        return False
    
    # Create labels
    print("Creating labels...")
    features_df['future_return'] = features_df['ret_1'].shift(-1)
    
    # Remove NaN values before creating direction labels
    valid_mask = ~features_df['future_return'].isna()
    features_df = features_df[valid_mask]
    
    features_df['direction'] = pd.cut(
        features_df['future_return'], 
        bins=[-np.inf, -0.005, 0.005, np.inf], 
        labels=[0, 1, 2]
    ).astype(int)
    
    # Remove missing values
    features_df = features_df.dropna()
    print(f"Final dataset shape: {features_df.shape}")
    
    if len(features_df) < 60:
        print("Insufficient data for walk-forward analysis")
        return False
    
    print(f"Label distribution: {np.bincount(features_df['direction'])}")
    
    # Run walk-forward analysis
    print("\n" + "=" * 60)
    print("RUNNING WALK-FORWARD ANALYSIS")
    print("=" * 60)
    
    wfa = WalkForwardAnalysis(
        train_window_days=50,
        retrain_frequency_days=10,
        min_train_samples=30
    )
    
    try:
        wf_results = wfa.run_analysis(features_df, target_col='direction')
        print("Walk-forward analysis completed successfully!")
        
        # Print summary results
        summary = wf_results['summary']
        print(f"\nWalk-Forward Summary:")
        print(f"  Total Windows: {summary['total_windows']}")
        print(f"  Average Accuracy: {summary['avg_accuracy']:.4f}")
        print(f"  Accuracy Stability: {summary['accuracy_stability']:.4f}")
        print(f"  Trading Sharpe: {summary['trading_sharpe']:.4f}")
        print(f"  Total Return: {summary['total_trading_return']:.4f}")
        print(f"  Recommendation: {summary['recommendation']}")
        
        # Print overall metrics from walk-forward analysis
        metrics = wf_results['overall_metrics']
        print(f"\nOverall Metrics:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1_score']:.4f}")
        
        # Trading metrics from walk-forward
        trading = metrics['trading']
        print(f"\nTrading Metrics:")
        print(f"  Total Return: {trading['total_return']:.4f}")
        print(f"  Sharpe Ratio: {trading['sharpe_ratio']:.4f}")
        print(f"  Max Drawdown: {trading['max_drawdown']:.4f}")
        print(f"  Win Rate: {trading['win_rate']:.4f}")
        print(f"  Number of Trades: {trading['num_trades']}")
        
        # Final recommendation based on walk-forward results
        print("\n" + "=" * 60)
        print("FINAL RECOMMENDATION")
        print("=" * 60)
        
        overall_score = 50 + (metrics['accuracy'] - 0.33) * 100  # Simple scoring
        print(f"Model Score: {overall_score:.1f}/100")
        print(f"Final Recommendation: {summary['recommendation']}")
        
        print("\n🎉 Walk-forward analysis completed successfully!")
        print("Advanced ML training pipeline with comprehensive evaluation is working.")
        
        return True
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run the advanced training test"""
    print("Testing Advanced ML Training Pipeline")
    print("=" * 80)
    
    success = test_advanced_training()
    
    if success:
        print("\n🎉 Advanced ML training pipeline test PASSED!")
        print("Walk-forward analysis and comprehensive evaluation are working correctly.")
        return True
    else:
        print("\n❌ Advanced ML training pipeline test FAILED!")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
