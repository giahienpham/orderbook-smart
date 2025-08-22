#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from booksmart.backtest.walk_forward import WalkForwardAnalysis
from booksmart.backtest.evaluation import ComprehensiveEvaluator
from booksmart.features.stocks_features import make_stock_features
from booksmart.ingest.stocks_yf import fetch_ohlcv

def test_walk_forward_analysis():
    """Test walk-forward analysis with real market data"""
    print("=" * 60)
    print("TESTING WALK-FORWARD ANALYSIS")
    print("=" * 60)
    
    # Get some real data
    print("Fetching market data...")
    symbol = "AAPL"
    data = fetch_ohlcv(symbol, lookback_days=180, interval="1d")
    
    if data is None or len(data) < 50:
        print("Insufficient data, creating synthetic dataset...")
        data = create_synthetic_market_data(180)
    
    print(f"Raw data shape: {data.shape}")
    
    # Create features
    print("Creating features...")
    try:
        features_df = make_stock_features(data)
        print(f"Features created: {features_df.shape}")
    except Exception as e:
        print(f"Feature creation failed: {e}")
        return False
    
    # Create labels (direction prediction)
    print("Creating labels...")
    features_df['future_return'] = features_df['ret_1'].shift(-1)
    
    # Remove NaN values before creating direction labels
    valid_mask = ~features_df['future_return'].isna()
    features_df = features_df[valid_mask]
    
    features_df['direction'] = pd.cut(
        features_df['future_return'], 
        bins=[-np.inf, -0.002, 0.002, np.inf], 
        labels=[0, 1, 2]
    ).astype(int)
    
    # Remove rows with missing values
    features_df = features_df.dropna()
    print(f"Final dataset shape: {features_df.shape}")
    
    if len(features_df) < 60:
        print("Insufficient data for walk-forward analysis")
        return False
    
    # Run walk-forward analysis
    print("\nRunning walk-forward analysis...")
    wfa = WalkForwardAnalysis(
        train_window_days=30,  # Smaller window for test
        retrain_frequency_days=5,
        min_train_samples=20
    )
    
    try:
        results = wfa.run_analysis(features_df, target_col='direction')
        print("Walk-forward analysis completed successfully!")
        
        # Print summary
        summary = results['summary']
        print(f"\nSUMMARY:")
        print(f"Total windows: {summary['total_windows']}")
        print(f"Average accuracy: {summary['avg_accuracy']:.4f}")
        print(f"Accuracy stability: {summary['accuracy_stability']:.4f}")
        print(f"Trading Sharpe: {summary['trading_sharpe']:.4f}")
        print(f"Total trading return: {summary['total_trading_return']:.4f}")
        print(f"Model degradation: {summary['model_degradation']:.6f}")
        print(f"Recommendation: {summary['recommendation']}")
        
        # Print overall metrics
        metrics = results['overall_metrics']
        print(f"\nOVERALL METRICS:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1_score']:.4f}")
        print(f"Average confidence: {metrics['avg_confidence']:.4f}")
        print(f"Total predictions: {metrics['total_predictions']}")
        
        # Print stability metrics
        stability = metrics['stability']
        print(f"\nSTABILITY METRICS:")
        print(f"Accuracy std: {stability['accuracy_std']:.4f}")
        print(f"Accuracy trend: {stability['accuracy_trend']:.6f}")
        print(f"Min window accuracy: {stability['min_window_accuracy']:.4f}")
        print(f"Max window accuracy: {stability['max_window_accuracy']:.4f}")
        
        # Print trading metrics
        trading = metrics['trading']
        print(f"\nTRADING METRICS:")
        print(f"Total return: {trading['total_return']:.4f}")
        print(f"Number of trades: {trading['num_trades']}")
        print(f"Win rate: {trading['win_rate']:.4f}")
        print(f"Sharpe ratio: {trading['sharpe_ratio']:.4f}")
        print(f"Max drawdown: {trading['max_drawdown']:.4f}")
        
        return True
        
    except Exception as e:
        print(f"Walk-forward analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_comprehensive_evaluator():
    """Test comprehensive model evaluation"""
    print("\n" + "=" * 60)
    print("TESTING COMPREHENSIVE EVALUATOR")
    print("=" * 60)
    
    # Create synthetic test data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic predictions and actual values
    y_true = np.random.choice([0, 1, 2], n_samples, p=[0.3, 0.4, 0.3])
    
    # Make predictions somewhat correlated with true values
    y_pred = y_true.copy()
    noise_indices = np.random.choice(n_samples, size=int(0.3 * n_samples), replace=False)
    y_pred[noise_indices] = np.random.choice([0, 1, 2], len(noise_indices))
    
    # Generate probability matrix
    y_proba = np.random.dirichlet([1, 1, 1], n_samples)
    for i in range(n_samples):
        # Make probabilities somewhat consistent with predictions
        y_proba[i, y_pred[i]] = max(y_proba[i, y_pred[i]], 0.5)
        y_proba[i] = y_proba[i] / y_proba[i].sum()  # Renormalize
    
    # Generate synthetic returns
    returns = np.random.normal(0, 0.01, n_samples)
    
    # Generate timestamps
    timestamps = pd.date_range(start='2023-01-01', periods=n_samples, freq='D')
    
    print(f"Generated synthetic data: {n_samples} samples")
    print(f"True class distribution: {np.bincount(y_true)}")
    print(f"Predicted class distribution: {np.bincount(y_pred)}")
    print(f"Accuracy: {np.mean(y_true == y_pred):.4f}")
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluator = ComprehensiveEvaluator()
    
    try:
        results = evaluator.evaluate_model_performance(
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            returns=returns,
            timestamps=timestamps
        )
        
        print("Comprehensive evaluation completed successfully!")
        
        # Print classification metrics
        classification = results['classification']
        print(f"\nCLASSIFICATION METRICS:")
        print(f"Accuracy: {classification['accuracy']:.4f}")
        print(f"Precision: {classification['precision']:.4f}")
        print(f"Recall: {classification['recall']:.4f}")
        print(f"F1 Score: {classification['f1_score']:.4f}")
        print(f"AUC Score: {classification['auc_score']:.4f}")
        print(f"Log Loss: {classification['log_loss']:.4f}")
        
        # Print confidence metrics
        confidence = results['confidence']
        print(f"\nCONFIDENCE METRICS:")
        print(f"Average confidence: {confidence['avg_confidence']:.4f}")
        print(f"Confidence std: {confidence['confidence_std']:.4f}")
        print(f"Overconfidence bias: {confidence['overconfidence_bias']:.4f}")
        print(f"Confidence-accuracy correlation: {confidence['confidence_accuracy_corr']:.4f}")
        
        # Print trading metrics
        trading = results['trading']
        print(f"\nTRADING METRICS:")
        print(f"Total return: {trading['total_return']:.4f}")
        print(f"Annualized return: {trading['annualized_return']:.4f}")
        print(f"Sharpe ratio: {trading['sharpe_ratio']:.4f}")
        print(f"Information ratio: {trading['information_ratio']:.4f}")
        print(f"Max drawdown: {trading['max_drawdown']:.4f}")
        print(f"Win rate: {trading['win_rate']:.4f}")
        print(f"Profit factor: {trading['profit_factor']:.4f}")
        
        # Print temporal metrics
        temporal = results['temporal']
        print(f"\nTEMPORAL METRICS:")
        print(f"Monthly accuracy mean: {temporal['monthly_accuracy_mean']:.4f}")
        print(f"Monthly accuracy std: {temporal['monthly_accuracy_std']:.4f}")
        print(f"Accuracy trend: {temporal['accuracy_trend']:.6f}")
        print(f"Stability score: {temporal['stability_score']:.4f}")
        print(f"Performance decay: {temporal['performance_decay']:.6f}")
        
        # Print risk metrics
        risk = results['risk']
        print(f"\nRISK METRICS:")
        print(f"Average entropy: {risk['avg_entropy']:.4f}")
        print(f"Uncertainty-accuracy correlation: {risk['uncertainty_accuracy_correlation']:.4f}")
        print(f"Tail risk accuracy: {risk['tail_risk_accuracy']:.4f}")
        print(f"Robustness score: {risk['robustness_score']:.4f}")
        
        # Print information metrics
        information = results['information']
        print(f"\nINFORMATION METRICS:")
        print(f"Mutual information: {information['mutual_information']:.4f}")
        print(f"Information gain: {information['information_gain']:.4f}")
        print(f"Normalized mutual info: {information['normalized_mutual_info']:.4f}")
        
        # Print overall score
        print(f"\nOVERALL SCORE: {results['overall_score']:.2f}/100")
        
        return True
        
    except Exception as e:
        print(f"Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_synthetic_market_data(n_days: int) -> pd.DataFrame:
    """Create synthetic market data for testing"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='D')
    
    # Generate price series with some autocorrelation
    returns = np.random.normal(0, 0.02, n_days)
    returns[0] = 0
    
    # Add some momentum
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1] + np.random.normal(0, 0.01)
    
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLC data
    highs = prices * (1 + np.abs(np.random.normal(0, 0.01, n_days)))
    lows = prices * (1 - np.abs(np.random.normal(0, 0.01, n_days)))
    opens = prices * (1 + np.random.normal(0, 0.005, n_days))
    
    volumes = np.random.lognormal(15, 0.5, n_days)
    
    data = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    return data

def main():
    """Run all tests"""
    print("Testing Walk-Forward Analysis and Comprehensive Evaluation")
    print("=" * 80)
    
    success_count = 0
    total_tests = 2
    
    # Test 1: Walk-forward analysis
    if test_walk_forward_analysis():
        success_count += 1
        print("✓ Walk-forward analysis test PASSED")
    else:
        print("✗ Walk-forward analysis test FAILED")
    
    # Test 2: Comprehensive evaluator
    if test_comprehensive_evaluator():
        success_count += 1
        print("✓ Comprehensive evaluator test PASSED")
    else:
        print("✗ Comprehensive evaluator test FAILED")
    
    print("\n" + "=" * 80)
    print(f"SUMMARY: {success_count}/{total_tests} tests passed")
    
    if success_count == total_tests:
        print("🎉 All tests passed! Walk-forward analysis is ready for Jane Street.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
