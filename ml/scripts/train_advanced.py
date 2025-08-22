#!/usr/bin/env python3

import argparse
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from booksmart.features.stocks_features import make_stock_features
from booksmart.ingest.stocks_yf import fetch_ohlcv
from booksmart.backtest.walk_forward import WalkForwardAnalysis
from booksmart.backtest.evaluation import ComprehensiveEvaluator
from booksmart.backtest.simple_bt import enhanced_direction_backtest

def main():
    parser = argparse.ArgumentParser(description='Advanced ML training with walk-forward analysis')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--lookback-days', type=int, default=500, help='Days of historical data')
    parser.add_argument('--interval', type=str, default='1d', help='Data interval')
    parser.add_argument('--train-window', type=int, default=120, help='Training window days')
    parser.add_argument('--retrain-freq', type=int, default=21, help='Retrain frequency days')
    parser.add_argument('--min-samples', type=int, default=50, help='Minimum training samples')
    parser.add_argument('--output-dir', type=str, default='models', help='Output directory')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ADVANCED ML TRAINING WITH WALK-FORWARD ANALYSIS")
    print("=" * 80)
    print(f"Symbol: {args.symbol}")
    print(f"Lookback days: {args.lookback_days}")
    print(f"Interval: {args.interval}")
    print(f"Training window: {args.train_window} days")
    print(f"Retrain frequency: {args.retrain_freq} days")
    print(f"Minimum samples: {args.min_samples}")
    
    # Fetch data
    print(f"\nFetching {args.symbol} data...")
    data = fetch_ohlcv(args.symbol, lookback_days=args.lookback_days, interval=args.interval)
    
    if data is None or len(data) < args.min_samples * 2:
        print("Insufficient data available")
        return
    
    print(f"Raw data shape: {data.shape}")
    
    # Create features
    print("Creating features...")
    try:
        features_df = make_stock_features(data)
        print(f"Features created: {features_df.shape}")
    except Exception as e:
        print(f"Error creating features: {e}")
        return
    
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
    
    if len(features_df) < args.min_samples * 3:
        print("Insufficient data for walk-forward analysis")
        return
    
    print(f"Label distribution: {np.bincount(features_df['direction'])}")
    
    # Run walk-forward analysis
    print("\n" + "=" * 60)
    print("RUNNING WALK-FORWARD ANALYSIS")
    print("=" * 60)
    
    wfa = WalkForwardAnalysis(
        train_window_days=args.train_window,
        retrain_frequency_days=args.retrain_freq,
        min_train_samples=args.min_samples
    )
    
    try:
        wf_results = wfa.run_analysis(features_df, target_col='direction')
        print("Walk-forward analysis completed successfully!")
        
        # Print summary results
        print_walk_forward_summary(wf_results)
        
        # Run comprehensive evaluation
        print("\n" + "=" * 60)
        print("RUNNING COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        # Aggregate all predictions from walk-forward windows
        all_predictions = np.concatenate([r['predictions'] for r in wf_results['window_results']])
        all_actual = np.concatenate([r['actual'] for r in wf_results['window_results']])
        all_probs = np.vstack([r['probabilities'] for r in wf_results['window_results']])
        all_dates = []
        for r in wf_results['window_results']:
            all_dates.extend(r['test_dates'])
        all_timestamps = pd.DatetimeIndex(all_dates)
        
        # Get corresponding returns
        returns_data = features_df.loc[all_timestamps, 'future_return'].values
        
        evaluator = ComprehensiveEvaluator()
        comprehensive_results = evaluator.evaluate_model_performance(
            y_true=all_actual,
            y_pred=all_predictions,
            y_proba=all_probs,
            returns=returns_data,
            timestamps=all_timestamps
        )
        
        print_comprehensive_evaluation(comprehensive_results)
        
        # Enhanced backtest
        print("\n" + "=" * 60)
        print("RUNNING ENHANCED BACKTEST")
        print("=" * 60)
        
        confidence_scores = np.max(all_probs, axis=1)
        backtest_results = enhanced_direction_backtest(
            predictions=all_predictions,
            actual=all_actual,
            returns=returns_data,
            confidence=confidence_scores,
            timestamps=all_timestamps
        )
        
        print_enhanced_backtest_results(backtest_results)
        
        # Save results
        save_advanced_results(
            args.output_dir, args.symbol, wf_results, 
            comprehensive_results, backtest_results
        )
        
        print("\n" + "=" * 60)
        print("FINAL RECOMMENDATION")
        print("=" * 60)
        
        overall_score = comprehensive_results['overall_score']
        recommendation = get_final_recommendation(
            wf_results['summary'], comprehensive_results, backtest_results
        )
        
        print(f"Overall Model Score: {overall_score:.1f}/100")
        print(f"Recommendation: {recommendation}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

def print_walk_forward_summary(results):
    """Print walk-forward analysis summary"""
    summary = results['summary']
    metrics = results['overall_metrics']
    
    print(f"Total Windows: {summary['total_windows']}")
    print(f"Average Accuracy: {summary['avg_accuracy']:.4f}")
    print(f"Accuracy Stability: {summary['accuracy_stability']:.4f}")
    print(f"Trading Sharpe: {summary['trading_sharpe']:.4f}")
    print(f"Total Return: {summary['total_trading_return']:.4f}")
    print(f"Model Degradation: {summary['model_degradation']:.6f}")
    print(f"WF Recommendation: {summary['recommendation']}")
    
    print(f"\nStability Metrics:")
    stability = metrics['stability']
    print(f"  Accuracy Trend: {stability['accuracy_trend']:.6f}")
    print(f"  Min Window Accuracy: {stability['min_window_accuracy']:.4f}")
    print(f"  Max Window Accuracy: {stability['max_window_accuracy']:.4f}")

def print_comprehensive_evaluation(results):
    """Print comprehensive evaluation results"""
    # Classification metrics
    classification = results['classification']
    print(f"Classification Metrics:")
    print(f"  Accuracy: {classification['accuracy']:.4f}")
    print(f"  Precision: {classification['precision']:.4f}")
    print(f"  Recall: {classification['recall']:.4f}")
    print(f"  F1 Score: {classification['f1_score']:.4f}")
    print(f"  AUC Score: {classification['auc_score']:.4f}")
    print(f"  Log Loss: {classification['log_loss']:.4f}")
    
    # Confidence metrics
    confidence = results['confidence']
    print(f"\nConfidence Metrics:")
    print(f"  Average Confidence: {confidence['avg_confidence']:.4f}")
    print(f"  Overconfidence Bias: {confidence['overconfidence_bias']:.4f}")
    print(f"  ECE: {confidence['calibration']['expected_calibration_error']:.4f}")
    
    # Trading metrics
    trading = results['trading']
    print(f"\nTrading Metrics:")
    print(f"  Annualized Return: {trading['annualized_return']:.4f}")
    print(f"  Sharpe Ratio: {trading['sharpe_ratio']:.4f}")
    print(f"  Information Ratio: {trading['information_ratio']:.4f}")
    print(f"  Max Drawdown: {trading['max_drawdown']:.4f}")
    print(f"  Win Rate: {trading['win_rate']:.4f}")
    print(f"  Profit Factor: {trading['profit_factor']:.4f}")
    
    # Risk metrics
    risk = results['risk']
    print(f"\nRisk Metrics:")
    print(f"  Tail Risk Accuracy: {risk['tail_risk_accuracy']:.4f}")
    print(f"  Robustness Score: {risk['robustness_score']:.4f}")
    print(f"  Average Entropy: {risk['avg_entropy']:.4f}")
    
    print(f"\nOverall Score: {results['overall_score']:.1f}/100")

def print_enhanced_backtest_results(results):
    """Print enhanced backtest results"""
    print(f"Enhanced Backtest Results:")
    print(f"  Total Return: {results['total_return']:.4f}")
    print(f"  Annualized Return: {results['annualized_return']:.4f}")
    print(f"  Volatility: {results['volatility']:.4f}")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"  Calmar Ratio: {results['calmar_ratio']:.4f}")
    print(f"  Max Drawdown: {results['max_drawdown']:.4f}")
    print(f"  Win Rate: {results['win_rate']:.4f}")
    print(f"  Number of Trades: {results['num_trades']}")
    print(f"  Activity Rate: {results['activity_rate']:.4f}")
    
    # Confidence analysis
    if 'confidence_analysis' in results:
        conf_analysis = results['confidence_analysis']
        print(f"\nConfidence Analysis:")
        print(f"  High Confidence Accuracy: {conf_analysis['high_confidence_accuracy']:.4f}")
        print(f"  Confidence-Return Correlation: {conf_analysis['confidence_return_correlation']:.4f}")

def save_advanced_results(output_dir, symbol, wf_results, comp_results, bt_results):
    """Save all results to files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save summary results
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"{output_dir}/advanced_results_{symbol}_{timestamp}.txt"
    
    with open(results_file, 'w') as f:
        f.write("ADVANCED ML TRAINING RESULTS\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Walk-Forward Analysis Summary:\n")
        summary = wf_results['summary']
        for key, value in summary.items():
            f.write(f"  {key}: {value}\n")
        
        f.write(f"\nOverall Model Score: {comp_results['overall_score']:.1f}/100\n")
        
        f.write(f"\nKey Metrics:\n")
        f.write(f"  Accuracy: {comp_results['classification']['accuracy']:.4f}\n")
        f.write(f"  Sharpe Ratio: {comp_results['trading']['sharpe_ratio']:.4f}\n")
        f.write(f"  Max Drawdown: {comp_results['trading']['max_drawdown']:.4f}\n")
        f.write(f"  Win Rate: {comp_results['trading']['win_rate']:.4f}\n")
        f.write(f"  Stability: {wf_results['overall_metrics']['stability']['accuracy_std']:.4f}\n")
    
    print(f"\nResults saved to: {results_file}")

def get_final_recommendation(wf_summary, comp_results, bt_results):
    """Generate final recommendation"""
    overall_score = comp_results['overall_score']
    sharpe = comp_results['trading']['sharpe_ratio']
    accuracy = comp_results['classification']['accuracy']
    stability = 1.0 / (1.0 + abs(wf_summary['accuracy_stability']))
    
    if overall_score >= 75 and sharpe > 1.5 and accuracy > 0.6 and stability > 0.8:
        return "STRONG BUY - Excellent model ready for production"
    elif overall_score >= 60 and sharpe > 1.0 and accuracy > 0.55 and stability > 0.7:
        return "BUY - Good model with solid performance"
    elif overall_score >= 45 and sharpe > 0.5 and accuracy > 0.5:
        return "HOLD - Model shows promise but needs improvement"
    else:
        return "AVOID - Model lacks consistent predictive power"

if __name__ == "__main__":
    main()
