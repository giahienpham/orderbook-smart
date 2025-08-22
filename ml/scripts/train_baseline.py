#!/usr/bin/env python3
"""Train and evaluate enhanced XGBoost model with microstructure features."""

import argparse
from pathlib import Path
import warnings

import pandas as pd
import numpy as np

from booksmart.features.dataset import build_xy, time_split
from booksmart.features.labeling import label_next_direction
from booksmart.features.stocks_features import make_stock_features
from booksmart.features.volatility import create_volatility_features
from booksmart.ingest.stocks_yf import fetch_ohlcv
from booksmart.models.xgb_model import XGBDirectionModel
from booksmart.backtest.simple_bt import simple_direction_backtest

warnings.filterwarnings('ignore')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--lookback-days", type=int, default=10, help="Lookback days")
    parser.add_argument("--interval", default="1m", help="Data interval")
    parser.add_argument("--model-path", default="models/enhanced_model.pkl", help="Model save path")
    parser.add_argument("--verbose", action="store_true", help="Verbose training")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} data...")
    df = fetch_ohlcv(args.symbol, interval=args.interval, lookback_days=args.lookback_days)
    print(f"Raw data shape: {df.shape}")

    print("Creating basic features...")
    df_feat = make_stock_features(df)
    print(f"After basic features: {df_feat.shape}")
    
    print("Adding volatility features...")
    df_feat = create_volatility_features(df_feat, price_col='close')
    print(f"After volatility features: {df_feat.shape}")

    print("Creating labels...")
    df_feat["label"] = label_next_direction(df_feat, horizon=5)
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"Final feature data shape: {df_feat.shape}")

    print("Splitting data...")
    train_df, val_df, test_df = time_split(df_feat)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Feature columns (excluding non-numeric and target)
    exclude_cols = ["ts", "open", "high", "low", "close", "volume", "label", 
                   "bid_est", "ask_est", "bid_vol_est", "ask_vol_est"]
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
    
    print(f"Using {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols[:10]):  # Show first 10
        print(f"  {i+1}. {col}")
    if len(feature_cols) > 10:
        print(f"  ... and {len(feature_cols) - 10} more")

    # Check for any remaining NaN or inf values
    train_feat_clean = train_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    val_feat_clean = val_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    test_feat_clean = test_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    X_train, y_train = build_xy(pd.concat([train_df[['label']], train_feat_clean], axis=1), 
                               feature_cols, "label")
    X_val, y_val = build_xy(pd.concat([val_df[['label']], val_feat_clean], axis=1), 
                           feature_cols, "label")
    X_test, y_test = build_xy(pd.concat([test_df[['label']], test_feat_clean], axis=1), 
                             feature_cols, "label")

    print("Label distribution:")
    print(f"  Train: {np.bincount(y_train + 1)}")  # Convert -1,0,1 to 0,1,2 for bincount
    print(f"  Val:   {np.bincount(y_val + 1)}")
    print(f"  Test:  {np.bincount(y_test + 1)}")

    print("Training enhanced model...")
    model = XGBDirectionModel(
        max_depth=8,
        eta=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.1,
        reg_lambda=0.1
    )
    
    model.train(
        X_train, y_train, 
        X_val, y_val, 
        feature_names=feature_cols,
        num_rounds=200, 
        early_stopping_rounds=20,
        verbose=args.verbose
    )

    print("\nEvaluating...")
    train_metrics = model.evaluate(X_train, y_train, detailed=True)
    val_metrics = model.evaluate(X_val, y_val, detailed=True)
    test_metrics = model.evaluate(X_test, y_test, detailed=True)

    print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    
    if 'avg_confidence' in test_metrics:
        print(f"Test avg confidence: {test_metrics['avg_confidence']:.4f}")
        print(f"High confidence accuracy: {test_metrics.get('high_confidence_accuracy', 'N/A')}")

    print("\nTop 10 Most Important Features:")
    top_features = model.get_feature_importance(10)
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\nBacktesting...")
    test_preds, test_confidence = model.predict_with_confidence(X_test)
    
    # Basic backtest
    bt_results = simple_direction_backtest(test_df, test_preds)
    print("\nBasic backtest results:")
    for k, v in bt_results.items():
        print(f"  {k}: {v:.4f}")
    
    # High-confidence backtest
    high_conf_mask = test_confidence > 0.6
    if np.sum(high_conf_mask) > 10:
        high_conf_preds = test_preds[high_conf_mask]
        high_conf_df = test_df.iloc[high_conf_mask].reset_index(drop=True)
        bt_results_hc = simple_direction_backtest(high_conf_df, high_conf_preds)
        print(f"\nHigh-confidence backtest ({np.sum(high_conf_mask)} trades):")
        for k, v in bt_results_hc.items():
            print(f"  {k}: {v:.4f}")

    # Feature analysis
    print(f"\nFeature correlation with returns:")
    returns = test_df['ret_1'].values
    feature_corrs = []
    for i, feat_name in enumerate(feature_cols[:10]):
        if i < X_test.shape[1]:
            corr = np.corrcoef(X_test[:, i], returns)[0, 1]
            if not np.isnan(corr):
                feature_corrs.append((feat_name, abs(corr)))
    
    feature_corrs.sort(key=lambda x: x[1], reverse=True)
    for feat_name, corr in feature_corrs[:5]:
        print(f"  {feat_name}: {corr:.4f}")

    # Save model
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_path)
    print(f"\nModel saved to {args.model_path}")
    
    # Save detailed results
    results_path = Path(args.model_path).parent / "training_results.txt"
    with open(results_path, 'w') as f:
        f.write(f"Enhanced XGBoost Training Results\n")
        f.write(f"==================================\n")
        f.write(f"Symbol: {args.symbol}\n")
        f.write(f"Data shape: {df_feat.shape}\n")
        f.write(f"Features: {len(feature_cols)}\n")
        f.write(f"Train accuracy: {train_metrics['accuracy']:.4f}\n")
        f.write(f"Val accuracy: {val_metrics['accuracy']:.4f}\n")
        f.write(f"Test accuracy: {test_metrics['accuracy']:.4f}\n")
        f.write(f"Sharpe ratio: {bt_results['sharpe_ratio']:.4f}\n")
        f.write(f"Max drawdown: {bt_results['max_drawdown']:.4f}\n")
        f.write(f"\nTop 10 Features:\n")
        for _, row in top_features.iterrows():
            f.write(f"  {row['feature']}: {row['importance']:.4f}\n")
    
    print(f"Results saved to {results_path}")

    # Optional plotting
    if args.plot:
        try:
            model.plot_training_curves()
            model.plot_feature_importance()
        except Exception as e:
            print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()