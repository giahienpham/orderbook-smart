#!/usr/bin/env python3
"""Train and evaluate baseline XGBoost model for stock direction prediction."""

import argparse
from pathlib import Path

import pandas as pd

from booksmart.features.dataset import build_xy, time_split
from booksmart.features.labeling import label_next_direction
from booksmart.features.stocks_features import make_stock_features
from booksmart.ingest.stocks_yf import fetch_ohlcv
from booksmart.models.xgb_model import XGBDirectionModel
from booksmart.backtest.simple_bt import simple_direction_backtest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="AAPL", help="Stock symbol")
    parser.add_argument("--lookback-days", type=int, default=5, help="Lookback days")
    parser.add_argument("--interval", default="1m", help="Data interval")
    parser.add_argument("--model-path", default="models/baseline.pkl", help="Model save path")
    args = parser.parse_args()

    print(f"Fetching {args.symbol} data...")
    df = fetch_ohlcv(args.symbol, interval=args.interval, lookback_days=args.lookback_days)
    print(f"Raw data shape: {df.shape}")

    print("Creating features...")
    df_feat = make_stock_features(df)
    df_feat["label"] = label_next_direction(df_feat, horizon=5)
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"Feature data shape: {df_feat.shape}")

    print("Splitting data...")
    train_df, val_df, test_df = time_split(df_feat)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Feature columns (excluding non-numeric and target)
    feature_cols = [col for col in df_feat.columns 
                   if col not in ["ts", "open", "high", "low", "close", "volume", "label"]]
    print(f"Feature columns: {feature_cols}")

    X_train, y_train = build_xy(train_df, feature_cols, "label")
    X_val, y_val = build_xy(val_df, feature_cols, "label")
    X_test, y_test = build_xy(test_df, feature_cols, "label")

    print("Training model...")
    model = XGBDirectionModel()
    model.train(X_train, y_train, X_val, y_val, num_rounds=50)

    print("Evaluating...")
    train_metrics = model.evaluate(X_train, y_train)
    val_metrics = model.evaluate(X_val, y_val)
    test_metrics = model.evaluate(X_test, y_test)

    print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val accuracy: {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy: {test_metrics['accuracy']:.4f}")

    print("Backtesting...")
    test_preds = model.predict(X_test)
    bt_results = simple_direction_backtest(test_df, test_preds)
    
    print(f"Backtest results:")
    for k, v in bt_results.items():
        print(f"  {k}: {v:.4f}")

    # Save model
    Path(args.model_path).parent.mkdir(exist_ok=True)
    model.save(args.model_path)
    print(f"Model saved to {args.model_path}")


if __name__ == "__main__":
    main()