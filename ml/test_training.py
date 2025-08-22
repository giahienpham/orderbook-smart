#!/usr/bin/env python3
"""Test the training script with fallback model to avoid XGBoost issues."""

import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.insert(0, '/Users/hienpham/Desktop/coding/orderbook-smart/ml')

from booksmart.features.dataset import build_xy, time_split
from booksmart.features.labeling import label_next_direction
from booksmart.features.stocks_features import make_stock_features
from booksmart.features.volatility import create_volatility_features
from booksmart.ingest.stocks_yf import fetch_ohlcv
from booksmart.models.fallback_model import FallbackDirectionModel
from booksmart.backtest.simple_bt import simple_direction_backtest

def test_training():
    print("=== Testing Enhanced Training with Fallback Model ===\n")
    
    print("Fetching AAPL data...")
    df = fetch_ohlcv("AAPL", interval="1m", lookback_days=3)
    print(f"Raw data shape: {df.shape}")

    print("Creating basic features...")
    df_feat = make_stock_features(df)
    print(f"After basic features: {df_feat.shape}")
    
    print("Adding volatility features...")
    df_feat = create_volatility_features(df_feat, price_col='close')
    print(f"After volatility features: {df_feat.shape}")

    print("Creating labels...")
    df_feat["label"] = label_next_direction(df_feat, horizon=3)  # Shorter horizon for small dataset
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"Final feature data shape: {df_feat.shape}")

    print("Splitting data...")
    train_df, val_df, test_df = time_split(df_feat, train_ratio=0.6, val_ratio=0.2)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Feature selection - use fewer features to avoid overfitting
    exclude_cols = ["ts", "open", "high", "low", "close", "volume", "label", 
                   "bid_est", "ask_est", "bid_vol_est", "ask_vol_est"]
    all_feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
    
    # Select most important feature types to reduce overfitting
    important_features = [col for col in all_feature_cols if any(keyword in col.lower() for keyword in 
                         ['ret_', 'rsi_', 'vol_roll', 'ema_', 'momentum', 'tod_min', 'range_pct'])]
    
    feature_cols = important_features[:20]  # Use only top 20 features
    print(f"Using {len(feature_cols)} selected features")

    # Clean data
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
    print(f"  Train: {np.bincount(y_train + 1)}")
    print(f"  Val:   {np.bincount(y_val + 1)}")
    print(f"  Test:  {np.bincount(y_test + 1)}")

    print("\nTraining RandomForest model with regularization...")
    model = FallbackDirectionModel(
        n_estimators=50,  # Fewer trees
        max_depth=4,      # Shallower trees
        min_samples_split=10,  # More conservative splitting
        min_samples_leaf=5,    # Larger leaf nodes
        max_features='sqrt',   # Feature subsampling
        random_state=42
    )
    
    model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols, verbose=True)

    print("\nEvaluating...")
    train_metrics = model.evaluate(X_train, y_train, detailed=True)
    val_metrics = model.evaluate(X_val, y_val, detailed=True)
    test_metrics = model.evaluate(X_test, y_test, detailed=True)

    print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    
    # Check for overfitting
    overfitting = train_metrics['accuracy'] - val_metrics['accuracy']
    print(f"Overfitting gap: {overfitting:.4f}")
    
    if overfitting > 0.2:
        print("⚠️  Warning: Model is overfitting. Consider:")
        print("   - Reducing number of features")
        print("   - Increasing regularization")
        print("   - Getting more data")
    else:
        print("✅ Model generalization looks reasonable")

    if 'avg_confidence' in test_metrics:
        print(f"Test avg confidence: {test_metrics['avg_confidence']:.4f}")

    print(f"\nTop {min(10, len(feature_cols))} Most Important Features:")
    top_features = model.get_feature_importance(min(10, len(feature_cols)))
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    print("\nBacktesting...")
    test_preds, test_confidence = model.predict_with_confidence(X_test)
    
    bt_results = simple_direction_backtest(test_df, test_preds)
    print("\nBacktest results:")
    for k, v in bt_results.items():
        if abs(v) < 1e10:  # Filter extreme values
            print(f"  {k}: {v:.4f}")

    # Feature analysis
    print(f"\nFeature correlation with returns:")
    if len(test_df) > 10:
        returns = test_df['ret_1'].values
        feature_corrs = []
        for i, feat_name in enumerate(feature_cols[:5]):
            if i < X_test.shape[1]:
                corr = np.corrcoef(X_test[:, i], returns)[0, 1]
                if not np.isnan(corr):
                    feature_corrs.append((feat_name, abs(corr)))
        
        feature_corrs.sort(key=lambda x: x[1], reverse=True)
        for feat_name, corr in feature_corrs:
            print(f"  {feat_name}: {corr:.4f}")

    print(f"\n🎉 Enhanced training pipeline test completed!")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Data: {df.shape[0]} bars -> {len(test_df)} test samples")
    print(f"Features: {len(feature_cols)} selected from {len(all_feature_cols)} generated")
    print(f"Model: RandomForest with regularization")
    print(f"Performance: {test_metrics['accuracy']:.1%} test accuracy")
    print(f"Overfitting: {overfitting:.3f} (train - val accuracy)")
    
    if test_metrics['accuracy'] > 0.4 and overfitting < 0.2:
        print("✅ Training pipeline is working well!")
        return True
    else:
        print("⚠️  Training pipeline needs improvement")
        return False

if __name__ == "__main__":
    test_training()
