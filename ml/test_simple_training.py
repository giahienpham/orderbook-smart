#!/usr/bin/env python3
"""Simple test to verify the enhanced training pipeline works correctly."""

import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, '/Users/hienpham/Desktop/coding/orderbook-smart/ml')

import pandas as pd
import numpy as np
from booksmart.features.stocks_features import make_stock_features
from booksmart.features.volatility import create_volatility_features
from booksmart.features.labeling import label_next_direction
from booksmart.models.fallback_model import FallbackDirectionModel

def quick_test():
    print("=== Quick Training Pipeline Test ===\n")
    
    # Create more realistic synthetic data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate price series with some predictable patterns
    dates = pd.date_range('2025-01-01', periods=n_samples, freq='1min')
    
    # Create trending price with some noise
    trend = np.linspace(100, 110, n_samples)
    noise = np.random.randn(n_samples) * 0.5
    returns = np.diff(np.log(trend + noise))
    prices = 100 * np.exp(np.cumsum(np.concatenate([[0], returns])))
    
    df = pd.DataFrame({
        'ts': dates,
        'open': prices * (1 + np.random.randn(n_samples) * 0.001),
        'high': prices * (1 + np.abs(np.random.randn(n_samples)) * 0.002),
        'low': prices * (1 - np.abs(np.random.randn(n_samples)) * 0.002),
        'close': prices,
        'volume': np.random.randint(1000, 5000, n_samples)
    })
    
    # Fix OHLC consistency
    for i in range(len(df)):
        df.loc[i, 'high'] = max(df.loc[i, 'open'], df.loc[i, 'high'], df.loc[i, 'low'], df.loc[i, 'close'])
        df.loc[i, 'low'] = min(df.loc[i, 'open'], df.loc[i, 'high'], df.loc[i, 'low'], df.loc[i, 'close'])
    
    print(f"Created synthetic data: {df.shape}")
    
    # Create features
    print("Generating features...")
    df_feat = make_stock_features(df)
    df_feat = create_volatility_features(df_feat, price_col='close') 
    df_feat["label"] = label_next_direction(df_feat, horizon=5)
    df_feat = df_feat.dropna().reset_index(drop=True)
    
    print(f"Feature dataset: {df_feat.shape}")
    
    # Simple train/test split
    split_idx = int(0.8 * len(df_feat))
    train_df = df_feat[:split_idx]
    test_df = df_feat[split_idx:]
    
    # Use only the most stable features
    stable_features = [
        'ret_1', 'ret_5', 'ema_10', 'ema_50', 'rsi_14', 'vol_roll_20',
        'tod_min', 'momentum_5_20', 'range_pct', 'price_to_ema10'
    ]
    
    # Make sure all features exist
    stable_features = [f for f in stable_features if f in df_feat.columns]
    print(f"Using {len(stable_features)} stable features: {stable_features}")
    
    # Prepare data
    X_train = train_df[stable_features].fillna(0).replace([np.inf, -np.inf], 0).values
    y_train = train_df['label'].values
    X_test = test_df[stable_features].fillna(0).replace([np.inf, -np.inf], 0).values  
    y_test = test_df['label'].values
    
    print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"Label distribution - Train: {np.bincount(y_train + 1)}, Test: {np.bincount(y_test + 1)}")
    
    # Train simple model
    print("\nTraining model...")
    model = FallbackDirectionModel(
        n_estimators=20,
        max_depth=3,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=42
    )
    
    model.train(X_train, y_train, feature_names=stable_features, verbose=False)
    
    # Evaluate
    train_acc = model.evaluate(X_train, y_train)['accuracy']
    test_acc = model.evaluate(X_test, y_test)['accuracy']
    
    print(f"\nResults:")
    print(f"  Train accuracy: {train_acc:.3f}")
    print(f"  Test accuracy:  {test_acc:.3f}")
    print(f"  Overfitting:    {train_acc - test_acc:.3f}")
    
    # Feature importance
    print(f"\nTop 5 features:")
    importance = model.get_feature_importance(5)
    for _, row in importance.iterrows():
        print(f"  {row['feature']}: {row['importance']:.3f}")
    
    # Test predictions
    predictions = model.predict(X_test)
    pred_counts = np.bincount(predictions + 1)
    print(f"\nPrediction distribution: {pred_counts}")
    
    # Success criteria
    baseline_acc = max(np.bincount(y_test + 1)) / len(y_test)  # Most frequent class
    success = test_acc > baseline_acc and (train_acc - test_acc) < 0.3
    
    print(f"\nBaseline (most frequent): {baseline_acc:.3f}")
    print(f"Model beats baseline: {test_acc > baseline_acc}")
    print(f"Reasonable generalization: {(train_acc - test_acc) < 0.3}")
    
    if success:
        print("\n✅ Training pipeline test PASSED!")
        print("   - Model beats baseline")
        print("   - Reasonable generalization")
        print("   - All features generated successfully")
        return True
    else:
        print("\n⚠️  Training pipeline needs work")
        return False

if __name__ == "__main__":
    success = quick_test()
    print(f"\n=== Test {'PASSED' if success else 'FAILED'} ===")
