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
from booksmart.backtest.simple_bt import simple_direction_backtest

def test_full_pipeline():
    print("=== Testing Full Enhanced ML Pipeline ===\n")
    print("Creating synthetic data...")
    dates = pd.date_range('2025-01-01', periods=500, freq='1min')
    np.random.seed(42)
    
    price_changes = np.random.randn(500) * 0.002
    price_changes[100:200] += 0.001  # Uptrend
    price_changes[300:400] -= 0.001  # Downtrend
    
    prices = 100 * np.cumprod(1 + price_changes)
    
    df = pd.DataFrame({
        'ts': dates,
        'open': prices * (1 + np.random.randn(500) * 0.0001),
        'high': prices * (1 + np.abs(np.random.randn(500)) * 0.0005),
        'low': prices * (1 - np.abs(np.random.randn(500)) * 0.0005),
        'close': prices,
        'volume': np.random.randint(1000, 10000, 500)
    })
    
    df['high'] = np.maximum.reduce([df['open'], df['high'], df['low'], df['close']])
    df['low'] = np.minimum.reduce([df['open'], df['high'], df['low'], df['close']])
    
    print(f"Synthetic data created: {df.shape}")
    
    print("Creating enhanced features...")
    df_feat = make_stock_features(df)
    print(f"After basic features: {df_feat.shape}")
    
    df_feat = create_volatility_features(df_feat, price_col='close')
    print(f"After volatility features: {df_feat.shape}")
    
    print("Creating labels...")
    df_feat["label"] = label_next_direction(df_feat, horizon=5)
    df_feat = df_feat.dropna().reset_index(drop=True)
    print(f"Final dataset: {df_feat.shape}")
    
    print("Splitting data...")
    train_df, val_df, test_df = time_split(df_feat)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    exclude_cols = ["ts", "open", "high", "low", "close", "volume", "label", 
                   "bid_est", "ask_est", "bid_vol_est", "ask_vol_est"]
    feature_cols = [col for col in df_feat.columns if col not in exclude_cols]
    print(f"Using {len(feature_cols)} features")
    
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
    
    try:
        import sys
        sys.path.append('/Users/hienpham/Desktop/coding/orderbook-smart/ml/booksmart/models')
        from xgb_model import XGBDirectionModel
        print("\nTraining XGBoost model...")
        model = XGBDirectionModel(max_depth=6, eta=0.1)
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols, 
                   num_rounds=100, verbose=False)
        model_type = "XGBoost"
    except Exception as e:
        print(f"XGBoost failed: {e}")
        from fallback_model import FallbackDirectionModel
        print("\nXGBoost not available, using RandomForest fallback...")
        model = FallbackDirectionModel(n_estimators=100, max_depth=6)
        model.train(X_train, y_train, X_val, y_val, feature_names=feature_cols, verbose=True)
        model_type = "RandomForest"
    
    print(f"\nEvaluating {model_type} model...")
    train_metrics = model.evaluate(X_train, y_train, detailed=True)
    val_metrics = model.evaluate(X_val, y_val, detailed=True)
    test_metrics = model.evaluate(X_test, y_test, detailed=True)
    
    print(f"Train accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Val accuracy:   {val_metrics['accuracy']:.4f}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    
    if 'avg_confidence' in test_metrics:
        print(f"Test avg confidence: {test_metrics['avg_confidence']:.4f}")
    
    print(f"\nTop 10 Most Important Features ({model_type}):")
    top_features = model.get_feature_importance(10)
    for _, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    print("\nBacktesting...")
    if hasattr(model, 'predict_with_confidence'):
        test_preds, test_confidence = model.predict_with_confidence(X_test)
    else:
        test_preds = model.predict(X_test)
        test_confidence = None
    
    bt_results = simple_direction_backtest(test_df, test_preds)
    print("\nBacktest results:")
    for k, v in bt_results.items():
        if abs(v) < 1e10:  
            print(f"  {k}: {v:.4f}")
    
    if test_confidence is not None:
        high_conf_mask = test_confidence > 0.6
        if np.sum(high_conf_mask) > 10:
            high_conf_preds = test_preds[high_conf_mask]
            high_conf_df = test_df.iloc[high_conf_mask].reset_index(drop=True)
            bt_results_hc = simple_direction_backtest(high_conf_df, high_conf_preds)
            print(f"\nHigh-confidence backtest ({np.sum(high_conf_mask)} trades):")
            for k, v in bt_results_hc.items():
                if abs(v) < 1e10:
                    print(f"  {k}: {v:.4f}")
    
    print(f"\n🎉 Full pipeline test completed successfully with {model_type}!")
    return True

if __name__ == "__main__":
    test_full_pipeline()
