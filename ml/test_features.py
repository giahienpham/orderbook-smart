import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

def test_basic_features():
    print("Testing basic feature generation...")
    
    # Mock some simple data
    dates = pd.date_range('2025-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'ts': dates,
        'open': 100 + np.cumsum(np.random.randn(100) * 0.01),
        'high': 100 + np.cumsum(np.random.randn(100) * 0.01) + 0.1,
        'low': 100 + np.cumsum(np.random.randn(100) * 0.01) - 0.1,
        'close': 100 + np.cumsum(np.random.randn(100) * 0.01),
        'volume': np.random.randint(1000, 10000, 100)
    })
    
    # Ensure high >= close >= low
    df['high'] = np.maximum(df['high'], df['close'])
    df['low'] = np.minimum(df['low'], df['close'])
    
    try:
        from booksmart.features.stocks_features import make_stock_features
        df_with_features = make_stock_features(df)
        print(f"✓ Basic features created: {df_with_features.shape}")
        print(f"  Features: {[col for col in df_with_features.columns if col not in ['ts', 'open', 'high', 'low', 'close', 'volume']][:5]}...")
        return True
    except Exception as e:
        print(f"✗ Basic features failed: {e}")
        return False

def test_volatility_features():
    print("Testing volatility features...")
    
    dates = pd.date_range('2025-01-01', periods=100, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(100) * 0.01)
    })
    
    try:
        from booksmart.features.volatility import create_volatility_features
        df_with_vol = create_volatility_features(df, price_col='close')
        print(f"✓ Volatility features created: {df_with_vol.shape}")
        vol_cols = [col for col in df_with_vol.columns if 'vol' in col.lower()]
        print(f"  Vol features: {vol_cols[:3]}...")
        return True
    except Exception as e:
        print(f"✗ Volatility features failed: {e}")
        return False

def test_microstructure_features():
    print("Testing microstructure features...")
    
    try:
        from booksmart.features.microstructure import microprice, effective_spread
        
        price = microprice(99.5, 100.5, 1000, 800)
        spread = effective_spread(100.0, 99.9)
        
        print(f"✓ Microstructure functions work: microprice={price:.3f}, spread={spread:.1f}bps")
        return True
    except Exception as e:
        print(f"✗ Microstructure features failed: {e}")
        return False

def test_labeling():
    print("Testing labeling...")
    
    dates = pd.date_range('2025-01-01', periods=50, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(50) * 0.01)
    })
    
    try:
        from booksmart.features.labeling import label_next_direction
        labels = label_next_direction(df, horizon=5)
        unique_labels = np.unique(labels.dropna())
        print(f"✓ Labeling works: unique labels {unique_labels}")
        return True
    except Exception as e:
        print(f"✗ Labeling failed: {e}")
        return False

def test_data_ingestion():
    print("Testing data ingestion (requires internet)...")
    
    try:
        from booksmart.ingest.stocks_yf import fetch_ohlcv
        df = fetch_ohlcv("AAPL", interval="1d", lookback_days=5)
        print(f"✓ Data ingestion works: {df.shape}")
        return True
    except Exception as e:
        print(f"✗ Data ingestion failed: {e}")
        return False

def test_backtest():
    print("Testing backtest...")
    
    dates = pd.date_range('2025-01-01', periods=50, freq='1min')
    np.random.seed(42)
    
    df = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(50) * 0.01),
        'ret_1': np.random.randn(50) * 0.01
    })
    
    predictions = np.random.choice([-1, 0, 1], size=50)
    
    try:
        from booksmart.backtest.simple_bt import simple_direction_backtest
        results = simple_direction_backtest(df, predictions)
        print(f"✓ Backtest works: Sharpe={results['sharpe_ratio']:.3f}")
        return True
    except Exception as e:
        print(f"✗ Backtest failed: {e}")
        return False

def main():
    print("=== Testing Enhanced ML Features ===\n")
    
    # Add current directory to Python path
    sys.path.insert(0, '/Users/hienpham/Desktop/coding/orderbook-smart/ml')
    
    tests = [
        test_basic_features,
        test_volatility_features,
        test_microstructure_features,
        test_labeling,
        test_data_ingestion,
        test_backtest
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print(" All tests passed! Your enhanced ML features are ready.")
        print("\nTo test XGBoost specifically:")
        print("1. Install libomp: brew install libomp")
        print("2. Run: python3 scripts/train_baseline.py --symbol AAPL --lookback-days 2")
    else:
        print(" Some tests failed. Check the errors above.")

if __name__ == "__main__":
    main()
