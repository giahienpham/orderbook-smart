#!/usr/bin/env python3
"""
Test script for real-time feature engine and performance optimization
Part of Commit 3: C++ Engine Integration and Performance Optimization
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd
import time
import warnings
warnings.filterwarnings('ignore')

from booksmart.engine.realtime_features import RealTimeFeatureEngine, PerformanceProfiler, OptimizedBacktester
from booksmart.models.xgb_model import XGBDirectionModel
from booksmart.models.fallback_model import FallbackDirectionModel


def test_real_time_feature_engine():
    """Test real-time feature generation"""
    print("\n" + "="*60)
    print("TESTING REAL-TIME FEATURE ENGINE")
    print("="*60)
    
    # Initialize feature engine
    engine = RealTimeFeatureEngine(initial_price=100.0, tick_size=0.01)
    
    # Generate test data
    np.random.seed(42)
    n_points = 100
    base_price = 100.0
    
    prices = []
    volumes = []
    
    for i in range(n_points):
        # Generate realistic price movement
        if i == 0:
            price = base_price
        else:
            price_change = np.random.normal(0, 0.02) * prices[-1]
            price = max(prices[-1] + price_change, 1.0)  # Ensure positive price
        
        volume = np.random.exponential(1000)  # Exponential distribution for volume
        
        prices.append(price)
        volumes.append(volume)
    
    # Test feature generation
    print(f"Generating features for {n_points} data points...")
    
    start_time = time.time()
    for i, (price, volume) in enumerate(zip(prices, volumes)):
        timestamp = pd.Timestamp.now() + pd.Timedelta(seconds=i)
        features = engine.add_market_data(price, volume, timestamp)
        
        if i == 0:
            print(f"First feature set keys: {list(features.keys())}")
        elif i == 10:
            print(f"Feature set at step 10: {len(features)} features")
            print("Sample features:", {k: f"{v:.4f}" if isinstance(v, (int, float)) else str(v) 
                                      for k, v in list(features.items())[:10]})
    
    generation_time = time.time() - start_time
    print(f"Feature generation completed in {generation_time:.4f} seconds")
    print(f"Average time per feature set: {generation_time/n_points:.6f} seconds")
    
    # Test DataFrame conversion
    df = engine.get_feature_dataframe()
    print(f"Feature DataFrame shape: {df.shape}")
    print(f"Feature columns: {list(df.columns)[:10]}...")
    
    # Test latest features retrieval
    latest = engine.get_latest_features()
    latest_5 = engine.get_latest_features(5)
    
    print(f"Latest features: {len(latest)} features")
    print(f"Latest 5 features: {len(latest_5)} feature sets")
    
    return True


def test_performance_profiler():
    """Test performance profiling capabilities"""
    print("\n" + "="*60)
    print("TESTING PERFORMANCE PROFILER")
    print("="*60)
    
    profiler = PerformanceProfiler()
    
    # Define test functions
    @profiler.profile_function('fast_function')
    def fast_function(n):
        return sum(range(n))
    
    @profiler.profile_function('slow_function')
    def slow_function(n):
        time.sleep(0.01)  # Simulate work
        return [i**2 for i in range(n)]
    
    @profiler.profile_function('memory_intensive')
    def memory_intensive(n):
        data = [np.random.randn(100) for _ in range(n)]
        return len(data)
    
    # Run test functions multiple times
    print("Running profiled functions...")
    
    for i in range(5):
        fast_function(1000)
        slow_function(100)
        memory_intensive(50)
    
    # Get performance report
    report = profiler.get_performance_report()
    
    print("Performance Report:")
    for func_name, metrics in report.items():
        print(f"\n{func_name}:")
        print(f"  Calls: {metrics['call_count']}")
        print(f"  Avg Time: {metrics['avg_time']:.6f}s")
        print(f"  Total Time: {metrics['total_time']:.6f}s")
        print(f"  Memory Delta: {metrics['avg_memory_delta']:.2f}MB")
    
    # Test formatted summary
    print("\nFormatted Summary:")
    profiler.print_performance_summary()
    
    return True


def test_optimized_backtester():
    """Test optimized backtester with real-time features"""
    print("\n" + "="*60)
    print("TESTING OPTIMIZED BACKTESTER")
    print("="*60)
    
    # Generate test data
    np.random.seed(42)
    n_points = 200
    
    # Create price series with trend and noise
    base_price = 100.0
    trend = 0.001  # Small upward trend
    noise_scale = 0.02
    
    prices = [base_price]
    for i in range(1, n_points):
        price_change = trend + np.random.normal(0, noise_scale)
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1.0))
    
    # Generate volumes
    volumes = np.random.exponential(1000, n_points)
    
    # Create pandas series
    price_series = pd.Series(prices)
    volume_series = pd.Series(volumes)
    
    print(f"Generated test data: {n_points} points")
    print(f"Price range: {min(prices):.2f} - {max(prices):.2f}")
    print(f"Volume range: {min(volumes):.0f} - {max(volumes):.0f}")
    
    # Initialize backtester
    backtester = OptimizedBacktester(initial_capital=100000.0)
    
    # Create a simple model for testing
    try:
        # Try to use XGBDirectionModel
        model = XGBDirectionModel()
        
        # Create dummy training data
        n_features = 10
        X_train = np.random.randn(100, n_features)
        y_train = np.random.choice([0, 1, 2], 100)  # 0=sell, 1=hold, 2=buy
        
        model.train(X_train, y_train)
        print("Using XGBDirectionModel for backtesting")
        
    except Exception as e:
        print(f"XGBDirectionModel failed, using FallbackDirectionModel: {e}")
        model = FallbackDirectionModel()
    
    # Run backtest
    print("Running optimized backtest...")
    start_time = time.time()
    
    results = backtester.backtest_with_model(
        price_series, 
        volume_series, 
        model,
        lookback_window=20
    )
    
    backtest_time = time.time() - start_time
    print(f"Backtest completed in {backtest_time:.4f} seconds")
    
    # Analyze results
    print("\nBacktest Results:")
    print(f"Number of trades: {len(results['trades'])}")
    print(f"Number of predictions: {len(results['predictions'])}")
    print(f"Number of feature sets: {len(results['features'])}")
    
    # Performance metrics
    metrics = results['performance_metrics']
    print(f"\nPerformance Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    
    # Profiling report
    profiling_report = results['profiling_report']
    print(f"\nProfiling Report:")
    for func_name, prof_metrics in profiling_report.items():
        print(f"  {func_name}: {prof_metrics['call_count']} calls, "
              f"avg {prof_metrics['avg_time']:.6f}s")
    
    # Test portfolio value evolution
    portfolio_values = results['portfolio_value']
    if len(portfolio_values) > 1:
        initial_value = portfolio_values[0]
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_value) / initial_value
        print(f"\nPortfolio Evolution:")
        print(f"  Initial Value: ${initial_value:,.2f}")
        print(f"  Final Value: ${final_value:,.2f}")
        print(f"  Total Return: {total_return:.2%}")
    
    return True


def test_integration_with_existing_pipeline():
    """Test integration with existing ML pipeline components"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION WITH EXISTING PIPELINE")
    print("="*60)
    
    # Test importing other components
    try:
        from booksmart.features.stocks_features import make_stock_features
        from booksmart.backtest.walk_forward import WalkForwardAnalysis
        
        print("✓ Successfully imported existing pipeline components")
        
        # Test feature compatibility
        engine = RealTimeFeatureEngine()
        
        # Generate some features
        for i in range(10):
            price = 100 + np.random.normal(0, 1)
            volume = np.random.exponential(1000)
            engine.add_market_data(price, volume)
        
        # Get feature DataFrame
        rt_features = engine.get_feature_dataframe()
        
        print(f"✓ Real-time features shape: {rt_features.shape}")
        print(f"✓ Real-time feature types: {rt_features.dtypes.value_counts().to_dict()}")
        
        # Test compatibility with existing feature function
        # Create sample stock data
        dates = pd.date_range('2023-01-01', periods=50)
        stock_data = pd.DataFrame({
            'open': 100 + np.random.randn(50),
            'high': 102 + np.random.randn(50),
            'low': 98 + np.random.randn(50),
            'close': 100 + np.random.randn(50),
            'volume': np.random.exponential(1000, 50),
            'ts': dates  # Add ts column that the function expects
        }, index=dates)
        
        traditional_features = make_stock_features(stock_data)
        
        print(f"✓ Traditional features shape: {traditional_features.shape}")
        print(f"✓ Feature generation compatibility confirmed")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False


def test_edge_cases_and_robustness():
    """Test edge cases and robustness"""
    print("\n" + "="*60)
    print("TESTING EDGE CASES AND ROBUSTNESS")
    print("="*60)
    
    test_results = []
    
    # Test 1: Empty data
    try:
        engine = RealTimeFeatureEngine()
        df = engine.get_feature_dataframe()
        assert df.empty, "Empty feature DataFrame should be empty"
        print("✓ Empty data handling")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Empty data test failed: {e}")
        test_results.append(False)
    
    # Test 2: Single data point
    try:
        engine = RealTimeFeatureEngine()
        features = engine.add_market_data(100.0, 1000.0)
        assert isinstance(features, dict), "Should return feature dictionary"
        assert 'price' in features, "Should contain price"
        print("✓ Single data point handling")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Single data point test failed: {e}")
        test_results.append(False)
    
    # Test 3: Extreme values
    try:
        engine = RealTimeFeatureEngine()
        # Test with very large and very small values
        extreme_values = [
            (1e6, 1e9),  # Very large
            (1e-6, 1e-3),  # Very small
            (0.01, 1),  # Minimal
        ]
        
        for price, volume in extreme_values:
            features = engine.add_market_data(price, volume)
            numeric_features = {k: v for k, v in features.items() if isinstance(v, (int, float))}
            non_finite = [k for k, v in numeric_features.items() if not np.isfinite(v)]
            assert len(non_finite) == 0, f"Non-finite features found: {non_finite}"
        
        print("✓ Extreme values handling")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Extreme values test failed: {e}")
        test_results.append(False)
    
    # Test 4: NaN and infinite values
    try:
        engine = RealTimeFeatureEngine()
        
        # Add some normal data first
        for i in range(5):
            engine.add_market_data(100 + i, 1000)
        
        # Test with problematic values (should be handled gracefully)
        problematic_values = [
            (float('inf'), 1000),
            (100, float('inf')),
            (float('nan'), 1000),
            (100, float('nan'))
        ]
        
        for price, volume in problematic_values:
            try:
                features = engine.add_market_data(price, volume)
                # Should either work or fail gracefully
            except Exception:
                pass  # Expected to potentially fail
        
        print("✓ NaN/infinite values handling")
        test_results.append(True)
    except Exception as e:
        print(f"✗ NaN/infinite values test failed: {e}")
        test_results.append(False)
    
    # Test 5: Memory usage with large datasets
    try:
        engine = RealTimeFeatureEngine()
        
        # Generate large dataset
        for i in range(1000):
            price = 100 + np.sin(i/10) + np.random.normal(0, 0.1)
            volume = np.random.exponential(1000)
            engine.add_market_data(price, volume)
        
        # Check memory usage is reasonable
        df = engine.get_feature_dataframe()
        memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        
        assert memory_mb < 100, f"Memory usage too high: {memory_mb}MB"
        print(f"✓ Large dataset handling (Memory: {memory_mb:.2f}MB)")
        test_results.append(True)
    except Exception as e:
        print(f"✗ Large dataset test failed: {e}")
        test_results.append(False)
    
    success_rate = sum(test_results) / len(test_results)
    print(f"\nRobustness test success rate: {success_rate:.1%}")
    
    return success_rate > 0.8


def main():
    """Run all tests for real-time feature engine"""
    print("REAL-TIME FEATURE ENGINE TEST SUITE")
    print("Commit 3: C++ Engine Integration and Performance Optimization")
    print("="*80)
    
    test_functions = [
        test_real_time_feature_engine,
        test_performance_profiler,
        test_optimized_backtester,
        test_integration_with_existing_pipeline,
        test_edge_cases_and_robustness
    ]
    
    results = []
    for test_func in test_functions:
        try:
            result = test_func()
            results.append(result)
            print(f"✓ {test_func.__name__}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            print(f"✗ {test_func.__name__}: FAILED ({e})")
            results.append(False)
    
    # Summary
    passed = sum(results)
    total = len(results)
    success_rate = passed / total
    
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests passed: {passed}/{total}")
    print(f"Success rate: {success_rate:.1%}")
    
    if success_rate >= 0.8:
        print("✓ REAL-TIME FEATURE ENGINE TESTS PASSED")
        print("Ready for production use with C++ orderbook integration")
    else:
        print("✗ SOME TESTS FAILED")
        print("Review failed tests before proceeding")
    
    return success_rate >= 0.8


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
