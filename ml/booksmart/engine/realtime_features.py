import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import warnings
from ..engine.orderbook_engine import HighFrequencySimulator, OrderRequest, OrderType, Side


class RealTimeFeatureEngine:
    """Real-time feature engineering using orderbook simulation"""
    
    def __init__(self, initial_price: float = 100.0, tick_size: float = 0.01):
        self.simulator = HighFrequencySimulator(initial_price, tick_size)
        self.feature_history = []
        self.price_history = []
        
    def add_market_data(self, price: float, volume: float, timestamp: Optional[pd.Timestamp] = None):
        """Add market data point and generate features"""
        if timestamp is None:
            timestamp = pd.Timestamp.now()
        
        # Create synthetic orders based on price movement
        orders = self._generate_synthetic_orders(price, volume)
        
        # Execute simulation step
        step_result = self.simulator.step(orders)
        
        # Extract features
        features = self._extract_features(step_result, price, volume, timestamp)
        
        self.feature_history.append(features)
        self.price_history.append(price)
        
        return features
    
    def _generate_synthetic_orders(self, price: float, volume: float) -> List[OrderRequest]:
        """Generate synthetic orders based on price and volume"""
        orders = []
        
        # Determine order side based on price movement
        if len(self.price_history) > 0:
            price_change = price - self.price_history[-1]
            
            if price_change > 0:
                # Price going up, more buy orders
                side = Side.BID
                order_size = volume * 0.6
            elif price_change < 0:
                # Price going down, more sell orders
                side = Side.ASK
                order_size = volume * 0.6
            else:
                # No change, balanced
                orders.append(OrderRequest(OrderType.MARKET, Side.BID, 0, volume * 0.3))
                orders.append(OrderRequest(OrderType.MARKET, Side.ASK, 0, volume * 0.3))
                return orders
            
            orders.append(OrderRequest(OrderType.MARKET, side, 0, order_size))
        
        return orders
    
    def _extract_features(self, step_result: Dict, price: float, volume: float, timestamp: pd.Timestamp) -> Dict:
        """Extract comprehensive features from simulation step"""
        features = {
            'timestamp': timestamp,
            'price': price,
            'volume': volume,
        }
        
        # Ensure input values are finite
        if not np.isfinite(price):
            price = 100.0  # Default price
        if not np.isfinite(volume):
            volume = 1000.0  # Default volume
        
        # Orderbook features
        book_state = step_result['book_state']
        best_bid = book_state.get('best_bid', 0) or 0
        best_ask = book_state.get('best_ask', 0) or 0
        
        # Handle None values from book state
        if best_bid is None or not np.isfinite(best_bid):
            best_bid = price * 0.999  # Default to slightly below current price
        if best_ask is None or not np.isfinite(best_ask):
            best_ask = price * 1.001  # Default to slightly above current price
            
        spread = best_ask - best_bid if best_ask > best_bid else 0
        mid_price = (best_ask + best_bid) / 2 if best_ask > 0 and best_bid > 0 else price
        
        # Ensure all values are finite
        spread = spread if np.isfinite(spread) else 0.0
        mid_price = mid_price if np.isfinite(mid_price) else price
        
        features.update({
            'best_bid': best_bid,
            'best_ask': best_ask,
            'spread': spread,
            'mid_price': mid_price
        })
        
        # Microstructure features
        micro_features = step_result['features']
        features.update({
            'vwap': micro_features.get('vwap', price) or price,
            'order_flow_imbalance': micro_features.get('order_flow_imbalance', 0) or 0,
            'avg_price_impact': micro_features.get('avg_price_impact', 0) or 0,
            'price_impact_volatility': micro_features.get('price_impact_volatility', 0) or 0,
            'total_volume': micro_features.get('total_volume', volume) or volume,
            'trade_count': micro_features.get('trade_count', 1) or 1,
            'avg_fill_rate': micro_features.get('avg_fill_rate', 1.0) or 1.0
        })
        
        # Ensure all microstructure features are finite
        for key in ['vwap', 'order_flow_imbalance', 'avg_price_impact', 'price_impact_volatility', 'total_volume', 'avg_fill_rate']:
            if not np.isfinite(features[key]):
                features[key] = 0.0
        
        # Technical features if we have history
        if len(self.feature_history) > 0:
            tech_features = self._calculate_technical_features()
            features.update(tech_features)
        
        # Final check - ensure all numeric features are finite
        for key, value in features.items():
            if isinstance(value, (int, float)) and not np.isfinite(value):
                features[key] = 0.0
        
        return features
    
    def _calculate_technical_features(self) -> Dict:
        """Calculate technical features from history"""
        if len(self.feature_history) < 2:
            return {}
        
        recent_features = self.feature_history[-10:]  # Last 10 observations
        
        # Price-based features
        prices = [f.get('price', 0) for f in recent_features if f.get('price') is not None]
        returns = np.diff(prices) / np.array(prices[:-1]) if len(prices) > 1 else [0]
        
        # Volume-based features
        volumes = [f.get('volume', 0) for f in recent_features if f.get('volume') is not None]
        
        # Spread features
        spreads = [f.get('spread', 0) for f in recent_features if f.get('spread') is not None]
        
        features = {}
        
        # Returns features
        if len(returns) > 0 and all(np.isfinite(returns)):
            features.update({
                'return_mean': np.mean(returns),
                'return_std': np.std(returns),
                'return_skew': self._safe_skew(returns),
                'return_kurtosis': self._safe_kurtosis(returns)
            })
        
        # Volume features
        if len(volumes) > 1 and all(np.isfinite(volumes)):
            features.update({
                'volume_mean': np.mean(volumes),
                'volume_std': np.std(volumes),
                'volume_trend': self._calculate_trend(volumes)
            })
        
        # Spread features
        if len(spreads) > 1 and all(np.isfinite(spreads)):
            features.update({
                'spread_mean': np.mean(spreads),
                'spread_std': np.std(spreads),
                'spread_trend': self._calculate_trend(spreads)
            })
        
        # Momentum features
        if len(prices) >= 3 and all(np.isfinite(prices)):
            features.update({
                'momentum_3': (prices[-1] - prices[-4]) / prices[-4] if len(prices) >= 4 and prices[-4] != 0 else 0,
                'momentum_5': (prices[-1] - prices[-6]) / prices[-6] if len(prices) >= 6 and prices[-6] != 0 else 0
            })
        
        return features
    
    def _safe_skew(self, data: List[float]) -> float:
        """Calculate skewness safely"""
        try:
            from scipy import stats
            return float(stats.skew(data))
        except:
            return 0.0
    
    def _safe_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis safely"""
        try:
            from scipy import stats
            return float(stats.kurtosis(data))
        except:
            return 0.0
    
    def _calculate_trend(self, data: List[float]) -> float:
        """Calculate trend using linear regression"""
        if len(data) < 2:
            return 0.0
        
        # Remove any non-finite values
        clean_data = [x for x in data if np.isfinite(x)]
        if len(clean_data) < 2:
            return 0.0
        
        x = np.arange(len(clean_data))
        try:
            slope = np.polyfit(x, clean_data, 1)[0]
            return float(slope) if np.isfinite(slope) else 0.0
        except:
            return 0.0
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """Get all features as DataFrame"""
        if not self.feature_history:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.feature_history)
        df.set_index('timestamp', inplace=True)
        return df
    
    def get_latest_features(self, n: int = 1) -> Dict:
        """Get latest n feature sets"""
        if not self.feature_history:
            return {}
        
        if n == 1:
            return self.feature_history[-1]
        else:
            return self.feature_history[-n:]


class PerformanceProfiler:
    """Performance profiling for ML and orderbook operations"""
    
    def __init__(self):
        self.timings = {}
        self.memory_usage = {}
        self.call_counts = {}
    
    def profile_function(self, func_name: str):
        """Decorator for profiling function performance"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                import time
                import psutil
                import os
                
                # Get initial memory
                process = psutil.Process(os.getpid())
                initial_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Time the function
                start_time = time.perf_counter()
                result = func(*args, **kwargs)
                end_time = time.perf_counter()
                
                # Get final memory
                final_memory = process.memory_info().rss / 1024 / 1024  # MB
                
                # Record metrics
                execution_time = end_time - start_time
                memory_delta = final_memory - initial_memory
                
                if func_name not in self.timings:
                    self.timings[func_name] = []
                    self.memory_usage[func_name] = []
                    self.call_counts[func_name] = 0
                
                self.timings[func_name].append(execution_time)
                self.memory_usage[func_name].append(memory_delta)
                self.call_counts[func_name] += 1
                
                return result
            return wrapper
        return decorator
    
    def get_performance_report(self) -> Dict:
        """Get comprehensive performance report"""
        report = {}
        
        for func_name in self.timings:
            timings = self.timings[func_name]
            memory_usage = self.memory_usage[func_name]
            
            report[func_name] = {
                'call_count': self.call_counts[func_name],
                'total_time': sum(timings),
                'avg_time': np.mean(timings),
                'min_time': min(timings),
                'max_time': max(timings),
                'std_time': np.std(timings),
                'avg_memory_delta': np.mean(memory_usage),
                'max_memory_delta': max(memory_usage) if memory_usage else 0,
                'total_memory_delta': sum(memory_usage)
            }
        
        return report
    
    def print_performance_summary(self):
        """Print formatted performance summary"""
        report = self.get_performance_report()
        
        print("=" * 80)
        print("PERFORMANCE PROFILING REPORT")
        print("=" * 80)
        
        for func_name, metrics in report.items():
            print(f"\nFunction: {func_name}")
            print(f"  Calls: {metrics['call_count']}")
            print(f"  Total Time: {metrics['total_time']:.4f}s")
            print(f"  Avg Time: {metrics['avg_time']:.4f}s")
            print(f"  Min/Max Time: {metrics['min_time']:.4f}s / {metrics['max_time']:.4f}s")
            print(f"  Time Std: {metrics['std_time']:.4f}s")
            print(f"  Avg Memory Delta: {metrics['avg_memory_delta']:.2f}MB")
            print(f"  Max Memory Delta: {metrics['max_memory_delta']:.2f}MB")


class OptimizedBacktester:
    """High-performance backtester using C++ orderbook engine"""
    
    def __init__(self, initial_capital: float = 1000000.0):
        self.feature_engine = RealTimeFeatureEngine()
        self.profiler = PerformanceProfiler()
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trade_log = []
        
    @property
    def profiled_predict(self):
        """Profiled prediction method"""
        return self.profiler.profile_function('predict')
    
    @property
    def profiled_feature_generation(self):
        """Profiled feature generation method"""
        return self.profiler.profile_function('feature_generation')
    
    def backtest_with_model(self, 
                           price_data: pd.Series, 
                           volume_data: pd.Series,
                           model,
                           lookback_window: int = 50) -> Dict:
        """Run optimized backtest with ML model"""
        
        results = {
            'trades': [],
            'portfolio_value': [],
            'positions': [],
            'features': [],
            'predictions': [],
            'performance_metrics': {}
        }
        
        @self.profiler.profile_function('feature_generation')
        def generate_features(price, volume, timestamp):
            return self.feature_engine.add_market_data(price, volume, timestamp)
        
        @self.profiler.profile_function('predict')
        def make_prediction(features_df):
            if len(features_df) < lookback_window:
                return 1  # Neutral
            
            # Use the latest features for prediction
            latest_features = features_df.iloc[-lookback_window:].select_dtypes(include=[np.number])
            
            # Handle missing values
            latest_features = latest_features.fillna(0)
            
            if latest_features.shape[1] == 0:
                return 1  # Neutral if no numeric features
            
            try:
                # Make prediction
                prediction = model.predict(latest_features.iloc[-1:].values)
                return prediction[0] if hasattr(prediction, '__getitem__') else prediction
            except Exception as e:
                warnings.warn(f"Prediction failed: {e}")
                return 1  # Neutral on error
        
        # Main backtest loop
        for timestamp, (price, volume) in enumerate(zip(price_data, volume_data)):
            ts = pd.Timestamp.now() + pd.Timedelta(seconds=timestamp)
            
            # Generate features
            features = generate_features(price, volume, ts)
            results['features'].append(features)
            
            # Get features DataFrame
            features_df = self.feature_engine.get_feature_dataframe()
            
            # Make prediction if we have enough data
            if len(features_df) >= lookback_window:
                prediction = make_prediction(features_df)
                results['predictions'].append(prediction)
                
                # Execute trade based on prediction
                trade_result = self._execute_trade(prediction, price, volume, timestamp)
                if trade_result:
                    results['trades'].append(trade_result)
            else:
                results['predictions'].append(1)  # Neutral
            
            # Record portfolio state
            portfolio_value = self._calculate_portfolio_value(price)
            results['portfolio_value'].append(portfolio_value)
            results['positions'].append(dict(self.positions))
        
        # Calculate performance metrics
        results['performance_metrics'] = self._calculate_performance_metrics(results)
        results['profiling_report'] = self.profiler.get_performance_report()
        
        return results
    
    def _execute_trade(self, prediction: int, price: float, volume: float, timestamp: int) -> Optional[Dict]:
        """Execute trade based on prediction"""
        position_size = 100  # Fixed position size for simplicity
        
        if prediction == 2:  # Buy signal
            if self.capital >= position_size * price:
                self.capital -= position_size * price
                self.positions['long'] = self.positions.get('long', 0) + position_size
                
                return {
                    'timestamp': timestamp,
                    'action': 'buy',
                    'price': price,
                    'size': position_size,
                    'capital_remaining': self.capital
                }
        
        elif prediction == 0:  # Sell signal
            if self.positions.get('long', 0) >= position_size:
                self.capital += position_size * price
                self.positions['long'] = self.positions.get('long', 0) - position_size
                
                return {
                    'timestamp': timestamp,
                    'action': 'sell',
                    'price': price,
                    'size': position_size,
                    'capital_remaining': self.capital
                }
        
        return None
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value"""
        long_value = self.positions.get('long', 0) * current_price
        return self.capital + long_value
    
    def _calculate_performance_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive performance metrics"""
        portfolio_values = results['portfolio_value']
        
        if len(portfolio_values) < 2:
            return {}
        
        # Returns calculation
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        # Basic metrics
        total_return = (portfolio_values[-1] - self.initial_capital) / self.initial_capital
        volatility = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = np.mean(returns) / volatility if volatility > 0 else 0
        
        # Drawdown calculation
        cumulative_returns = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns / running_max - 1
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Trade statistics
        trades = results['trades']
        num_trades = len(trades)
        
        if num_trades > 0:
            buy_trades = [t for t in trades if t['action'] == 'buy']
            sell_trades = [t for t in trades if t['action'] == 'sell']
            
            # Calculate PnL for completed round trips
            pnl_list = []
            for sell_trade in sell_trades:
                # Find matching buy trade (simplified)
                for buy_trade in buy_trades:
                    if buy_trade['timestamp'] < sell_trade['timestamp']:
                        pnl = (sell_trade['price'] - buy_trade['price']) * sell_trade['size']
                        pnl_list.append(pnl)
                        break
            
            win_rate = len([pnl for pnl in pnl_list if pnl > 0]) / len(pnl_list) if pnl_list else 0
            avg_pnl = np.mean(pnl_list) if pnl_list else 0
        else:
            win_rate = 0
            avg_pnl = 0
        
        return {
            'total_return': total_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'final_capital': portfolio_values[-1]
        }
