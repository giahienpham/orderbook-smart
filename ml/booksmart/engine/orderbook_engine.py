import sys
import os

# Try to import the C++ orderbook engine
try:
    # This would be the pybind11 compiled module
    import orderbook_engine_cpp
    CPP_ENGINE_AVAILABLE = True
except ImportError:
    CPP_ENGINE_AVAILABLE = False

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class Side(Enum):
    BID = 0
    ASK = 1


class OrderType(Enum):
    LIMIT = 0
    MARKET = 1
    CANCEL = 2


@dataclass
class OrderRequest:
    type: OrderType
    side: Side
    price: float
    size: float


@dataclass
class Fill:
    price: float
    size: float


@dataclass
class OrderResult:
    total_filled: float
    notional: float
    fills: List[Fill]


class PythonOrderBook:
    """Pure Python implementation of orderbook for fallback"""
    
    def __init__(self):
        self.bids = {}  # price -> size
        self.asks = {}  # price -> size
        
    def execute_market(self, side: Side, size: float) -> OrderResult:
        """Execute market order"""
        fills = []
        remaining_size = size
        total_filled = 0.0
        notional = 0.0
        
        if side == Side.BID:  # Buying, hit asks
            sorted_asks = sorted(self.asks.items())
            for price, available_size in sorted_asks:
                if remaining_size <= 0:
                    break
                    
                fill_size = min(remaining_size, available_size)
                fills.append(Fill(price=price, size=fill_size))
                
                total_filled += fill_size
                notional += price * fill_size
                remaining_size -= fill_size
                
                # Update book
                self.asks[price] -= fill_size
                if self.asks[price] <= 0:
                    del self.asks[price]
                    
        else:  # Selling, hit bids
            sorted_bids = sorted(self.bids.items(), reverse=True)
            for price, available_size in sorted_bids:
                if remaining_size <= 0:
                    break
                    
                fill_size = min(remaining_size, available_size)
                fills.append(Fill(price=price, size=fill_size))
                
                total_filled += fill_size
                notional += price * fill_size
                remaining_size -= fill_size
                
                # Update book
                self.bids[price] -= fill_size
                if self.bids[price] <= 0:
                    del self.bids[price]
        
        return OrderResult(total_filled=total_filled, notional=notional, fills=fills)
    
    def execute_limit(self, side: Side, price: float, size: float) -> OrderResult:
        """Execute limit order"""
        # First try to match against existing orders
        if side == Side.BID:
            # Check if we can hit any asks
            matching_asks = [p for p in self.asks.keys() if p <= price]
            if matching_asks:
                # Execute as market order up to the limit price
                market_result = self.execute_market(side, size)
                return market_result
            else:
                # Add to book
                if price in self.bids:
                    self.bids[price] += size
                else:
                    self.bids[price] = size
                return OrderResult(total_filled=0.0, notional=0.0, fills=[])
        else:
            # Check if we can hit any bids
            matching_bids = [p for p in self.bids.keys() if p >= price]
            if matching_bids:
                # Execute as market order down to the limit price
                market_result = self.execute_market(side, size)
                return market_result
            else:
                # Add to book
                if price in self.asks:
                    self.asks[price] += size
                else:
                    self.asks[price] = size
                return OrderResult(total_filled=0.0, notional=0.0, fills=[])
    
    def cancel(self, side: Side, price: float, size: float):
        """Cancel order"""
        if side == Side.BID and price in self.bids:
            self.bids[price] = max(0, self.bids[price] - size)
            if self.bids[price] <= 0:
                del self.bids[price]
        elif side == Side.ASK and price in self.asks:
            self.asks[price] = max(0, self.asks[price] - size)
            if self.asks[price] <= 0:
                del self.asks[price]
    
    def best_bid(self) -> Optional[float]:
        """Get best bid price"""
        if not self.bids:
            return None
        return max(self.bids.keys())
    
    def best_ask(self) -> Optional[float]:
        """Get best ask price"""
        if not self.asks:
            return None
        return min(self.asks.keys())
    
    def get_book_state(self) -> Dict:
        """Get current book state"""
        return {
            'bids': dict(self.bids),
            'asks': dict(self.asks),
            'best_bid': self.best_bid(),
            'best_ask': self.best_ask(),
            'spread': self.best_ask() - self.best_bid() if self.best_bid() and self.best_ask() else None
        }


class OrderBookEngine:
    """High-level interface that uses C++ engine if available, else Python fallback"""
    
    def __init__(self, use_cpp: bool = True):
        self.use_cpp = use_cpp and CPP_ENGINE_AVAILABLE
        
        if self.use_cpp:
            try:
                self.engine = orderbook_engine_cpp.LimitOrderBook()
                print("Using C++ orderbook engine")
            except Exception as e:
                print(f"Failed to initialize C++ engine: {e}, falling back to Python")
                self.use_cpp = False
                self.engine = PythonOrderBook()
        else:
            self.engine = PythonOrderBook()
            print("Using Python orderbook engine")
    
    def execute_market(self, side: Side, size: float) -> OrderResult:
        """Execute market order"""
        if self.use_cpp:
            # Convert to C++ types and call
            cpp_side = 0 if side == Side.BID else 1
            result = self.engine.execute_market(cpp_side, size)
            # Convert back to Python types
            return self._convert_cpp_result(result)
        else:
            return self.engine.execute_market(side, size)
    
    def execute_limit(self, side: Side, price: float, size: float) -> OrderResult:
        """Execute limit order"""
        if self.use_cpp:
            cpp_side = 0 if side == Side.BID else 1
            result = self.engine.execute_limit(cpp_side, price, size)
            return self._convert_cpp_result(result)
        else:
            return self.engine.execute_limit(side, price, size)
    
    def cancel(self, side: Side, price: float, size: float):
        """Cancel order"""
        if self.use_cpp:
            cpp_side = 0 if side == Side.BID else 1
            self.engine.cancel(cpp_side, price, size)
        else:
            self.engine.cancel(side, price, size)
    
    def best_bid(self) -> Optional[float]:
        """Get best bid"""
        if self.use_cpp:
            result = self.engine.best_bid()
            return result if result != 0 else None
        else:
            return self.engine.best_bid()
    
    def best_ask(self) -> Optional[float]:
        """Get best ask"""
        if self.use_cpp:
            result = self.engine.best_ask()
            return result if result != 0 else None
        else:
            return self.engine.best_ask()
    
    def get_book_state(self) -> Dict:
        """Get current book state"""
        if self.use_cpp:
            # For C++ engine, we need to extract the state differently
            return {
                'best_bid': self.best_bid(),
                'best_ask': self.best_ask(),
                'spread': self.best_ask() - self.best_bid() if self.best_bid() and self.best_ask() else None,
                'engine_type': 'cpp'
            }
        else:
            state = self.engine.get_book_state()
            state['engine_type'] = 'python'
            return state
    
    def _convert_cpp_result(self, cpp_result) -> OrderResult:
        """Convert C++ result to Python OrderResult"""
        if hasattr(cpp_result, 'fills'):
            fills = [Fill(price=f.price, size=f.size) for f in cpp_result.fills]
        else:
            fills = []
        
        return OrderResult(
            total_filled=getattr(cpp_result, 'total_filled', 0.0),
            notional=getattr(cpp_result, 'notional', 0.0),
            fills=fills
        )


class HighFrequencySimulator:
    """High-frequency trading simulator using the orderbook engine"""
    
    def __init__(self, initial_price: float = 100.0, tick_size: float = 0.01):
        self.orderbook = OrderBookEngine()
        self.initial_price = initial_price
        self.tick_size = tick_size
        self.current_time = 0
        self.trade_history = []
        self.book_states = []
        
        # Initialize the book with some liquidity
        self._initialize_book()
    
    def _initialize_book(self):
        """Initialize orderbook with some liquidity"""
        mid_price = self.initial_price
        
        # Add bids (buy orders) below mid price
        for i in range(1, 11):
            price = mid_price - i * self.tick_size
            size = np.random.uniform(100, 1000)
            self.orderbook.execute_limit(Side.BID, price, size)
        
        # Add asks (sell orders) above mid price
        for i in range(1, 11):
            price = mid_price + i * self.tick_size
            size = np.random.uniform(100, 1000)
            self.orderbook.execute_limit(Side.ASK, price, size)
    
    def simulate_market_impact(self, side: Side, size: float) -> Dict:
        """Simulate market impact of an order"""
        initial_state = self.orderbook.get_book_state()
        
        # Execute the order
        result = self.orderbook.execute_market(side, size)
        
        final_state = self.orderbook.get_book_state()
        
        # Calculate impact
        if initial_state['best_bid'] and initial_state['best_ask']:
            initial_mid = (initial_state['best_bid'] + initial_state['best_ask']) / 2
        else:
            initial_mid = self.initial_price
            
        if final_state['best_bid'] and final_state['best_ask']:
            final_mid = (final_state['best_bid'] + final_state['best_ask']) / 2
        else:
            final_mid = self.initial_price
        
        price_impact = final_mid - initial_mid
        
        # Calculate VWAP
        if result.fills:
            vwap = result.notional / result.total_filled
        else:
            vwap = 0.0
        
        return {
            'order_size': size,
            'filled_size': result.total_filled,
            'fill_rate': result.total_filled / size if size > 0 else 0,
            'vwap': vwap,
            'price_impact': price_impact,
            'initial_spread': initial_state['spread'],
            'final_spread': final_state['spread'],
            'num_fills': len(result.fills),
            'fills': result.fills
        }
    
    def generate_microstructure_features(self, lookback_trades: int = 10) -> Dict:
        """Generate microstructure features from recent trading"""
        if len(self.trade_history) < lookback_trades:
            return {}
        
        recent_trades = self.trade_history[-lookback_trades:]
        
        # Volume-weighted metrics
        total_volume = sum(t['filled_size'] for t in recent_trades)
        vwap = sum(t['vwap'] * t['filled_size'] for t in recent_trades) / total_volume if total_volume > 0 else 0
        
        # Order flow imbalance
        buy_volume = sum(t['filled_size'] for t in recent_trades if t.get('side') == Side.BID)
        sell_volume = sum(t['filled_size'] for t in recent_trades if t.get('side') == Side.ASK)
        order_flow_imbalance = (buy_volume - sell_volume) / (buy_volume + sell_volume) if (buy_volume + sell_volume) > 0 else 0
        
        # Price impact metrics
        avg_price_impact = np.mean([t['price_impact'] for t in recent_trades])
        price_impact_vol = np.std([t['price_impact'] for t in recent_trades])
        
        # Current book state
        current_state = self.orderbook.get_book_state()
        
        return {
            'vwap': vwap,
            'order_flow_imbalance': order_flow_imbalance,
            'avg_price_impact': avg_price_impact,
            'price_impact_volatility': price_impact_vol,
            'current_spread': current_state.get('spread', 0),
            'total_volume': total_volume,
            'trade_count': len(recent_trades),
            'avg_fill_rate': np.mean([t['fill_rate'] for t in recent_trades])
        }
    
    def step(self, orders: List[OrderRequest]) -> Dict:
        """Execute a simulation step with given orders"""
        step_results = []
        
        for order in orders:
            if order.type == OrderType.MARKET:
                impact = self.simulate_market_impact(order.side, order.size)
                impact['side'] = order.side
                impact['timestamp'] = self.current_time
                self.trade_history.append(impact)
                step_results.append(impact)
            
            elif order.type == OrderType.LIMIT:
                result = self.orderbook.execute_limit(order.side, order.price, order.size)
                step_results.append({
                    'type': 'limit',
                    'side': order.side,
                    'price': order.price,
                    'size': order.size,
                    'filled': result.total_filled,
                    'timestamp': self.current_time
                })
        
        # Record book state
        book_state = self.orderbook.get_book_state()
        book_state['timestamp'] = self.current_time
        self.book_states.append(book_state)
        
        # Generate features
        features = self.generate_microstructure_features()
        
        self.current_time += 1
        
        return {
            'step_results': step_results,
            'book_state': book_state,
            'features': features,
            'timestamp': self.current_time
        }
