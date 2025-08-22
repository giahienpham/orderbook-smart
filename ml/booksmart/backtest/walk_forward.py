import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import warnings
from ..models.xgb_model import XGBDirectionModel
from ..models.fallback_model import FallbackDirectionModel

class WalkForwardAnalysis:
    def __init__(self, 
                 train_window_days: int = 252,
                 retrain_frequency_days: int = 21,
                 min_train_samples: int = 100):
        self.train_window_days = train_window_days
        self.retrain_frequency_days = retrain_frequency_days
        self.min_train_samples = min_train_samples
        self.models = []
        self.results = []
        
    def run_analysis(self, data: pd.DataFrame, target_col: str = 'direction') -> Dict:
        """Run walk-forward analysis on the dataset"""
        data = data.copy().sort_index()
        
        if len(data) < self.min_train_samples * 2:
            raise ValueError(f"Insufficient data: need at least {self.min_train_samples * 2} samples")
        
        # Prepare feature columns (exclude target and non-numeric columns)
        feature_cols = []
        for col in data.columns:
            if col != target_col and not col.startswith('future_') and col != 'direction' and col != 'ts':
                try:
                    # Check if column is numeric and contains no timestamps
                    sample_data = data[col].iloc[:5].dropna()
                    if len(sample_data) > 0:
                        pd.to_numeric(sample_data)
                        feature_cols.append(col)
                except (ValueError, TypeError):
                    print(f"Skipping non-numeric column: {col}")
                    continue  # Skip non-numeric columns
        
        print(f"Selected {len(feature_cols)} feature columns: {feature_cols[:5]}...")
        
        results = []
        model_performances = []
        
        # Calculate walk-forward windows
        start_idx = self.train_window_days
        retrain_counter = 0
        model = None
        
        for current_idx in range(start_idx, len(data), self.retrain_frequency_days):
            end_idx = min(current_idx + self.retrain_frequency_days, len(data))
            
            # Training window
            train_start = max(0, current_idx - self.train_window_days)
            train_data = data.iloc[train_start:current_idx]
            
            # Test window
            test_data = data.iloc[current_idx:end_idx]
            
            if len(train_data) < self.min_train_samples or len(test_data) == 0:
                continue
                
            # Prepare training data
            X_train = train_data[feature_cols].values
            y_train = train_data[target_col].values
            
            # Prepare test data
            X_test = test_data[feature_cols].values
            y_test = test_data[target_col].values
            
            # Train model (or retrain)
            if model is None or retrain_counter % 1 == 0:  # Retrain every window
                try:
                    model = XGBDirectionModel()
                    model.train(X_train, y_train, feature_names=feature_cols)
                except Exception as e:
                    warnings.warn(f"XGBoost failed, using fallback: {e}")
                    model = FallbackDirectionModel()
                    model.train(X_train, y_train, feature_names=feature_cols)
            
            # Make predictions
            predictions = model.predict(X_test)
            probs = model.predict_proba(X_test)
            
            # Store results for this window
            window_result = {
                'window_start': train_data.index[0],
                'window_end': train_data.index[-1],
                'test_start': test_data.index[0],
                'test_end': test_data.index[-1],
                'train_samples': len(train_data),
                'test_samples': len(test_data),
                'predictions': predictions,
                'actual': y_test,
                'probabilities': probs,
                'test_dates': test_data.index.tolist()
            }
            
            results.append(window_result)
            
            # Evaluate model performance on this window
            accuracy = np.mean(predictions == y_test)
            model_performances.append({
                'window': retrain_counter,
                'accuracy': accuracy,
                'test_samples': len(test_data),
                'train_samples': len(train_data)
            })
            
            retrain_counter += 1
        
        # Aggregate results
        all_predictions = np.concatenate([r['predictions'] for r in results])
        all_actual = np.concatenate([r['actual'] for r in results])
        all_probs = np.vstack([r['probabilities'] for r in results])
        
        overall_metrics = self._calculate_comprehensive_metrics(
            all_actual, all_predictions, all_probs, results
        )
        
        return {
            'overall_metrics': overall_metrics,
            'window_results': results,
            'model_performances': model_performances,
            'summary': self._create_summary(overall_metrics, model_performances)
        }
    
    def _calculate_comprehensive_metrics(self, y_true, y_pred, y_proba, window_results) -> Dict:
        """Calculate comprehensive evaluation metrics"""
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, log_loss
        
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        # Confidence metrics
        confidence_scores = np.max(y_proba, axis=1)
        avg_confidence = np.mean(confidence_scores)
        
        # Log loss (if all classes present)
        try:
            logloss = log_loss(y_true, y_proba)
        except ValueError:
            logloss = np.nan
        
        # Temporal stability
        window_accuracies = []
        for result in window_results:
            window_acc = np.mean(result['predictions'] == result['actual'])
            window_accuracies.append(window_acc)
        
        stability_metrics = {
            'accuracy_std': np.std(window_accuracies),
            'accuracy_trend': self._calculate_trend(window_accuracies),
            'min_window_accuracy': np.min(window_accuracies),
            'max_window_accuracy': np.max(window_accuracies)
        }
        
        # Trading simulation metrics
        trading_metrics = self._calculate_trading_metrics(y_true, y_pred, confidence_scores)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'log_loss': logloss,
            'avg_confidence': avg_confidence,
            'stability': stability_metrics,
            'trading': trading_metrics,
            'total_predictions': len(y_true),
            'class_distribution': {
                'actual': np.bincount(y_true),
                'predicted': np.bincount(y_pred)
            }
        }
    
    def _calculate_trading_metrics(self, y_true, y_pred, confidence_scores) -> Dict:
        """Calculate trading-specific metrics"""
        # Simple trading simulation: go long on up predictions, short on down predictions
        # Direction mapping: 0=down, 1=flat, 2=up
        
        returns = []
        trades = []
        
        for i in range(len(y_pred)):
            actual_direction = y_true[i]
            predicted_direction = y_pred[i]
            confidence = confidence_scores[i]
            
            # Simple return based on prediction accuracy
            if predicted_direction == 2:  # Predicted up
                trade_return = 1.0 if actual_direction == 2 else -0.5 if actual_direction == 0 else 0
            elif predicted_direction == 0:  # Predicted down
                trade_return = 1.0 if actual_direction == 0 else -0.5 if actual_direction == 2 else 0
            else:  # Predicted flat or uncertain
                trade_return = 0
            
            # Weight by confidence
            weighted_return = trade_return * confidence
            returns.append(weighted_return)
            trades.append(abs(trade_return) > 0)
        
        returns = np.array(returns)
        total_return = np.sum(returns)
        num_trades = np.sum(trades)
        win_rate = np.mean([r > 0 for r in returns if r != 0]) if num_trades > 0 else 0
        
        # Risk metrics
        returns_std = np.std(returns) if len(returns) > 1 else 0
        sharpe_ratio = total_return / returns_std if returns_std > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        return {
            'total_return': total_return,
            'num_trades': num_trades,
            'win_rate': win_rate,
            'avg_return_per_trade': total_return / num_trades if num_trades > 0 else 0,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'returns_std': returns_std
        }
    
    def _calculate_trend(self, values) -> float:
        """Calculate trend in values using linear regression slope"""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def _create_summary(self, metrics: Dict, model_performances: List[Dict]) -> Dict:
        """Create summary of walk-forward analysis"""
        return {
            'total_windows': len(model_performances),
            'avg_accuracy': metrics['accuracy'],
            'accuracy_stability': metrics['stability']['accuracy_std'],
            'trading_sharpe': metrics['trading']['sharpe_ratio'],
            'total_trading_return': metrics['trading']['total_return'],
            'model_degradation': metrics['stability']['accuracy_trend'],
            'recommendation': self._get_recommendation(metrics)
        }
    
    def _get_recommendation(self, metrics: Dict) -> str:
        """Provide recommendation based on metrics"""
        accuracy = metrics['accuracy']
        stability = metrics['stability']['accuracy_std']
        sharpe = metrics['trading']['sharpe_ratio']
        
        if accuracy > 0.6 and stability < 0.1 and sharpe > 1.0:
            return "STRONG_BUY - Excellent model performance with high stability"
        elif accuracy > 0.55 and stability < 0.15 and sharpe > 0.5:
            return "BUY - Good model performance with acceptable stability"
        elif accuracy > 0.5 and stability < 0.2:
            return "HOLD - Model shows some predictive power but monitor closely"
        else:
            return "AVOID - Model lacks consistent predictive power"
