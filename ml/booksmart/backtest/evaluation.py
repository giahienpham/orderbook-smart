import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings

class ComprehensiveEvaluator:
    """Comprehensive model evaluation with Jane Street-style metrics"""
    
    def __init__(self):
        self.metrics_cache = {}
    
    def evaluate_model_performance(self, 
                                 y_true: np.ndarray, 
                                 y_pred: np.ndarray, 
                                 y_proba: np.ndarray,
                                 returns: Optional[np.ndarray] = None,
                                 timestamps: Optional[pd.DatetimeIndex] = None) -> Dict:
        """Comprehensive model evaluation"""
        
        results = {}
        
        # Classification metrics
        results['classification'] = self._classification_metrics(y_true, y_pred, y_proba)
        
        # Confidence and calibration metrics
        results['confidence'] = self._confidence_metrics(y_true, y_pred, y_proba)
        
        # Trading performance metrics
        if returns is not None:
            results['trading'] = self._trading_metrics(y_true, y_pred, y_proba, returns)
        
        # Temporal stability metrics
        if timestamps is not None:
            results['temporal'] = self._temporal_metrics(y_true, y_pred, timestamps)
        
        # Risk metrics
        results['risk'] = self._risk_metrics(y_true, y_pred, y_proba)
        
        # Information theoretic metrics
        results['information'] = self._information_metrics(y_true, y_pred, y_proba)
        
        # Overall score
        results['overall_score'] = self._calculate_overall_score(results)
        
        return results
    
    def _classification_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """Standard classification metrics"""
        from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                                   log_loss, roc_auc_score, confusion_matrix)
        
        accuracy = accuracy_score(y_true, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
        
        # Handle log loss safely
        try:
            logloss = log_loss(y_true, y_proba)
        except ValueError:
            logloss = np.nan
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Multi-class AUC (if applicable)
        try:
            if len(np.unique(y_true)) > 2:
                auc = roc_auc_score(y_true, y_proba, multi_class='ovr', average='weighted')
            else:
                auc = roc_auc_score(y_true, y_proba[:, 1] if y_proba.shape[1] > 1 else y_proba)
        except (ValueError, IndexError):
            auc = np.nan
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'log_loss': logloss,
            'auc_score': auc,
            'confusion_matrix': cm.tolist(),
            'class_balance': self._calculate_class_balance(y_true, y_pred)
        }
    
    def _confidence_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """Confidence and calibration metrics"""
        confidence_scores = np.max(y_proba, axis=1)
        
        # Basic confidence stats
        avg_confidence = np.mean(confidence_scores)
        confidence_std = np.std(confidence_scores)
        
        # Confidence vs accuracy relationship
        correct_predictions = (y_true == y_pred)
        
        # Calibration analysis
        calibration_results = self._analyze_calibration(correct_predictions, confidence_scores)
        
        # Overconfidence metric
        overconfidence = np.mean(confidence_scores[~correct_predictions]) - np.mean(confidence_scores[correct_predictions])
        
        return {
            'avg_confidence': avg_confidence,
            'confidence_std': confidence_std,
            'calibration': calibration_results,
            'overconfidence_bias': overconfidence,
            'confidence_accuracy_corr': np.corrcoef(confidence_scores, correct_predictions.astype(float))[0, 1]
        }
    
    def _trading_metrics(self, y_true, y_pred, y_proba, returns) -> Dict:
        """Trading performance metrics"""
        confidence_scores = np.max(y_proba, axis=1)
        
        # Generate trading signals
        trading_returns = self._generate_trading_returns(y_true, y_pred, confidence_scores, returns)
        
        # Performance metrics
        total_return = np.sum(trading_returns)
        num_trades = np.sum(np.abs(trading_returns) > 1e-6)
        
        if len(trading_returns) > 1:
            volatility = np.std(trading_returns)
            sharpe_ratio = np.mean(trading_returns) / volatility if volatility > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Win rate
        positive_returns = trading_returns[trading_returns > 0]
        negative_returns = trading_returns[trading_returns < 0]
        win_rate = len(positive_returns) / num_trades if num_trades > 0 else 0
        
        # Risk metrics
        max_drawdown = self._calculate_max_drawdown(trading_returns)
        var_95 = np.percentile(trading_returns, 5) if len(trading_returns) > 0 else 0
        
        # Information ratio (excess return vs tracking error)
        benchmark_return = np.mean(returns) if returns is not None else 0
        excess_returns = trading_returns - benchmark_return
        tracking_error = np.std(excess_returns) if len(excess_returns) > 1 else 0
        info_ratio = np.mean(excess_returns) / tracking_error if tracking_error > 0 else 0
        
        return {
            'total_return': total_return,
            'annualized_return': total_return * 252 / len(trading_returns) if len(trading_returns) > 0 else 0,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'information_ratio': info_ratio,
            'max_drawdown': max_drawdown,
            'var_95': var_95,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'avg_return_per_trade': total_return / num_trades if num_trades > 0 else 0,
            'profit_factor': np.sum(positive_returns) / abs(np.sum(negative_returns)) if len(negative_returns) > 0 else np.inf
        }
    
    def _temporal_metrics(self, y_true, y_pred, timestamps) -> Dict:
        """Temporal stability and drift metrics"""
        # Create time-based windows
        df = pd.DataFrame({
            'actual': y_true,
            'predicted': y_pred,
            'timestamp': timestamps
        })
        
        # Monthly performance
        df['month'] = df['timestamp'].dt.to_period('M')
        monthly_accuracy = df.groupby('month').apply(
            lambda x: np.mean(x['actual'] == x['predicted'])
        )
        
        # Trend analysis
        accuracy_trend = self._calculate_trend(monthly_accuracy.values)
        accuracy_stability = np.std(monthly_accuracy.values)
        
        # Regime detection
        regime_analysis = self._detect_performance_regimes(monthly_accuracy.values)
        
        return {
            'monthly_accuracy_mean': monthly_accuracy.mean(),
            'monthly_accuracy_std': monthly_accuracy.std(),
            'accuracy_trend': accuracy_trend,
            'stability_score': 1.0 / (1.0 + accuracy_stability),  # Higher is more stable
            'regime_analysis': regime_analysis,
            'performance_decay': self._calculate_performance_decay(monthly_accuracy.values)
        }
    
    def _risk_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """Risk and robustness metrics"""
        confidence_scores = np.max(y_proba, axis=1)
        correct_predictions = (y_true == y_pred)
        
        # Model uncertainty
        entropy = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
        avg_entropy = np.mean(entropy)
        
        # Prediction consistency (low entropy should correlate with high accuracy)
        uncertainty_accuracy_corr = np.corrcoef(entropy, correct_predictions.astype(float))[0, 1]
        
        # Tail risk: performance in low-confidence predictions
        low_conf_mask = confidence_scores < np.percentile(confidence_scores, 20)
        tail_accuracy = np.mean(correct_predictions[low_conf_mask]) if np.sum(low_conf_mask) > 0 else 0
        
        # Robustness score
        robustness_score = self._calculate_robustness_score(correct_predictions, confidence_scores)
        
        return {
            'avg_entropy': avg_entropy,
            'uncertainty_accuracy_correlation': uncertainty_accuracy_corr,
            'tail_risk_accuracy': tail_accuracy,
            'robustness_score': robustness_score,
            'prediction_diversity': len(np.unique(y_pred)) / len(np.unique(y_true))
        }
    
    def _information_metrics(self, y_true, y_pred, y_proba) -> Dict:
        """Information theoretic metrics"""
        from sklearn.metrics import mutual_info_score
        
        # Mutual information
        mi_score = mutual_info_score(y_true, y_pred)
        
        # Entropy metrics
        true_entropy = stats.entropy(np.bincount(y_true))
        pred_entropy = stats.entropy(np.bincount(y_pred))
        
        # Information gain
        info_gain = true_entropy - mi_score
        
        # Normalized mutual information
        norm_mi = mi_score / max(true_entropy, pred_entropy) if max(true_entropy, pred_entropy) > 0 else 0
        
        return {
            'mutual_information': mi_score,
            'information_gain': info_gain,
            'normalized_mutual_info': norm_mi,
            'true_entropy': true_entropy,
            'predicted_entropy': pred_entropy
        }
    
    def _generate_trading_returns(self, y_true, y_pred, confidence_scores, returns) -> np.ndarray:
        """Generate trading returns based on predictions"""
        trading_returns = np.zeros(len(y_pred))
        
        for i in range(len(y_pred)):
            actual_direction = y_true[i]
            predicted_direction = y_pred[i]
            confidence = confidence_scores[i]
            
            # Simple strategy: bet on direction with confidence weighting
            if predicted_direction == 2:  # Up prediction
                signal = 1.0
            elif predicted_direction == 0:  # Down prediction  
                signal = -1.0
            else:  # Flat or uncertain
                signal = 0.0
            
            # Weight signal by confidence
            position_size = signal * confidence
            
            # Calculate return (simplified)
            if returns is not None and len(returns) > i:
                trading_returns[i] = position_size * returns[i]
            else:
                # Use direction-based return
                actual_return = 0.01 if actual_direction == 2 else -0.01 if actual_direction == 0 else 0
                trading_returns[i] = position_size * actual_return
        
        return trading_returns
    
    def _calculate_max_drawdown(self, returns) -> float:
        """Calculate maximum drawdown"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = cumulative - running_max
        return np.min(drawdowns) if len(drawdowns) > 0 else 0
    
    def _calculate_trend(self, values) -> float:
        """Calculate trend using linear regression"""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return slope
    
    def _analyze_calibration(self, correct_predictions, confidence_scores, n_bins=10) -> Dict:
        """Analyze prediction calibration"""
        # Create confidence bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        calibration_errors = []
        bin_sizes = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidence_scores > bin_lower) & (confidence_scores <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = correct_predictions[in_bin].mean()
                avg_confidence_in_bin = confidence_scores[in_bin].mean()
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                calibration_errors.append(calibration_error)
                bin_sizes.append(prop_in_bin)
        
        # Expected Calibration Error (ECE)
        ece = np.average(calibration_errors, weights=bin_sizes) if calibration_errors else 0
        
        return {
            'expected_calibration_error': ece,
            'bin_calibration_errors': calibration_errors,
            'bin_sizes': bin_sizes
        }
    
    def _detect_performance_regimes(self, values) -> Dict:
        """Detect performance regimes using changepoint detection"""
        if len(values) < 4:
            return {'num_regimes': 1, 'regime_changes': []}
        
        # Simple changepoint detection using variance changes
        changes = []
        for i in range(2, len(values) - 2):
            before_var = np.var(values[:i])
            after_var = np.var(values[i:])
            if abs(before_var - after_var) > 0.01:  # Threshold for significant change
                changes.append(i)
        
        return {
            'num_regimes': len(changes) + 1,
            'regime_changes': changes
        }
    
    def _calculate_performance_decay(self, values) -> float:
        """Calculate performance decay rate"""
        if len(values) < 2:
            return 0.0
        
        # Fit exponential decay
        x = np.arange(len(values))
        try:
            # Simple linear trend as proxy for decay
            slope = np.polyfit(x, values, 1)[0]
            return -slope  # Negative slope indicates decay
        except:
            return 0.0
    
    def _calculate_robustness_score(self, correct_predictions, confidence_scores) -> float:
        """Calculate model robustness score"""
        # Robustness: high accuracy even with medium confidence
        medium_conf_mask = (confidence_scores >= 0.4) & (confidence_scores <= 0.7)
        if np.sum(medium_conf_mask) > 0:
            medium_conf_accuracy = np.mean(correct_predictions[medium_conf_mask])
            return medium_conf_accuracy
        return 0.0
    
    def _calculate_class_balance(self, y_true, y_pred) -> Dict:
        """Calculate class balance metrics"""
        true_dist = np.bincount(y_true) / len(y_true)
        pred_dist = np.bincount(y_pred, minlength=len(true_dist)) / len(y_pred)
        
        # KL divergence between distributions
        kl_div = stats.entropy(true_dist, pred_dist)
        
        return {
            'true_distribution': true_dist.tolist(),
            'predicted_distribution': pred_dist.tolist(),
            'kl_divergence': kl_div
        }
    
    def _calculate_overall_score(self, results) -> float:
        """Calculate overall model score (0-100)"""
        weights = {
            'accuracy': 0.3,
            'sharpe_ratio': 0.2,
            'stability': 0.2,
            'calibration': 0.15,
            'information': 0.15
        }
        
        score = 0.0
        
        # Accuracy component
        if 'classification' in results:
            score += weights['accuracy'] * results['classification']['accuracy'] * 100
        
        # Trading performance component
        if 'trading' in results:
            sharpe = results['trading']['sharpe_ratio']
            sharpe_score = min(100, max(0, (sharpe + 2) * 25))  # Scale Sharpe to 0-100
            score += weights['sharpe_ratio'] * sharpe_score
        
        # Stability component
        if 'temporal' in results:
            stability = results['temporal']['stability_score']
            score += weights['stability'] * stability * 100
        
        # Calibration component
        if 'confidence' in results:
            ece = results['confidence']['calibration']['expected_calibration_error']
            calibration_score = max(0, 100 - ece * 100)
            score += weights['calibration'] * calibration_score
        
        # Information component
        if 'information' in results:
            norm_mi = results['information']['normalized_mutual_info']
            score += weights['information'] * norm_mi * 100
        
        return min(100, max(0, score))
