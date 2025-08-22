from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from scipy import optimize
from sklearn.preprocessing import StandardScaler


class GARCHModel:
    """GARCH(1,1) model with MLE estimation."""
    
    def __init__(self):
        self.omega = None
        self.alpha = None
        self.beta = None
        self.fitted = False
        
    def _log_likelihood(self, params: np.ndarray, returns: np.ndarray) -> float:
        """Negative log-likelihood for GARCH(1,1)."""
        omega, alpha, beta = params
        
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e6
            
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)  # Initial variance
        
        log_likelihood = 0.0
        
        for t in range(1, n):
            sigma2[t] = omega + alpha * returns[t-1]**2 + beta * sigma2[t-1]
            if sigma2[t] <= 0:
                return 1e6
            log_likelihood += -0.5 * (np.log(2 * np.pi) + np.log(sigma2[t]) + returns[t]**2 / sigma2[t])
            
        return -log_likelihood
    
    def fit(self, returns: np.ndarray) -> None:
        """Fit GARCH model using maximum likelihood estimation."""
        # Initial parameter guess
        initial_params = [0.0001, 0.1, 0.85]
        
        # omega > 0, alpha >= 0, beta >= 0, alpha + beta < 1
        bounds = [(1e-8, 1), (0, 1), (0, 1)]
        constraints = {'type': 'ineq', 'fun': lambda x: 0.999 - x[1] - x[2]}
        
        result = optimize.minimize(
            self._log_likelihood,
            initial_params,
            args=(returns,),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            self.omega, self.alpha, self.beta = result.x
            self.fitted = True
        else:
            self.omega = 0.0001
            self.alpha = 0.1
            self.beta = 0.85
            self.fitted = True
    
    def forecast(self, returns: np.ndarray, horizon: int = 1) -> np.ndarray:
        """Forecast conditional variance."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        n = len(returns)
        sigma2 = np.zeros(n + horizon)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * sigma2[t-1]
        
        for t in range(n, n + horizon):
            if t == n:
                sigma2[t] = self.omega + self.alpha * returns[-1]**2 + self.beta * sigma2[t-1]
            else:
                sigma2[t] = self.omega + (self.alpha + self.beta) * sigma2[t-1]
        
        return np.sqrt(sigma2[-horizon:])
    
    def conditional_variance(self, returns: np.ndarray) -> np.ndarray:
        """Calculate conditional variance series."""
        if not self.fitted:
            raise ValueError("Model must be fitted first")
            
        n = len(returns)
        sigma2 = np.zeros(n)
        sigma2[0] = np.var(returns)
        
        for t in range(1, n):
            sigma2[t] = self.omega + self.alpha * returns[t-1]**2 + self.beta * sigma2[t-1]
            
        return sigma2


class RealizedVolatility:
    """Realized volatility measures for high-frequency data."""
    
    @staticmethod
    def realized_variance(returns: np.ndarray) -> float:
        """Standard realized variance."""
        return np.sum(returns**2)
    
    @staticmethod
    def bipower_variation(returns: np.ndarray) -> float:
        """Bipower variation - robust to jumps."""
        n = len(returns)
        if n < 2:
            return 0.0
        
        mu1 = np.sqrt(2/np.pi)  # E[|Z|] for standard normal Z
        
        return (mu1**(-2)) * (np.pi/2) * np.sum(
            np.abs(returns[1:]) * np.abs(returns[:-1])
        )
    
    @staticmethod
    def tripower_quarticity(returns: np.ndarray) -> float:
        """Tripower quarticity for jump-robust inference."""
        n = len(returns)
        if n < 3:
            return 0.0
            
        mu_4_3 = 2**(2/3) * (np.gamma(7/6) / np.gamma(1/2))
        
        return n * (mu_4_3**(-3)) * np.sum(
            np.abs(returns[2:])**(4/3) * 
            np.abs(returns[1:-1])**(4/3) * 
            np.abs(returns[:-2])**(4/3)
        )
    
    @staticmethod
    def jump_test(returns: np.ndarray, alpha: float = 0.05) -> Tuple[bool, float]:
        """Test for jumps using BNS (2004) statistic."""
        rv = RealizedVolatility.realized_variance(returns)
        bv = RealizedVolatility.bipower_variation(returns)
        tq = RealizedVolatility.tripower_quarticity(returns)
        
        if tq <= 0 or bv <= 0:
            return False, 0.0
        
        n = len(returns)
        numerator = np.sqrt(n) * (rv - bv)
        denominator = np.sqrt((np.pi**2 / 4 + np.pi - 5) * tq)
        
        if denominator == 0:
            return False, 0.0
            
        z_stat = numerator / denominator
        
        from scipy import stats
        critical_value = stats.norm.ppf(1 - alpha/2)
        
        has_jump = np.abs(z_stat) > critical_value
        return has_jump, z_stat


class VolatilityClustering:
    """Methods to detect and model volatility clustering."""
    
    @staticmethod
    def ljung_box_test(residuals: np.ndarray, lags: int = 10) -> Tuple[float, float]:
        """Ljung-Box test for serial correlation in squared residuals."""
        n = len(residuals)
        squared_residuals = residuals**2
        
        acf = np.zeros(lags + 1)
        mean_sq = np.mean(squared_residuals)
        
        for k in range(lags + 1):
            if k == 0:
                acf[k] = 1.0
            else:
                covariance = np.mean(
                    (squared_residuals[:-k] - mean_sq) * 
                    (squared_residuals[k:] - mean_sq)
                )
                variance = np.var(squared_residuals)
                acf[k] = covariance / variance if variance > 0 else 0
        
        lb_stat = n * (n + 2) * np.sum([
            acf[k]**2 / (n - k) for k in range(1, lags + 1)
        ])
        
        # p-value
        from scipy import stats
        p_value = 1 - stats.chi2.cdf(lb_stat, lags)
        
        return lb_stat, p_value
    
    @staticmethod
    def volatility_clustering_index(returns: np.ndarray, window: int = 20) -> np.ndarray:
        """Calculate volatility clustering index over rolling windows."""
        n = len(returns)
        vci = np.zeros(n)
        
        abs_returns = np.abs(returns)
        
        for i in range(window, n):
            window_data = abs_returns[i-window:i]
            current_vol = abs_returns[i]
            
            recent_avg = np.mean(window_data)
            if recent_avg > 0:
                vci[i] = current_vol / recent_avg
            else:
                vci[i] = 1.0
                
        return vci


class VolatilityFeatures:
    """Generate volatility-based features for ML models."""
    
    def __init__(self, lookback_windows: list = None):
        if lookback_windows is None:
            self.lookback_windows = [5, 10, 20, 50]
        else:
            self.lookback_windows = lookback_windows
        self.garch_model = GARCHModel()
        
    def create_features(self, returns: pd.Series) -> pd.DataFrame:
        """Create comprehensive volatility features."""
        features = pd.DataFrame(index=returns.index)
        
        for window in self.lookback_windows:
            features[f'vol_ewm_{window}'] = returns.ewm(span=window).std() * np.sqrt(252)
            features[f'vol_roll_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            
            features[f'skew_{window}'] = returns.rolling(window).skew()
            features[f'kurt_{window}'] = returns.rolling(window).kurt()
        # GARCH features
        if len(returns.dropna()) > 50:
            clean_returns = returns.dropna().values
            try:
                self.garch_model.fit(clean_returns)
                garch_vol = self.garch_model.conditional_variance(clean_returns)
                
                garch_series = pd.Series(
                    np.sqrt(garch_vol) * np.sqrt(252), 
                    index=returns.dropna().index
                )
                features['garch_vol'] = garch_series.reindex(returns.index)
                
                # GARCH forecast
                try:
                    forecast = self.garch_model.forecast(clean_returns, horizon=1)[0]
                    features['garch_forecast'] = forecast * np.sqrt(252)
                except:
                    features['garch_forecast'] = np.nan
                    
            except:
                features['garch_vol'] = np.nan
                features['garch_forecast'] = np.nan
        
        # Volatility clustering
        clustering_idx = VolatilityClustering.volatility_clustering_index(
            returns.values, window=20
        )
        features['vol_clustering'] = clustering_idx
        
        # Volatility regime features
        features['vol_regime'] = self._volatility_regime(returns)
        
        if 'vol_roll_5' in features.columns and 'vol_roll_20' in features.columns:
            features['vol_ratio_5_20'] = (
                features['vol_roll_5'] / (features['vol_roll_20'] + 1e-8)
            )
        
        return features.fillna(method='ffill').fillna(0)
    
    def _volatility_regime(self, returns: pd.Series, threshold: float = 1.5) -> pd.Series:
        """Identify volatility regime (low/high) based on rolling statistics."""
        vol_20 = returns.rolling(20).std()
        vol_100 = returns.rolling(100).std()
        
        regime = pd.Series(index=returns.index, dtype=int)
        regime[:] = 0  # Normal regime
        
        # High regime
        regime[vol_20 > threshold * vol_100] = 1
        # Low regime  
        regime[vol_20 < vol_100 / threshold] = -1
        return regime
    
    def _parkinson_volatility(self, high: pd.Series, low: pd.Series, window: int) -> pd.Series:
        """Parkinson range-based volatility estimator."""
        log_hl_ratio = np.log(high / low)
        return np.sqrt(log_hl_ratio.rolling(window).mean() / (4 * np.log(2))) * np.sqrt(252)


def create_volatility_features(df: pd.DataFrame, price_col: str = 'close') -> pd.DataFrame:
    out = df.copy()
    
    returns = out[price_col].pct_change().dropna()
    
    vol_features = VolatilityFeatures()
    features_df = vol_features.create_features(returns)
    
    # Remove overlapping columns before joining
    overlapping_cols = [col for col in features_df.columns if col in out.columns]
    if overlapping_cols:
        features_df = features_df.drop(columns=overlapping_cols)
    
    out = out.join(features_df, how='left')
    
    out['vol_breakout'] = (
        (out.get('vol_roll_5', 0) > 1.5 * out.get('vol_roll_20', 1)).astype(int)
    )
    
    out['vol_contraction'] = (
        (out.get('vol_roll_5', 1) < 0.7 * out.get('vol_roll_20', 1)).astype(int)
    )
    
    return out.fillna(method='ffill')