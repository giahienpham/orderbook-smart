from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Install with: brew install libomp && pip install xgboost")

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False


class XGBDirectionModel:
    
    def __init__(self, **xgb_params: Any):
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is not available. Install with: brew install libomp && pip install xgboost")
            
        self.params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "gamma": 0.1,
            "min_child_weight": 1,
            "reg_alpha": 0.01,
            "reg_lambda": 0.01,
            "seed": 42,
            **xgb_params
        }
        self.model = None
        self.feature_names = None
        self.feature_importance_ = None
        self.training_history = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None,
              num_rounds: int = 100, early_stopping_rounds: int = 10,
              verbose: bool = False) -> None:
        y_train_mapped = y_train + 1
        
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        dtrain = xgb.DMatrix(X_train, label=y_train_mapped, feature_names=self.feature_names)
        evals = [(dtrain, "train")]
        
        if X_val is not None and y_val is not None:
            y_val_mapped = y_val + 1
            dval = xgb.DMatrix(X_val, label=y_val_mapped, feature_names=self.feature_names)
            evals.append((dval, "val"))
        
        evals_result = {}
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=verbose
        )
        
        self.training_history = evals_result
        
        importance_dict = self.model.get_score(importance_type='gain')
        self.feature_importance_ = pd.DataFrame([
            {'feature': feat, 'importance': importance_dict.get(feat, 0.0)}
            for feat in self.feature_names
        ]).sort_values('importance', ascending=False)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        probs = self.model.predict(dtest)
        # Convert back: 0,1,2 -> -1,0,1
        return np.argmax(probs, axis=1) - 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        return self.model.predict(dtest)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(X)
        predictions = np.argmax(probs, axis=1) - 1
        confidence = np.max(probs, axis=1)
        return predictions, confidence
    
    def evaluate(self, X, y, detailed=False):
        """Evaluate model performance"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        y_pred = self.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        result = {"accuracy": accuracy}
        
        if detailed:
            unique_classes = np.unique(y)
            if len(unique_classes) == 1:
                # Handle edge case where only one class is present
                result["note"] = f"Only one class present: {unique_classes[0]}"
                result["precision"] = accuracy
                result["recall"] = accuracy
                result["f1"] = accuracy
            else:
                class_names = ["down", "flat", "up"]
                available_names = [class_names[i] for i in unique_classes]
                
                report = classification_report(y, y_pred, target_names=available_names, output_dict=True, zero_division=0)
                result.update(report)
        
        return result
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.feature_importance_ is None:
            raise ValueError("Model not trained")
        return self.feature_importance_.head(top_n)
    
    def plot_feature_importance(self, top_n: int = 20, figsize: Tuple[int, int] = (10, 8)) -> None:
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return
            
        if self.feature_importance_ is None:
            raise ValueError("Model not trained")
        
        top_features = self.get_feature_importance(top_n)
        
        plt.figure(figsize=figsize)
        sns.barplot(data=top_features, y='feature', x='importance')
        plt.title(f'Top {top_n} Feature Importance (XGBoost Gain)')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_training_curves(self, figsize: Tuple[int, int] = (12, 4)) -> None:
        if not PLOTTING_AVAILABLE:
            print("Plotting not available. Install matplotlib and seaborn.")
            return
            
        if self.training_history is None:
            raise ValueError("No training history available")
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        train_loss = self.training_history['train']['mlogloss']
        axes[0].plot(train_loss, label='Train')
        if 'val' in self.training_history:
            val_loss = self.training_history['val']['mlogloss']
            axes[0].plot(val_loss, label='Validation')
        axes[0].set_title('Training Loss')
        axes[0].set_xlabel('Iteration')
        axes[0].set_ylabel('Log Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        if self.feature_importance_ is not None:
            top_10 = self.feature_importance_.head(10)
            sns.barplot(data=top_10, y='feature', x='importance', ax=axes[1])
            axes[1].set_title('Top 10 Features')
        
        plt.tight_layout()
        plt.show()
    
    def analyze_predictions(self, X: np.ndarray, y: np.ndarray, 
                          feature_names: List[str] = None) -> pd.DataFrame:
        predictions, confidence = self.predict_with_confidence(X)
        
        analysis_df = pd.DataFrame({
            'actual': y,
            'predicted': predictions,
            'confidence': confidence,
            'correct': (y == predictions).astype(int)
        })
        
        if feature_names is None:
            feature_names = self.feature_names
        
        for i, feat_name in enumerate(feature_names[:10]):
            if i < X.shape[1]:
                analysis_df[f'feat_{feat_name}'] = X[:, i]
        
        return analysis_df
    
    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model not trained")
        
        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_,
            "training_history": self.training_history
        }
        
        with open(path, "wb") as f:
            pickle.dump(model_data, f)
    
    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.params = data["params"]
            self.feature_names = data.get("feature_names")
            self.feature_importance_ = data.get("feature_importance")
            self.training_history = data.get("training_history")