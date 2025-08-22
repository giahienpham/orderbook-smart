from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


class FallbackDirectionModel:
    
    def __init__(self, **rf_params: Any):
        self.params = {
            "n_estimators": 100,
            "max_depth": 8,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42,
            **rf_params
        }
        self.model = RandomForestClassifier(**self.params)
        self.feature_names = None
        self.feature_importance_ = None
        self.fitted = False
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              feature_names: List[str] = None,
              num_rounds: int = 100, early_stopping_rounds: int = 10,
              verbose: bool = False) -> None:
        
        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        self.model.fit(X_train, y_train)
        self.fitted = True
        
        importance_scores = self.model.feature_importances_
        self.feature_importance_ = pd.DataFrame([
            {'feature': feat, 'importance': importance_scores[i]}
            for i, feat in enumerate(self.feature_names)
        ]).sort_values('importance', ascending=False)
        
        if verbose:
            print(f"RandomForest trained with {len(self.feature_names)} features")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not trained")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Model not trained")
        return self.model.predict_proba(X)
    
    def predict_with_confidence(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        probs = self.predict_proba(X)
        predictions = self.model.predict(X)
        confidence = np.max(probs, axis=1)
        return predictions, confidence
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, detailed: bool = False) -> dict:
        y_pred = self.predict(X)
        
        acc = accuracy_score(y, y_pred)
        
        unique_classes = np.unique(y)
        if len(unique_classes) == 1:
            # Handle edge case where only one class is present
            metrics = {
                "accuracy": acc,
                "note": f"Only one class present: {unique_classes[0]}",
                "confusion_matrix": [[len(y)]]
            }
        else:
            class_names = ["down", "flat", "up"]
            available_names = [class_names[i] for i in unique_classes]
            
            report = classification_report(y, y_pred, target_names=available_names, output_dict=True, zero_division=0)
            cm = confusion_matrix(y, y_pred)
            
            metrics = {
                "accuracy": acc,
                "classification_report": report,
                "confusion_matrix": cm.tolist()
            }
        
        if detailed:
            _, confidence = self.predict_with_confidence(X)
            metrics["avg_confidence"] = np.mean(confidence)
            metrics["min_confidence"] = np.min(confidence)
        
        return metrics
    
    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        if self.feature_importance_ is None:
            raise ValueError("Model not trained")
        return self.feature_importance_.head(top_n)
    
    def save(self, path: str | Path) -> None:
        if not self.fitted:
            raise ValueError("Model not trained")
        
        model_data = {
            "model": self.model,
            "params": self.params,
            "feature_names": self.feature_names,
            "feature_importance": self.feature_importance_
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
            self.fitted = True
