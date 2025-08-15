from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


class XGBDirectionModel:
    """Simple XGBoost classifier for stock direction prediction."""
    
    def __init__(self, **xgb_params: Any):
        self.params = {
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "max_depth": 6,
            "eta": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 42,
            **xgb_params
        }
        self.model = None
        self.feature_names = None
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              num_rounds: int = 100, early_stopping_rounds: int = 10) -> None:
        # Convert labels: -1,0,1 -> 0,1,2 for XGBoost
        y_train_mapped = y_train + 1
        
        dtrain = xgb.DMatrix(X_train, label=y_train_mapped)
        evals = [(dtrain, "train")]
        
        if X_val is not None and y_val is not None:
            y_val_mapped = y_val + 1
            dval = xgb.DMatrix(X_val, label=y_val_mapped)
            evals.append((dval, "val"))
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=num_rounds,
            evals=evals,
            early_stopping_rounds=early_stopping_rounds,
            verbose_eval=False
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X)
        probs = self.model.predict(dtest)
        # Convert back: 0,1,2 -> -1,0,1
        return np.argmax(probs, axis=1) - 1
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained")
        
        dtest = xgb.DMatrix(X)
        return self.model.predict(dtest)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        y_pred = self.predict(X)
        acc = accuracy_score(y, y_pred)
        report = classification_report(y, y_pred, target_names=["down", "flat", "up"], output_dict=True)
        return {"accuracy": acc, "classification_report": report}
    
    def save(self, path: str | Path) -> None:
        if self.model is None:
            raise ValueError("Model not trained")
        
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "params": self.params}, f)
    
    def load(self, path: str | Path) -> None:
        with open(path, "rb") as f:
            data = pickle.load(f)
            self.model = data["model"]
            self.params = data["params"]