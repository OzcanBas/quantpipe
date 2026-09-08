from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import numpy as np


class PredictorModel():
    def __init__(self):
        self.model = LogisticRegression(random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False

    def fit(self, X_train, y_train):
        X_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_scaled, y_train)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)


class PredictorModelXG():
    def __init__(
            self, 
            n_estimators, 
            max_depth, 
            learning_rate, 
            min_child_weight, 
            subsample, 
            colsample_bytree
            ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_child_weight=self.min_child_weight,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            reg_alpha=0.1,
            reg_lambda=5.0,
            random_state=42,
            n_jobs=-1,
        )
        self.is_fitted = False

    def fit(self, X_train, y_train):
        y_binary = (y_train == 1).astype(int)
        self.model.fit(X_train, y_binary)
        self.is_fitted = True

    def predict(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        prediction = self.model.predict(X)
        return prediction * 2 - 1

    def predict_proba(self, X):
        if not self.is_fitted:
            raise ValueError("Model must be fitted before predictions")
        return self.model.predict_proba(X)
