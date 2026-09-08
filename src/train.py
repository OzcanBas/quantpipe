from .data import DataHandler
from .model import PredictorModelXG
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import optuna

import numpy as np


def train_model(ticker="AAPL", windows=None, split_date="2023-01-01"):
    if windows is None:
        windows = [10]
    # load and prepare the data
    data_handler = DataHandler(ticker)
    data_handler.download_data()
    features, target = data_handler.build_feature_matrix(windows)

    # split into training and held-out test sets by date
    feature_index = data_handler.features.index
    train_mask = feature_index < split_date
    test_mask = feature_index >= split_date

    X_train_full, X_test_full = features[train_mask], features[test_mask]
    y_train_full, y_test_full = target[train_mask], target[test_mask]

    tscv = TimeSeriesSplit(n_splits=5, max_train_size=504)

    def objective(trial):
        # hyperparameter tuning
        n_estimators = trial.suggest_int('n_estimators', 100, 200)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.05, 0.5)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)


        accuracies = []
        # cross-validate only on the training portion
        for train_idx, test_idx in tscv.split(X_train_full):
            X_train, X_test = X_train_full[train_idx], X_train_full[test_idx]
            Y_train, Y_test = y_train_full[train_idx], y_train_full[test_idx]
            model = PredictorModelXG(
                n_estimators=n_estimators, 
                max_depth=max_depth, 
                learning_rate=learning_rate,
                min_child_weight=min_child_weight,
                subsample=subsample,
                colsample_bytree=colsample_bytree
            )

            model.fit(X_train, Y_train)
            Y_prediction = model.predict(X_test)
            accuracies.append(accuracy_score(Y_test, Y_prediction))

        return np.mean(accuracies)

    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=100)

    # now get the best parameters and train the final model
    best_n_estimators = study.best_params['n_estimators']
    best_max_depth = study.best_params['max_depth']
    best_learning_rate = study.best_params['learning_rate']
    best_min_child_weight = study.best_params['min_child_weight']
    best_subsample = study.best_params['subsample']
    best_colsample_bytree = study.best_params['colsample_bytree']
    print(study.best_params)

    model = PredictorModelXG(
        best_n_estimators, 
        best_max_depth, 
        best_learning_rate,
        best_min_child_weight,
        best_subsample,
        best_colsample_bytree
        )

    # train final model on all training data
    model.fit(X_train_full[-504:], y_train_full[-504:])

    # evaluate on held-out test data the model has never seen
    y_pred_test = model.predict(X_test_full)
    test_acc = accuracy_score(y_test_full, y_pred_test)
    print(f"Out-of-sample Accuracy: {test_acc:.3f}")
    print(f'length of y_test: {len(y_test_full)}')

    return model, data_handler, split_date
