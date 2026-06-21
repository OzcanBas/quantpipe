from .data import DataHandler
from .model import PredictorModelXG
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

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

    model = PredictorModelXG()
    accuracies = []

    # cross-validate only on the training portion
    tscv = TimeSeriesSplit(n_splits=5)
    for train_idx, test_idx in tscv.split(X_train_full):
        X_train, X_test = X_train_full[train_idx], X_train_full[test_idx]
        Y_train, Y_test = y_train_full[train_idx], y_train_full[test_idx]
        model.fit(X_train, Y_train)
        Y_prediction = model.predict(X_test)
        accuracies.append(accuracy_score(Y_test, Y_prediction))

    print(f"Average CV Accuracy: {np.mean(accuracies):.3f}")
    print(f"Per-fold Accuracies: {[f'{a:.3f}' for a in accuracies]}")

    # train final model on all training data
    model.fit(X_train_full, y_train_full)

    # evaluate on held-out test data the model has never seen
    y_pred_test = model.predict(X_test_full)
    test_acc = accuracy_score(y_test_full, y_pred_test)
    print(f"Out-of-sample Accuracy: {test_acc:.3f}")

    return model, data_handler, split_date
