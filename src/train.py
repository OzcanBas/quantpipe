from .data import build_pooled_dataset
from .model import PredictorModelXG
from .utils import set_seed
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import optuna

import numpy as np

set_seed(42)


DEFAULT_POOL_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA"]
ROLLING_WINDOW_DAYS = 504  # ~2 trading years


def _date_rolling_splits(dates, n_splits, max_train_days):
    """TimeSeriesSplit over unique calendar dates, mapped back to row indices.

    Needed because the pooled panel has multiple tickers sharing the same
    date - a plain TimeSeriesSplit over row order would not respect
    chronology once rows aren't one-per-day. max_train_days keeps the same
    "trading days" meaning as before, regardless of how many tickers are
    pooled.
    """
    unique_dates = np.sort(dates.unique())
    date_tscv = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_days)

    for train_date_idx, test_date_idx in date_tscv.split(unique_dates):
        train_dates = set(unique_dates[train_date_idx])
        test_dates = set(unique_dates[test_date_idx])
        train_mask = dates.isin(train_dates).to_numpy()
        test_mask = dates.isin(test_dates).to_numpy()
        yield np.where(train_mask)[0], np.where(test_mask)[0]


def train_model(
        target_ticker="AAPL",
        pool_tickers=None,
        windows=None,
        split_date="2023-01-01",
        refresh_cache=False
        ):
    if windows is None:
        windows = [10]
    if pool_tickers is None:
        pool_tickers = DEFAULT_POOL_TICKERS
    if target_ticker not in pool_tickers:
        pool_tickers = [target_ticker] + list(pool_tickers)

    # build features for every pooled ticker (each computed from its own price history)
    pooled, feature_cols, handlers = build_pooled_dataset(
        pool_tickers,
        windows,
        refresh_cache=refresh_cache,
    )
    target_handler = handlers[target_ticker]

    # apply the SAME cutoff date to every ticker, not just the target, so no
    # ticker's post-split-date data (which would reveal the test period's
    # market regime) can leak into training
    train_pool = pooled[pooled["Date"] < split_date].reset_index(drop=True)
    target_test = pooled[(pooled["ticker"] == target_ticker) & (pooled["Date"] >= split_date)].reset_index(drop=True)

    X_train_full = train_pool[feature_cols].to_numpy()
    y_train_full = train_pool["target"].to_numpy().astype(int)
    train_dates = train_pool["Date"]

    X_test_full = target_test[feature_cols].to_numpy()
    y_test_full = target_test["target"].to_numpy().astype(int)

    def objective(trial):
        # hyperparameter tuning
        n_estimators = trial.suggest_int('n_estimators', 100, 200)
        max_depth = trial.suggest_int('max_depth', 2, 10)
        learning_rate = trial.suggest_float('learning_rate', 0.05, 0.5)
        min_child_weight = trial.suggest_int('min_child_weight', 1, 20)
        subsample = trial.suggest_float('subsample', 0.5, 1.0)
        colsample_bytree = trial.suggest_float('colsample_bytree', 0.5, 1.0)

        accuracies = []
        # cross-validate only on the training portion, split by date
        for train_idx, test_idx in _date_rolling_splits(train_dates, n_splits=5, max_train_days=ROLLING_WINDOW_DAYS):
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

    print(study.best_params)

    model = PredictorModelXG(**study.best_params)

    # train final model on the most recent ROLLING_WINDOW_DAYS calendar dates,
    # pooled across all tickers
    recent_dates = np.sort(train_dates.unique())[-ROLLING_WINDOW_DAYS:]
    recent_mask = train_dates.isin(recent_dates).to_numpy()
    model.fit(X_train_full[recent_mask], y_train_full[recent_mask])

    # evaluate on the target ticker's held-out test data the model has never seen
    y_pred_test = model.predict(X_test_full)
    test_acc = accuracy_score(y_test_full, y_pred_test)
    print(f"Out-of-sample Accuracy ({target_ticker}): {test_acc:.3f}")
    print(f"length of y_test: {len(y_test_full)}")

    return model, target_handler, split_date
