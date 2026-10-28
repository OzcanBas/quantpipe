# Quantitative Trading Pipeline

This is a modular equity trading system that downloads historical price data, engineers technical and cross-asset features, trains a classification model to predict next-day price direction, and evaluates the strategy through walk-forward backtesting.

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

The first run downloads data from Yahoo Finance and caches the raw responses in `data/raw/` (git-ignored). Later runs load from the cache, so results are repeatable. Pass `refresh_cache=True` to `train_model` / `DataHandler` to re-download.

`experiments.ipynb` is an interactive, cell-by-cell version of `main.py`, plus a statistical comparison against a baseline model.

## Project Structure

| File | Description |
|------|-------------|
| `src/data.py` | Data ingestion, caching, cleaning, feature engineering, and the leak-free train/test split |
| `src/model.py` | Model definitions (logistic regression, XGBoost) |
| `src/train.py` | Hyperparameter tuning with date-based time-series cross-validation, and final model training |
| `src/backtest.py` | Walk-forward backtester with portfolio tracking and performance metrics |
| `src/utils.py` | Random seed setup for reproducibility |
| `main.py` | Entry point: trains the model and runs the backtest |
| `experiments.ipynb` | Notebook version of the pipeline, with baseline comparison and McNemar test |

## Features

**Technical indicators**, computed per window of 5, 10 and 20 days:
- **N-day returns**: percentage price change over n days, capturing short- to medium-term momentum
- **N-day rolling volatility**: standard deviation of daily returns over a rolling window, measuring recent price uncertainty
- **N-day price trend**: relative distance between the current price and its moving average `(Close - MA) / MA`, identifying overbought/oversold conditions relative to the recent trend
- **Volume ratio**: current volume relative to the n-day average, highlighting unusual trading activity

**Fixed-window indicators**:
- **RSI (14-day)**: relative strength index, comparing the magnitude of recent gains to losses to identify overbought (>70) or oversold (<30) conditions
- **MACD (12/26/9)**: difference between fast (12-day) and slow (26-day) exponential moving averages, plus its 9-day signal line and histogram, capturing momentum shifts

**Cross-asset signals** (SPY, VIX, TLT):
- 5-day and 20-day returns for each asset, providing macro context: broad market direction (SPY), implied volatility (VIX), and bond market risk appetite (TLT)

**Target**: `+1` if the next day's close is above today's close, otherwise `-1`. The horizon is set by `target_horizon` in `main.py`.

## Model

**XGBoost** is the model used by the pipeline. It is a gradient-boosted tree classifier with fixed regularisation (`reg_alpha=0.1`, `reg_lambda=5.0`), suited to nonlinear relationships between tabular features. A **logistic regression** class (`PredictorModel`) is included in `src/model.py` as a linear alternative, but the training pipeline does not use it.

**Training data**: the model is trained on a pooled panel of six large-cap tech stocks (AAPL, MSFT, GOOGL, AMZN, META, NVDA). Each stock's features come from its own price history. Pooling gives about 6× more training rows than a single stock. Testing and backtesting use only the target ticker (AAPL).

**Hyperparameter tuning**: Optuna (TPE sampler, seeded, 100 trials) tunes `n_estimators`, `max_depth`, `learning_rate`, `min_child_weight`, `subsample` and `colsample_bytree` to maximise mean cross-validation accuracy. Cross-validation uses 5 folds split by **calendar date**, not row position, so all tickers' rows for a given day land in the same fold. Each training window is capped at the most recent 504 trading days (about 2 years). Training rows whose label depends on a price inside the validation fold are removed, to prevent leakage.

**Train/test split**:
- Hyperparameters are tuned on 2015–2022 data.
- The final model is fit on the most recent 504 trading days before the split (2021–2022), pooled across all tickers.
- The model is evaluated out of sample on AAPL from 2023-01-01 to 2025-01-01.

Training rows whose label would use a price after the split date are excluded.

Tuned hyperparameters: `n_estimators=150`, `max_depth=9`, `learning_rate=0.112`, `min_child_weight=4`, `subsample=0.754`, `colsample_bytree=0.793` (best CV accuracy 52.1%).

## Backtest

- A signal is generated from each day's close and executed at the **next day's open**, to avoid look-ahead bias.
- Long on a `+1` prediction, short on `-1`. Because the model always predicts one of the two, the strategy is always in the market.
- 95% of portfolio value is allocated to each position. Transaction cost is 1 bp per side. Starting capital is $100,000.

## Results

Out-of-sample performance (2023–2024), AAPL, 501 trading days:

| Metric | Strategy | Buy & hold AAPL |
|--------|----------|-----------------|
| Total return | 28.52% | 98.78% |
| Annualised return | 13.51% | — |
| Annualised volatility | 22.07% | 23.04% |
| Sharpe ratio | 0.684 | 1.622 |
| Maximum drawdown | -29.30% | -16.50% |
| Beta (vs S&P 500) | -0.178 | — |
| Alpha (CAPM, cumulative over the period) | 28.38% | — |
| Trades (winning / losing) | 193 (100 / 93) | — |

**Classification accuracy vs a no-information baseline** (from `experiments.ipynb`):

| Model | Out-of-sample accuracy |
|-------|------------------------|
| XGBoost | 50.9% |
| Baseline: always predict the training majority class (up) | 56.1% |

The test period had 281 up days and 220 down days. On the 266 days where exactly one of the two models was right, XGBoost was right on 120 and the baseline on 146. An exact McNemar test gives p = 0.125, so there is **no statistically significant difference** between the trained model and the baseline.

**Interpretation**: the model shows no out-of-sample predictive edge. The positive strategy return and alpha should not be read as skill:
- The beta is negative because the strategy was frequently short during a strong bull market.
- With a negative beta, CAPM's expected return is close to zero, so the alpha is almost equal to the total return.
- Buy-and-hold beat the strategy on total return, Sharpe ratio and drawdown.
- A 100/93 win/loss split over 193 trades is consistent with chance in a single two-year window.

The value of the project is in the pipeline itself (leak-free pooled cross-validation, reproducible data, a realistic execution assumption, and honest statistical evaluation), not in the strategy's returns.

## Evaluation

- **Sharpe ratio**: annualised mean daily return divided by annualised volatility. It is computed without subtracting the risk-free rate.
- **Maximum drawdown**: largest peak-to-trough decline in portfolio value, quantifying worst-case downside.
- **Beta / alpha**: beta is computed against S&P 500 (`^GSPC`) daily returns. Alpha is the CAPM excess return over the whole backtest period, using the average 10-year Treasury yield (`^TNX`) as the risk-free rate.
- **Buy-and-hold comparison**: benchmark to determine whether the model adds value beyond passive exposure.
- **Baseline + McNemar test**: checks whether the model's accuracy is significantly different from always predicting the majority class.

## Limitations

**Data**
- **Small dataset.** Daily data from 2015 to 2025 gives about 2,500 rows per stock. After the test split and the 504-day training window, the final model sees only about 3,000 pooled rows. Financial returns have a very low signal-to-noise ratio, so this is too little data to learn a reliable pattern.
- **Correlated, hand-picked training universe.** The six pooled stocks are all large-cap US tech and move together, so the effective sample size is much smaller than 6×. A model trained on them may not generalise to other sectors or stocks.
- **Limited feature set.** Only price- and volume-based technical indicators plus three cross-asset returns are used. There are no fundamentals, earnings dates, news or sentiment, options data, or macro releases. These are often what actually moves a stock.
- **Free data source.** Yahoo Finance via `yfinance` is an unofficial API. 

**Model and target**
- **Model choice is constrained by data size.** XGBoost on tabular features was chosen because neural networks and sequence models (LSTMs, transformers) need far more data to avoid overfitting. Tree models still don't capture temporal structure beyond what the hand-built features encode.
- **The next-day direction target is very noisy.** Day-to-day price moves are dominated by noise. A longer horizon (e.g. 5 days, already supported through `target_horizon`) may give a smoother, more predictable target. The backtester would then need a matching holding period, because it currently re-trades daily.
- **Direction only, not magnitude.** Every day is weighted equally whether the move is 0.1% or 5%. Days where the close is unchanged are labelled as down. The model can be right most days and still lose money on the few large moves.
- **No confidence threshold.** The model's predicted probabilities are not used. The strategy goes fully long or short even when the model is near 50/50, instead of staying flat when uncertain.
- **Overfitting risk from hyperparameter search.** 100 trials maximising a noisy CV accuracy (52.1%) will partly fit noise, so the best CV score is optimistic. The selected `max_depth=9` is a deep tree for this little data. The tuning objective (accuracy) also isn't the same as the goal (risk-adjusted profit).

**Evaluation and backtest**
- **Single test period and single stock.** There is one two-year out-of-sample window (a strong bull market for AAPL) with no rolling retraining. The results are highly regime-dependent and not enough to judge the strategy's performance.
- **Label and trade mismatch.** The model predicts close-to-close direction, but trades are executed open-to-open. The overnight gap is part of the realised return but not of the label.
- **Optimistic trading assumptions.** Costs are 1 bp per side, with no bid-ask spread, slippage or market impact. Short-sale borrow fees, margin requirements and interest on cash are ignored. Fractional shares are allowed.
- **Metric simplifications.** The Sharpe ratio doesn't subtract the risk-free rate. The risk-free proxy is the 10-year Treasury yield rather than a short-term T-bill rate. Alpha is cumulative over the period, not annualised.

## Future Work

- Predict a 5-day horizon with a matching holding period in the backtester
- Use predicted probabilities to stay flat when the model is uncertain
- Rolling walk-forward retraining over multiple test windows
- Daily prediction mode with rolling retraining on live market data
- Multi-asset portfolio support with position sizing
- Integration with portfolio optimisation project (Anderson Acceleration-based constrained optimisation)
