# Quantitative Trading Pipeline

This is a modular equity trading system that downloads historical price data, engineers technical and cross-asset features, trains a classification model to predict next-day price direction, and evaluates the strategy through walk-forward backtesting.

## Quickstart

```bash
pip install -r requirements.txt
python main.py
```

## Project Structure

| File | Description |
|------|-------------|
| `data.py` | Data ingestion, cleaning, and feature engineering |
| `model.py` | Model definitions (logistic regression, XGBoost) |
| `train.py` | Training pipeline with time-series cross-validation |
| `backtest.py` | Walk-forward backtester with portfolio tracking |
| `main.py` | Entry point — trains the model and runs the backtest |

## Features

**Technical indicators** (computed per window of 5, 10, 20 days):
- **N-day returns** — percentage price change over n days, capturing short- to medium-term momentum
- **N-day rolling volatility** — standard deviation of daily returns over a rolling window, measuring recent price uncertainty
- **N-day price trend** — relative distance between the current price and its moving average `(Close - MA) / MA`, identifying overbought/oversold conditions relative to the recent trend
- **RSI** — relative strength index, comparing the magnitude of recent gains to losses to identify overbought (>70) or oversold (<30) conditions
- **MACD** — difference between fast (12-day) and slow (26-day) exponential moving averages, capturing momentum shifts
- **Volume ratio** — current volume relative to the n-day average, highlighting unusual trading activity

**Cross-asset signals** (SPY, VIX, TLT):
- 5-day and 20-day returns for each asset, providing macro context — broad market direction (SPY), implied volatility (VIX), and bond market risk appetite (TLT)

## Model

The pipeline supports two classifiers:
- **Logistic regression** — linear baseline, requires feature scaling
- **XGBoost** (default) — gradient-boosted trees with regularisation (`max_depth=2`, `reg_lambda=5.0`), better suited to the nonlinear relationships in financial features

Training uses `TimeSeriesSplit` (5 folds) to respect temporal ordering — future data never leaks into training folds. The model is trained on 2015–2023 and evaluated out-of-sample on 2023–2025.

## Results

Out-of-sample performance (2023–2025), AAPL:

| Metric | Value |
|--------|-------|
| Out-of-sample accuracy | 51.3% |
| Strategy return | 12.26% |
| Buy & hold return | 101.65% |
| Sharpe ratio | 0.39 |
| Maximum drawdown | -20.89% |

The model achieves a positive but modest edge. The strategy underperforms buy-and-hold during a strong bull market, which is expected for a daily direction model — the value is in risk-adjusted returns and the ability to go short during drawdowns.

## Evaluation

- **Sharpe ratio** — annualised return per unit of volatility, the standard risk-adjusted performance measure
- **Maximum drawdown** — largest peak-to-trough decline in portfolio value, quantifying worst-case downside
- **Buy-and-hold comparison** — benchmark to determine whether the model adds value beyond passive exposure

## Future Work

- Daily prediction mode with rolling retraining on live market data
- Multi-asset portfolio support with position sizing
- Integration with portfolio optimisation project (Anderson Acceleration-based constrained optimisation)
