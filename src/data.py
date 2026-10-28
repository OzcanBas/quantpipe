import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = PROJECT_ROOT / "data" / "raw"


def _cache_key(ticker, start, end, interval, auto_adjust):
    safe_ticker = ticker.replace("^", "idx_").replace("/", "_").replace("=", "_")
    adjust = "adjusted" if auto_adjust else "unadjusted"
    return f"{safe_ticker}_{start}_{end}_{interval}_{adjust}.pkl"


def _normalize_downloaded_data(data, ticker):
    if isinstance(data.columns, pd.MultiIndex):
        if ticker in data.columns.get_level_values(-1):
            data = data.xs(ticker, axis=1, level=-1)
        elif ticker in data.columns.get_level_values(0):
            data = data.xs(ticker, axis=1, level=0)

    return data.sort_index()


def load_price_data(
        ticker,
        start="2015-01-01",
        end="2025-01-01",
        interval="1d",
        auto_adjust=False,
        refresh_cache=False
        ):
    """Load raw Yahoo Finance data, caching the exact response for repeatable runs."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / _cache_key(ticker, start, end, interval, auto_adjust)

    if cache_path.exists() and not refresh_cache:
        logger.info(f"{ticker}: Loading raw data from cache: {cache_path}")
        return pd.read_pickle(cache_path)

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=auto_adjust,
        threads=False,
        progress=True,
    )

    if data is None or data.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker or network connection.")

    data = _normalize_downloaded_data(data, ticker)
    data.to_pickle(cache_path)
    logger.info(f"{ticker}: Cached raw data at {cache_path}")
    return data


class DataHandler():
    def __init__(self, ticker, start="2015-01-01", end="2025-01-01", refresh_cache=False):
        self.ticker = ticker
        self.start = start
        self.end = end
        self.refresh_cache = refresh_cache

    def __str__(self):
        return f"DataHandler({self.ticker})"

    def download_data(self):
        self.data = load_price_data(
            self.ticker,
            start=self.start,
            end=self.end,
            interval="1d",
            auto_adjust=False,
            refresh_cache=self.refresh_cache,
        )

    def n_day_returns(self, n):
        return self.data["Close"].pct_change(n)

    def n_day_rolling_mean(self, n):
        return self.data["Close"].rolling(n).mean()

    def n_day_rolling_volatility(self, n):
        daily_return = self.data["Close"].pct_change()
        n_day_rolling_volatility = daily_return.rolling(n).std()
        return n_day_rolling_volatility

    def n_day_price_trend(self, n):
        moving_average = self.n_day_rolling_mean(n)
        return (self.data["Close"] - moving_average)/moving_average
    
    def rsi(self, n=14):
        delta = self.data["Close"].diff()
        gains = delta.clip(lower=0)
        losses = -delta.clip(upper=0)
        avg_gain = gains.ewm(span=n, min_periods=n).mean()
        avg_loss = losses.ewm(span=n, min_periods=n).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def macd(self, fast=12, slow=26, signal=9):
        ema_fast = self.data["Close"].ewm(span=fast, min_periods=fast).mean()
        ema_slow = self.data["Close"].ewm(span=slow, min_periods=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
        return macd_line, signal_line, macd_line - signal_line
    
    def volume_ratio(self, n):
        return self.data["Volume"] / self.data["Volume"].rolling(n).mean()
    
    def add_cross_asset_features(self, df):
        cross_assets = {
            "SPY": "spy",
            "^VIX": "vix",
            "TLT": "tlt"
        }

        for ticker, prefix in cross_assets.items():
            data = load_price_data(
                ticker,
                start=self.start,
                end=self.end,
                interval="1d",
                auto_adjust=False,
                refresh_cache=self.refresh_cache,
            )
            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.squeeze()

            df[f"{prefix}_5d_returns"] = close.pct_change(5)
            df[f"{prefix}_20d_returns"] = close.pct_change(20)

        return df


    def clean_missing(self):
        missing_before = self.data.isna().sum().sum()

        if missing_before > 0:
            logger.info(f"{self.ticker}: Cleaning {missing_before} missing values using forward fill.")

        self.data = self.data.ffill()

        missing_after = self.data.isna().sum().sum()

        if missing_after == 0:
            logger.info(f"{self.ticker}: Missing data cleaned successfully.")
        else:
            logger.warning(f"{self.ticker}: Still has {missing_after} missing values after cleaning.")

        return self.data


    def log_outliers(self, n=20, threshold=1):
        df = self.data.copy()
        returns = df["Close"].pct_change()

        rolling_mean = returns.rolling(n).mean()
        rolling_std = returns.rolling(n).std()

        z_score = (returns - rolling_mean)/rolling_std
        df["z_score"] = z_score

        outliers = df[df["z_score"].abs() > threshold]

        return outliers
    
    def create_target_column(self, horizon=1):
        future_prices = self.data["Close"].shift(-horizon)
        target = pd.Series(index=self.data.index, dtype="float")
        target[future_prices > self.data["Close"]] = 1
        target[future_prices <= self.data["Close"]] = -1
        target[future_prices.isna()] = np.nan
        return target

    def build_feature_frame(self, windows=(10, 20), target_horizon=1):
        self.clean_missing()
        df = self.data.copy()

        for w in windows:
            df[f"{w}d_returns"] = self.n_day_returns(w)
            df[f"{w}d_rolling_volatility"] = self.n_day_rolling_volatility(w)
            df[f"{w}d_price_trend"] = self.n_day_price_trend(w)
            df[f"{w}d_volume"] = self.volume_ratio(w)

        df["rsi"] = self.rsi(14)
        macd_line, signal_line, macd_hist = self.macd()
        df["macd"] = macd_line
        df["macd_signal"] = signal_line
        df["macd_hist"] = macd_hist

        self.add_cross_asset_features(df)

        df["target"] = self.create_target_column(target_horizon)

        feature_cols = [c for c in df.columns if c not in self.data.columns and c != "target"]
        self.features = df[feature_cols + ["target"]].dropna()
        return self.features

    def build_feature_matrix(self, windows=[10, 20], target_horizon=1):
        df_model = self.build_feature_frame(windows, target_horizon)
        feature_cols = [c for c in df_model.columns if c != "target"]
        X = df_model[feature_cols].to_numpy()
        y = df_model["target"].to_numpy().astype(int)
        return X, y

def build_train_test_dataset(
        target_ticker,
        tickers,
        windows=(10, 20),
        target_horizon=1,
        start_date="2015-01-01",
        end_date="2025-01-01",
        split_date="2023-01-01",
        refresh_cache=False
    ):
    """Build features from full history, then split off leak-free training rows."""

    if target_horizon < 1:
        raise ValueError("target_horizon must be at least 1")

    frames = []
    handlers = {}
    split_timestamp = pd.Timestamp(split_date)

    for ticker in tickers:
        handler = DataHandler(ticker, start=start_date, end=end_date, refresh_cache=refresh_cache)
        handler.download_data()
        feat = handler.build_feature_frame(windows, target_horizon).copy()
        target_dates = pd.Series(handler.data.index, index=handler.data.index).shift(-target_horizon)
        feat["target_date"] = target_dates.reindex(feat.index)
        feat.index.name = "Date"
        feat = feat.reset_index()
        feat["ticker"] = ticker

        # A target uses a price target_horizon trading rows later. Exclude
        # the final training rows for this ticker so their labels stay before
        # the test period, even when the split falls on a weekend or holiday.
        pre_split_dates = handler.data.index[handler.data.index < split_timestamp]
        safe_train_dates = pre_split_dates[:-target_horizon]
        feat["train_eligible"] = feat["Date"].isin(safe_train_dates)

        frames.append(feat)
        handlers[ticker] = handler

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.sort_values(["Date", "ticker"]).reset_index(drop=True)

    feature_cols = [c for c in pooled.columns if c not in ("Date", "ticker", "target", "target_date", "train_eligible")]

    pooled_train = pooled[pooled["train_eligible"]].drop(columns="train_eligible").reset_index(drop=True)
    pooled_test = pooled[(pooled["ticker"] == target_ticker) & (pooled["Date"] >= split_timestamp)]
    pooled_test = pooled_test.drop(columns="train_eligible").reset_index(drop=True)

    return pooled_train, pooled_test, feature_cols, handlers
