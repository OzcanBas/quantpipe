import yfinance as yf
import pandas as pd
import numpy as np

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class DataHandler():
    def __init__(self, ticker, start="2015-01-01", end="2025-01-01"):
        self.ticker = ticker
        self.start = start
        self.end = end

    def __str__(self):
        return f"DataHandler({self.ticker})"

    def download_data(self):
        data = yf.download(
            self.ticker,
            start=self.start,
            end=self.end,
            interval="1d",
            auto_adjust=False,
            threads=True,
            progress=True,
        )

        if data is None or data.empty:
            raise ValueError(f"No data returned for {self.ticker}. Check ticker or network connection.")

        if isinstance(data.columns, pd.MultiIndex):
            if self.ticker in data.columns.get_level_values(-1):
                data = data.xs(self.ticker, axis=1, level=-1)
            elif self.ticker in data.columns.get_level_values(0):
                data = data.xs(self.ticker, axis=1, level=0)

        self.data = data

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
            data = yf.download(ticker, start=self.start, end=self.end)
            close = data["Close"]

            if isinstance(close, pd.DataFrame):
                close = close.squeeze()

            df[f"{prefix}_5d_returns"] = close.pct_change(5)
            df[f"{prefix}_20d_returns"] = close.pct_change(20)

        return df


    def clean_missing(self):
        missing_before = self.data.isna().sum().sum()

        if missing_before > 0:
            logger.info(f"{self.ticker}: Cleaning {missing_before} missing values using forward/backward fill.")

        self.data = self.data.ffill().bfill()

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

    def build_feature_frame(self, windows=[10, 20], target_horizon=1):
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
