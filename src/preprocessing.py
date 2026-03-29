"""
src/preprocessing.py — Feature scaling, sequence building, train/val/test splits
"""
import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass
from sklearn.preprocessing import (
    RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer
)
import joblib
from pathlib import Path
from loguru import logger

from .features.technical_indicators import TechnicalIndicators

@dataclass
class ProcessedData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: List[str]
    scaler: object
    target_scaler: object
    train_dates: pd.DatetimeIndex
    val_dates: pd.DatetimeIndex
    test_dates: pd.DatetimeIndex
    raw_close: pd.Series

class MarketPreprocessor:
    SCALER_TYPES = {
        "robust": RobustScaler,
        "minmax": MinMaxScaler,
        "standard": StandardScaler,
        "quantile": QuantileTransformer,
    }

    def __init__(
        self,
        window: int = 60,
        horizon: int = 1,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        feature_scaler: str = "robust",
        target_scaler: str = "minmax",
        scaler_save_path: Optional[Path] = None,
    ):
        self.window = window
        self.horizon = horizon
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.feature_scaler_type = feature_scaler
        self.target_scaler_type = target_scaler
        self.scaler_save_path = scaler_save_path
        self._feature_scaler = None
        self._target_scaler = None

    def fit_transform(self, df: pd.DataFrame, target_col: str = "Close") -> ProcessedData:
        logger.info("Starting preprocessing pipeline ...")

        # Build derived features before sequence construction.
        df = TechnicalIndicators.add_all(df)
        df = self._add_lag_features(df, target_col, lags=[1, 2, 3, 5, 10, 20])
        df = self._add_rolling_stats(df, target_col)
        df = self._add_calendar_features(df)

        initial_len = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_len - len(df)} NaN rows after feature engineering.")

        # Exclude raw/target fields from model features.
        exclude = [target_col, "Open", "High", "Low", "Volume",
                   "spike_flag", "log_volume", "returns", "log_returns"]
        feature_cols = [c for c in df.columns if c not in exclude and not c.startswith("Adj")]

        df[feature_cols] = self._cap_outliers(df[feature_cols])

        # Convert to contiguous float arrays for downstream transforms.
        X_raw = df[feature_cols].values.astype(np.float32)
        y_raw = df[target_col].values.astype(np.float32)

        # Build sequences before splitting to preserve temporal ordering.
        X_seq, y_seq, dates_seq = self._build_sequences(X_raw, y_raw, df.index)

        # Split chronologically into train/validation/test.
        n_seq = len(X_seq)
        train_end = int(n_seq * self.train_ratio)
        val_end = int(n_seq * (self.train_ratio + self.val_ratio))

        X_train_raw, y_train_raw = X_seq[:train_end], y_seq[:train_end]
        X_val_raw, y_val_raw = X_seq[train_end:val_end], y_seq[train_end:val_end]
        X_test_raw, y_test_raw = X_seq[val_end:], y_seq[val_end:]

        # Fit scalers on train split only to avoid leakage.
        self._feature_scaler = self.SCALER_TYPES[self.feature_scaler_type]()
        self._target_scaler = self.SCALER_TYPES[self.target_scaler_type](feature_range=(0, 1)) \
            if self.target_scaler_type == "minmax" else self.SCALER_TYPES[self.target_scaler_type]()

        n_features = len(feature_cols)
        
        # Scale feature tensors.
        X_train_scaled = self._feature_scaler.fit_transform(X_train_raw.reshape(-1, n_features)).reshape(X_train_raw.shape)
        X_val_scaled = self._feature_scaler.transform(X_val_raw.reshape(-1, n_features)).reshape(X_val_raw.shape)
        X_test_scaled = self._feature_scaler.transform(X_test_raw.reshape(-1, n_features)).reshape(X_test_raw.shape)

        # Scale regression targets.
        y_train_scaled = self._target_scaler.fit_transform(y_train_raw.reshape(-1, 1)).ravel()
        y_val_scaled = self._target_scaler.transform(y_val_raw.reshape(-1, 1)).ravel()
        y_test_scaled = self._target_scaler.transform(y_test_raw.reshape(-1, 1)).ravel()

        if self.scaler_save_path:
            self._save_scalers(self.scaler_save_path)

        return ProcessedData(
            X_train=X_train_scaled, y_train=y_train_scaled,
            X_val=X_val_scaled,     y_val=y_val_scaled,
            X_test=X_test_scaled,   y_test=y_test_scaled,
            feature_names=feature_cols,
            scaler=self._feature_scaler,
            target_scaler=self._target_scaler,
            train_dates=dates_seq[:train_end],
            val_dates=dates_seq[train_end:val_end],
            test_dates=dates_seq[val_end:],
            raw_close=df[target_col],
        )

    def inverse_transform_target(self, y: np.ndarray) -> np.ndarray:
        return self._target_scaler.inverse_transform(y.reshape(-1, 1)).ravel()

    def _build_sequences(self, X: np.ndarray, y: np.ndarray, index: pd.DatetimeIndex):
        seq_X, targ_y, dates = [], [], []
        for i in range(self.window, len(X) - self.horizon + 1):
            seq_X.append(X[i - self.window: i])
            targ_y.append(y[i + self.horizon - 1])
            dates.append(index[i + self.horizon - 1])
        return np.array(seq_X, dtype=np.float32), np.array(targ_y, dtype=np.float32), pd.DatetimeIndex(dates)

    def _add_lag_features(self, df: pd.DataFrame, col: str, lags: List[int]) -> pd.DataFrame:
        for lag in lags:
            df[f"lag_{col}_{lag}"] = df[col].shift(lag)
            df[f"lag_ret_{lag}"] = df["returns"].shift(lag)
        return df

    def _add_rolling_stats(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        for window in [5, 10, 20, 60]:
            df[f"roll_mean_{window}"] = df[col].rolling(window).mean()
            df[f"roll_std_{window}"] = df[col].rolling(window).std()
            df[f"roll_skew_{window}"] = df[col].rolling(window).skew()
            df[f"roll_kurt_{window}"] = df[col].rolling(window).kurt()
            df[f"roll_max_{window}"] = df[col].rolling(window).max()
            df[f"roll_min_{window}"] = df[col].rolling(window).min()
        return df

    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["cal_day_of_week"] = df.index.dayofweek / 4.0
        df["cal_month"] = (df.index.month - 1) / 11.0
        df["cal_quarter"] = (df.index.quarter - 1) / 3.0
        df["cal_day_of_year_sin"] = np.sin(2 * np.pi * df.index.dayofyear / 365)
        df["cal_day_of_year_cos"] = np.cos(2 * np.pi * df.index.dayofyear / 365)
        df["cal_week_sin"] = np.sin(2 * np.pi * df.index.dayofweek / 5)
        df["cal_week_cos"] = np.cos(2 * np.pi * df.index.dayofweek / 5)
        return df

    def _cap_outliers(self, df: pd.DataFrame, lower_q=0.01, upper_q=0.99) -> pd.DataFrame:
        lower = df.quantile(lower_q)
        upper = df.quantile(upper_q)
        return df.clip(lower=lower, upper=upper, axis=1)

    def _save_scalers(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._feature_scaler, path / "feature_scaler.pkl")
        joblib.dump(self._target_scaler, path / "target_scaler.pkl")

    def load_scalers(self, path: Path):
        self._feature_scaler = joblib.load(path / "feature_scaler.pkl")
        self._target_scaler = joblib.load(path / "target_scaler.pkl")