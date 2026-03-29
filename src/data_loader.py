"""
src/data_loader.py — Multi-source financial data ingestion with caching
"""
import os
import time
import hashlib
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import yfinance as yf
from loguru import logger

warnings.filterwarnings("ignore")


class MarketDataLoader:
    """
    Fetches OHLCV data from multiple sources with:
    - Disk-based caching (avoid redundant API calls)
    - Data validation & gap filling
    - Corporate action adjustment
    - Multi-ticker batch download
    """

    CACHE_DIR = Path("data/raw")
    CACHE_EXPIRY_HOURS = 24

    def __init__(self, cache: bool = True, cache_dir: Optional[Path] = None):
        self.cache = cache
        self.cache_dir = cache_dir or self.CACHE_DIR
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ──────────────────────────────────────────────────────────────

    def load(
        self,
        ticker: str,
        start: str = "2018-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
        adjust: bool = True,
    ) -> pd.DataFrame:
        """Load single ticker OHLCV data."""
        end = end or datetime.today().strftime("%Y-%m-%d")
        cache_key = self._cache_key(ticker, start, end, interval)
        cache_path = self.cache_dir / f"{cache_key}.parquet"

        if self.cache and self._is_cache_valid(cache_path):
            logger.info(f"[CACHE HIT] {ticker} ({start} → {end})")
            df = pd.read_parquet(cache_path)
        else:
            logger.info(f"[FETCHING] {ticker} from yfinance ...")
            df = self._fetch_yfinance(ticker, start, end, interval, adjust)
            if self.cache:
                df.to_parquet(cache_path)

        df = self._validate_and_clean(df, ticker)
        return df

    def load_batch(
        self,
        tickers: List[str],
        start: str = "2018-01-01",
        end: Optional[str] = None,
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """Load multiple tickers in parallel."""
        end = end or datetime.today().strftime("%Y-%m-%d")
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.load(ticker, start, end, interval)
            except Exception as e:
                logger.warning(f"Failed to load {ticker}: {e}")
        return result

    def load_market_context(
        self, start: str = "2018-01-01", end: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load macro/market context indicators:
        - SPY (S&P 500 proxy)
        - QQQ (NASDAQ proxy)
        - VIX (fear index)
        - TLT (bond proxy)
        - DXY (dollar index via UUP)
        - GLD (gold)
        """
        context_tickers = {"SPY": "sp500", "QQQ": "nasdaq", "^VIX": "vix",
                           "TLT": "bonds", "UUP": "dxy", "GLD": "gold"}
        frames = {}
        for tkr, name in context_tickers.items():
            try:
                df = self.load(tkr, start, end)
                frames[name] = df["Close"].rename(name)
            except Exception as e:
                logger.warning(f"Could not load context ticker {tkr}: {e}")

        if frames:
            ctx = pd.concat(frames.values(), axis=1)
            ctx = ctx.ffill().bfill()
            # Compute returns for stationarity
            ctx_ret = ctx.pct_change().add_suffix("_ret")
            return pd.concat([ctx, ctx_ret], axis=1)
        return pd.DataFrame()

    def load_with_context(
        self,
        ticker: str,
        start: str = "2018-01-01",
        end: Optional[str] = None,
    ) -> pd.DataFrame:
        """Load ticker OHLCV merged with market context features."""
        ticker_df = self.load(ticker, start, end)
        ctx_df = self.load_market_context(start, end)
        if not ctx_df.empty:
            merged = ticker_df.join(ctx_df, how="left")
            merged = merged.ffill().bfill()
            return merged
        return ticker_df

    # ── Private helpers ─────────────────────────────────────────────────────────

    def _fetch_yfinance(
        self, ticker: str, start: str, end: str, interval: str, adjust: bool
    ) -> pd.DataFrame:
        """
        Use yf.download() — more reliable than yf.Ticker().history() for
        futures (GC=F, CL=F), crypto (BTC-USD), and equities alike.
        yf.Ticker().history() can return None internally on newer yfinance
        versions when the chart API returns an empty response.
        """
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=adjust,
            progress=False,
            threads=False,
        )

        if df is None or df.empty:
            raise ValueError(
                f"No data returned for '{ticker}'. "
                f"Check the symbol is correct and you have internet access."
            )

        # yf.download with a single ticker returns MultiIndex columns in newer
        # versions — flatten: ('Close', 'AAPL') -> 'Close'
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [lvl0 for lvl0, _ in df.columns]

        # Normalise index timezone
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # Capitalise column names
        df.columns = [str(c).strip().capitalize() for c in df.columns]

        # Futures & some ETFs can have zero/missing Volume
        if "Volume" not in df.columns:
            df["Volume"] = 0
        else:
            df["Volume"] = df["Volume"].fillna(0)

        for col in ["Open", "High", "Low", "Close"]:
            if col not in df.columns:
                raise ValueError(f"Column '{col}' missing for {ticker}. Got: {list(df.columns)}")

        return df[["Open", "High", "Low", "Close", "Volume"]].copy()

    def _validate_and_clean(self, df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Remove zero prices, fill small gaps, flag anomalies."""
        # Remove rows with zero/negative prices
        price_cols = ["Open", "High", "Low", "Close"]
        mask = (df[price_cols] <= 0).any(axis=1)
        if mask.sum() > 0:
            logger.warning(f"{ticker}: Removed {mask.sum()} rows with non-positive prices")
            df = df[~mask]

        # Asset-specific gap fill:
        # Crypto (BTC-USD) trades 24/7 — use calendar days, not business days.
        # Equities / futures — use business-day frequency.
        is_crypto = "BTC" in ticker.upper() or "ETH" in ticker.upper()
        freq = "D" if is_crypto else "B"
        try:
            df = df.asfreq(freq)
        except Exception:
            pass   # sparse index; skip resampling
        df = df.ffill(limit=3)

        # Spike threshold: crypto moves 20 %+ legitimately; use 40 % for crypto
        spike_thresh = 0.40 if is_crypto else 0.20
        returns = df["Close"].pct_change().abs()
        spikes = returns > spike_thresh
        if spikes.sum() > 0:
            logger.warning(
                f"{ticker}: {spikes.sum()} spikes detected (>{spike_thresh*100:.0f}% daily)"
            )
        df["spike_flag"] = spikes.astype(int)

        # Log-transform volume for better distribution
        df["log_volume"] = np.log1p(df["Volume"])

        # Add basic return columns
        df["returns"] = df["Close"].pct_change()
        df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))

        df.dropna(subset=["Close", "returns"], inplace=True)
        return df

    def _cache_key(self, ticker: str, start: str, end: str, interval: str) -> str:
        raw = f"{ticker}_{start}_{end}_{interval}"
        return hashlib.md5(raw.encode()).hexdigest()[:12] + f"_{ticker.replace('^','')}"

    def _is_cache_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        return age_hours < self.CACHE_EXPIRY_HOURS


class MultiTimeframeLoader(MarketDataLoader):
    """
    Extends base loader to support multi-timeframe analysis.
    Aligns 1d, 1wk, 1mo data into a single synchronized DataFrame.
    """

    TIMEFRAMES = {"daily": "1d", "weekly": "1wk", "monthly": "1mo"}

    def load_multi_timeframe(
        self, ticker: str, start: str = "2018-01-01"
    ) -> Dict[str, pd.DataFrame]:
        result = {}
        for name, interval in self.TIMEFRAMES.items():
            try:
                df = self.load(ticker, start=start, interval=interval)
                result[name] = df
            except Exception as e:
                logger.warning(f"Could not load {interval} for {ticker}: {e}")
        return result

    def align_timeframes(
        self, frames: Dict[str, pd.DataFrame], base: str = "daily"
    ) -> pd.DataFrame:
        """Align weekly/monthly features to daily index via forward-fill."""
        base_df = frames[base].copy()
        for tf_name, df in frames.items():
            if tf_name == base:
                continue
            suffix = f"_{tf_name[:1].upper()}"  # _W or _M
            renamed = df.add_suffix(suffix)
            base_df = base_df.join(renamed, how="left").ffill()
        return base_df
