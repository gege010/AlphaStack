"""
src/features/technical_indicators.py
30+ technical indicators across momentum, trend, volatility, and volume categories.
"""
import numpy as np
import pandas as pd
from loguru import logger
from typing import Optional


class TechnicalIndicators:
    """
    Computes a comprehensive set of technical indicators.
    All methods are pure functions — they return the input DataFrame
    with new columns added. Column names follow the pattern: `ti_{name}`.
    """

    @staticmethod
    def add_all(df: pd.DataFrame, config=None) -> pd.DataFrame:
        """Add all technical indicator groups at once."""
        df = df.copy()
        df = TechnicalIndicators.add_momentum(df)
        df = TechnicalIndicators.add_trend(df)
        df = TechnicalIndicators.add_volatility(df)
        df = TechnicalIndicators.add_volume(df)
        df = TechnicalIndicators.add_price_transforms(df)
        
        # Integrasi Indikator Kustom
        df = TechnicalIndicators.add_chang_vwap_ema_ribbon(df)
        
        logger.info(f"Added technical indicators. Columns: {df.shape[1]}")
        return df

    # ── Momentum ─────────────────────────────────────────────────────────────

    @staticmethod
    def add_momentum(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # RSI (14, 7, 21)
        for period in [7, 14, 21]:
            df[f"ti_rsi_{period}"] = TechnicalIndicators._rsi(close, period)

        # Stochastic Oscillator
        df["ti_stoch_k"], df["ti_stoch_d"] = TechnicalIndicators._stochastic(
            high, low, close, 14, 3
        )

        # Williams %R
        df["ti_williams_r"] = TechnicalIndicators._williams_r(high, low, close, 14)

        # Rate of Change (ROC)
        for period in [5, 10, 20]:
            df[f"ti_roc_{period}"] = close.pct_change(period) * 100

        # Commodity Channel Index (CCI)
        df["ti_cci"] = TechnicalIndicators._cci(high, low, close, 20)

        # Money Flow Index (MFI)
        df["ti_mfi"] = TechnicalIndicators._mfi(high, low, close, df["Volume"], 14)

        # Momentum
        df["ti_momentum_10"] = close - close.shift(10)

        return df

    # ── Trend ─────────────────────────────────────────────────────────────────

    @staticmethod
    def add_trend(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # MACD
        df["ti_macd"], df["ti_macd_signal"], df["ti_macd_hist"] = \
            TechnicalIndicators._macd(close, 12, 26, 9)

        # EMA family
        for period in [9, 21, 50, 100, 200]:
            df[f"ti_ema_{period}"] = close.ewm(span=period, adjust=False).mean()

        # SMA family
        for period in [10, 20, 50, 200]:
            df[f"ti_sma_{period}"] = close.rolling(period).mean()

        # ADX (Average Directional Index)
        df["ti_adx"], df["ti_di_plus"], df["ti_di_minus"] = \
            TechnicalIndicators._adx(high, low, close, 14)

        # Price relative to moving averages
        df["ti_close_vs_sma50"] = (close / df["ti_sma_50"]) - 1
        df["ti_close_vs_sma200"] = (close / df["ti_sma_200"]) - 1

        # Golden/Death cross signal
        df["ti_golden_cross"] = (df["ti_sma_50"] > df["ti_sma_200"]).astype(int)

        # Parabolic SAR (simplified)
        df["ti_psar"] = TechnicalIndicators._parabolic_sar(high, low, close)

        # Ichimoku Cloud
        tenkan, kijun, senkou_a, senkou_b = TechnicalIndicators._ichimoku(high, low)
        df["ti_ichimoku_tenkan"] = tenkan
        df["ti_ichimoku_kijun"] = kijun
        df["ti_ichimoku_cloud_green"] = (senkou_a > senkou_b).astype(int)

        return df

    # ── Volatility ────────────────────────────────────────────────────────────

    @staticmethod
    def add_volatility(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        # Bollinger Bands
        for period in [20, 50]:
            upper, mid, lower, bw, pct_b = TechnicalIndicators._bollinger(close, period)
            df[f"ti_bb_upper_{period}"] = upper
            df[f"ti_bb_lower_{period}"] = lower
            df[f"ti_bb_width_{period}"] = bw
            df[f"ti_bb_pct_{period}"] = pct_b

        # ATR (Average True Range)
        for period in [7, 14, 21]:
            df[f"ti_atr_{period}"] = TechnicalIndicators._atr(high, low, close, period)

        # Historical Volatility (rolling std of log returns)
        log_ret = np.log(close / close.shift(1))
        for period in [10, 20, 30]:
            df[f"ti_hist_vol_{period}"] = log_ret.rolling(period).std() * np.sqrt(252) * 100

        # Keltner Channel
        df["ti_kc_upper"], df["ti_kc_lower"] = TechnicalIndicators._keltner(
            high, low, close, 20
        )

        # Donchian Channel
        df["ti_donchian_high"] = high.rolling(20).max()
        df["ti_donchian_low"] = low.rolling(20).min()
        df["ti_donchian_width"] = df["ti_donchian_high"] - df["ti_donchian_low"]

        return df

    # ── Volume ────────────────────────────────────────────────────────────────

    @staticmethod
    def add_volume(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # On-Balance Volume (OBV)
        df["ti_obv"] = TechnicalIndicators._obv(close, volume)

        # Volume-Weighted Average Price (VWAP) rolling
        df["ti_vwap"] = TechnicalIndicators._vwap(high, low, close, volume, 20)

        # Chaikin Money Flow
        df["ti_cmf"] = TechnicalIndicators._cmf(high, low, close, volume, 20)

        # Volume oscillator
        vol_short = volume.ewm(span=5).mean()
        vol_long = volume.ewm(span=20).mean()
        df["ti_vol_osc"] = ((vol_short - vol_long) / vol_long) * 100

        # Volume SMA ratios
        df["ti_vol_sma20"] = volume.rolling(20).mean()
        df["ti_vol_ratio"] = volume / df["ti_vol_sma20"]  # > 1 = above avg volume

        # Accumulation/Distribution Line
        df["ti_adl"] = TechnicalIndicators._adl(high, low, close, volume)

        return df

    # ── Price Transforms ──────────────────────────────────────────────────────

    @staticmethod
    def add_price_transforms(df: pd.DataFrame) -> pd.DataFrame:
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        open_ = df["Open"]

        # Candlestick features
        df["ti_body"] = close - open_
        df["ti_body_pct"] = df["ti_body"] / open_
        df["ti_upper_shadow"] = high - df[["Close", "Open"]].max(axis=1)
        df["ti_lower_shadow"] = df[["Close", "Open"]].min(axis=1) - low
        df["ti_candle_range"] = high - low

        # Doji detection
        df["ti_doji"] = (df["ti_body"].abs() <= 0.1 * df["ti_candle_range"]).astype(int)

        # Hammer pattern
        df["ti_hammer"] = (
            (df["ti_lower_shadow"] > 2 * df["ti_body"].abs()) &
            (df["ti_upper_shadow"] < 0.1 * df["ti_candle_range"])
        ).astype(int)

        # Gap detection
        df["ti_gap_up"] = (open_ > df["Close"].shift(1) * 1.01).astype(int)
        df["ti_gap_down"] = (open_ < df["Close"].shift(1) * 0.99).astype(int)

        return df

    # ── Personal Custom Indicators ────────────────────────────────────────────

    @staticmethod
    def add_chang_vwap_ema_ribbon(df: pd.DataFrame) -> pd.DataFrame:
        """
        Integrasi [Chang] VWAP System + EMA Ribbon.
        Kombinasi pita EMA untuk mendeteksi tren berlapis dan VWAP untuk konfirmasi volume.
        """
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        # 1. EMA Ribbon (Pita EMA menggunakan deret Fibonacci)
        ribbon_periods = [13, 21, 34, 55, 89]
        for period in ribbon_periods:
            df[f"ti_chang_ema_ribbon_{period}"] = close.ewm(span=period, adjust=False).mean()

        # Ribbon Alignment Signal (Tren Bullish murni jika terurut sempurna: 13 > 21 > 34 > 55 > 89)
        bullish_alignment = (df["ti_chang_ema_ribbon_13"] > df["ti_chang_ema_ribbon_21"]) & \
                            (df["ti_chang_ema_ribbon_21"] > df["ti_chang_ema_ribbon_34"]) & \
                            (df["ti_chang_ema_ribbon_34"] > df["ti_chang_ema_ribbon_55"]) & \
                            (df["ti_chang_ema_ribbon_55"] > df["ti_chang_ema_ribbon_89"])
        df["ti_chang_ribbon_bullish"] = bullish_alignment.astype(int)

        # 2. VWAP System (Rolling 20-bar proxy untuk data harian)
        typical_price = (high + low + close) / 3
        vwap = (typical_price * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)
        df["ti_chang_vwap_line"] = vwap

        # 3. Sinyal Konfirmasi Chang (Harga menembus VWAP + Ribbon Bullish)
        df["ti_chang_golden_signal"] = ((close > vwap) & bullish_alignment).astype(int)

        return df

    # ── Indicator Implementations ─────────────────────────────────────────────

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        loss = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _macd(close: pd.Series, fast=12, slow=26, signal=9):
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        sig = macd.ewm(span=signal, adjust=False).mean()
        return macd, sig, macd - sig

    @staticmethod
    def _bollinger(close: pd.Series, period=20, std_mult=2.0):
        mid = close.rolling(period).mean()
        std = close.rolling(period).std()
        upper = mid + std_mult * std
        lower = mid - std_mult * std
        bw = (upper - lower) / mid
        pct_b = (close - lower) / (upper - lower + 1e-10)
        return upper, mid, lower, bw, pct_b

    @staticmethod
    def _atr(high, low, close, period=14) -> pd.Series:
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        return tr.ewm(com=period - 1, adjust=False).mean()

    @staticmethod
    def _stochastic(high, low, close, k_period=14, d_period=3):
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def _williams_r(high, low, close, period=14) -> pd.Series:
        highest_high = high.rolling(period).max()
        lowest_low = low.rolling(period).min()
        return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

    @staticmethod
    def _cci(high, low, close, period=20) -> pd.Series:
        typical_price = (high + low + close) / 3
        mean_dev = typical_price.rolling(period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        return (typical_price - typical_price.rolling(period).mean()) / (0.015 * mean_dev + 1e-10)

    @staticmethod
    def _adx(high, low, close, period=14):
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs()
        ], axis=1).max(axis=1)
        dm_plus = ((high - high.shift(1)).clip(lower=0)
                   .where((high - high.shift(1)) > (low.shift(1) - low), 0))
        dm_minus = ((low.shift(1) - low).clip(lower=0)
                    .where((low.shift(1) - low) > (high - high.shift(1)), 0))
        atr = tr.ewm(com=period - 1, adjust=False).mean()
        di_plus = 100 * dm_plus.ewm(com=period - 1, adjust=False).mean() / (atr + 1e-10)
        di_minus = 100 * dm_minus.ewm(com=period - 1, adjust=False).mean() / (atr + 1e-10)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10)
        adx = dx.ewm(com=period - 1, adjust=False).mean()
        return adx, di_plus, di_minus

    @staticmethod
    def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        signed_vol = volume * np.sign(close.diff())
        return signed_vol.cumsum()

    @staticmethod
    def _vwap(high, low, close, volume, period=20) -> pd.Series:
        typical_price = (high + low + close) / 3
        return (typical_price * volume).rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)

    @staticmethod
    def _cmf(high, low, close, volume, period=20) -> pd.Series:
        mf_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
        mf_volume = mf_multiplier * volume
        return mf_volume.rolling(period).sum() / (volume.rolling(period).sum() + 1e-10)

    @staticmethod
    def _mfi(high, low, close, volume, period=14) -> pd.Series:
        typical_price = (high + low + close) / 3
        raw_mf = typical_price * volume
        pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0).rolling(period).sum()
        neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0).rolling(period).sum()
        mfr = pos_mf / (neg_mf + 1e-10)
        return 100 - (100 / (1 + mfr))

    @staticmethod
    def _adl(high, low, close, volume) -> pd.Series:
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        return (clv * volume).cumsum()

    @staticmethod
    def _keltner(high, low, close, period=20, mult=2.0):
        ema = close.ewm(span=period, adjust=False).mean()
        atr = TechnicalIndicators._atr(high, low, close, period)
        return ema + mult * atr, ema - mult * atr

    @staticmethod
    def _parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series,
                       af_start=0.02, af_step=0.02, af_max=0.20) -> pd.Series:
        """Simplified Parabolic SAR."""
        psar = close.copy()
        bull = True
        af = af_start
        ep = low.iloc[0]
        hp = high.iloc[0]
        lp = low.iloc[0]

        for i in range(2, len(close)):
            prev_psar = psar.iloc[i - 1]
            if bull:
                psar.iloc[i] = prev_psar + af * (hp - prev_psar)
                psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2])
                if low.iloc[i] < psar.iloc[i]:
                    bull = False
                    psar.iloc[i] = hp
                    lp = low.iloc[i]
                    af = af_start
                else:
                    if high.iloc[i] > hp:
                        hp = high.iloc[i]
                        af = min(af + af_step, af_max)
            else:
                psar.iloc[i] = prev_psar + af * (lp - prev_psar)
                psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2])
                if high.iloc[i] > psar.iloc[i]:
                    bull = True
                    psar.iloc[i] = lp
                    hp = high.iloc[i]
                    af = af_start
                else:
                    if low.iloc[i] < lp:
                        lp = low.iloc[i]
                        af = min(af + af_step, af_max)
        return psar

    @staticmethod
    def _ichimoku(high: pd.Series, low: pd.Series,
                  tenkan_period=9, kijun_period=26, senkou_b_period=52):
        def midpoint(h, l, p):
            return (h.rolling(p).max() + l.rolling(p).min()) / 2
        tenkan = midpoint(high, low, tenkan_period)
        kijun = midpoint(high, low, kijun_period)
        senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
        senkou_b = midpoint(high, low, senkou_b_period).shift(kijun_period)
        return tenkan, kijun, senkou_a, senkou_b