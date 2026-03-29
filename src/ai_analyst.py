"""
src/ai_analyst.py
LLM-powered market analyst that generates professional commentary
on predictions, technical signals, and portfolio risk.
"""
import json
import textwrap
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class AnalystReport:
    ticker: str
    date: str
    current_price: float
    predicted_price: float
    predicted_return_pct: float
    confidence_interval: tuple
    signal: str                  # BUY / SELL / HOLD
    signal_strength: str         # STRONG / MODERATE / WEAK
    key_technicals: Dict[str, float]
    commentary: str
    risks: List[str]
    opportunities: List[str]
    sentiment: str               # BULLISH / BEARISH / NEUTRAL


class AIAnalyst:
    """
    Generates professional market commentary using either:
    1. Anthropic Claude API (if API key is set)
    2. Rule-based template fallback (always available)

    The commentary synthesizes:
    - Model prediction with uncertainty bounds
    - Key technical indicator readings
    - Recent price action context
    - Risk/opportunity assessment
    """

    SIGNAL_THRESHOLDS = {
        "strong_buy":  0.03,
        "buy":         0.01,
        "hold_upper":  0.01,
        "hold_lower": -0.01,
        "sell":       -0.01,
        "strong_sell": -0.03,
    }

    def __init__(self, api_key: Optional[str] = None, use_llm: bool = True):
        self.api_key = api_key
        self.use_llm = use_llm and api_key is not None
        if self.use_llm:
            logger.info("AIAnalyst: LLM mode enabled (Anthropic Claude)")
        else:
            logger.info("AIAnalyst: Rule-based template mode")

    def generate_report(
        self,
        ticker: str,
        current_price: float,
        predicted_price: float,
        lower_bound: float,
        upper_bound: float,
        technical_indicators: Dict[str, float],
        recent_prices: Optional[pd.Series] = None,
        backtest_metrics: Optional[Dict] = None,
    ) -> AnalystReport:
        """Generate a full analyst report for a ticker prediction."""

        pred_return = (predicted_price - current_price) / current_price
        signal, strength = self._classify_signal(pred_return, technical_indicators)
        sentiment = self._classify_sentiment(technical_indicators)

        # Extract key technicals for display
        key_ti = self._extract_key_technicals(technical_indicators)

        # Generate commentary
        if self.use_llm:
            commentary = self._llm_commentary(
                ticker, current_price, predicted_price, pred_return,
                lower_bound, upper_bound, key_ti, signal, sentiment, backtest_metrics
            )
        else:
            commentary = self._template_commentary(
                ticker, current_price, predicted_price, pred_return,
                lower_bound, upper_bound, key_ti, signal, sentiment
            )

        risks = self._identify_risks(technical_indicators, pred_return)
        opportunities = self._identify_opportunities(technical_indicators, pred_return)

        return AnalystReport(
            ticker=ticker,
            date=datetime.now().strftime("%Y-%m-%d"),
            current_price=current_price,
            predicted_price=predicted_price,
            predicted_return_pct=round(pred_return * 100, 2),
            confidence_interval=(round(lower_bound, 2), round(upper_bound, 2)),
            signal=signal,
            signal_strength=strength,
            key_technicals=key_ti,
            commentary=commentary,
            risks=risks,
            opportunities=opportunities,
            sentiment=sentiment,
        )

    def _llm_commentary(
        self, ticker, current_price, predicted_price, pred_return,
        lower, upper, key_ti, signal, sentiment, backtest_metrics
    ) -> str:
        """Call Anthropic Claude API for professional commentary."""
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.api_key)

            ti_summary = "\n".join([f"  - {k}: {v:.2f}" for k, v in key_ti.items()])
            bt_summary = ""
            if backtest_metrics:
                bt_summary = f"""
Backtest Performance:
  - Total Return: {backtest_metrics.get('total_return', 0):.1f}%
  - Sharpe Ratio: {backtest_metrics.get('sharpe_ratio', 0):.2f}
  - Win Rate: {backtest_metrics.get('win_rate', 0):.1f}%
  - Max Drawdown: {backtest_metrics.get('max_drawdown', 0):.1f}%
"""

            prompt = f"""You are a senior quantitative analyst at a hedge fund. 
Write a concise, professional market analysis for {ticker}.

Current Price: ${current_price:.2f}
Predicted Price (next session): ${predicted_price:.2f}
Predicted Return: {pred_return*100:+.2f}%
95% Confidence Interval: [${lower:.2f}, ${upper:.2f}]
Model Signal: {signal}
Overall Sentiment: {sentiment}

Key Technical Indicators:
{ti_summary}
{bt_summary}

Write 3-4 sentences of professional commentary. Cover:
1. Price action context and what the model is signaling
2. Which technical indicators support or contradict the prediction
3. Key risk factors and level of conviction

Be specific, data-driven, and avoid generic phrases. Write in present tense."""

            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text

        except Exception as e:
            logger.warning(f"LLM commentary failed: {e}. Falling back to template.")
            return self._template_commentary(
                ticker, current_price, predicted_price, pred_return,
                lower, upper, key_ti, signal, sentiment
            )

    def _template_commentary(
        self, ticker, current_price, predicted_price, pred_return,
        lower, upper, key_ti, signal, sentiment
    ) -> str:
        """Rule-based commentary generation."""
        rsi = key_ti.get("RSI_14", 50)
        macd_hist = key_ti.get("MACD_Hist", 0)
        bb_pct = key_ti.get("BB_Pct_20", 0.5)
        adx = key_ti.get("ADX", 25)

        # RSI context
        if rsi > 70:
            rsi_text = f"RSI at {rsi:.0f} signals overbought conditions"
        elif rsi < 30:
            rsi_text = f"RSI at {rsi:.0f} indicates oversold territory"
        else:
            rsi_text = f"RSI at {rsi:.0f} remains in neutral territory"

        # MACD context
        macd_text = "MACD histogram is positive, supporting bullish momentum" \
            if macd_hist > 0 else "MACD histogram has turned negative, suggesting weakening momentum"

        # BB context
        if bb_pct > 0.8:
            bb_text = "price is trading near the upper Bollinger Band, indicating potential resistance"
        elif bb_pct < 0.2:
            bb_text = "price is near the lower Bollinger Band, which may offer support"
        else:
            bb_text = "price is trading within Bollinger Band bounds"

        # Trend strength
        trend_text = f"ADX at {adx:.0f} indicates {'strong' if adx > 25 else 'weak'} trend conditions"

        return (
            f"{ticker} is currently trading at ${current_price:.2f}, with the ensemble model "
            f"projecting a move to ${predicted_price:.2f} "
            f"({pred_return*100:+.2f}%) in the next session "
            f"(95% CI: ${lower:.2f}–${upper:.2f}). "
            f"The overall signal is {signal} with {sentiment.lower()} bias. "
            f"{rsi_text}, while {macd_text}. "
            f"Additionally, {bb_text} and {trend_text}. "
            f"Traders should monitor these levels closely and manage risk accordingly."
        )

    def _classify_signal(
        self, pred_return: float, ti: Dict
    ) -> tuple:
        """Classify signal as BUY/SELL/HOLD with strength."""
        t = self.SIGNAL_THRESHOLDS
        if pred_return >= t["strong_buy"]:
            return "BUY", "STRONG"
        elif pred_return >= t["buy"]:
            return "BUY", "MODERATE"
        elif pred_return <= t["strong_sell"]:
            return "SELL", "STRONG"
        elif pred_return <= t["sell"]:
            return "SELL", "MODERATE"
        else:
            return "HOLD", "WEAK"

    def _classify_sentiment(self, ti: Dict) -> str:
        bullish_signals = 0
        bearish_signals = 0

        rsi = ti.get("ti_rsi_14", 50)
        if rsi > 55: bullish_signals += 1
        elif rsi < 45: bearish_signals += 1

        macd_hist = ti.get("ti_macd_hist", 0)
        if macd_hist > 0: bullish_signals += 1
        else: bearish_signals += 1

        golden_cross = ti.get("ti_golden_cross", 0)
        if golden_cross == 1: bullish_signals += 1
        else: bearish_signals += 1

        if bullish_signals > bearish_signals:
            return "BULLISH"
        elif bearish_signals > bullish_signals:
            return "BEARISH"
        return "NEUTRAL"

    def _extract_key_technicals(self, ti: Dict) -> Dict[str, float]:
        mapping = {
            "RSI_14": "ti_rsi_14",
            "MACD_Hist": "ti_macd_hist",
            "BB_Pct_20": "ti_bb_pct_20",
            "ADX": "ti_adx",
            "ATR_14": "ti_atr_14",
            "Stoch_K": "ti_stoch_k",
            "OBV": "ti_obv",
            "MFI": "ti_mfi",
        }
        return {
            display: round(float(ti.get(col, 0)), 3)
            for display, col in mapping.items()
            if col in ti
        }

    def _identify_risks(self, ti: Dict, pred_return: float) -> List[str]:
        risks = []
        rsi = ti.get("ti_rsi_14", 50)
        vol = ti.get("ti_hist_vol_20", 0)
        adx = ti.get("ti_adx", 25)

        if rsi > 70:
            risks.append("Overbought RSI — potential mean reversion")
        if rsi < 30:
            risks.append("Oversold RSI — risk of further downside if support breaks")
        if vol > 30:
            risks.append(f"Elevated historical volatility ({vol:.1f}%) — wider than expected price swings")
        if adx < 20:
            risks.append("Weak trend (ADX < 20) — signal reliability reduced in choppy market")
        if abs(pred_return) > 0.05:
            risks.append("Large predicted move — ensure appropriate position sizing")
        if not risks:
            risks.append("Market conditions appear stable — monitor for regime change")
        return risks

    def _identify_opportunities(self, ti: Dict, pred_return: float) -> List[str]:
        opps = []
        rsi = ti.get("ti_rsi_14", 50)
        cmf = ti.get("ti_cmf", 0)
        bb_pct = ti.get("ti_bb_pct_20", 0.5)
        golden = ti.get("ti_golden_cross", 0)

        if rsi < 35:
            opps.append("Oversold bounce potential — consider scaled entry")
        if cmf > 0.1:
            opps.append("Positive Chaikin Money Flow — institutional accumulation signal")
        if bb_pct < 0.15:
            opps.append("Price at lower Bollinger Band — mean reversion opportunity")
        if golden == 1:
            opps.append("SMA 50/200 golden cross active — long-term trend confirmed bullish")
        if pred_return > 0.02:
            opps.append(f"Strong model conviction: {pred_return*100:+.1f}% predicted return")
        if not opps:
            opps.append("Wait for clearer signal confirmation before entry")
        return opps
