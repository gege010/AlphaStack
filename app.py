"""
app.py — Streamlit Dashboard for AlphaStack
Run: streamlit run app.py
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import yfinance as yf
from typing import Dict, Optional
import time
import random

# Project asset universe.
from config import TRAIN_TICKERS, UNIVERSE

st.set_page_config(
    page_title="AlphaStack",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom dashboard theme.
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

  html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
  .stApp { background-color: #0a0e1a; color: #e2e8f0; }

  .metric-card {
    background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
    border: 1px solid #374151;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
    transition: transform 0.2s;
  }
  .metric-card:hover { transform: translateY(-2px); }
  .metric-label { font-size: 0.72rem; color: #9ca3af; letter-spacing: 0.12em; text-transform: uppercase; }
  .metric-value { font-size: 1.7rem; font-weight: 700; font-family: 'JetBrains Mono', monospace; margin-top: 4px; }
  .metric-delta { font-size: 0.78rem; margin-top: 4px; }

  .signal-buy  { color: #10b981; }
  .signal-sell { color: #ef4444; }
  .signal-hold { color: #f59e0b; }

  .section-header {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.2em;
    text-transform: uppercase; color: #6366f1;
    border-bottom: 1px solid #1e293b;
    padding-bottom: 6px; margin-bottom: 1rem; margin-top: 1.5rem;
  }
  .commentary-box {
    background: #111827;
    border-left: 3px solid #6366f1;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-size: 0.9rem;
    line-height: 1.7;
    color: #cbd5e1;
    font-style: italic;
  }
  .tag { display: inline-block; padding: 2px 10px; border-radius: 20px;
         font-size: 0.7rem; font-weight: 600; letter-spacing: 0.05em; margin: 2px; }
  .tag-risk { background: #7f1d1d; color: #fca5a5; }
  .tag-opp  { background: #064e3b; color: #6ee7b7; }
</style>
""", unsafe_allow_html=True)


# Demo helpers. Replace with real model inference in production.

@st.cache_data(ttl=300)
def fetch_price_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch OHLCV data via yfinance."""
    try:
        df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Failed to fetch data for {ticker}: {e}")
        return pd.DataFrame()


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute a core set of indicators inline for the dashboard."""
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # Relative Strength Index (14).
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD (12, 26, 9).
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    # Bollinger Bands (20, 2).
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    df["BB_Upper"] = sma20 + 2 * std20
    df["BB_Lower"] = sma20 - 2 * std20
    df["BB_Mid"] = sma20

    # Average True Range (14).
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(com=13, adjust=False).mean()

    # Exponential moving averages.
    for p in [9, 21, 50, 200]:
        df[f"EMA{p}"] = close.ewm(span=p, adjust=False).mean()

    # Volume ratio against 20-day average.
    df["Vol_SMA20"] = df["Volume"].rolling(20).mean()
    df["Vol_Ratio"] = df["Volume"] / (df["Vol_SMA20"] + 1e-10)

    # On-Balance Volume.
    df["OBV"] = (df["Volume"] * np.sign(close.diff())).cumsum()

    # Stochastic oscillator.
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    df["Stoch_K"] = 100 * (close - lowest) / (highest - lowest + 1e-10)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()

    return df.dropna()


def simulate_predictions(df: pd.DataFrame, horizon: int = 1) -> Dict:
    """
    Simulate model output for demo purposes.
    Replace with: load actual trained model → run inference.
    """
    close = df["Close"].values
    recent = close[-30:]
    current = close[-1]

    # Approximate short-term drift from recent price action.
    trend = (recent[-1] / recent[-5] - 1)
    daily_pred_return = trend * 0.1 + np.random.normal(0, 0.008)

    # Simulate up to 30 business-day forward path.
    future_prices = [current]
    for i in range(30):
        r = np.random.normal(daily_pred_return, 0.015)
        future_prices.append(future_prices[-1] * (1 + r))

    # Use horizon-specific target point.
    pred_price = future_prices[horizon]
    pred_return = (pred_price / current) - 1

    # Expand uncertainty with longer prediction horizon.
    uncertainty = abs(pred_return) * 0.5 + (0.01 * np.sqrt(horizon))
    lower = pred_price * (1 - 1.96 * uncertainty)
    upper = pred_price * (1 + 1.96 * uncertainty)

    # Simulate per-model variation for comparison chart (Only BiLSTM and XGBoost).
    lstm_pred   = pred_price * (1 + np.random.normal(0, 0.003 * np.sqrt(horizon)))
    xgb_pred    = pred_price * (1 + np.random.normal(0, 0.004 * np.sqrt(horizon)))

    return {
        "current": current,
        "predicted": pred_price,
        "lower": lower,
        "upper": upper,
        "return_pct": pred_return * 100,
        "lstm": lstm_pred,
        "xgb": xgb_pred,
        "uncertainty": uncertainty * 100,
        "future_prices": future_prices,
        "signal": "BUY" if pred_return > 0.015 else ("SELL" if pred_return < -0.015 else "HOLD"),
        "confidence": round(random.uniform(60, 90), 1),
        "horizon": horizon,  # Needed for horizon-aligned plotting.
    }


def simulate_backtest_metrics() -> Dict:
    return {
        "total_return":      round(random.uniform(18, 65), 1),
        "annualized_return": round(random.uniform(10, 30), 1),
        "sharpe_ratio":      round(random.uniform(1.2, 2.4), 2),
        "sortino_ratio":     round(random.uniform(1.5, 3.0), 2),
        "max_drawdown":     -round(random.uniform(8, 22), 1),
        "win_rate":          round(random.uniform(52, 70), 1),
        "num_trades":        random.randint(40, 120),
        "profit_factor":     round(random.uniform(1.3, 2.5), 2),
        "calmar_ratio":      round(random.uniform(0.9, 2.1), 2),
    }


def generate_commentary(ticker, pred_info, df) -> str:
    rsi = df["RSI"].iloc[-1]
    macd_h = df["MACD_Hist"].iloc[-1]
    signal = pred_info["signal"]

    rsi_note = f"RSI at {rsi:.0f} ({'overbought' if rsi>70 else 'oversold' if rsi<30 else 'neutral'})"
    macd_note = "MACD positive — momentum building" if macd_h > 0 else "MACD negative — momentum weakening"

    signal_map = {
        "BUY":  f"{ticker} shows bullish alignment across model ensemble",
        "SELL": f"{ticker} exhibits bearish pressure in ensemble consensus",
        "HOLD": f"{ticker} signals are mixed — market indecision persists",
    }

    return (
        f"{signal_map[signal]}. The ensemble model projects ${pred_info['predicted']:.2f} "
        f"({pred_info['return_pct']:+.2f}%) with a 95% confidence interval of "
        f"[${pred_info['lower']:.2f}–${pred_info['upper']:.2f}]. "
        f"{rsi_note}. {macd_note}. "
        f"Model agreement score: {pred_info['confidence']:.0f}%."
    )


def candlestick_chart(df: pd.DataFrame, pred_info: Dict, ticker: str, show_bb: bool, show_ema: bool) -> go.Figure:
    last_n = df.tail(120)
    horizon = pred_info["horizon"]
    
    # Generate forecast dates up to selected horizon.
    future_dates = pd.date_range(
        start=last_n.index[-1] + timedelta(days=1), periods=horizon, freq="B"
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.6, 0.2, 0.2],
        vertical_spacing=0.03,
    )

    # Price candles.
    fig.add_trace(go.Candlestick(
        x=last_n.index, open=last_n["Open"], high=last_n["High"],
        low=last_n["Low"], close=last_n["Close"],
        name="Price",
        increasing=dict(fillcolor="#10b981", line=dict(color="#10b981", width=1)),
        decreasing=dict(fillcolor="#ef4444", line=dict(color="#ef4444", width=1)),
    ), row=1, col=1)

    # Optional Bollinger overlay.
    if show_bb:
        fig.add_trace(go.Scatter(x=last_n.index, y=last_n["BB_Upper"],
            line=dict(color="#6366f1", width=1, dash="dot"), name="BB Upper",
            showlegend=False, opacity=0.6), row=1, col=1)
        fig.add_trace(go.Scatter(x=last_n.index, y=last_n["BB_Lower"],
            line=dict(color="#6366f1", width=1, dash="dot"), name="BB Lower",
            fill="tonexty", fillcolor="rgba(99,102,241,0.05)",
            showlegend=False, opacity=0.6), row=1, col=1)

    # Optional EMA overlays.
    if show_ema:
        for ema, color in [(21, "#f59e0b"), (50, "#06b6d4"), (200, "#a78bfa")]:
            if f"EMA{ema}" in last_n:
                fig.add_trace(go.Scatter(
                    x=last_n.index, y=last_n[f"EMA{ema}"],
                    line=dict(color=color, width=1),
                    name=f"EMA{ema}", opacity=0.8
                ), row=1, col=1)

    # Forecast cone truncated to selected horizon.
    future_mean = pred_info["future_prices"][1:horizon+1]
    lowers = [p * (1 - 0.015 * (i+1)**0.5) for i, p in enumerate(future_mean)]
    uppers = [p * (1 + 0.015 * (i+1)**0.5) for i, p in enumerate(future_mean)]

    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=uppers + lowers[::-1],
        fill="toself", fillcolor="rgba(99,102,241,0.12)",
        line=dict(color="rgba(0,0,0,0)"),
        name="95% CI", showlegend=True,
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=future_dates, y=future_mean,
        line=dict(color="#818cf8", width=2, dash="dash"),
        name="Ensemble Forecast",
        marker=dict(size=3),
    ), row=1, col=1)

    # Highlight target point at horizon end.
    fig.add_trace(go.Scatter(
        x=[future_dates[-1]], y=[pred_info["predicted"]],
        mode="markers",
        marker=dict(size=10, color="#818cf8", symbol="diamond",
                    line=dict(color="white", width=1.5)),
        name=f"T+{horizon} Target",
    ), row=1, col=1)

    # Volume panel.
    colors = ["#10b981" if c >= o else "#ef4444"
              for c, o in zip(last_n["Close"], last_n["Open"])]
    fig.add_trace(go.Bar(
        x=last_n.index, y=last_n["Volume"],
        marker_color=colors, name="Volume", opacity=0.7,
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=last_n.index, y=last_n["Vol_SMA20"],
        line=dict(color="#f59e0b", width=1.5), name="Vol MA20", opacity=0.8,
    ), row=2, col=1)

    # RSI panel.
    fig.add_trace(go.Scatter(
        x=last_n.index, y=last_n["RSI"],
        line=dict(color="#22d3ee", width=1.5), name="RSI",
    ), row=3, col=1)
    for level, color in [(70, "#ef4444"), (30, "#10b981"), (50, "#6b7280")]:
        fig.add_hline(y=level, line=dict(color=color, width=1, dash="dot"),
                      row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1117",
        height=620,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", y=1.02, x=0, font=dict(size=10)),
        font=dict(family="JetBrains Mono", size=11, color="#9ca3af"),
    )
    fig.update_yaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
    fig.update_xaxes(gridcolor="#1e293b", zerolinecolor="#1e293b")
    return fig


def model_comparison_chart(pred_info: Dict, current: float) -> go.Figure:
    models = ["BiLSTM-Attn", "XGBoost", "Ensemble"]
    preds = [pred_info["lstm"], pred_info["xgb"], pred_info["predicted"]]
    colors = ["#6366f1", "#f59e0b", "#10b981"]
    returns = [(p - current) / current * 100 for p in preds]

    fig = go.Figure()
    for m, r, c in zip(models, returns, colors):
        fig.add_trace(go.Bar(
            x=[m], y=[r],
            marker_color=c, name=m,
            text=[f"{r:+.2f}%"], textposition="outside",
            textfont=dict(color=c, size=12, family="JetBrains Mono"),
        ))

    fig.add_hline(y=0, line=dict(color="#475569", width=1))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1117",
        height=220, showlegend=False,
        margin=dict(l=0, r=0, t=10, b=0),
        yaxis=dict(title="Predicted Return (%)", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        font=dict(family="JetBrains Mono", size=11, color="#9ca3af"),
        bargap=0.35,
    )
    return fig


def portfolio_equity_chart(returns_series: pd.Series) -> go.Figure:
    equity = (1 + returns_series).cumprod() * 100_000
    bm = (1 + returns_series.rolling(1).mean().fillna(0) * 0.7).cumprod() * 100_000  # Lightweight benchmark proxy.

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity.index, y=equity,
        line=dict(color="#6366f1", width=2),
        fill="tonexty", fillcolor="rgba(99,102,241,0.06)",
        name="AI Strategy",
    ))
    fig.add_trace(go.Scatter(
        x=bm.index, y=bm,
        line=dict(color="#475569", width=1.5, dash="dot"),
        name="Buy & Hold",
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0d1117",
        height=260,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", y=1.02, font=dict(size=10)),
        yaxis=dict(title="Portfolio Value ($)", gridcolor="#1e293b"),
        xaxis=dict(gridcolor="#1e293b"),
        font=dict(family="JetBrains Mono", size=11, color="#9ca3af"),
        hovermode="x unified",
    )
    return fig


def technical_gauge(value: float, name: str, min_v: float, max_v: float,
                    thresholds: list) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={"text": name, "font": {"size": 12, "color": "#9ca3af"}},
        gauge={
            "axis": {"range": [min_v, max_v], "tickcolor": "#475569"},
            "bar": {"color": "#6366f1", "thickness": 0.3},
            "bgcolor": "#1e293b",
            "steps": [
                {"range": [min_v, thresholds[0]], "color": "#ef4444"},
                {"range": [thresholds[0], thresholds[1]], "color": "#1e293b"},
                {"range": [thresholds[1], max_v], "color": "#10b981"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": value,
            },
        },
        number={"font": {"size": 20, "family": "JetBrains Mono", "color": "#e2e8f0"}},
    ))
    fig.update_layout(
        paper_bgcolor="#0a0e1a",
        height=160,
        margin=dict(l=15, r=15, t=40, b=5),
        font=dict(color="#9ca3af"),
    )
    return fig


def multi_ticker_heatmap(tickers: list) -> go.Figure:
    """Performance heatmap across multiple tickers."""
    data = {}
    for tkr in tickers:
        try:
            # Use source symbol from configured universe.
            yf_symbol = UNIVERSE[tkr][0] 
            
            # Download using provider-compatible symbol.
            df = yf.download(yf_symbol, period="1mo", interval="1d", progress=False, auto_adjust=True)
            if not df.empty:
                close = df["Close"].squeeze()
                data[tkr] = {
                    "1D": close.pct_change().iloc[-1] * 100,
                    "1W": (close.iloc[-1] / close.iloc[-5] - 1) * 100 if len(close) >= 5 else 0,
                    "1M": (close.iloc[-1] / close.iloc[0] - 1) * 100,
                }
        except:
            pass

    if not data:
        return go.Figure()

    df_heat = pd.DataFrame(data).T
    fig = go.Figure(go.Heatmap(
        z=df_heat.values,
        x=df_heat.columns.tolist(),
        y=df_heat.index.tolist(),
        colorscale=[[0, "#ef4444"], [0.5, "#1e293b"], [1, "#10b981"]],
        zmid=0,
        text=[[f"{v:.2f}%" for v in row] for row in df_heat.values],
        texttemplate="%{text}",
        textfont={"size": 12, "family": "JetBrains Mono"},
        showscale=True,
        colorbar=dict(tickfont=dict(color="#9ca3af", size=10)),
    ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0e1a",
        plot_bgcolor="#0a0e1a",
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="JetBrains Mono", size=11, color="#9ca3af"),
        xaxis=dict(side="top"),
    )
    return fig


def main():
    # Sidebar controls.
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding: 1rem 0 0.5rem'>
          <span style='font-size:1.8rem'>📈</span>
          <div style='font-family:Sora;font-weight:700;font-size:1.1rem;color:#e2e8f0;margin-top:4px'>
            AlphaStack
          </div>
          <div style='font-size:0.7rem;color:#6b7280;margin-top:2px;font-family:JetBrains Mono'>
            Ensemble Deep Learning
          </div>
        </div>
        <hr style='border-color:#1e293b;margin:12px 0'>
        """, unsafe_allow_html=True)

        st.markdown("<div class='section-header'>UNIVERSE</div>", unsafe_allow_html=True)
        # Keep selector aligned with training universe.
        ticker = st.selectbox(
            "Primary Ticker",
            options=TRAIN_TICKERS,
            index=0,
        )
        horizon = st.select_slider(
            "Prediction Horizon",
            options=["1D", "5D", "10D", "30D"],
            value="1D",
        )

        st.markdown("<div class='section-header'>CHART OVERLAYS</div>", unsafe_allow_html=True)
        show_bb = st.toggle("Bollinger Bands", value=True)
        show_ema = st.toggle("EMA Lines (21/50/200)", value=True)

        st.markdown("<div class='section-header'>MODELS</div>", unsafe_allow_html=True)
        active_models = st.multiselect(
            "Active ensemble members",
            ["BiLSTM-Attention", "XGBoost"],
            default=["BiLSTM-Attention", "XGBoost"],
        )

        st.markdown("<div class='section-header'>WATCHLIST</div>", unsafe_allow_html=True)
        # Reuse training universe for watchlist defaults.
        watchlist = st.multiselect(
            "Tickers for heatmap",
            options=TRAIN_TICKERS,
            default=TRAIN_TICKERS[:5] if len(TRAIN_TICKERS) >= 5 else TRAIN_TICKERS,
        )

        st.markdown("""
        <hr style='border-color:#1e293b;margin:12px 0'>
        <div style='font-size:0.65rem;color:#374151;text-align:center;line-height:1.5'>
          ⚠️ For educational purposes only.<br>Not financial advice.
        </div>
        """, unsafe_allow_html=True)

    # Resolve provider symbol from configured ticker key.
    yf_symbol = UNIVERSE[ticker][0]

    with st.spinner(f"Fetching {ticker} ({yf_symbol}) data..."):
        # Fetch by provider symbol (e.g., CL=F), not display key (e.g., OIL).
        df = fetch_price_data(yf_symbol)

    if df.empty:
        st.error("Could not load data. Check ticker and internet connection.")
        return

    df = compute_indicators(df)
    pred = simulate_predictions(df, horizon=int(horizon[:-1]))
    bt = simulate_backtest_metrics()
    commentary = generate_commentary(ticker, pred, df)

    current_price = df["Close"].iloc[-1]
    prev_close = df["Close"].iloc[-2]
    daily_ret = (current_price / prev_close - 1) * 100
    vol_20 = df["Close"].pct_change().rolling(20).std().iloc[-1] * np.sqrt(252) * 100

    # Header bar.
    sig_color = {"BUY": "#10b981", "SELL": "#ef4444", "HOLD": "#f59e0b"}[pred["signal"]]
    st.markdown(f"""
    <div style='display:flex;align-items:center;justify-content:space-between;
                padding:12px 16px;background:#111827;border-radius:12px;
                border:1px solid #1e293b;margin-bottom:1.2rem'>
      <div>
        <span style='font-size:1.8rem;font-weight:700;font-family:Sora;color:#f1f5f9'>
          {ticker}
        </span>
        <span style='margin-left:12px;font-size:1rem;color:#9ca3af;font-family:JetBrains Mono'>
          ${current_price:.2f}
        </span>
        <span style='margin-left:8px;font-size:0.9rem;color:{"#10b981" if daily_ret>=0 else "#ef4444"};
                     font-family:JetBrains Mono'>
          {daily_ret:+.2f}% today
        </span>
      </div>
      <div style='display:flex;gap:12px;align-items:center'>
        <div style='background:#0a0e1a;border:1px solid {sig_color};border-radius:20px;
                    padding:4px 18px;'>
          <span style='color:{sig_color};font-weight:700;font-size:0.85rem;letter-spacing:0.1em'>
            {pred["signal"]} · {horizon}
          </span>
        </div>
        <div style='font-size:0.75rem;color:#6b7280;font-family:JetBrains Mono'>
          Confidence: {pred["confidence"]:.0f}%
        </div>
        <div style='font-size:0.75rem;color:#6b7280;font-family:JetBrains Mono'>
          {datetime.now().strftime("%Y-%m-%d %H:%M")}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Top KPI row.
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "PREDICTED", f"${pred['predicted']:.2f}",
         f"{pred['return_pct']:+.2f}%", "#818cf8"),
        (k2, "95% CI LOWER", f"${pred['lower']:.2f}", "lower bound", "#6b7280"),
        (k3, "95% CI UPPER", f"${pred['upper']:.2f}", "upper bound", "#6b7280"),
        (k4, "RSI (14)", f"{df['RSI'].iloc[-1]:.1f}",
         "overbought" if df["RSI"].iloc[-1]>70 else "oversold" if df["RSI"].iloc[-1]<30 else "neutral",
         "#f59e0b"),
        (k5, "HIST VOL 20D", f"{vol_20:.1f}%", "annualized", "#22d3ee"),
        (k6, "ATR (14)", f"${df['ATR'].iloc[-1]:.2f}", "avg true range", "#a78bfa"),
    ]
    for col, label, val, delta, color in kpis:
        with col:
            st.markdown(f"""
            <div class='metric-card'>
              <div class='metric-label'>{label}</div>
              <div class='metric-value' style='color:{color}'>{val}</div>
              <div class='metric-delta' style='color:#6b7280'>{delta}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Main chart and model comparison.
    chart_col, meta_col = st.columns([3, 1])

    with chart_col:
        st.markdown("<div class='section-header'>PRICE CHART · TECHNICAL ANALYSIS · FORECAST</div>",
                    unsafe_allow_html=True)
        fig_main = candlestick_chart(df, pred, ticker, show_bb, show_ema)
        st.plotly_chart(fig_main, width="stretch", config={"displayModeBar": False})

    with meta_col:
        st.markdown("<div class='section-header'>MODEL ENSEMBLE</div>", unsafe_allow_html=True)
        fig_models = model_comparison_chart(pred, current_price)
        st.plotly_chart(fig_models, width="stretch", config={"displayModeBar": False})

        st.markdown("<div class='section-header'>MOMENTUM GAUGES</div>", unsafe_allow_html=True)
        g1, g2 = st.columns(2)
        with g1:
            st.plotly_chart(
                technical_gauge(df["RSI"].iloc[-1], "RSI", 0, 100, [30, 70]),
                width="stretch", config={"displayModeBar": False}
            )
        with g2:
            stoch = df["Stoch_K"].iloc[-1]
            st.plotly_chart(
                technical_gauge(stoch, "Stoch %K", 0, 100, [20, 80]),
                width="stretch", config={"displayModeBar": False}
            )

        # Model-generated narrative summary.
        st.markdown("<div class='section-header'>AI ANALYST</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='commentary-box'>"{commentary}"</div>
        """, unsafe_allow_html=True)

    # MACD and OBV row.
    macd_col, obv_col = st.columns(2)

    with macd_col:
        st.markdown("<div class='section-header'>MACD</div>", unsafe_allow_html=True)
        last_n = df.tail(120)
        fig_macd = go.Figure()
        colors = ["#10b981" if v >= 0 else "#ef4444" for v in last_n["MACD_Hist"]]
        fig_macd.add_trace(go.Bar(x=last_n.index, y=last_n["MACD_Hist"],
            marker_color=colors, name="Histogram", opacity=0.8))
        fig_macd.add_trace(go.Scatter(x=last_n.index, y=last_n["MACD"],
            line=dict(color="#6366f1", width=1.5), name="MACD"))
        fig_macd.add_trace(go.Scatter(x=last_n.index, y=last_n["MACD_Signal"],
            line=dict(color="#f59e0b", width=1.5, dash="dot"), name="Signal"))
        fig_macd.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
            height=200, margin=dict(l=0, r=0, t=5, b=0),
            legend=dict(orientation="h", y=1.1, font=dict(size=9)),
            yaxis=dict(gridcolor="#1e293b"),
            xaxis=dict(gridcolor="#1e293b"),
            font=dict(family="JetBrains Mono", size=10, color="#9ca3af"),
        )
        st.plotly_chart(fig_macd, width="stretch", config={"displayModeBar": False})

    with obv_col:
        st.markdown("<div class='section-header'>ON-BALANCE VOLUME</div>", unsafe_allow_html=True)
        fig_obv = go.Figure()
        fig_obv.add_trace(go.Scatter(
            x=last_n.index, y=last_n["OBV"],
            line=dict(color="#22d3ee", width=2),
            fill="tozeroy", fillcolor="rgba(34,211,238,0.06)",
            name="OBV",
        ))
        fig_obv.update_layout(
            template="plotly_dark", paper_bgcolor="#0a0e1a", plot_bgcolor="#0d1117",
            height=200, margin=dict(l=0, r=0, t=5, b=0),
            yaxis=dict(gridcolor="#1e293b"),
            xaxis=dict(gridcolor="#1e293b"),
            font=dict(family="JetBrains Mono", size=10, color="#9ca3af"),
            showlegend=False,
        )
        st.plotly_chart(fig_obv, width="stretch", config={"displayModeBar": False})

    # Backtest and heatmap row.
    bt_col, heat_col = st.columns([3, 2])

    with bt_col:
        st.markdown("<div class='section-header'>BACKTEST PERFORMANCE</div>", unsafe_allow_html=True)

        # Simulated equity curve for UI demo.
        np.random.seed(42)
        ret = df["Close"].pct_change().dropna()
        strategy_multiplier = 1.3
        sim_ret = ret * strategy_multiplier + np.random.normal(0, 0.002, len(ret))
        st.plotly_chart(
            portfolio_equity_chart(sim_ret),
            width="stretch", config={"displayModeBar": False}
        )

        # Backtest KPIs.
        b1, b2, b3, b4, b5 = st.columns(5)
        bt_kpis = [
            (b1, "TOTAL RETURN", f"{bt['total_return']:+.1f}%",
             "#10b981" if bt["total_return"] > 0 else "#ef4444"),
            (b2, "SHARPE", f"{bt['sharpe_ratio']:.2f}", "#818cf8"),
            (b3, "MAX DD", f"{bt['max_drawdown']:.1f}%", "#ef4444"),
            (b4, "WIN RATE", f"{bt['win_rate']:.1f}%", "#22d3ee"),
            (b5, "TRADES", f"{bt['num_trades']}", "#f59e0b"),
        ]
        for col, label, val, color in bt_kpis:
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                  <div class='metric-label'>{label}</div>
                  <div class='metric-value' style='color:{color};font-size:1.2rem'>{val}</div>
                </div>
                """, unsafe_allow_html=True)

    with heat_col:
        st.markdown("<div class='section-header'>WATCHLIST PERFORMANCE</div>",
                    unsafe_allow_html=True)
        if watchlist:
            with st.spinner("Loading watchlist..."):
                fig_heat = multi_ticker_heatmap(watchlist)
            if fig_heat.data:
                st.plotly_chart(fig_heat, width="stretch",
                                config={"displayModeBar": False})

        # Risk and opportunity tags.
        st.markdown("<div class='section-header'>RISK FACTORS</div>", unsafe_allow_html=True)
        risks = [
            "Monitor for overbought conditions",
            "Elevated implied volatility regime",
            "Check for earnings calendar",
            "Watch macro data releases",
        ]
        opps = [
            "Momentum aligned with trend",
            "Volume confirms price action",
            "Strong sector rotation signal",
        ]
        risk_html = "".join([f"<span class='tag tag-risk'>⚠ {r}</span>" for r in risks])
        opp_html = "".join([f"<span class='tag tag-opp'>✓ {o}</span>" for o in opps])
        st.markdown(risk_html + "<br><br>" + opp_html, unsafe_allow_html=True)

    # Footer.
    st.markdown("""
    <hr style='border-color:#1e293b;margin:2rem 0 0.5rem'>
    <div style='display:flex;justify-content:space-between;align-items:center;
                font-size:0.65rem;color:#374151;font-family:JetBrains Mono;padding-bottom:1rem'>
      <div>AlphaStack · BiLSTM-Attention + XGBoost Ensemble</div>
      <div>Data: Yahoo Finance · For educational purposes only · Not financial advice</div>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()