# 📈 AlphaStack — Quantitative Ensemble Deep Learning System

A production-grade AI/ML system for financial market prediction across **commodities, crypto, and equities** — featuring GPU-accelerated deep learning, custom VWAP-EMA ribbon signals, a strictly isolated time-series preprocessing engine, and an interactive Dockerized dashboard.

---

## 🚀 Quick Start 

### Option A: Docker Deployment (Recommended)
Launch the fully containerized environment and dashboard in two commands:
```bash
# 1. Build the image
docker build -t alphastack .

# 2. Run the container
docker run -p 8501:8501 alphastack
```
*Access the dashboard at `http://localhost:8501`*

### Option B: Local Python Environment
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train all 6 assets  ← single command, no arguments needed
python train_model.py

# 3. Launch interactive dashboard
streamlit run app.py
```
`train_model.py` auto-detects your GPU, trains all assets sequentially with Mixed Precision (AMP), and outputs a metrics summary.

---

## 🎯 Universe — 6 Assets

| Key    | Symbol    | Asset Class | Notes |
|--------|-----------|-------------|-------|
| `GOLD` | `GC=F`    | Commodity   | Gold futures |
| `BTC`  | `BTC-USD` | Crypto      | 24/7 trading, high volatility |
| `OIL`  | `CL=F`    | Commodity   | WTI crude futures |
| `AAPL` | `AAPL`    | Equity      | Apple Inc. |
| `MSFT` | `MSFT`    | Equity      | Microsoft Corp. |
| `NVDA` | `NVDA`    | Equity      | NVIDIA Corp. |

---

## 🏗️ Architecture

```text
AlphaStack/
├── Dockerfile                      ← Containerization config
├── train_model.py                  ← ONE COMMAND training entry point
├── app.py                          ← Streamlit interactive dashboard
├── config.py                       ← Central config (tickers, GPU, hyperparams)
├── src/
│   ├── data_loader.py              ← yfinance API + caching mechanisms
│   ├── preprocessing.py            ← Sequence-first, leak-proof MinMax scaling
│   ├── ai_analyst.py               ← Rule-based & LLM market commentary
│   ├── models/
│   │   └── model_lstm.py           ← BiLSTM + Multi-head Attention + MC Dropout
│   ├── training/
│   │   └── trainer.py              ← GPU training loop: AMP, OneCycleLR, Early Stopping
│   ├── features/
│   │   └── technical_indicators.py ← 30+ standard TIs + Custom VWAP-EMA Signals
│   └── evaluation/
│       └── backtesting.py          ← Vectorized backtester + forecasting metrics
```

---

## ⚡ Training Strategy — Speed + Accuracy

| Component | Choice | Why |
|-----------|--------|-----|
| **Architecture** | BiLSTM-Attention | Best speed/accuracy trade-off for sequential time series |
| **Scheduler** | OneCycleLR | Converges 40–60% faster than cosine over 150 epochs |
| **Precision** | FP16 AMP | ~2× GPU throughput, prevents memory overflow on raw prices |
| **Optimizer** | Fused AdamW | ~15% faster CUDA kernel execution |
| **Compilation** | `torch.compile` | Extra speedup on PyTorch ≥ 2.0 (Linux/WSL native) |
| **cuDNN** | `benchmark=True` | Auto-selects fastest convolution kernel for the active GPU |
| **XGBoost** | `device=cuda` | GPU-accelerated histogram tree building |
| **Ensemble** | Weighted avg | Zero extra training — weights derived via softmax(-MAE_val) |

### Expected training time

| GPU | Per-asset | All 6 assets |
|-----|-----------|-------------|
| RTX 4090 | ~20 sec | ~2 min |
| RTX 4050 / 3060 | ~30 sec | ~3 min |
| CPU only | ~4 min | ~25 min |

---

## 🧠 Model Details

**BiLSTM-Attention** (primary)
- 2× Bidirectional LSTM blocks with residual projections.
- Multi-head self-attention (4 heads) over the temporal dimension.
- Monte Carlo Dropout (30 forward passes) → Generates 95% confidence intervals.
- ~2.5M parameters, trains in < 30 seconds per asset on modern GPUs.

**XGBoost** (complement)
- Features: `[last_step | mean | std | min | max]` × all TA columns.
- GPU histogram-based tree building (`tree_method="hist"`, `device="cuda"`).
- Captures complex, non-linear interactions between technical indicators.

**Weighted Ensemble**
- Weight = softmax(−MAE_val) → better-performing model gets higher combination weight.
- MC Dropout CI propagated to the final ensemble prediction.

---

## 📊 Features Engineering

**Custom Alpha Signals:**
- **VWAP-EMA Ribbon**: Proprietary signal combining rolling VWAP with a Fibonacci-sequence EMA Ribbon (13, 21, 34, 55, 89) for strict trend alignment and volume confirmation.

**30+ Standard Technical Indicators:**
- **Momentum**: RSI (7/14/21), Stochastic %K/%D, Williams %R, ROC (5/10/20), CCI, MFI, Momentum
- **Trend**: MACD (12/26/9), EMA (9/21/50/100/200), SMA (10/20/50/200), ADX, Parabolic SAR, Ichimoku
- **Volatility**: Bollinger Bands (20/50), ATR (7/14/21), Historical Vol (10/20/30D), Keltner, Donchian
- **Volume**: OBV, VWAP, CMF, Volume oscillator, ADL
- **Price**: Candlestick body/shadow, Doji, Hammer, Gap detection
- **Calendar**: Day-of-week/month/quarter (sin/cos encoded)
- **Lags & Rolling**: 1/2/3/5/10/20-bar price/return lags; 5/10/20/60-bar mean, std, skew, kurtosis

---

## ⚙️ Advanced CLI Options

```bash
# Train specific assets only
python train_model.py --tickers GOLD BTC OIL

# Override training hyperparameters
python train_model.py --epochs 60 --window 30 --horizon 1

# Skip assets that already have checkpoints
python train_model.py --skip-existing

# Force re-download all data (ignore cache)
python train_model.py --no-cache
```

---

## 🖥️ System Requirements

- **CUDA 11.8+** recommended (NVIDIA) for full acceleration.
- **Minimum VRAM**: 4 GB.
- **Apple Silicon**: MPS backend supported automatically.
- **OS**: Linux/WSL2 recommended for `torch.compile` support (gracefully skips on Windows).

```bash
# Verify GPU is detected
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## ⚠️ Disclaimer

For educational and portfolio showcase purposes only. Not financial advice.