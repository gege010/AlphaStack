"""
config.py — Central configuration for Market Prediction AI
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
import torch

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models_saved"
REPORTS_DIR = ROOT_DIR / "reports"
LOGS_DIR = ROOT_DIR / "logs"

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ── GPU detection & global CUDA optimisations ──────────────────────────────────
def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        # Allow cuDNN to auto-tune kernel selection for this hardware
        torch.backends.cudnn.benchmark = True
        # Keep determinism off for max throughput
        torch.backends.cudnn.deterministic = False
        # Enable TF32 on Ampere+ GPUs (free ~3× speedup on matmuls)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        return torch.device("cuda")
    if torch.backends.mps.is_available():          # Apple Silicon
        return torch.device("mps")
    return torch.device("cpu")                     # CPU fallback (warns at runtime)

DEVICE: torch.device = _resolve_device()

# ── Target universe ────────────────────────────────────────────────────────────
# Maps a short key → (yfinance symbol, display name, asset class)
UNIVERSE: Dict[str, tuple] = {
    "GOLD":  ("GC=F",    "Gold Futures",       "commodity"),
    "BTC":   ("BTC-USD", "Bitcoin",            "crypto"),
    "OIL":   ("CL=F",    "WTI Crude Oil",      "commodity"),
    "AAPL":  ("AAPL",    "Apple Inc.",         "equity"),
    "MSFT":  ("MSFT",    "Microsoft Corp.",    "equity"),
    "NVDA":  ("NVDA",    "NVIDIA Corp.",       "equity"),
}
# Ordered list of keys (used by train_model.py)
TRAIN_TICKERS: List[str] = list(UNIVERSE.keys())


# ── Data Config ────────────────────────────────────────────────────────────────
@dataclass
class DataConfig:
    start_date: str = "2018-01-01"    # ~6 years of history
    end_date: Optional[str] = None    # None = today
    interval: str = "1d"
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    target_col: str = "Close"
    # Window: 40 bars gives enough context without blowing memory/time
    feature_window: int = 40
    # Predict next-day close by default
    horizon: int = 1


# ── Feature Config ─────────────────────────────────────────────────────────────
@dataclass
class FeatureConfig:
    use_technical: bool = True
    use_sentiment: bool = False    # Requires API key
    use_macro: bool = False        # Requires Alpha Vantage key
    use_multi_timeframe: bool = False  # Weekly/monthly alignment (slower)


# ── BiLSTM-Attention Config ────────────────────────────────────────────────────
# Balanced architecture: accurate enough, trains in < 3 min/ticker on GPU.
@dataclass
class LSTMConfig:
    hidden_dim: int = 192          # Sweet spot: large enough, not wasteful
    num_layers: int = 2            # 2 stacked BiLSTM blocks (vs 3 before)
    dropout: float = 0.25
    bidirectional: bool = True
    attention_heads: int = 4
    use_layer_norm: bool = True
    mc_dropout_samples: int = 30   # MC Dropout inference passes


# ── Transformer Config (TFT — optional, heavier) ──────────────────────────────
@dataclass
class TransformerConfig:
    d_model: int = 128
    nhead: int = 8
    num_encoder_layers: int = 3
    dropout: float = 0.1
    use_positional_encoding: bool = True
    use_gating: bool = True


# ── Training Config — tuned for GPU speed + accuracy ──────────────────────────
# Strategy: OneCycleLR achieves near-optimal weights in 50–80 epochs,
# far faster than cosine annealing over 150 epochs.
@dataclass
class TrainingConfig:
    batch_size: int = 128          # Larger batch → better GPU utilisation
    epochs: int = 80               # OneCycleLR converges early; ES cuts further
    learning_rate: float = 5e-4    # Base LR for OneCycleLR (peak = 10×)
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    early_stopping_patience: int = 15   # Stop early if plateau
    lr_scheduler: str = "onecycle"      # Best scheduler for fast convergence
    loss_fn: str = "huber"              # Robust to outliers (crypto spikes etc.)
    optimizer: str = "adamw"
    use_mixed_precision: bool = True    # AMP: ~2× speedup on CUDA with zero loss
    compile_model: bool = True          # torch.compile() on PyTorch ≥ 2.0
    seed: int = 42
    num_workers: int = 4               # DataLoader workers
    pin_memory: bool = True            # Pinned memory for faster H2D transfer


# ── Ensemble Config ────────────────────────────────────────────────────────────
# Weighted ensemble (no meta-training) → zero extra training time.
# Weights derived from inverse-val-loss of each base model.
@dataclass
class EnsembleConfig:
    base_models: List[str] = field(default_factory=lambda: ["lstm", "xgboost"])
    # "ridge" or "weighted_avg" — weighted_avg needs no training step
    meta_learner: str = "weighted_avg"
    use_uncertainty_weighting: bool = True
    xgb_n_estimators: int = 400
    xgb_max_depth: int = 5
    xgb_learning_rate: float = 0.06
    xgb_subsample: float = 0.8
    xgb_colsample: float = 0.75


# ── Backtest Config ────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    initial_capital: float = 100_000.0
    commission: float = 0.001          # 0.1 %
    slippage: float = 0.0005
    position_sizing: str = "kelly"
    max_position_pct: float = 0.20
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.15
    risk_free_rate: float = 0.04


# ── API Config ─────────────────────────────────────────────────────────────────
@dataclass
class APIConfig:
    alpha_vantage_key: Optional[str] = None
    news_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None


# ── Master Config ──────────────────────────────────────────────────────────────
@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    api: APIConfig = field(default_factory=APIConfig)

    def load_env(self):
        from dotenv import load_dotenv
        import os
        load_dotenv()
        self.api.alpha_vantage_key = os.getenv("ALPHA_VANTAGE_KEY")
        self.api.news_api_key = os.getenv("NEWS_API_KEY")
        self.api.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        return self


# Singleton
cfg = Config()
