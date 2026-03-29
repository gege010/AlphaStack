"""CLI entrypoint for end-to-end model training across selected tickers."""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import xgboost as xgb
from loguru import logger

# Project imports.
from config import cfg, DEVICE, UNIVERSE, TRAIN_TICKERS, MODELS_DIR, LOGS_DIR
from src.data_loader import MarketDataLoader
from src.preprocessing import MarketPreprocessor, ProcessedData
from src.models.model_lstm import BiLSTMAttentionModel
from src.training.trainer import ModelTrainer
from src.evaluation.backtesting import compute_all_metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AlphaStack — train all assets with one command.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python train_model.py --tickers GOLD BTC --epochs 60",
    )
    p.add_argument(
        "--tickers", nargs="+", default=TRAIN_TICKERS,
        choices=list(UNIVERSE.keys()),
        metavar="TICKER",
        help=f"Assets to train. Default: all ({', '.join(TRAIN_TICKERS)})",
    )
    p.add_argument("--epochs",  type=int, default=None, help="Override max epoch count")
    p.add_argument("--horizon", type=int, default=None, help="Prediction horizon in days (default: 1)")
    p.add_argument("--window",  type=int, default=None, help="Lookback window in bars (default: 40)")
    p.add_argument("--no-cache", action="store_true", help="Force re-download all data")
    p.add_argument(
        "--skip-existing", action="store_true",
        help="Skip a ticker if its checkpoint already exists",
    )
    return p.parse_args()


def setup(args: argparse.Namespace) -> None:
    """Seeds, logging, optional CLI overrides."""
    import random
    seed = cfg.training.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    # Apply CLI overrides to runtime config.
    if args.epochs:
        cfg.training.epochs = args.epochs
    if args.horizon:
        cfg.data.horizon = args.horizon
    if args.window:
        cfg.data.feature_window = args.window

    LOGS_DIR.mkdir(exist_ok=True)
    logger.remove()
    logger.add(sys.stdout, format="<green>{time:HH:mm:ss}</green> | {message}", level="INFO")
    logger.add(
        LOGS_DIR / "train_{time:YYYYMMDD_HHmmss}.log",
        rotation="50 MB", level="DEBUG"
    )


def load_and_preprocess(
    ticker_key: str, use_cache: bool
) -> Tuple[ProcessedData, MarketPreprocessor]:
    """
    Fetch OHLCV, engineer 30+ TA features, scale, build sequences.
    Returns ProcessedData (train/val/test) and fitted preprocessor.
    """
    yf_symbol, display_name, asset_class = UNIVERSE[ticker_key]
    logger.info(f"  → Data: {display_name} ({yf_symbol}) [{asset_class}]")

    loader = MarketDataLoader(cache=use_cache)
    raw_df = loader.load(
        yf_symbol,
        start=cfg.data.start_date,
        end=cfg.data.end_date,
    )

    scaler_path = MODELS_DIR / ticker_key / "scalers"
    preprocessor = MarketPreprocessor(
        window=cfg.data.feature_window,
        horizon=cfg.data.horizon,
        train_ratio=cfg.data.train_ratio,
        val_ratio=cfg.data.val_ratio,
        scaler_save_path=scaler_path,
    )
    data = preprocessor.fit_transform(raw_df)

    logger.info(
        f"  → Sequences  train={data.X_train.shape}  "
        f"val={data.X_val.shape}  test={data.X_test.shape}"
    )
    return data, preprocessor


def train_lstm(
    data: ProcessedData, ticker_key: str
) -> BiLSTMAttentionModel:
    """
    Build + train BiLSTM-Attention on GPU.
    OneCycleLR + AMP + fused AdamW + optional torch.compile.
    """
    input_dim = data.X_train.shape[-1]
    model = BiLSTMAttentionModel(
        input_dim=input_dim,
        hidden_dim=cfg.lstm.hidden_dim,
        num_layers=cfg.lstm.num_layers,
        dropout=cfg.lstm.dropout,
        bidirectional=cfg.lstm.bidirectional,
        attention_heads=cfg.lstm.attention_heads,
    )

    trainer = ModelTrainer(
        model=model,
        config=cfg,
        save_dir=MODELS_DIR / ticker_key,
        model_name="bilstm",
    )
    trainer.fit(data.X_train, data.y_train, data.X_val, data.y_val)
    return trainer.model


def train_xgboost(
    data: ProcessedData, ticker_key: str
) -> xgb.XGBRegressor:
    """
    Train XGBoost on the last-step (most recent bar) of each sequence.
    Captures non-linear interactions between TA indicators.
    Fast: ~20-30 s per ticker on CPU.
    """
    logger.info("  → XGBoost: building flat features from last sequence step …")
    # Use last step plus sequence statistics as tabular features.
    X_flat_train = _flatten_sequences(data.X_train)
    X_flat_val   = _flatten_sequences(data.X_val)

    ec = cfg.ensemble
    model_path = MODELS_DIR / ticker_key / "xgb_model.json"

    booster = xgb.XGBRegressor(
        n_estimators=ec.xgb_n_estimators,
        max_depth=ec.xgb_max_depth,
        learning_rate=ec.xgb_learning_rate,
        subsample=ec.xgb_subsample,
        colsample_bytree=ec.xgb_colsample,
        min_child_weight=3,
        reg_alpha=0.05,
        reg_lambda=1.0,
        n_jobs=-1,
        random_state=cfg.training.seed,
        early_stopping_rounds=30,
        eval_metric="rmse",
        tree_method="hist",       # Histogram tree builder is faster for large tabular data.
        device="cuda" if DEVICE.type == "cuda" else "cpu",  # Use CUDA backend when available.
    )
    booster.fit(
        X_flat_train, data.y_train,
        eval_set=[(X_flat_val, data.y_val)],
        verbose=False,
    )
    booster.save_model(str(model_path))
    logger.info(f"  → XGBoost saved to {model_path}")
    return booster


def _flatten_sequences(X: np.ndarray) -> np.ndarray:
    """
    Summarise each (window, features) sequence into a flat vector:
    [last_step, mean, std, min, max]  →  5 × n_features columns.
    This preserves temporal statistics without blowing up dimensionality.
    """
    last  = X[:, -1, :]
    mean  = X.mean(axis=1)
    std   = X.std(axis=1)
    mn    = X.min(axis=1)
    mx    = X.max(axis=1)
    return np.concatenate([last, mean, std, mn, mx], axis=1)


def build_ensemble_weights(
    lstm_model: BiLSTMAttentionModel,
    xgb_model: xgb.XGBRegressor,
    data: ProcessedData,
    preprocessor: MarketPreprocessor,
) -> Dict[str, float]:
    """
    Compute ensemble weights from *validation set* performance.
    Weight = softmax(−val_loss) so better models get higher weight.
    No extra training round needed.
    """
    lstm_model.eval()
    X_val_t = torch.from_numpy(data.X_val).float().to(DEVICE)

    with torch.no_grad():
        lstm_raw, _ = lstm_model(X_val_t)
    lstm_pred = preprocessor.inverse_transform_target(
        lstm_raw.cpu().numpy().squeeze()
    )

    xgb_pred = preprocessor.inverse_transform_target(
        xgb_model.predict(_flatten_sequences(data.X_val))
    )
    y_true = preprocessor.inverse_transform_target(data.y_val)

    from src.evaluation.backtesting import mae as _mae
    loss_lstm = _mae(y_true, lstm_pred)
    loss_xgb  = _mae(y_true, xgb_pred)

    # Convert validation losses to normalized ensemble weights.
    scores  = np.array([-loss_lstm, -loss_xgb])
    scores -= scores.max()                       # Improve softmax numerical stability.
    weights = np.exp(scores) / np.exp(scores).sum()

    w = {"lstm": float(weights[0]), "xgb": float(weights[1])}
    logger.info(f"  → Ensemble weights: BiLSTM={w['lstm']:.3f}  XGBoost={w['xgb']:.3f}")
    return w


def ensemble_predict(
    lstm_model:  BiLSTMAttentionModel,
    xgb_model:   xgb.XGBRegressor,
    weights:     Dict[str, float],
    X_seq:       np.ndarray,
    preprocessor: MarketPreprocessor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Produce weighted ensemble prediction + 95 % CI from MC Dropout.
    Returns: (mean_pred, lower_95, upper_95)  — all in original price scale.
    """
    lstm_model.eval()
    X_t = torch.from_numpy(X_seq).float().to(DEVICE)

    # Estimate predictive uncertainty via MC dropout sampling.
    mc_mean, mc_std, _ = lstm_model.predict_with_uncertainty(
        X_t, n_samples=cfg.lstm.mc_dropout_samples
    )
    lstm_pred = preprocessor.inverse_transform_target(mc_mean.squeeze())
    lstm_std  = preprocessor.inverse_transform_target(mc_std.squeeze()) - \
                preprocessor.inverse_transform_target(np.zeros_like(mc_std.squeeze()))

    xgb_pred = preprocessor.inverse_transform_target(
        xgb_model.predict(_flatten_sequences(X_seq))
    )

    ensemble_pred = weights["lstm"] * lstm_pred + weights["xgb"] * xgb_pred

    # Build 95% interval from LSTM uncertainty term.
    lower = ensemble_pred - 1.96 * np.abs(lstm_std)
    upper = ensemble_pred + 1.96 * np.abs(lstm_std)
    return ensemble_pred, lower, upper


def evaluate(
    lstm_model: BiLSTMAttentionModel,
    xgb_model: xgb.XGBRegressor,
    weights: Dict[str, float],
    data: ProcessedData,
    preprocessor: MarketPreprocessor,
    ticker_key: str,
) -> Dict:
    """Evaluate all three models on held-out test set, print table, save JSON."""
    y_true = preprocessor.inverse_transform_target(data.y_test)

    # Evaluate BiLSTM.
    lstm_model.eval()
    X_t = torch.from_numpy(data.X_test).float().to(DEVICE)
    with torch.no_grad():
        lstm_raw, _ = lstm_model(X_t)
    lstm_pred = preprocessor.inverse_transform_target(lstm_raw.cpu().numpy().squeeze())

    # Evaluate XGBoost.
    xgb_pred = preprocessor.inverse_transform_target(
        xgb_model.predict(_flatten_sequences(data.X_test))
    )

    # Evaluate weighted ensemble.
    ens_pred, _, _ = ensemble_predict(
        lstm_model, xgb_model, weights, data.X_test, preprocessor
    )

    results = {
        "BiLSTM":   compute_all_metrics(y_true, lstm_pred),
        "XGBoost":  compute_all_metrics(y_true, xgb_pred),
        "Ensemble": compute_all_metrics(y_true, ens_pred),
    }

    # Log compact metrics table.
    cols = ["rmse", "mae", "mape", "directional_accuracy", "r2"]
    header = f"  {'Model':<12}" + "".join(f"  {c.upper()[:8]:>9}" for c in cols)
    logger.info(f"\n  ── Test results: {ticker_key} {'─'*40}")
    logger.info(header)
    for model_name, m in results.items():
        row = f"  {model_name:<12}" + "".join(f"  {m[c]:>9.4f}" for c in cols)
        logger.info(row)

    # Persist per-ticker evaluation metrics.
    save_path = MODELS_DIR / ticker_key / "eval_results.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  → Results saved: {save_path}")
    return results


def train_ticker(ticker_key: str, args: argparse.Namespace) -> Dict:
    yf_symbol, display_name, asset_class = UNIVERSE[ticker_key]
    ckpt_path = MODELS_DIR / ticker_key / "bilstm_best.pt"

    # Skip retraining when checkpoint already exists.
    if args.skip_existing and ckpt_path.exists():
        logger.info(f"  [SKIP] {ticker_key} checkpoint found. Use --no-skip to retrain.")
        return {}

    sep = "═" * 62
    logger.info(f"\n{sep}")
    logger.info(f"  ASSET  : {ticker_key:6s}  {display_name}  [{asset_class}]")
    logger.info(f"  SYMBOL : {yf_symbol}")
    logger.info(f"  DEVICE : {DEVICE}  |  AMP: {cfg.training.use_mixed_precision}")
    logger.info(f"  SCHED  : {cfg.training.lr_scheduler.upper()}  |  "
                f"epochs={cfg.training.epochs}  batch={cfg.training.batch_size}  "
                f"window={cfg.data.feature_window}  horizon={cfg.data.horizon}")
    logger.info(sep)

    t_start = time.perf_counter()

    # Step 1: load and preprocess data.
    data, preprocessor = load_and_preprocess(ticker_key, use_cache=not args.no_cache)

    # Step 2: train BiLSTM.
    logger.info("  [1/3] BiLSTM-Attention …")
    lstm_model = train_lstm(data, ticker_key)

    # Step 3: train XGBoost.
    logger.info("  [2/3] XGBoost …")
    xgb_model = train_xgboost(data, ticker_key)

    # Step 4: compute ensemble weights.
    logger.info("  [3/3] Computing ensemble weights …")
    weights = build_ensemble_weights(lstm_model, xgb_model, data, preprocessor)

    # Persist ensemble weights for inference.
    w_path = MODELS_DIR / ticker_key / "ensemble_weights.json"
    with open(w_path, "w") as f:
        json.dump(weights, f, indent=2)

    # Step 5: evaluate on test split.
    results = evaluate(lstm_model, xgb_model, weights, data, preprocessor, ticker_key)

    elapsed = time.perf_counter() - t_start
    logger.success(f"  ✔  {ticker_key} complete in {elapsed:.1f}s\n")
    return results


def main() -> None:
    args = parse_args()
    setup(args)

    # Startup banner.
    gpu_info = (
        torch.cuda.get_device_name(0) if DEVICE.type == "cuda"
        else "MPS" if DEVICE.type == "mps"
        else "CPU ⚠  (no GPU detected)"
    )
    logger.info(
        f"\n"
        f"  ╔{'═'*58}╗\n"
        f"  ║  AlphaStack — Training Pipeline          ║\n"
        f"  ║  Device  : {gpu_info:<46}║\n"
        f"  ║  Assets  : {', '.join(args.tickers):<46}║\n"
        f"  ║  Method  : BiLSTM-Attn + XGBoost → Weighted Ensemble    ║\n"
        f"  ║  Sched   : OneCycleLR (AMP + torch.compile if CUDA)      ║\n"
        f"  ╚{'═'*58}╝"
    )

    # Train each selected ticker.
    all_results = {}
    wall_start  = time.perf_counter()

    for i, ticker_key in enumerate(args.tickers, 1):
        logger.info(f"\n  [{i}/{len(args.tickers)}] Training {ticker_key} …")
        try:
            all_results[ticker_key] = train_ticker(ticker_key, args)
        except Exception as e:
            logger.error(f"  ✘ {ticker_key} failed: {e}")
            import traceback; traceback.print_exc()
            continue

    total = time.perf_counter() - wall_start

    # Print aggregate summary.
    logger.info(f"\n{'═'*62}")
    logger.info("  TRAINING COMPLETE — SUMMARY")
    logger.info(f"{'═'*62}")
    logger.info(f"  {'Ticker':<8}  {'Model':<10}  {'RMSE':>8}  {'Dir.Acc%':>9}  {'R²':>8}")
    logger.info(f"  {'─'*8}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*8}")
    for tkr, res in all_results.items():
        if not res:
            continue
        ens = res.get("Ensemble", {})
        logger.info(
            f"  {tkr:<8}  {'Ensemble':<10}  "
            f"{ens.get('rmse',0):>8.4f}  "
            f"{ens.get('directional_accuracy',0):>9.1f}  "
            f"{ens.get('r2',0):>8.4f}"
        )
    logger.info(f"{'─'*62}")
    logger.info(f"  Total wall-clock: {total:.1f}s ({total/60:.1f} min)")
    logger.info(f"  Models saved to : {MODELS_DIR}/")
    logger.info(f"{'═'*62}\n")

    # Save aggregate training summary.
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"total_seconds": total, "results": all_results}, f, indent=2)
    logger.success(f"  Summary → {summary_path}")


if __name__ == "__main__":
    main()
