"""Training utilities: mixed precision, schedulers, and checkpointing."""
import time
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
)
from loguru import logger

from config import DEVICE


def _check_and_report_device() -> None:
    """Log GPU info; emit a loud warning if training falls back to CPU."""
    if DEVICE.type == "cuda":
        prop = torch.cuda.get_device_properties(0)
        total_gb = prop.total_memory / 1024 ** 3
        logger.info(
            f"GPU  : {prop.name}  |  VRAM: {total_gb:.1f} GB  "
            f"|  CUDA {torch.version.cuda}  |  cuDNN {torch.backends.cudnn.version()}"
        )
        logger.info(f"AMP  : {'enabled (fp16)' if torch.cuda.is_bf16_supported() else 'enabled (fp16)'}")
        logger.info(f"tf32 : matmul={torch.backends.cuda.matmul.allow_tf32}  "
                    f"cudnn={torch.backends.cudnn.allow_tf32}")
    elif DEVICE.type == "mps":
        logger.info("Device: Apple Silicon MPS (Metal Performance Shaders)")
    else:
        logger.warning(
            "⚠  No GPU detected — training on CPU.  "
            "Expected training time will be 10-30× longer.  "
            "Install CUDA-enabled PyTorch: https://pytorch.org/get-started/locally/"
        )


class HuberLoss(nn.Module):
    def __init__(self, delta: float = 1.0):
        super().__init__()
        self.delta = delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.huber_loss(pred, target, delta=self.delta)


class QuantileLoss(nn.Module):
    def __init__(self, quantiles: List[float] = (0.1, 0.5, 0.9)):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        q_tensors = [preds.get("q10"), preds.get("pred"), preds.get("q90")]
        total = torch.tensor(0.0, device=target.device)
        for q, qt in zip(self.quantiles, q_tensors):
            if qt is None:
                continue
            errors = target.unsqueeze(-1) - qt
            total = total + torch.mean(torch.max(q * errors, (q - 1) * errors))
        return total


class EarlyStopping:
    def __init__(self, patience: int = 15, min_delta: float = 1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best = float("inf")
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class ModelTrainer:
    """
    Unified trainer supporting BiLSTM (tuple output) and TFT (dict output).
    All tensors are moved to `DEVICE` (GPU if available).
    """

    LOSS_FNS = {"mse": nn.MSELoss, "mae": nn.L1Loss, "huber": HuberLoss}

    def __init__(
        self,
        model: nn.Module,
        config,
        save_dir: Path,
        model_name: str = "model",
        use_tft_loss: bool = False,
    ):
        _check_and_report_device()

        self.cfg = config
        self.save_dir = Path(save_dir)
        self.model_name = model_name
        self.use_tft_loss = use_tft_loss
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = DEVICE

        # Enable torch.compile only where backend support is stable.
        if (
            config.training.compile_model
            and self.device.type == "cuda"
            and hasattr(torch, "compile")
            and sys.platform != "win32"
        ):
            try:
                model = torch.compile(model, mode="reduce-overhead")
                logger.info("torch.compile() applied — fused kernel mode active.")
            except Exception as e:
                logger.warning(f"torch.compile() skipped: {e}")
        elif sys.platform == "win32" and config.training.compile_model:
            logger.info("Windows detected: Skipping torch.compile() (Triton not supported).")

        self.model = model.to(self.device)

        # Select base loss function.
        self.criterion = (
            QuantileLoss() if use_tft_loss
            else self.LOSS_FNS.get(config.training.loss_fn, HuberLoss)()
        )

        # Use fused AdamW on CUDA for better throughput.
        _optim_kw = dict(lr=config.training.learning_rate, weight_decay=config.training.weight_decay)
        if self.device.type == "cuda":
            _optim_kw["fused"] = True
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **_optim_kw)

        # GradScaler is active only for CUDA mixed-precision training.
        amp_enabled = config.training.use_mixed_precision and self.device.type == "cuda"
        self.scaler = GradScaler(device="cuda", enabled=amp_enabled)
        self.amp_enabled = amp_enabled

        self.history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [], "lr": []
        }
        self.best_val_loss = float("inf")
        self.best_epoch = 0

    def fit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray,   y_val: np.ndarray,
    ) -> Dict[str, List[float]]:
        cfg = self.cfg.training
        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader   = self._make_loader(X_val,   y_val,   shuffle=False)

        scheduler    = self._build_scheduler(len(train_loader), cfg)
        early_stop   = EarlyStopping(patience=cfg.early_stopping_patience)

        n_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logger.info(
            f"{'─'*58}\n"
            f"  Model   : {self.model_name}  ({n_params:,} params)\n"
            f"  Device  : {self.device}\n"
            f"  AMP     : {self.amp_enabled}\n"
            f"  Epochs  : {cfg.epochs}  |  Batch: {cfg.batch_size}  "
            f"|  LR: {cfg.learning_rate}  |  Sched: {cfg.lr_scheduler}\n"
            f"  Train   : {len(X_train):,}  |  Val: {len(X_val):,}\n"
            f"{'─'*58}"
        )

        t0 = time.perf_counter()
        for epoch in range(1, cfg.epochs + 1):
            train_loss = self._train_epoch(train_loader, scheduler, cfg)
            val_loss   = self._eval_epoch(val_loader)

            lr_now = self.optimizer.param_groups[0]["lr"]
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["lr"].append(lr_now)

            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch    = epoch
                self._save_checkpoint("best")

            if epoch % 5 == 0 or epoch == 1:
                elapsed = time.perf_counter() - t0
                eta = elapsed / epoch * (cfg.epochs - epoch)
                logger.info(
                    f"  [{epoch:3d}/{cfg.epochs}]  "
                    f"train={train_loss:.5f}  val={val_loss:.5f}  "
                    f"lr={lr_now:.2e}  best={self.best_val_loss:.5f}  "
                    f"ETA {eta:.0f}s"
                )

            if early_stop(val_loss):
                logger.info(
                    f"  Early stop @ epoch {epoch} "
                    f"(patience {cfg.early_stopping_patience}). "
                    f"Best epoch: {self.best_epoch}."
                )
                break

        total = time.perf_counter() - t0
        logger.success(
            f"  Done in {total:.1f}s — best val loss {self.best_val_loss:.5f} "
            f"(epoch {self.best_epoch})"
        )
        self._load_checkpoint("best")
        self._save_history()
        return self.history

    def _train_epoch(self, loader: DataLoader, scheduler, cfg) -> float:
        self.model.train()
        total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=self.device.type, enabled=self.amp_enabled):
                out = self.model(Xb)
                loss = self._compute_loss(out, yb)

            if self.amp_enabled:
                # AMP path: scaler handles backward and optimizer step.
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Non-AMP path for CPU/MPS devices.
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.gradient_clip)
                self.optimizer.step()

            if isinstance(scheduler, OneCycleLR):
                scheduler.step()

            total += loss.item() * Xb.size(0)
        return total / len(loader.dataset)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> float:
        self.model.eval()
        total = 0.0
        for Xb, yb in loader:
            Xb, yb = Xb.to(self.device, non_blocking=True), yb.to(self.device, non_blocking=True)
            # Use autocast on CUDA; keep no_grad context on other devices.
            ctx = autocast(device_type=self.device.type, enabled=self.amp_enabled)                   if self.device.type == "cuda" else torch.no_grad()
            with ctx:
                out = self.model(Xb)
                loss = self._compute_loss(out, yb)
            total += loss.item() * Xb.size(0)
        return total / len(loader.dataset)

    def _compute_loss(self, out, yb: torch.Tensor) -> torch.Tensor:
        if self.use_tft_loss:
            return self.criterion(out, yb)
        pred = out[0] if isinstance(out, tuple) else out["pred"]
        return self.criterion(pred.squeeze(-1), yb)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
        ds = TensorDataset(
            torch.from_numpy(X).float(),
            torch.from_numpy(y).float(),
        )
        # Keep worker setup compatible across platforms/backends.
        nw = self.cfg.training.num_workers if self.device.type != "mps" else 0
        return DataLoader(
            ds,
            batch_size=self.cfg.training.batch_size,
            shuffle=shuffle,
            num_workers=nw,
            pin_memory=(self.device.type == "cuda"),
            persistent_workers=(nw > 0),
        )

    def _build_scheduler(self, steps_per_epoch: int, cfg):
        if cfg.lr_scheduler == "onecycle":
            # OneCycleLR warmup followed by cosine decay.
            return OneCycleLR(
                self.optimizer,
                max_lr=cfg.learning_rate * 10,
                steps_per_epoch=steps_per_epoch,
                epochs=cfg.epochs,
                pct_start=0.25,         # 25% warmup.
                div_factor=10,
                final_div_factor=1000,
                anneal_strategy="cos",
            )
        if cfg.lr_scheduler == "cosine":
            return CosineAnnealingLR(self.optimizer, T_max=cfg.epochs, eta_min=1e-6)
        if cfg.lr_scheduler == "plateau":
            return ReduceLROnPlateau(self.optimizer, patience=8, factor=0.5, min_lr=1e-6)
        return CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)

    def _save_checkpoint(self, tag: str) -> None:
        path = self.save_dir / f"{self.model_name}_{tag}.pt"
        # Save original module state when torch.compile wraps the model.
        state = (
            self.model._orig_mod.state_dict()
            if hasattr(self.model, "_orig_mod")
            else self.model.state_dict()
        )
        torch.save({"epoch": self.best_epoch, "model_state": state,
                    "val_loss": self.best_val_loss}, path)

    def _load_checkpoint(self, tag: str) -> None:
        path = self.save_dir / f"{self.model_name}_{tag}.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device, weights_only=True)
            target = (
                self.model._orig_mod
                if hasattr(self.model, "_orig_mod")
                else self.model
            )
            target.load_state_dict(ckpt["model_state"])

    def _save_history(self) -> None:
        path = self.save_dir / f"{self.model_name}_history.json"
        with open(path, "w") as f:
            json.dump(self.history, f, indent=2)
