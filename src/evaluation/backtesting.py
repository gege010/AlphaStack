"""
src/evaluation/metrics.py + backtesting.py — Comprehensive model evaluation
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional
from dataclasses import dataclass, field
from loguru import logger


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION METRICS
# ══════════════════════════════════════════════════════════════════════════════

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2 + 1e-10
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100)

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """% of times the predicted direction matches actual direction."""
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return float(np.mean(true_dir == pred_dir) * 100)

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / (ss_tot + 1e-10))

def theil_u(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Theil's U2 — values < 1 beat naive forecast."""
    naive = y_true[:-1]
    actual = y_true[1:]
    pred = y_pred[1:]
    return float(
        np.sqrt(np.mean((actual - pred) ** 2)) /
        (np.sqrt(np.mean((actual - naive) ** 2)) + 1e-10)
    )

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
        "theil_u": theil_u(y_true, y_pred),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTING ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Trade:
    entry_date: pd.Timestamp
    exit_date: Optional[pd.Timestamp]
    entry_price: float
    exit_price: Optional[float]
    direction: int           # +1 long, -1 short
    shares: float
    pnl: float = 0.0
    return_pct: float = 0.0
    hit_stop_loss: bool = False
    hit_take_profit: bool = False


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    num_trades: int
    avg_trade_return: float
    avg_holding_days: float
    portfolio_values: pd.Series
    trades: list = field(default_factory=list)
    benchmark_return: float = 0.0
    alpha: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {k: v for k, v in self.__dict__.items()
                if isinstance(v, (int, float))}


class VectorizedBacktester:
    """
    Event-driven backtester with:
    - Transaction costs + slippage
    - Stop-loss / take-profit orders
    - Kelly criterion position sizing
    - Long and short support
    - Performance attribution
    """

    def __init__(
        self,
        initial_capital: float = 100_000,
        commission: float = 0.001,
        slippage: float = 0.0005,
        stop_loss_pct: float = 0.05,
        take_profit_pct: float = 0.15,
        max_position_pct: float = 0.2,
        risk_free_rate: float = 0.04,
        position_sizing: str = "kelly",
    ):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.max_position_pct = max_position_pct
        self.risk_free_rate = risk_free_rate
        self.position_sizing = position_sizing

    def run(
        self,
        prices: pd.Series,
        predictions: np.ndarray,
        signal_threshold: float = 0.005,
        allow_short: bool = True,
    ) -> BacktestResult:
        """
        Run backtest given price series and model predictions.

        Signal logic:
        - Buy if predicted_return > threshold
        - Sell/Short if predicted_return < -threshold
        - Hold otherwise

        prices: pd.Series with DatetimeIndex
        predictions: np.ndarray of predicted prices (same length as prices)
        """
        prices = prices.copy()
        pred_returns = pd.Series(
            (predictions - prices.values) / prices.values,
            index=prices.index
        )

        capital = self.initial_capital
        position = 0.0          # number of shares
        entry_price = 0.0
        entry_date = None
        trades = []
        portfolio_values = []

        for i, (date, price) in enumerate(prices.items()):
            pred_ret = pred_returns.iloc[i]
            port_value = capital + position * price
            portfolio_values.append(port_value)

            # Check stop-loss / take-profit
            if position != 0 and entry_price > 0:
                current_ret = (price - entry_price) / entry_price * np.sign(position)
                hit_sl = current_ret <= -self.stop_loss_pct
                hit_tp = current_ret >= self.take_profit_pct

                if hit_sl or hit_tp:
                    proceeds = self._execute_close(position, price)
                    pnl = proceeds - abs(position) * entry_price * np.sign(position)
                    trades.append(Trade(
                        entry_date=entry_date, exit_date=date,
                        entry_price=entry_price, exit_price=price,
                        direction=int(np.sign(position)), shares=abs(position),
                        pnl=pnl, return_pct=current_ret * 100,
                        hit_stop_loss=hit_sl, hit_take_profit=hit_tp,
                    ))
                    capital += proceeds
                    position = 0.0
                    continue

            # Generate new signal
            signal = 0
            if pred_ret > signal_threshold:
                signal = 1
            elif pred_ret < -signal_threshold and allow_short:
                signal = -1

            # Execute trade
            if signal != 0 and position == 0:
                shares = self._size_position(capital, price, signal, pred_ret)
                cost = self._execute_open(shares * signal, price)
                capital -= cost
                position = shares * signal
                entry_price = price
                entry_date = date

            elif signal == 0 and position != 0:
                # Close position
                proceeds = self._execute_close(position, price)
                ret = (price - entry_price) / entry_price * np.sign(position)
                trades.append(Trade(
                    entry_date=entry_date, exit_date=date,
                    entry_price=entry_price, exit_price=price,
                    direction=int(np.sign(position)), shares=abs(position),
                    pnl=proceeds - abs(position) * entry_price,
                    return_pct=ret * 100,
                ))
                capital += proceeds
                position = 0.0

        # Close any open position at end
        if position != 0:
            final_price = prices.iloc[-1]
            proceeds = self._execute_close(position, final_price)
            capital += proceeds

        portfolio_series = pd.Series(portfolio_values, index=prices.index)
        result = self._compute_statistics(portfolio_series, trades, prices)
        logger.info(
            f"Backtest complete | Total return: {result.total_return:.2f}% | "
            f"Sharpe: {result.sharpe_ratio:.3f} | Trades: {result.num_trades}"
        )
        return result

    def _size_position(self, capital: float, price: float,
                       signal: int, pred_ret: float) -> float:
        """Position sizing using Kelly criterion or fixed fraction."""
        max_shares = (capital * self.max_position_pct) / price
        if self.position_sizing == "kelly":
            # Simplified Kelly: f = (edge / odds) capped at max
            win_prob = min(max(abs(pred_ret) * 10, 0.3), 0.7)
            kelly_f = (win_prob * 2 - 1) / 1.0   # win/loss ratio assumed 1
            kelly_f = max(0.05, min(kelly_f, self.max_position_pct))
            return min((capital * kelly_f) / price, max_shares)
        else:
            return max_shares * 0.5   # Fixed 50% of max

    def _execute_open(self, shares: float, price: float) -> float:
        """Cost to open: price + slippage + commission."""
        slipped = price * (1 + self.slippage * np.sign(shares))
        gross = abs(shares) * slipped
        return gross * (1 + self.commission)

    def _execute_close(self, position: float, price: float) -> float:
        """Proceeds from closing: price - slippage - commission."""
        slipped = price * (1 - self.slippage * np.sign(position))
        gross = abs(position) * slipped
        return gross * (1 - self.commission)

    def _compute_statistics(
        self, portfolio: pd.Series, trades: list, prices: pd.Series
    ) -> BacktestResult:
        returns = portfolio.pct_change().dropna()
        total_ret = (portfolio.iloc[-1] / portfolio.iloc[0] - 1) * 100
        n_years = len(returns) / 252

        ann_ret = ((1 + total_ret / 100) ** (1 / max(n_years, 0.01)) - 1) * 100
        rf_daily = self.risk_free_rate / 252
        excess = returns - rf_daily
        sharpe = (excess.mean() / (excess.std() + 1e-10)) * np.sqrt(252)

        downside = returns[returns < 0].std() + 1e-10
        sortino = (excess.mean() / downside) * np.sqrt(252)

        # Max drawdown
        cum = (1 + returns).cumprod()
        peak = cum.cummax()
        drawdown = ((cum - peak) / peak)
        max_dd = drawdown.min() * 100

        calmar = ann_ret / (abs(max_dd) + 1e-10)

        # Trade stats
        pnls = [t.pnl for t in trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p <= 0]
        win_rate = len(winners) / max(len(trades), 1) * 100
        profit_factor = sum(winners) / (abs(sum(losers)) + 1e-10)
        avg_ret = np.mean([t.return_pct for t in trades]) if trades else 0.0

        # Holding period
        holding_days = []
        for t in trades:
            if t.exit_date and t.entry_date:
                holding_days.append((t.exit_date - t.entry_date).days)

        # Benchmark (buy & hold)
        bm_ret = (prices.iloc[-1] / prices.iloc[0] - 1) * 100

        return BacktestResult(
            total_return=total_ret,
            annualized_return=ann_ret,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            num_trades=len(trades),
            avg_trade_return=avg_ret,
            avg_holding_days=np.mean(holding_days) if holding_days else 0,
            portfolio_values=portfolio,
            trades=trades,
            benchmark_return=bm_ret,
            alpha=ann_ret - bm_ret * (1 / max(n_years, 0.01)),
        )
