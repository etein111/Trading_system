"""Portfolio-level risk analytics and target-weight policy.

The implementation follows common open-source portfolio-risk patterns used by
PyPortfolioOpt, Riskfolio-Lib, and Qlib portfolio analysis: historical
covariance, marginal risk contribution, concentration and deterministic stress
scenarios. It remains dependency-light so paper decisions are auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class PortfolioRiskDecision:
    target_weights: Dict[str, float]
    adjusted: bool
    active_scale: float
    reasons: list
    before: Dict[str, object]
    after: Dict[str, object]

    def to_dict(self) -> Dict[str, object]:
        return {
            "target_weights": self.target_weights,
            "adjusted": self.adjusted,
            "active_scale": self.active_scale,
            "reasons": self.reasons,
            "before": self.before,
            "after": self.after,
        }


def _history(prices: pd.DataFrame, asof=None, lookback: int = 252) -> pd.DataFrame:
    if prices.empty:
        return prices
    frame = prices.sort_index()
    if asof is not None:
        frame = frame.loc[frame.index <= pd.Timestamp(asof)]
    return frame.tail(int(lookback))


def _weights_from_positions(
    positions: Dict[str, float],
    cash: float,
    prices: pd.DataFrame,
    asof=None,
) -> Tuple[float, Dict[str, float], Dict[str, float]]:
    history = _history(prices, asof=asof, lookback=2)
    if history.empty:
        return float(cash), {}, {}
    latest = history.iloc[-1]
    values = {
        symbol: float(quantity) * float(latest[symbol])
        for symbol, quantity in positions.items()
        if symbol in latest.index and pd.notna(latest[symbol]) and float(latest[symbol]) > 0
    }
    equity = float(cash) + sum(values.values())
    if equity <= 0:
        return equity, {}, values
    return equity, {symbol: value / equity for symbol, value in values.items()}, values


def risk_metrics_for_weights(
    prices: pd.DataFrame,
    weights: Dict[str, float],
    benchmark: str = "SPY",
    asof=None,
    lookback: int = 252,
) -> Dict[str, object]:
    """Estimate risk for fixed weights using historical returns."""

    history = _history(prices, asof=asof, lookback=lookback)
    active = {symbol: float(weight) for symbol, weight in weights.items() if abs(float(weight)) > 1e-12}
    symbols = [symbol for symbol in active if symbol in history.columns]
    if benchmark not in history.columns or not symbols:
        return {
            "asof": str(history.index[-1]) if not history.empty else None,
            "cash_weight": float(1.0 - sum(active.values())),
            "gross_exposure": float(sum(abs(value) for value in active.values())),
            "net_exposure": float(sum(active.values())),
            "active_exposure": float(sum(value for symbol, value in active.items() if symbol != benchmark)),
            "concentration_hhi": float(sum(value * value for value in active.values())),
            "max_position_weight": float(max((abs(value) for value in active.values()), default=0.0)),
            "annualized_volatility": 0.0,
            "beta": 0.0,
            "max_drawdown": 0.0,
            "risk_contributions": {},
            "stress": {},
            "alerts": ["insufficient_price_history"],
        }

    columns = list(dict.fromkeys(symbols + [benchmark]))
    returns = history[columns].pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    asset_returns = returns[symbols]
    covariance = asset_returns.cov().fillna(0.0).to_numpy(dtype=float) * 252.0
    vector = np.asarray([active[symbol] for symbol in symbols], dtype=float)
    variance = float(vector @ covariance @ vector)
    volatility = float(np.sqrt(max(variance, 0.0)))

    benchmark_variance = float(returns[benchmark].var(ddof=1) * 252.0) if len(returns) > 1 else 0.0
    covariance_benchmark = asset_returns.apply(
        lambda series: float(series.cov(returns[benchmark]) * 252.0)
    ).to_numpy(dtype=float)
    beta = float(vector @ covariance_benchmark / benchmark_variance) if benchmark_variance > 1e-12 else 0.0

    portfolio_daily = asset_returns.mul(vector, axis=1).sum(axis=1)
    curve = (1.0 + portfolio_daily).cumprod()
    drawdown = curve / curve.cummax() - 1.0
    marginal = covariance @ vector / volatility if volatility > 1e-12 else np.zeros_like(vector)
    contribution_pct = vector * marginal / volatility if volatility > 1e-12 else np.zeros_like(vector)
    risk_contributions = {symbol: float(value) for symbol, value in zip(symbols, contribution_pct)}
    asset_beta = {
        symbol: float(covariance_benchmark[index] / benchmark_variance)
        if benchmark_variance > 1e-12 else 0.0
        for index, symbol in enumerate(symbols)
    }
    stress = {
        "benchmark_down_5pct": float(sum(active[symbol] * asset_beta[symbol] * -0.05 for symbol in symbols)),
        "benchmark_down_10pct": float(sum(active[symbol] * asset_beta[symbol] * -0.10 for symbol in symbols)),
        "benchmark_down_20pct": float(sum(active[symbol] * asset_beta[symbol] * -0.20 for symbol in symbols)),
        "all_positions_down_10pct": float(-0.10 * sum(abs(active[symbol]) for symbol in symbols)),
        "active_positions_down_20pct": float(-0.20 * sum(abs(active[symbol]) for symbol in symbols if symbol != benchmark)),
    }
    return {
        "asof": str(history.index[-1]),
        "cash_weight": float(1.0 - sum(active.values())),
        "gross_exposure": float(sum(abs(value) for value in active.values())),
        "net_exposure": float(sum(active.values())),
        "active_exposure": float(sum(value for symbol, value in active.items() if symbol != benchmark)),
        "concentration_hhi": float(sum(value * value for value in active.values())),
        "max_position_weight": float(max((abs(value) for value in active.values()), default=0.0)),
        "annualized_volatility": volatility,
        "beta": beta,
        "max_drawdown": float(-drawdown.min()) if not drawdown.empty else 0.0,
        "risk_contributions": risk_contributions,
        "stress": stress,
        "alerts": [],
    }


def analyze_account(account, prices: pd.DataFrame, benchmark: str = "SPY", asof=None, lookback: int = 252, config=None) -> Dict[str, object]:
    """Return a current account risk report suitable for paper-run output."""

    equity, weights, values = _weights_from_positions(account.positions, account.cash, prices, asof=asof)
    metrics = risk_metrics_for_weights(prices, weights, benchmark=benchmark, asof=asof, lookback=lookback)
    metrics["equity"] = float(equity)
    metrics["cash"] = float(account.cash)
    metrics["position_values"] = values
    metrics["weights"] = weights
    thresholds = {
        "max_portfolio_volatility": float(getattr(config, "max_portfolio_volatility", 0.35)),
        "max_portfolio_beta": float(getattr(config, "max_portfolio_beta", 1.25)),
        "portfolio_stress_loss_limit": float(getattr(config, "portfolio_stress_loss_limit", 0.20)),
        "max_single_active_weight": float(getattr(config, "max_single_active_weight", 0.06)),
    }
    alerts = list(metrics.get("alerts", []))
    if metrics.get("annualized_volatility", 0.0) > thresholds["max_portfolio_volatility"]:
        alerts.append("portfolio_volatility_limit")
    if metrics.get("beta", 0.0) > thresholds["max_portfolio_beta"]:
        alerts.append("portfolio_beta_limit")
    active_weights = [
        abs(weight)
        for symbol, weight in metrics.get("weights", {}).items()
        if symbol != benchmark
    ]
    metrics["max_active_position_weight"] = max(active_weights, default=0.0)
    if metrics["max_active_position_weight"] > thresholds["max_single_active_weight"]:
        alerts.append("position_concentration_limit")
    if metrics.get("stress", {}).get("benchmark_down_20pct", 0.0) < -thresholds["portfolio_stress_loss_limit"]:
        alerts.append("benchmark_stress_limit")
    metrics["thresholds"] = thresholds
    metrics["alerts"] = sorted(set(alerts))
    return metrics


def adjust_target_weights(prices: pd.DataFrame, target_weights: Dict[str, float], config, risk_state=None, asof=None) -> PortfolioRiskDecision:
    """Cap and scale active target weights until portfolio limits are respected."""

    core = getattr(config, "core_symbol", "SPY")
    active_limit = float(getattr(config, "active_weight", 0.20))
    if risk_state is not None:
        active_limit = min(active_limit, float(getattr(risk_state, "active_cap", active_limit)))
    adjusted = {symbol: max(0.0, float(weight)) for symbol, weight in target_weights.items() if float(weight) > 0}
    reasons = []

    for symbol in list(adjusted):
        if symbol != core:
            cap = float(getattr(config, "max_single_active_weight", 0.06))
            if adjusted[symbol] > cap:
                adjusted[symbol] = cap
                reasons.append(f"cap_position:{symbol}")
    active_symbols = [symbol for symbol in adjusted if symbol != core]
    active_total = sum(adjusted.get(symbol, 0.0) for symbol in active_symbols)
    if active_total > active_limit and active_total > 0:
        factor = active_limit / active_total
        for symbol in active_symbols:
            adjusted[symbol] *= factor
        reasons.append("cap_active_exposure")

    def metrics(weights):
        return risk_metrics_for_weights(
            prices,
            weights,
            benchmark=getattr(config, "benchmark", "SPY"),
            asof=asof,
            lookback=int(getattr(config, "risk_lookback_days", 252)),
        )

    before = metrics(adjusted)
    scale = 1.0
    max_vol = float(getattr(config, "max_portfolio_volatility", 0.35))
    max_beta = float(getattr(config, "max_portfolio_beta", 1.25))
    stress_limit = float(getattr(config, "portfolio_stress_loss_limit", 0.20))
    for _ in range(20):
        current = metrics(adjusted)
        stress_loss = abs(float(current.get("stress", {}).get("benchmark_down_20pct", 0.0)))
        if current.get("annualized_volatility", 0.0) <= max_vol and current.get("beta", 0.0) <= max_beta and stress_loss <= stress_limit:
            break
        for symbol in active_symbols:
            adjusted[symbol] *= 0.90
        scale *= 0.90
        reasons.append("scale_active_for_portfolio_risk")
    after = metrics(adjusted)
    return PortfolioRiskDecision(
        target_weights=adjusted,
        adjusted=bool(reasons),
        active_scale=float(scale),
        reasons=sorted(set(reasons)),
        before=before,
        after=after,
    )
