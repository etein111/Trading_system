import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .leakage import history_asof
from .risk import RiskManager
from .strategy import RegimeStrategy


@dataclass
class BacktestResult:
    equity: pd.DataFrame
    trades: pd.DataFrame
    metrics: Dict[str, float]

    def save(self, output_dir: str) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.equity.to_csv(directory / "equity.csv", index=False)
        self.trades.to_csv(directory / "trades.csv", index=False)
        with open(directory / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(self.metrics, handle, indent=2)


def _metrics(equity: pd.Series, turnover: float, trades: int) -> Dict[str, float]:
    if len(equity) < 2:
        return {}
    daily = equity.pct_change().fillna(0.0)
    years = max(len(equity) / 252.0, 1.0 / 252.0)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1.0 / years) - 1.0
    high = equity.cummax()
    drawdown = equity / high - 1.0
    annual_vol = daily.std(ddof=0) * np.sqrt(252.0)
    sharpe = (daily.mean() / daily.std(ddof=0) * np.sqrt(252.0)) if daily.std(ddof=0) else 0.0
    return {
        "starting_equity": float(equity.iloc[0]),
        "ending_equity": float(equity.iloc[-1]),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "max_drawdown": float(-drawdown.min()),
        "annual_volatility": float(annual_vol),
        "sharpe": float(sharpe),
        "average_daily_turnover": float(turnover / max(len(equity), 1)),
        "weight_change_events": float(trades),
    }


def run_backtest(
    prices: pd.DataFrame,
    config,
    fundamentals: pd.DataFrame = None,
    selector_backend=None,
    candidate_symbols=None,
) -> BacktestResult:
    config.validate()
    prices = prices.sort_index().dropna(how="all")
    strategy = RegimeStrategy(
        config,
        fundamentals=fundamentals,
        selector_backend=selector_backend,
        candidates=candidate_symbols,
    )
    risk = RiskManager(config)
    dates = list(prices.index)
    equity = config.starting_cash
    previous_weights: Dict[str, float] = {}
    high_water_prices: Dict[str, float] = {}
    equity_rows: List[Dict[str, float]] = []
    trade_rows: List[Dict[str, object]] = []
    total_turnover = 0.0
    events = 0
    equity_high_watermark = config.starting_cash

    if dates:
        equity_rows.append(
            {
                "date": dates[0],
                "equity": equity,
                "drawdown": 0.0,
                "active_weight": 0.0,
                "core_weight": 0.0,
                "cash_weight": 1.0,
            }
        )

    for index in range(1, len(dates)):
        date = dates[index]
        previous_date = dates[index - 1]
        history = history_asof(prices, date)
        risk_state = risk.state()
        target = strategy.target_weights(history, risk_state, current_weights=previous_weights)

        # Apply a trailing stop to active holdings. A stop removes the target for the next session.
        for symbol in list(target):
            if symbol == config.core_symbol or symbol not in high_water_prices:
                continue
            current_price = float(prices.loc[date, symbol])
            if risk.stop_triggered(current_price, high_water_prices[symbol]):
                del target[symbol]

        all_symbols = set(previous_weights) | set(target)
        turnover = sum(abs(target.get(symbol, 0.0) - previous_weights.get(symbol, 0.0)) for symbol in all_symbols)
        returns = prices.loc[date] / prices.loc[previous_date] - 1.0
        portfolio_return = 0.0
        for symbol, weight in previous_weights.items():
            if symbol in returns and pd.notna(returns[symbol]):
                portfolio_return += weight * float(returns[symbol])
        transaction_cost = turnover * config.transaction_cost
        equity *= 1.0 + portfolio_return - transaction_cost
        risk.update(equity)

        for symbol in all_symbols:
            before = previous_weights.get(symbol, 0.0)
            after = target.get(symbol, 0.0)
            if abs(after - before) > 1e-12:
                events += 1
                trade_rows.append(
                    {
                        "date": date,
                        "signal_asof": previous_date,
                        "execution_date": date,
                        "symbol": symbol,
                        "previous_weight": before,
                        "target_weight": after,
                        "reason": "rebalance_or_signal",
                    }
                )

        for symbol, weight in target.items():
            if weight > 0 and symbol in prices.columns:
                current_price = float(prices.loc[date, symbol])
                if previous_weights.get(symbol, 0.0) <= 0:
                    high_water_prices[symbol] = current_price
                else:
                    high_water_prices[symbol] = max(high_water_prices.get(symbol, current_price), current_price)
        for symbol in list(high_water_prices):
            if target.get(symbol, 0.0) <= 0:
                del high_water_prices[symbol]

        total_turnover += turnover
        previous_weights = target
        equity_high_watermark = max(equity_high_watermark, equity)
        equity_rows.append(
            {
                "date": date,
                "equity": equity,
                "drawdown": max(0.0, 1.0 - equity / equity_high_watermark),
                "active_weight": sum(weight for symbol, weight in target.items() if symbol != config.core_symbol),
                "core_weight": target.get(config.core_symbol, 0.0),
                "cash_weight": max(0.0, 1.0 - sum(target.values())),
            }
        )

    equity_frame = pd.DataFrame(equity_rows)
    trade_frame = pd.DataFrame(trade_rows)
    metrics = _metrics(equity_frame["equity"], total_turnover, events) if not equity_frame.empty else {}
    return BacktestResult(equity_frame, trade_frame, metrics)

