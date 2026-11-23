from typing import Dict

import numpy as np
import pandas as pd

from .backtest import BacktestResult, _metrics, run_backtest
from .leakage import audit_trade_timestamps


def buy_and_hold(prices: pd.DataFrame, symbol: str, starting_cash: float, transaction_cost: float) -> Dict[str, float]:
    if symbol not in prices.columns:
        return {}
    series = prices[symbol].dropna()
    if len(series) < 2:
        return {}
    equity = starting_cash * (1.0 - transaction_cost) * series / series.iloc[0]
    return _metrics(equity, transaction_cost, 1)


def yearly_metrics(equity: pd.DataFrame) -> pd.DataFrame:
    if equity.empty:
        return pd.DataFrame()
    frame = equity.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    rows = []
    for year, group in frame.groupby(frame["date"].dt.year):
        values = group["equity"].reset_index(drop=True)
        peak = values.cummax()
        rows.append(
            {
                "year": int(year),
                "start_equity": float(values.iloc[0]),
                "end_equity": float(values.iloc[-1]),
                "return": float(values.iloc[-1] / values.iloc[0] - 1.0),
                "max_drawdown": float(-(values / peak - 1.0).min()),
            }
        )
    return pd.DataFrame(rows)


def run_research(
    prices: pd.DataFrame,
    config,
    fundamentals: pd.DataFrame = None,
    selector_backend=None,
    candidate_symbols=None,
) -> Dict[str, object]:
    result: BacktestResult = run_backtest(
        prices,
        config,
        fundamentals=fundamentals,
        selector_backend=selector_backend,
        candidate_symbols=candidate_symbols,
    )
    benchmark = buy_and_hold(prices, config.benchmark, config.starting_cash, config.transaction_cost)
    qqq = buy_and_hold(prices, "QQQ", config.starting_cash, config.transaction_cost) if "QQQ" in prices.columns else {}
    years = yearly_metrics(result.equity)
    audit = audit_trade_timestamps(result.trades)
    return {
        "strategy": result.metrics,
        "benchmark_buy_and_hold": benchmark,
        "qqq_buy_and_hold": qqq,
        "comparison": {
            "cagr_difference": result.metrics.get("cagr", 0.0) - benchmark.get("cagr", 0.0),
            "max_drawdown_difference": result.metrics.get("max_drawdown", 0.0) - benchmark.get("max_drawdown", 0.0),
        },
        "lookahead_audit": audit,
        "yearly": years.to_dict(orient="records"),
        "backtest": result,
    }

