from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class SystemConfig:
    starting_cash: float = 40000.0
    benchmark: str = "SPY"
    core_symbol: str = "SPY"
    active_symbols: List[str] = field(default_factory=lambda: ["QQQ", "XLK", "MSFT", "AMZN", "NVDA", "PLTR"])

    core_weight_risk_on: float = 0.60
    core_weight_risk_off: float = 0.40
    reserve_weight: float = 0.20
    active_weight: float = 0.20
    max_active_positions: int = 4
    max_single_active_weight: float = 0.06

    short_window: int = 20
    medium_window: int = 50
    long_window: int = 200
    momentum_window: int = 60
    stop_distance: float = 0.10
    max_trade_risk: float = 0.005

    drawdown_review: float = 0.10
    drawdown_reduce: float = 0.15
    drawdown_circuit_breaker: float = 0.20
    transaction_cost_bps: float = 5.0
    min_paper_trade_notional: float = 25.0

    selection_min_history: int = 252
    selection_min_price: float = 5.0
    selection_momentum_lookback: int = 252
    selection_momentum_skip: int = 21
    selection_momentum_short: int = 60
    selection_volatility_window: int = 63

    risk_lookback_days: int = 252
    max_portfolio_volatility: float = 0.35
    max_portfolio_beta: float = 1.25
    portfolio_stress_loss_limit: float = 0.20

    @classmethod
    def from_yaml(cls, path: str) -> "SystemConfig":
        with open(path, "r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        allowed = {key: value for key, value in raw.items() if key in cls.__dataclass_fields__}
        return cls(**allowed)

    @property
    def transaction_cost(self) -> float:
        return self.transaction_cost_bps / 10000.0

    def validate(self) -> None:
        if self.starting_cash <= 0:
            raise ValueError("starting_cash must be positive")
        if not 0 < self.active_weight <= 1:
            raise ValueError("active_weight must be in (0, 1]")
        if not 0 < self.max_single_active_weight <= self.active_weight:
            raise ValueError("max_single_active_weight is invalid")
        if self.core_weight_risk_on + self.active_weight + self.reserve_weight > 1.000001:
            raise ValueError("risk-on target weights exceed 100%")
        if not 0 < self.stop_distance < 1:
            raise ValueError("stop_distance must be between 0 and 1")
        if not 0 < self.max_trade_risk < 1:
            raise ValueError("max_trade_risk must be between 0 and 1")

