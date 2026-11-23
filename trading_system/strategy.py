from typing import Dict

import pandas as pd

from .selector import SelectionConfig
from .selector_backend import PriceFactorBackend, SelectorBackend
from .portfolio_risk import adjust_target_weights


class RegimeStrategy:
    """Long-only medium-term strategy with a simple market regime filter."""

    def __init__(self, config, fundamentals=None, selector_backend: SelectorBackend = None, candidates=None):
        self.config = config
        self.fundamentals = fundamentals
        self.candidates = list(candidates) if candidates is not None else None
        selection_config = SelectionConfig(
            min_history=config.selection_min_history,
            min_price=config.selection_min_price,
            momentum_lookback=config.selection_momentum_lookback,
            momentum_skip=config.selection_momentum_skip,
            momentum_short=config.selection_momentum_short,
            volatility_window=config.selection_volatility_window,
            top_n=config.max_active_positions,
        )
        self.selector_backend = selector_backend or PriceFactorBackend(selection_config)

    def _risk_on(self, history: pd.DataFrame) -> bool:
        benchmark = history[self.config.benchmark].dropna()
        required = max(self.config.long_window, self.config.medium_window) + 1
        if len(benchmark) < required:
            return False
        sma_short = benchmark.rolling(self.config.short_window).mean().iloc[-1]
        sma_medium = benchmark.rolling(self.config.medium_window).mean().iloc[-1]
        sma_long = benchmark.rolling(self.config.long_window).mean().iloc[-1]
        return bool(benchmark.iloc[-1] > sma_long and sma_short > sma_medium)

    def target_weights(self, history: pd.DataFrame, risk_state, current_weights=None) -> Dict[str, float]:
        if history.empty or self.config.benchmark not in history:
            return {}
        weights: Dict[str, float] = {}
        risk_on = self._risk_on(history)
        if risk_on:
            core_weight = self.config.core_weight_risk_on
        else:
            core_weight = self.config.core_weight_risk_off
        if self.config.core_symbol in history.columns:
            weights[self.config.core_symbol] = core_weight

        if not risk_on or risk_state.active_cap <= 0:
            return weights

        selected_frame = self.selector_backend.select(
            history,
            candidates=self.candidates or self.config.active_symbols,
            asof=history.index[-1],
            fundamentals=self.fundamentals,
        )
        if selected_frame.empty or "eligible" not in selected_frame.columns:
            return weights
        selected = selected_frame.loc[selected_frame["eligible"]].head(self.config.max_active_positions)["symbol"].tolist()
        if not selected:
            return weights
        risk_based_cap = self.config.max_trade_risk / self.config.stop_distance
        per_position = min(
            self.config.max_single_active_weight,
            risk_based_cap,
            risk_state.active_cap / len(selected),
        )
        for symbol in selected:
            weights[symbol] = per_position
        decision = adjust_target_weights(
            history, weights, self.config, risk_state=risk_state, asof=history.index[-1]
        )
        return decision.target_weights

