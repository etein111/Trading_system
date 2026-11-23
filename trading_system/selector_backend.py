from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd

from .selector import PointInTimeSelector, SelectionConfig


class SelectorBackend(ABC):
    """Common interface for all selection engines."""

    name = "abstract"

    @abstractmethod
    def select(
        self,
        prices: pd.DataFrame,
        candidates: Iterable[str],
        asof,
        fundamentals: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        raise NotImplementedError


class PriceFactorBackend(SelectorBackend):
    name = "price_factors"

    def __init__(self, config: Optional[SelectionConfig] = None):
        self.selector = PointInTimeSelector(config)

    def select(self, prices, candidates, asof, fundamentals=None):
        return self.selector.select(prices, candidates=candidates, asof=asof, fundamentals=fundamentals)


class QlibScoreBackend(SelectorBackend):
    """Consume scores exported by a Qlib research job.

    The adapter deliberately does not import Qlib. Qlib trains and exports a
    score table; this runtime layer applies the same point-in-time boundary as
    the rule-based selector. The score file needs columns:
    ``symbol,score,available_at``.
    """

    name = "qlib_scores"

    def __init__(self, scores: pd.DataFrame, enforce_price_eligibility: bool = False, price_config: Optional[SelectionConfig] = None):
        required = {"symbol", "score", "available_at"}
        if not required.issubset(scores.columns):
            raise ValueError("Qlib scores must contain symbol, score and available_at columns")
        self.scores = scores.copy()
        self.scores["available_at"] = pd.to_datetime(self.scores["available_at"])
        self.enforce_price_eligibility = bool(enforce_price_eligibility)
        self.price_selector = PointInTimeSelector(price_config or SelectionConfig())

    def select(self, prices, candidates, asof, fundamentals=None):
        asof = pd.Timestamp(asof)
        allowed = set(candidates)
        scores = self.scores[
            (self.scores["available_at"] <= asof) & self.scores["symbol"].isin(allowed)
        ].sort_values("available_at")
        scores = scores.drop_duplicates("symbol", keep="last")
        if scores.empty:
            return pd.DataFrame(columns=["symbol", "score", "eligible", "mode"])
        scores = scores.copy()
        if self.enforce_price_eligibility:
            price_ranked = self.price_selector.select(
                prices, candidates=allowed, asof=asof, fundamentals=fundamentals
            )
            eligible_symbols = set(price_ranked.loc[price_ranked["eligible"], "symbol"])
            scores["eligible"] = scores["symbol"].isin(eligible_symbols)
        else:
            scores["eligible"] = True
        scores["mode"] = self.name
        return scores.sort_values("score", ascending=False).reset_index(drop=True)


class BlendedSelector(SelectorBackend):
    """Blend independent backends after each has passed its own as-of filter."""

    name = "blended"

    def __init__(self, backends, weights=None):
        if not backends:
            raise ValueError("at least one selector backend is required")
        self.backends = list(backends)
        self.weights = list(weights or [1.0] * len(self.backends))
        if len(self.backends) != len(self.weights):
            raise ValueError("backends and weights must have the same length")

    def select(self, prices, candidates, asof, fundamentals=None):
        frames = []
        for backend, weight in zip(self.backends, self.weights):
            frame = backend.select(prices, candidates, asof, fundamentals)
            if frame.empty or "score" not in frame.columns:
                continue
            current = frame[["symbol", "score", "eligible"]].copy()
            current["score"] = current["score"].rank(pct=True).fillna(0.5)
            current["weighted_score"] = current["score"] * float(weight)
            current = current.rename(
                columns={
                    "score": backend.name + "_score",
                    "weighted_score": backend.name + "_weighted_score",
                }
            )
            frames.append(current)
        if not frames:
            return pd.DataFrame(columns=["symbol", "score", "eligible", "mode"])
        merged = frames[0]
        for frame in frames[1:]:
            merged = merged.merge(frame, on="symbol", how="outer", suffixes=("", "_next"))
        weighted_columns = [column for column in merged.columns if column.endswith("_weighted_score")]
        merged["score"] = merged[weighted_columns].fillna(0.0).sum(axis=1)
        eligible_columns = [column for column in merged.columns if column == "eligible" or column.endswith("_eligible")]
        merged["eligible"] = merged[eligible_columns].fillna(False).astype(bool).all(axis=1)
        merged["mode"] = self.name
        return merged.sort_values(["eligible", "score"], ascending=[False, False]).reset_index(drop=True)

