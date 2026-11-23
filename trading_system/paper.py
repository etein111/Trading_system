import json
import numpy as np
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PaperOrder:
    symbol: str
    side: str
    quantity: float
    reference_price: float
    target_weight: float


@dataclass
class PaperAccount:
    cash: float
    positions: Dict[str, float] = field(default_factory=dict)
    last_prices: Dict[str, float] = field(default_factory=dict)
    high_watermark: float = 0.0
    realized_pnl: float = 0.0
    trade_count: int = 0
    updated_at: str = ""

    @classmethod
    def new(cls, cash: float) -> "PaperAccount":
        return cls(cash=float(cash), high_watermark=float(cash))

    @classmethod
    def load(cls, path: str, initial_cash: float) -> "PaperAccount":
        location = Path(path)
        if not location.exists():
            return cls.new(initial_cash)
        with open(location, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return cls(
            cash=float(raw.get("cash", initial_cash)),
            positions={key: float(value) for key, value in raw.get("positions", {}).items()},
            last_prices={key: float(value) for key, value in raw.get("last_prices", {}).items()},
            high_watermark=float(raw.get("high_watermark", initial_cash)),
            realized_pnl=float(raw.get("realized_pnl", 0.0)),
            trade_count=int(raw.get("trade_count", 0)),
            updated_at=str(raw.get("updated_at", "")),
        )

    def save(self, path: str) -> None:
        location = Path(path)
        location.parent.mkdir(parents=True, exist_ok=True)
        with open(location, "w", encoding="utf-8") as handle:
            json.dump(asdict(self), handle, indent=2)

    def equity(self, prices: Optional[Dict[str, float]] = None) -> float:
        mark_prices = prices or self.last_prices
        return float(self.cash + sum(self.positions.get(symbol, 0.0) * mark_prices.get(symbol, 0.0) for symbol in self.positions))

    def mark_to_market(self, prices: Dict[str, float]) -> float:
        self.last_prices.update({key: float(value) for key, value in prices.items()})
        current_equity = self.equity(prices)
        self.high_watermark = max(self.high_watermark, current_equity)
        return current_equity

    def execute_target(
        self,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
        transaction_cost: float = 0.0005,
        slippage: float = 0.0005,
        min_trade_notional: float = 25.0,
        timestamp: Optional[str] = None,
        whole_shares: bool = False,
    ) -> List[PaperOrder]:
        """Fill target-weight orders locally at latest reference prices."""
        current_equity = self.mark_to_market(prices)
        orders: List[PaperOrder] = []
        all_symbols = set(self.positions) | set(target_weights)
        for symbol in sorted(all_symbols):
            price = float(prices.get(symbol, 0.0))
            if price <= 0:
                continue
            current_shares = float(self.positions.get(symbol, 0.0))
            target_shares = current_equity * float(target_weights.get(symbol, 0.0)) / price
            delta = target_shares - current_shares
            if whole_shares:
                delta = (np.floor(delta) if delta > 0 else -np.floor(abs(delta)))
            if abs(delta) < 1e-9 or abs(delta * price) < min_trade_notional:
                continue
            is_buy = delta > 0
            fill_price = price * (1.0 + slippage if is_buy else 1.0 - slippage)
            notional = abs(delta * fill_price)
            fee = notional * transaction_cost
            self.cash -= delta * fill_price + fee
            self.positions[symbol] = current_shares + delta
            if abs(self.positions[symbol]) < 1e-9:
                del self.positions[symbol]
            orders.append(
                PaperOrder(
                    symbol=symbol,
                    side="BUY" if is_buy else "SELL",
                    quantity=abs(delta),
                    reference_price=fill_price,
                    target_weight=float(target_weights.get(symbol, 0.0)),
                )
            )
        self.trade_count += len(orders)
        self.updated_at = timestamp or datetime.now(timezone.utc).isoformat()
        self.mark_to_market(prices)
        return orders

    def summary(self, prices: Optional[Dict[str, float]] = None) -> Dict[str, object]:
        current_equity = self.equity(prices)
        drawdown = max(0.0, 1.0 - current_equity / max(self.high_watermark, current_equity))
        return {
            "equity": current_equity,
            "cash": self.cash,
            "positions": dict(self.positions),
            "drawdown": drawdown,
            "high_watermark": self.high_watermark,
            "trade_count": self.trade_count,
            "updated_at": self.updated_at,
        }


def rebalance_orders(
    equity: float,
    current_positions: Dict[str, float],
    target_weights: Dict[str, float],
    prices: Dict[str, float],
    whole_shares: bool = False,
) -> List[PaperOrder]:
    """Create reviewable paper orders; this function never submits them."""
    orders: List[PaperOrder] = []
    for symbol in sorted(set(current_positions) | set(target_weights)):
        if symbol not in prices or prices[symbol] <= 0:
            continue
        delta_weight = target_weights.get(symbol, 0.0) - current_positions.get(symbol, 0.0)
        quantity = abs(equity * delta_weight / prices[symbol])
        if whole_shares:
            quantity = float(np.floor(quantity))
        if quantity <= 1e-9:
            continue
        orders.append(
            PaperOrder(
                symbol=symbol,
                side="BUY" if delta_weight > 0 else "SELL",
                quantity=quantity,
                reference_price=prices[symbol],
                target_weight=target_weights.get(symbol, 0.0),
            )
        )
    return orders

