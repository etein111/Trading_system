"""Run one scheduled trading analysis and optionally email it.

The script never submits broker orders. With --execute-paper it updates the
local PaperAccount using whole-share fills only.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from trading_system.config import SystemConfig
from trading_system.data import load_close_csv
from trading_system.email_report import send_gmail_report
from trading_system.paper import PaperAccount, rebalance_orders
from trading_system.portfolio_risk import analyze_account
from trading_system.risk import RiskManager
from trading_system.selector import PointInTimeSelector, SelectionConfig
from trading_system.selector_backend import BlendedSelector, PriceFactorBackend, QlibScoreBackend, TextScoreBackend
from trading_system.strategy import RegimeStrategy


def symbols_file(path: str | None):
    if not path:
        return None
    return [line.strip().upper() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def backend(args):
    backends = []
    weights = []
    if args.scores:
        backends.append(QlibScoreBackend(pd.read_csv(args.scores)))
        weights.append(1.0)
    if args.text_scores:
        backends.append(TextScoreBackend(pd.read_csv(args.text_scores), max_event_risk=args.text_max_event_risk))
        weights.append(args.text_weight)
    if not backends:
        selection = SelectionConfig(top_n=args.max_positions)
        return PriceFactorBackend(selection)
    return backends[0] if len(backends) == 1 else BlendedSelector(backends, weights=weights)


def run_refresh(args):
    if not args.refresh and not args.retrain:
        return
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts/build_expanded_public_data.py"),
        "--output-dir", str(Path(args.data).parent),
        "--cache-dir", str(Path(args.cache_dir)),
        "--start", "2010-01-01",
    ]
    if args.refresh:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    if args.retrain:
        symbols = [line.strip() for line in Path(args.candidates_file).read_text(encoding="utf-8").splitlines() if line.strip()]
        model_output = Path(args.model_output)
        train_command = [
            sys.executable, "-m", "trading_system", "train-qlib",
            "--prices", args.data, "--volumes", str(Path(args.data).parent / "volumes.csv"),
            "--symbols", *symbols, "--cache-dir", args.cache_dir,
            "--output", str(model_output), "--train-years", "8",
            "--minimum-train-samples", "500", "--evaluation-stride", "3",
            "--num-boost-round", "80", "--target-mode", "raw",
        ]
        subprocess.run(train_command, cwd=PROJECT_ROOT, check=True)
        args.scores = str(model_output / "qlib_scores.csv")


def main():
    parser = argparse.ArgumentParser(description="run a scheduled trading report")
    parser.add_argument("--session", choices=["pre", "intra", "post"], required=True)
    parser.add_argument("--config", default="config_qlib.yaml")
    parser.add_argument("--data", default="research/expanded_data/prices.csv")
    parser.add_argument("--state", default="paper/account.json")
    parser.add_argument("--log", default="paper/trades.csv")
    parser.add_argument("--scores", default="research/qlib_model_enhanced/qlib_scores.csv")
    parser.add_argument("--text-scores", default=None)
    parser.add_argument("--text-weight", type=float, default=0.35)
    parser.add_argument("--text-max-event-risk", type=float, default=0.65)
    parser.add_argument("--candidates-file", default="research/expanded_data/symbols.txt")
    parser.add_argument("--cache-dir", default="research/expanded_data/sec_cache")
    parser.add_argument("--output-dir", default="research/daily_reports")
    parser.add_argument("--to", default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--model-output", default="research/qlib_model_daily")
    parser.add_argument("--execute-paper", action="store_true")
    parser.add_argument("--no-send", action="store_true", help="generate the report without email delivery")
    parser.add_argument("--max-positions", type=int, default=4)
    args = parser.parse_args()

    run_refresh(args)
    config = SystemConfig.from_yaml(args.config)
    prices = load_close_csv(args.data)
    candidates = symbols_file(args.candidates_file) or config.active_symbols
    account = PaperAccount.load(args.state, config.starting_cash)
    latest_prices = {symbol: float(prices.iloc[-1][symbol]) for symbol in prices.columns if pd.notna(prices.iloc[-1][symbol])}
    equity = account.mark_to_market(latest_prices)
    risk = RiskManager(config)
    risk.equity = equity
    risk.high_watermark = max(account.high_watermark, equity)
    risk_state = risk.state()
    risk_report = analyze_account(account, prices, benchmark=config.benchmark, asof=prices.index[-1], config=config)
    selector = backend(args)
    strategy = RegimeStrategy(config, selector_backend=selector, candidates=candidates)
    current_weights = {
        symbol: float(account.positions.get(symbol, 0.0) * latest_prices.get(symbol, 0.0) / equity)
        for symbol in account.positions if equity > 0 and latest_prices.get(symbol, 0.0) > 0
    }
    target = strategy.target_weights(prices, risk_state, current_weights=current_weights)
    orders = rebalance_orders(equity, current_weights, target, latest_prices, whole_shares=True)
    if args.execute_paper:
        account.execute_target(target, latest_prices, transaction_cost=config.transaction_cost, whole_shares=True, timestamp=str(prices.index[-1]))
        account.save(args.state)

    selected = selector.select(prices, candidates=candidates, asof=prices.index[-1])
    selected_rows = selected.head(max(args.max_positions, 10)).to_dict(orient="records")
    report = {
        "session": args.session,
        "run_at": datetime.now().astimezone().isoformat(),
        "data_asof": str(prices.index[-1]),
        "market_risk_on": strategy._risk_on(prices),
        "risk_state": risk_state.__dict__,
        "portfolio_risk": risk_report,
        "target_weights": target,
        "orders": [order.__dict__ for order in orders],
        "selected": selected_rows,
        "account": account.summary(latest_prices),
        "whole_share_policy": True,
        "executed_paper": bool(args.execute_paper),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / f"{args.session}_{pd.Timestamp(prices.index[-1]).date()}.json"
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    body = [
        f"Trading System {args.session.upper()} report",
        f"Data asof: {prices.index[-1]}",
        f"Market risk-on: {report['market_risk_on']}",
        f"Equity: {report['account']['equity']:.2f}",
        f"Drawdown: {report['account']['drawdown']:.2%}",
        f"Portfolio volatility: {risk_report.get('annualized_volatility', 0.0):.2%}",
        f"Portfolio beta: {risk_report.get('beta', 0.0):.3f}",
        f"Risk alerts: {', '.join(risk_report.get('alerts', [])) or 'none'}",
        "",
        "Target weights:",
        json.dumps(target, indent=2),
        "",
        "Whole-share recommendations:",
    ]
    if orders:
        body.extend([f"{order.side} {order.quantity:.0f} {order.symbol} -> {order.target_weight:.2%}" for order in orders])
    else:
        body.append("No orders")
    body.extend(["", "Top candidates:"])
    body.extend([f"{row.get('symbol')} score={row.get('score', row.get('qlib_scores_score', 'n/a'))} eligible={row.get('eligible')}" for row in selected_rows[:10]])
    recipient = args.to or os.environ.get("GMAIL_TO")
    if not args.no_send:
        if not recipient:
            raise RuntimeError("set GMAIL_TO or pass --to")
        send_gmail_report(
            subject=f"Trading System {args.session.upper()} {prices.index[-1].date()}",
            body="\n".join(body),
            recipients=[recipient],
            attachments=[str(report_path)],
        )
    print(json.dumps({"report": str(report_path.resolve()), "emailed_to": recipient if not args.no_send else None, "orders": [order.__dict__ for order in orders]}, indent=2, default=str))


if __name__ == "__main__":
    main()
