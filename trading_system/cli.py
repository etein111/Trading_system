import argparse
import json
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from .backtest import run_backtest
from .config import SystemConfig
from .data import download_yfinance, generate_demo_prices, load_close_csv, load_fundamentals_csv, load_volume_csv
from .evaluate import run_research
from .paper import PaperAccount
from .public_training import (
    TrainingConfig,
    build_latest_features,
    build_sec_fundamentals,
    build_training_samples,
    train_walk_forward,
)
from .risk import RiskManager
from .strategy import RegimeStrategy
from .selector import PointInTimeSelector, SelectionConfig
from .selector_backend import QlibScoreBackend


def _config_path(value):
    return value or str(Path(__file__).resolve().parents[1] / "config.yaml")


def _load_symbols_file(value):
    if not value:
        return None
    return [line.strip().upper() for line in Path(value).read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")]


def _run(prices, config, output):
    result = run_backtest(prices, config)
    result.save(output)
    print(json.dumps(result.metrics, indent=2))
    print("saved:", str(Path(output).resolve()))


def main():
    parser = argparse.ArgumentParser(description="Personal trading system research CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run on deterministic demo data")
    demo.add_argument("--days", type=int, default=1500)
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--config", default=None)
    demo.add_argument("--output", default="artifacts")

    backtest = subparsers.add_parser("backtest", help="run on a close-price CSV")
    backtest.add_argument("--data", required=True)
    backtest.add_argument("--fundamentals", default=None)
    backtest.add_argument("--config", default=None)
    backtest.add_argument("--output", default="artifacts")
    backtest.add_argument("--qlib-scores", default=None, help="historical Qlib scores CSV")
    backtest.add_argument("--candidates-file", default=None, help="one candidate symbol per line")
    backtest.add_argument("--enforce-price-eligibility", action="store_true")

    research = subparsers.add_parser("research", help="run a chronological backtest and benchmark report")
    research.add_argument("--data", required=True)
    research.add_argument("--fundamentals", default=None)
    research.add_argument("--config", default=None)
    research.add_argument("--output", default="research")
    research.add_argument("--qlib-scores", default=None, help="historical Qlib scores CSV")
    research.add_argument("--candidates-file", default=None, help="one candidate symbol per line")
    research.add_argument("--enforce-price-eligibility", action="store_true")

    download = subparsers.add_parser("download", help="download adjusted closes with optional yfinance")
    download.add_argument("--symbols", nargs="+", required=True)
    download.add_argument("--start", default=None)
    download.add_argument("--end", default=None)
    download.add_argument("--output", default="data/prices.csv")

    signals = subparsers.add_parser("signals", help="print the latest target weights from a CSV")
    signals.add_argument("--data", required=True)
    signals.add_argument("--config", default=None)

    select = subparsers.add_parser("select", help="rank candidates using as-of price factors and optional fundamentals")
    select.add_argument("--data", required=True)
    select.add_argument("--fundamentals", default=None)
    select.add_argument("--asof", default=None)
    select.add_argument("--symbols", nargs="*", default=None)
    select.add_argument("--config", default=None)
    select.add_argument("--output", default=None)
    select.add_argument("--qlib-scores", default=None, help="use a Qlib score CSV with symbol,score,available_at")
    select.add_argument("--enforce-price-eligibility", action="store_true")

    train_public = subparsers.add_parser("train-public", help="train a point-in-time model from SEC and price data")
    train_public.add_argument("--prices", required=True)
    train_public.add_argument("--symbols", nargs="+", required=True)
    train_public.add_argument("--benchmark", default="SPY")
    train_public.add_argument("--cache-dir", default="research/sec_cache")
    train_public.add_argument("--output", default="research/public_model")
    train_public.add_argument("--horizon-days", type=int, default=60)
    train_public.add_argument("--top-k", type=int, default=4)
    train_public.add_argument("--evaluation-stride", type=int, default=3)

    train_qlib = subparsers.add_parser("train-qlib", help="train a point-in-time Qlib/LightGBM model")
    train_qlib.add_argument("--prices", required=True)
    train_qlib.add_argument("--symbols", nargs="+", required=True)
    train_qlib.add_argument("--benchmark", default="SPY")
    train_qlib.add_argument("--cache-dir", default="research/sec_cache")
    train_qlib.add_argument("--output", default="research/qlib_model_enhanced")
    train_qlib.add_argument("--volumes", default=None, help="optional raw volume CSV aligned to the price dates")
    train_qlib.add_argument("--horizon-days", type=int, default=60)
    train_qlib.add_argument("--train-years", type=int, default=8)
    train_qlib.add_argument("--minimum-train-samples", type=int, default=500)
    train_qlib.add_argument("--top-k", type=int, default=4)
    train_qlib.add_argument("--evaluation-stride", type=int, default=3)
    train_qlib.add_argument("--num-boost-round", type=int, default=80)
    train_qlib.add_argument("--target-mode", choices=["rank", "raw"], default="raw")
    train_qlib.add_argument(
        "--include-fundamentals",
        action="store_true",
        help="include SEC fundamental features; default uses technical features only",
    )
    train_qlib.add_argument("--no-cross-sectional-rank", action="store_true")
    train_qlib.add_argument("--holdout-start", default="2024-01-01")

    paper = subparsers.add_parser("paper-run", help="mark and rebalance a local simulated account")
    paper.add_argument("--state", default="paper/account.json")
    paper.add_argument("--log", default="paper/trades.csv")
    paper.add_argument("--config", default=None)
    paper.add_argument("--start", default=None)
    paper.add_argument("--end", default=None)
    paper.add_argument("--lookback-days", type=int, default=900)
    paper.add_argument("--slippage-bps", type=float, default=5.0)
    paper.add_argument("--data", default=None, help="local close-price CSV; avoids live data download")
    paper.add_argument("--qlib-scores", default=None, help="historical Qlib scores CSV")
    paper.add_argument("--candidates-file", default=None, help="one candidate symbol per line")
    paper.add_argument("--enforce-price-eligibility", action="store_true")

    args = parser.parse_args()
    if args.command == "download":
        prices = download_yfinance(args.symbols, start=args.start, end=args.end)
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        prices.to_csv(destination, index_label="date")
        print("saved:", str(destination.resolve()))
        return

    if args.command == "train-public":
        prices = load_close_csv(args.prices)
        fundamentals = build_sec_fundamentals(args.symbols, cache_dir=args.cache_dir)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        fundamentals.to_csv(output / "sec_fundamentals.csv", index=False)
        config_public = TrainingConfig(
            horizon_days=args.horizon_days,
            top_k=args.top_k,
            evaluation_stride=args.evaluation_stride,
        )
        samples = build_training_samples(
            prices,
            fundamentals,
            benchmark=args.benchmark,
            symbols=args.symbols,
            config=config_public,
        )
        latest = build_latest_features(prices, fundamentals, args.symbols)
        summary = train_walk_forward(samples, latest, str(output), config=config_public)
        summary["sec_failures"] = fundamentals.attrs.get("failures", {})
        (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    if args.command == "train-qlib":
        from .qlib_training import (
            QLIB_FEATURE_COLUMNS,
            TECHNICAL_FEATURE_COLUMNS,
            train_qlib_walk_forward,
        )

        prices = load_close_csv(args.prices)
        volumes = load_volume_csv(args.volumes) if args.volumes else None
        fundamentals = build_sec_fundamentals(args.symbols, cache_dir=args.cache_dir)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        fundamentals.to_csv(output / "sec_fundamentals.csv", index=False)
        config_qlib = TrainingConfig(
            horizon_days=args.horizon_days,
            train_years=args.train_years,
            minimum_train_samples=args.minimum_train_samples,
            top_k=args.top_k,
            evaluation_stride=args.evaluation_stride,
        )
        samples = build_training_samples(
            prices,
            fundamentals,
            benchmark=args.benchmark,
            symbols=args.symbols,
            config=config_qlib,
            volumes=volumes,
        )
        latest = build_latest_features(prices, fundamentals, args.symbols, volumes=volumes)
        samples.to_csv(output / "training_samples.csv", index=False)
        summary = train_qlib_walk_forward(
            samples,
            latest,
            str(output),
            config=config_qlib,
            feature_columns=QLIB_FEATURE_COLUMNS if args.include_fundamentals else TECHNICAL_FEATURE_COLUMNS,
            target_mode=args.target_mode,
            cross_sectional_rank=not args.no_cross_sectional_rank,
            num_boost_round=args.num_boost_round,
            holdout_start=args.holdout_start,
        )
        summary["sec_failures"] = fundamentals.attrs.get("failures", {})
        (output / "qlib_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
        return

    config = SystemConfig.from_yaml(_config_path(args.config))
    if args.command == "paper-run":
        candidate_symbols = _load_symbols_file(args.candidates_file)
        symbols = list(dict.fromkeys([config.benchmark, config.core_symbol] + (candidate_symbols or config.active_symbols)))
        start = args.start or (date.today() - timedelta(days=args.lookback_days)).isoformat()
        if args.data:
            prices = load_close_csv(args.data)
            if args.start:
                prices = prices.loc[prices.index >= pd.Timestamp(args.start)]
            if args.end:
                prices = prices.loc[prices.index <= pd.Timestamp(args.end)]
        else:
            prices = download_yfinance(symbols, start=start, end=args.end)
        account = PaperAccount.load(args.state, config.starting_cash)
        latest = prices.iloc[-1]
        latest_prices = {symbol: float(latest[symbol]) for symbol in prices.columns if pd.notna(latest[symbol])}
        equity = account.mark_to_market(latest_prices)
        risk = RiskManager(config)
        risk.equity = equity
        risk.high_watermark = max(account.high_watermark, equity)
        selector_backend = (
            QlibScoreBackend(
                pd.read_csv(args.qlib_scores),
                enforce_price_eligibility=args.enforce_price_eligibility,
            )
            if args.qlib_scores
            else None
        )
        target = RegimeStrategy(
            config, selector_backend=selector_backend, candidates=candidate_symbols
        ).target_weights(prices, risk.state())
        orders = account.execute_target(
            target,
            latest_prices,
            transaction_cost=config.transaction_cost,
            slippage=args.slippage_bps / 10000.0,
            min_trade_notional=config.min_paper_trade_notional,
            timestamp=str(prices.index[-1]),
        )
        account.save(args.state)
        if orders:
            rows = [
                {
                    "date": str(prices.index[-1]),
                    "symbol": order.symbol,
                    "side": order.side,
                    "quantity": order.quantity,
                    "reference_price": order.reference_price,
                    "target_weight": order.target_weight,
                }
                for order in orders
            ]
            log_path = Path(args.log)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            new_log = pd.DataFrame(rows)
            if log_path.exists():
                new_log = pd.concat([pd.read_csv(log_path), new_log], ignore_index=True)
            new_log.to_csv(log_path, index=False)
        print(
            json.dumps(
                {
                    "asof": str(prices.index[-1]),
                    "target_weights": target,
                    "orders": [order.__dict__ for order in orders],
                    "account": account.summary(latest_prices),
                },
                indent=2,
            )
        )
        return

    if args.command == "demo":
        symbols = [config.benchmark, config.core_symbol] + config.active_symbols
        prices = generate_demo_prices(symbols, days=args.days, seed=args.seed)
        Path(args.output).mkdir(parents=True, exist_ok=True)
        prices.to_csv(Path(args.output) / "demo_prices.csv", index_label="date")
        _run(prices, config, args.output)
    elif args.command == "backtest":
        prices = load_close_csv(args.data)
        fundamentals = load_fundamentals_csv(args.fundamentals) if args.fundamentals else None
        candidate_symbols = _load_symbols_file(args.candidates_file)
        selector_backend = (
            QlibScoreBackend(
                pd.read_csv(args.qlib_scores),
                enforce_price_eligibility=args.enforce_price_eligibility,
            )
            if args.qlib_scores
            else None
        )
        result = run_backtest(
            prices,
            config,
            fundamentals=fundamentals,
            selector_backend=selector_backend,
            candidate_symbols=candidate_symbols,
        )
        result.save(args.output)
        print(json.dumps(result.metrics, indent=2))
        print("saved:", str(Path(args.output).resolve()))
    elif args.command == "research":
        prices = load_close_csv(args.data)
        fundamentals = load_fundamentals_csv(args.fundamentals) if args.fundamentals else None
        candidate_symbols = _load_symbols_file(args.candidates_file)
        selector_backend = (
            QlibScoreBackend(
                pd.read_csv(args.qlib_scores),
                enforce_price_eligibility=args.enforce_price_eligibility,
            )
            if args.qlib_scores
            else None
        )
        summary = run_research(
            prices,
            config,
            fundamentals=fundamentals,
            selector_backend=selector_backend,
            candidate_symbols=candidate_symbols,
        )
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        summary["backtest"].save(str(output))
        summary.pop("backtest")
        with open(output / "research_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        pd.DataFrame(summary["yearly"]).to_csv(output / "yearly_metrics.csv", index=False)
        print(json.dumps(summary, indent=2))
    elif args.command == "select":
        prices = load_close_csv(args.data)
        fundamentals = load_fundamentals_csv(args.fundamentals) if args.fundamentals else None
        candidates = args.symbols or config.active_symbols
        if args.qlib_scores:
            qlib_scores = pd.read_csv(args.qlib_scores)
            ranked = QlibScoreBackend(
                qlib_scores, enforce_price_eligibility=args.enforce_price_eligibility
            ).select(
                prices, candidates=candidates, asof=args.asof or prices.index[-1], fundamentals=fundamentals
            )
        else:
            selection_config = SelectionConfig(
                min_history=config.selection_min_history,
                min_price=config.selection_min_price,
                momentum_lookback=config.selection_momentum_lookback,
                momentum_skip=config.selection_momentum_skip,
                momentum_short=config.selection_momentum_short,
                volatility_window=config.selection_volatility_window,
                top_n=config.max_active_positions,
            )
            ranked = PointInTimeSelector(selection_config).select(
                prices, candidates=candidates, asof=args.asof, fundamentals=fundamentals
            )
        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            ranked.to_csv(args.output, index=False)
        print(ranked.to_json(orient="records", date_format="iso", indent=2))
    else:
        prices = load_close_csv(args.data)
        risk = RiskManager(config)
        strategy = RegimeStrategy(config)
        weights = strategy.target_weights(prices, risk.state())
        print(json.dumps({"asof": str(prices.index[-1]), "target_weights": weights}, indent=2))

