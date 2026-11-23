"""Download a reproducible liquid US technology/growth research universe.

This is a research universe, not a claim that all names were tradable at every
historical date.  The training code still requires a symbol to have enough
history on each as-of date and records SEC download failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trading_system.data import download_yfinance
from trading_system.public_training import build_sec_fundamentals

SYMBOLS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "AMD", "AVGO", "ORCL", "CRM",
    "ADBE", "NFLX", "NOW", "PANW", "PLTR", "SNOW", "MU", "ANET", "INTU", "QCOM",
    "TXN", "IBM", "ADSK", "ADI", "AMAT", "CDNS", "SNPS", "KLAC", "LRCX", "MCHP",
    "MRVL", "ON", "FTNT", "CRWD", "DDOG", "ZS", "NET", "OKTA", "MDB", "TEAM",
    "VEEV", "HUBS", "TTD", "UBER", "DASH", "TSLA", "PYPL", "SQ", "COIN", "HOOD",
    "RBLX", "ROKU", "DOCU", "TWLO", "BILL", "WDAY", "PINS", "SNAP", "SPOT", "GPN",
    "FIS", "PAYC", "FICO",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="download expanded public training data")
    parser.add_argument("--output-dir", default="research/expanded_data")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    cache = Path(args.cache_dir or output / "sec_cache")
    symbols = list(SYMBOLS)
    prices = download_yfinance(["SPY"] + symbols, start=args.start, end=args.end)
    prices.to_csv(output / "prices.csv", index_label="date")
    (output / "symbols.txt").write_text("\n".join(symbols) + "\n", encoding="utf-8")
    fundamentals = build_sec_fundamentals(symbols, cache_dir=str(cache))
    fundamentals.to_csv(output / "fundamentals.csv", index=False)
    summary = {
        "price_rows": int(len(prices)),
        "price_columns": int(len(prices.columns)),
        "available_symbols": sorted(prices.columns.tolist()),
        "fundamental_symbols": int(fundamentals["symbol"].nunique()),
        "sec_failures": fundamentals.attrs.get("failures", {}),
        "start": str(prices.index.min()),
        "end": str(prices.index.max()),
    }
    (output / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
