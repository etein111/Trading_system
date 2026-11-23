from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def validate_close_matrix(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        raise ValueError("price data is empty")
    result = prices.copy()
    result.index = pd.to_datetime(result.index)
    result = result.sort_index()
    result = result[~result.index.duplicated(keep="last")]
    result = result.apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan).ffill()
    result = result.dropna(axis=1, how="all")
    if (result <= 0).any().any():
        raise ValueError("prices must be positive")
    return result.dropna(how="all")


def load_close_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    columns = {str(column).lower(): column for column in frame.columns}
    if {"date", "symbol", "close"}.issubset(columns):
        long = frame.rename(
            columns={columns["date"]: "date", columns["symbol"]: "symbol", columns["close"]: "close"}
        )
        matrix = long.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    else:
        date_column = columns.get("date", frame.columns[0])
        wide = frame.rename(columns={date_column: "date"}).set_index("date")
        matrix = wide
    return validate_close_matrix(matrix)


def load_volume_csv(path: str) -> pd.DataFrame:
    """Load a wide or long daily volume CSV without price positivity checks."""
    frame = pd.read_csv(path)
    columns = {str(column).lower(): column for column in frame.columns}
    if {"date", "symbol", "volume"}.issubset(columns):
        long = frame.rename(
            columns={columns["date"]: "date", columns["symbol"]: "symbol", columns["volume"]: "volume"}
        )
        matrix = long.pivot_table(index="date", columns="symbol", values="volume", aggfunc="last")
    else:
        date_column = columns.get("date", frame.columns[0])
        matrix = frame.rename(columns={date_column: "date"}).set_index("date")
    matrix.index = pd.to_datetime(matrix.index)
    matrix = matrix.sort_index().apply(pd.to_numeric, errors="coerce")
    matrix = matrix.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return matrix


def load_fundamentals_csv(path: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"symbol", "available_at"}
    if not required.issubset({str(column) for column in frame.columns}):
        raise ValueError("fundamentals CSV must contain symbol and available_at columns")
    frame["available_at"] = pd.to_datetime(frame["available_at"])
    return frame.sort_values(["symbol", "available_at"]).reset_index(drop=True)


def download_yfinance(symbols, start=None, end=None) -> pd.DataFrame:
    """Optional convenience loader. Install yfinance separately for live market data."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("download_yfinance requires: pip install yfinance") from exc
    raw = yf.download(
        list(symbols),
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("yfinance returned no data")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"]
    else:
        name = list(symbols)[0]
        close = raw[["Close"]].rename(columns={"Close": name})
    return validate_close_matrix(close)


def download_yfinance_volume(symbols, start=None, end=None) -> pd.DataFrame:
    """Download raw daily share volume with the same dates as the close loader."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise RuntimeError("download_yfinance_volume requires: pip install yfinance") from exc
    names = list(symbols)
    raw = yf.download(
        names,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )
    if raw.empty:
        raise ValueError("yfinance returned no volume data")
    if isinstance(raw.columns, pd.MultiIndex):
        volume = raw["Volume"]
    else:
        name = names[0]
        volume = raw[["Volume"]].rename(columns={"Volume": name})
    volume.index = pd.to_datetime(volume.index)
    volume = volume.sort_index().apply(pd.to_numeric, errors="coerce")
    volume = volume.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")
    return volume


def generate_demo_prices(symbols: Iterable[str], days: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Create deterministic data for smoke tests; never use it for investment decisions."""
    names = list(dict.fromkeys(symbols))
    if "SPY" not in names:
        names.insert(0, "SPY")
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    values = {}
    regime = np.sin(np.linspace(0, 8 * np.pi, days))
    for index, symbol in enumerate(names):
        drift = 0.00025 + index * 0.00002
        beta = 0.8 + (index % 4) * 0.15
        noise = rng.normal(0, 0.010 + (index % 3) * 0.002, days)
        returns = drift + beta * (0.00025 + 0.004 * regime / 3.0) + noise
        prices = 100 * np.exp(np.cumsum(returns))
        values[symbol] = prices
    return validate_close_matrix(pd.DataFrame(values, index=dates))

