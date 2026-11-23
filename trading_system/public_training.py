import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


SEC_HEADERS = {
    "User-Agent": os.environ.get(
        "SEC_USER_AGENT",
        "TradingSystem research contact research@example.com",
    )
}

SEC_TAGS = {
    "revenue": ["RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"],
    "net_income": ["NetIncomeLoss"],
    "assets": ["Assets"],
    "equity": ["StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
    "operating_cash_flow": ["NetCashProvidedByUsedInOperatingActivities"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment"],
}

FEATURE_COLUMNS = [
    "momentum_60",
    "momentum_252_ex_21",
    "trend_distance_200",
    "volatility_63",
    "drawdown_252",
    "roe",
    "roa",
    "fcf_margin",
    "revenue_growth",
    "earnings_growth",
]

ENHANCED_TECHNICAL_FEATURE_COLUMNS = [
    "momentum_20",
    "momentum_60",
    "momentum_126",
    "momentum_252_ex_21",
    "trend_distance_50",
    "trend_distance_200",
    "volatility_20",
    "volatility_63",
    "drawdown_252",
    "relative_momentum_60",
    "relative_momentum_252_ex_21",
    "volume_ratio_20",
    "volume_zscore_20",
    "volume_missing",
]


@dataclass
class TrainingConfig:
    horizon_days: int = 60
    train_years: int = 4
    minimum_train_samples: int = 150
    evaluation_stride: int = 3
    top_k: int = 4
    random_state: int = 7


class SecCompanyFactsClient:
    """Public SEC XBRL Company Facts client with local caching."""

    def __init__(self, cache_dir: str = "research/sec_cache", sleep_seconds: float = 0.2):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sleep_seconds = sleep_seconds
        self._ticker_map = None

    def ticker_map(self) -> Dict[str, str]:
        if self._ticker_map is not None:
            return self._ticker_map
        path = self.cache_dir / "company_tickers.json"
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
        else:
            response = requests.get("https://www.sec.gov/files/company_tickers.json", headers=SEC_HEADERS, timeout=30)
            response.raise_for_status()
            raw = response.json()
            path.write_text(json.dumps(raw), encoding="utf-8")
        self._ticker_map = {str(item["ticker"]).upper(): str(item["cik_str"]).zfill(10) for item in raw.values()}
        return self._ticker_map

    def facts(self, symbol: str) -> Dict:
        cik = self.ticker_map().get(symbol.upper())
        if cik is None:
            raise KeyError("SEC CIK not found for " + symbol)
        path = self.cache_dir / (cik + ".json")
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
        response = requests.get(
            "https://data.sec.gov/api/xbrl/companyfacts/CIK" + cik + ".json",
            headers=SEC_HEADERS,
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        path.write_text(json.dumps(payload), encoding="utf-8")
        time.sleep(self.sleep_seconds)
        return payload

    @staticmethod
    def _rows_for_tag(facts: Dict, tags: Iterable[str], metric: str) -> pd.DataFrame:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        for tag in tags:
            if tag not in us_gaap:
                continue
            units = us_gaap[tag].get("units", {})
            unit_name = "USD" if "USD" in units else next(iter(units), None)
            if unit_name is None:
                continue
            frame = pd.DataFrame(units[unit_name])
            if frame.empty or "filed" not in frame.columns or "end" not in frame.columns:
                continue
            frame["metric"] = metric
            frame["value"] = pd.to_numeric(frame["val"], errors="coerce")
            frame["available_at"] = pd.to_datetime(frame["filed"])
            frame["period_end"] = pd.to_datetime(frame["end"])
            return frame
        return pd.DataFrame()

    def annual_point_in_time(self, symbol: str) -> pd.DataFrame:
        facts = self.facts(symbol)
        frames = []
        for metric, tags in SEC_TAGS.items():
            frame = self._rows_for_tag(facts, tags, metric)
            if frame.empty:
                continue
            forms = frame.get("form", pd.Series(index=frame.index, dtype=object)).astype(str)
            frame = frame[forms.isin(["10-K", "10-K/A"])]
            if "start" in frame.columns:
                start = pd.to_datetime(frame["start"], errors="coerce")
                duration = (frame["period_end"] - start).dt.days
                # Annual flow facts generally cover 300–400 days.
                flow = metric not in {"assets", "equity"}
                if flow:
                    frame = frame[duration.between(300, 400, inclusive="both")]
            frame = frame[["available_at", "period_end", "value", "metric"]].dropna(subset=["value"])
            frames.append(frame)
        if not frames:
            return pd.DataFrame()
        long = pd.concat(frames, ignore_index=True)
        wide = long.pivot_table(
            index=["available_at", "period_end"],
            columns="metric",
            values="value",
            aggfunc="last",
        ).reset_index()
        wide["symbol"] = symbol.upper()
        wide = wide.sort_values(["period_end", "available_at"]).reset_index(drop=True)
        derived = []
        for _, row in wide.iterrows():
            prior = wide[
                (wide["period_end"] < row["period_end"])
                & (wide["available_at"] <= row["available_at"])
            ].sort_values("period_end").tail(1)
            prior_row = prior.iloc[0] if not prior.empty else None
            revenue = row.get("revenue", np.nan)
            income = row.get("net_income", np.nan)
            assets = row.get("assets", np.nan)
            equity = row.get("equity", np.nan)
            cfo = row.get("operating_cash_flow", np.nan)
            capex = row.get("capex", np.nan)
            record = row.to_dict()
            record["roe"] = income / equity if pd.notna(income) and pd.notna(equity) and equity != 0 else np.nan
            record["roa"] = income / assets if pd.notna(income) and pd.notna(assets) and assets != 0 else np.nan
            record["fcf_margin"] = (cfo - capex) / revenue if pd.notna(cfo) and pd.notna(capex) and pd.notna(revenue) and revenue != 0 else np.nan
            record["revenue_growth"] = (
                revenue / prior_row.get("revenue", np.nan) - 1.0
                if prior_row is not None and pd.notna(revenue) and pd.notna(prior_row.get("revenue", np.nan)) and prior_row.get("revenue") != 0
                else np.nan
            )
            record["earnings_growth"] = (
                income / prior_row.get("net_income", np.nan) - 1.0
                if prior_row is not None and pd.notna(income) and pd.notna(prior_row.get("net_income", np.nan)) and prior_row.get("net_income") != 0
                else np.nan
            )
            derived.append(record)
        return pd.DataFrame(derived)


def build_sec_fundamentals(symbols: Iterable[str], cache_dir: str) -> pd.DataFrame:
    client = SecCompanyFactsClient(cache_dir=cache_dir)
    frames = []
    failures = {}
    for symbol in symbols:
        try:
            frame = client.annual_point_in_time(symbol)
            if not frame.empty:
                frames.append(frame)
        except Exception as exc:  # keep a public data run resilient to one ticker
            failures[symbol] = str(exc)
    if not frames:
        raise RuntimeError("SEC returned no usable annual facts: " + json.dumps(failures))
    result = pd.concat(frames, ignore_index=True)
    result.attrs["failures"] = failures
    return result.sort_values(["available_at", "symbol"]).reset_index(drop=True)


def _technical_features(
    series: pd.Series,
    benchmark_series: Optional[pd.Series] = None,
    volume_series: Optional[pd.Series] = None,
) -> Optional[Dict[str, float]]:
    series = series.dropna()
    if len(series) < 253:
        return None
    current = float(series.iloc[-1])
    returns = series.pct_change().dropna()
    sma50 = float(series.rolling(50).mean().iloc[-1])
    sma200 = float(series.rolling(200).mean().iloc[-1])
    result = {
        "momentum_20": current / float(series.iloc[-21]) - 1.0,
        "momentum_60": current / float(series.iloc[-61]) - 1.0,
        "momentum_126": current / float(series.iloc[-127]) - 1.0,
        "momentum_252_ex_21": float(series.iloc[-22] / series.iloc[-253] - 1.0),
        "trend_distance_50": current / sma50 - 1.0,
        "trend_distance_200": current / sma200 - 1.0,
        "volatility_20": float(returns.tail(20).std(ddof=0)),
        "volatility_63": float(returns.tail(63).std(ddof=0)),
        "drawdown_252": current / float(series.tail(252).max()) - 1.0,
    }
    if benchmark_series is not None:
        benchmark = benchmark_series.reindex(series.index).ffill().dropna()
        if len(benchmark) >= 253:
            result["relative_momentum_60"] = result["momentum_60"] - (float(benchmark.iloc[-1] / benchmark.iloc[-61] - 1.0))
            result["relative_momentum_252_ex_21"] = result["momentum_252_ex_21"] - (float(benchmark.iloc[-22] / benchmark.iloc[-253] - 1.0))
        else:
            result["relative_momentum_60"] = np.nan
            result["relative_momentum_252_ex_21"] = np.nan
    else:
        result["relative_momentum_60"] = np.nan
        result["relative_momentum_252_ex_21"] = np.nan
    if volume_series is not None:
        volume = pd.to_numeric(volume_series.reindex(series.index), errors="coerce").dropna()
        if len(volume) >= 20 and float(volume.iloc[-1]) > 0:
            recent = volume.tail(20).astype(float)
            log_recent = np.log(recent.clip(lower=1.0))
            mean = float(log_recent.mean())
            std = float(log_recent.std(ddof=0))
            result["volume_ratio_20"] = float(volume.iloc[-1] / max(float(recent.median()), 1.0) - 1.0)
            result["volume_zscore_20"] = float((log_recent.iloc[-1] - mean) / std) if std > 0 else 0.0
            result["volume_missing"] = 0.0
        else:
            result["volume_ratio_20"] = np.nan
            result["volume_zscore_20"] = np.nan
            result["volume_missing"] = 1.0
    else:
        result["volume_ratio_20"] = np.nan
        result["volume_zscore_20"] = np.nan
        result["volume_missing"] = 1.0
    return result


def build_training_samples(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    benchmark: str = "SPY",
    symbols: Optional[Iterable[str]] = None,
    config: Optional[TrainingConfig] = None,
    volumes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    config = config or TrainingConfig()
    prices = prices.sort_index()
    symbols = list(symbols or [column for column in prices.columns if column != benchmark])
    dates = list(prices.index[252:-config.horizon_days:21])
    rows = []
    for date in dates:
        future_index = prices.index.get_loc(date) + config.horizon_days
        if future_index >= len(prices.index) or benchmark not in prices.columns:
            continue
        spy_return = prices[benchmark].iloc[future_index] / prices[benchmark].loc[date] - 1.0
        for symbol in symbols:
            if symbol not in prices.columns:
                continue
            series = prices[symbol].loc[:date]
            benchmark_series = prices[benchmark].loc[:date] if benchmark in prices.columns else None
            volume_series = volumes[symbol].loc[:date] if volumes is not None and symbol in volumes.columns else None
            technical = _technical_features(series, benchmark_series=benchmark_series, volume_series=volume_series)
            if technical is None:
                continue
            future_return = prices[symbol].iloc[future_index] / prices[symbol].loc[date] - 1.0
            known = fundamentals[
                (fundamentals["symbol"] == symbol)
                & (fundamentals["available_at"] <= date)
            ].sort_values("available_at").tail(1)
            row = {"feature_date": date, "label_end": prices.index[future_index], "symbol": symbol}
            row.update(technical)
            if known.empty:
                for name in ["roe", "roa", "fcf_margin", "revenue_growth", "earnings_growth"]:
                    row[name] = np.nan
            else:
                row.update(known.iloc[0].to_dict())
            row["asset_forward_return"] = float(future_return)
            row["benchmark_forward_return"] = float(spy_return)
            row["excess_return"] = float(future_return - spy_return)
            row["eligible"] = bool(row["trend_distance_200"] > 0 and row["momentum_60"] > 0)
            rows.append(row)
    return pd.DataFrame(rows)


def train_walk_forward(
    samples: pd.DataFrame,
    latest_features: pd.DataFrame,
    output_dir: str,
    config: Optional[TrainingConfig] = None,
) -> Dict[str, object]:
    config = config or TrainingConfig()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = samples.copy()
    samples["feature_date"] = pd.to_datetime(samples["feature_date"])
    samples["label_end"] = pd.to_datetime(samples["label_end"])
    latest_features = latest_features.copy()
    latest_features["feature_date"] = pd.to_datetime(latest_features["feature_date"])
    model = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=250,
                    max_depth=4,
                    min_samples_leaf=8,
                    random_state=config.random_state,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    predictions = []
    evaluation_dates = sorted(samples["feature_date"].unique())[:: config.evaluation_stride]
    for date in evaluation_dates:
        train = samples[(samples["label_end"] < date) & (samples["feature_date"] < date)]
        if config.train_years:
            train = train[train["feature_date"] >= date - pd.Timedelta(days=365 * config.train_years)]
        test = samples[samples["feature_date"] == date].copy()
        if len(train) < config.minimum_train_samples or test.empty:
            continue
        model.fit(train[FEATURE_COLUMNS], train["excess_return"])
        test["predicted_score"] = model.predict(test[FEATURE_COLUMNS])
        eligible = test[test["eligible"]].sort_values("predicted_score", ascending=False).head(config.top_k)
        if eligible.empty:
            continue
        row = {
            "date": date,
            "selected_count": len(eligible),
            "selected_return": float(eligible["asset_forward_return"].mean()),
            "benchmark_return": float(eligible["benchmark_forward_return"].iloc[0]),
            "selected_excess_return": float(eligible["excess_return"].mean()),
            "selected_symbols": ",".join(eligible["symbol"].tolist()),
        }
        predictions.append(row)
    prediction_frame = pd.DataFrame(predictions)
    if not prediction_frame.empty:
        prediction_frame["date"] = pd.to_datetime(prediction_frame["date"])
        prediction_frame.to_csv(output / "walk_forward_predictions.csv", index=False)
    samples.to_csv(output / "training_samples.csv", index=False)

    asof = latest_features["feature_date"].max()
    train_final = samples[(samples["label_end"] < asof) & (samples["feature_date"] < asof)]
    if config.train_years:
        train_final = train_final[train_final["feature_date"] >= asof - pd.Timedelta(days=365 * config.train_years)]
    if len(train_final) < config.minimum_train_samples:
        raise RuntimeError("not enough point-in-time samples for final model: " + str(len(train_final)))
    model.fit(train_final[FEATURE_COLUMNS], train_final["excess_return"])
    latest_features = latest_features.copy()
    latest_features["score"] = model.predict(latest_features[FEATURE_COLUMNS])
    latest_features["available_at"] = latest_features["feature_date"]
    latest_features[["symbol", "score", "available_at"]].to_csv(output / "model_scores.csv", index=False)
    with open(output / "model.pkl", "wb") as handle:
        pickle.dump(model, handle)

    summary = {
        "model": "RandomForestRegressor with median imputation",
        "feature_columns": FEATURE_COLUMNS,
        "training_samples": int(len(train_final)),
        "walk_forward_events": int(len(prediction_frame)),
        "top_k": config.top_k,
        "asof": str(asof),
        "mean_selected_excess_return": float(prediction_frame["selected_excess_return"].mean()) if not prediction_frame.empty else None,
        "hit_rate_vs_benchmark": float((prediction_frame["selected_excess_return"] > 0).mean()) if not prediction_frame.empty else None,
        "cumulative_selected_return": float((1.0 + prediction_frame["selected_return"]).prod() - 1.0) if not prediction_frame.empty else None,
        "cumulative_benchmark_return": float((1.0 + prediction_frame["benchmark_return"]).prod() - 1.0) if not prediction_frame.empty else None,
        "data_source": "SEC XBRL Company Facts + yfinance adjusted daily prices",
    }
    (output / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_latest_features(
    prices: pd.DataFrame,
    fundamentals: pd.DataFrame,
    symbols: Iterable[str],
    asof=None,
    volumes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    date = pd.Timestamp(asof or prices.index[-1])
    rows = []
    for symbol in symbols:
        if symbol not in prices.columns:
            continue
        benchmark_series = prices["SPY"].loc[:date] if "SPY" in prices.columns else None
        volume_series = volumes[symbol].loc[:date] if volumes is not None and symbol in volumes.columns else None
        technical = _technical_features(
            prices[symbol].loc[:date],
            benchmark_series=benchmark_series,
            volume_series=volume_series,
        )
        if technical is None:
            continue
        known = fundamentals[(fundamentals["symbol"] == symbol) & (fundamentals["available_at"] <= date)].sort_values("available_at").tail(1)
        row = {"feature_date": date, "symbol": symbol}
        row.update(technical)
        if known.empty:
            for name in ["roe", "roa", "fcf_margin", "revenue_growth", "earnings_growth"]:
                row[name] = np.nan
        else:
            row.update(known.iloc[0].to_dict())
        row["eligible"] = bool(row["trend_distance_200"] > 0 and row["momentum_60"] > 0)
        rows.append(row)
    return pd.DataFrame(rows)

