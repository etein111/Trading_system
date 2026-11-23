"""Point-in-time Qlib training for the personal trading system.

The source data remains ordinary pandas files.  This module adapts them to
Qlib's DataLoader/DataHandlerLP/DatasetH abstractions and adds the safeguards
needed for a small cross-sectional US equity data set: point-in-time labels,
purged validation, cross-sectional feature ranks, explicit missing indicators,
and a ranking-oriented target.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from qlib.data.dataset.loader import DataLoader
from qlib.model.base import Model

from .public_training import FEATURE_COLUMNS, TrainingConfig

FUNDAMENTAL_FEATURE_COLUMNS = ["roe", "roa", "fcf_margin", "revenue_growth", "earnings_growth"]
TECHNICAL_FEATURE_COLUMNS = FEATURE_COLUMNS[:5]
QLIB_FEATURE_COLUMNS = FEATURE_COLUMNS + [f"{column}_missing" for column in FUNDAMENTAL_FEATURE_COLUMNS]


class StaticDataLoader(DataLoader):
    """Qlib DataLoader backed by a prepared pandas frame."""

    def __init__(self, frame: pd.DataFrame):
        if not isinstance(frame.index, pd.MultiIndex):
            raise ValueError("Qlib frame index must be a datetime/instrument MultiIndex")
        if frame.index.names != ["datetime", "instrument"]:
            raise ValueError("Qlib frame index names must be datetime and instrument")
        self.frame = frame.sort_index()

    def load(self, instruments=None, start_time=None, end_time=None) -> pd.DataFrame:
        frame = self.frame
        if instruments is not None:
            names = [instruments] if isinstance(instruments, str) else list(instruments)
            frame = frame[frame.index.get_level_values("instrument").isin(names)]
        if start_time is not None:
            frame = frame[frame.index.get_level_values("datetime") >= pd.Timestamp(start_time)]
        if end_time is not None:
            frame = frame[frame.index.get_level_values("datetime") <= pd.Timestamp(end_time)]
        return frame.copy()


def _load_lightgbm():
    """Import LightGBM, loading a bundled macOS OpenMP runtime if needed."""

    try:
        import lightgbm as lgb  # type: ignore

        return lgb
    except OSError:
        import ctypes
        import sys

        candidates = [
            Path(sys.prefix) / "lib/python3.11/site-packages/sklearn/.dylibs/libomp.dylib",
            Path(__file__).resolve().parents[1]
            / ".venv/lib/python3.11/site-packages/sklearn/.dylibs/libomp.dylib",
        ]
        for candidate in candidates:
            if candidate.exists():
                ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
                break
        import lightgbm as lgb  # type: ignore

        return lgb


def _prepare_rows(
    rows: pd.DataFrame,
    feature_columns: Sequence[str],
    target_mode: str = "rank",
    cross_sectional_rank: bool = True,
) -> pd.DataFrame:
    """Add missing flags, rank-normalize features, and construct the label."""

    rows = rows.copy()
    rows["feature_date"] = pd.to_datetime(rows["feature_date"])
    rows["symbol"] = rows["symbol"].astype(str)
    for column in FUNDAMENTAL_FEATURE_COLUMNS:
        if column not in rows:
            rows[column] = np.nan
        rows[f"{column}_missing"] = rows[column].isna().astype(float)

    base_columns = [column for column in feature_columns if not column.endswith("_missing")]
    if cross_sectional_rank:
        for column in base_columns:
            values = pd.to_numeric(rows[column], errors="coerce")
            ranked = values.groupby(rows["feature_date"]).rank(pct=True)
            rows[column] = ranked.fillna(0.5).astype(float)
    else:
        for column in base_columns:
            rows[column] = pd.to_numeric(rows[column], errors="coerce")

    for column in feature_columns:
        if column.endswith("_missing"):
            rows[column] = pd.to_numeric(rows[column], errors="coerce").fillna(1.0).astype(float)

    if "excess_return" not in rows:
        rows["excess_return"] = np.nan
    labels = pd.to_numeric(rows["excess_return"], errors="coerce")
    if target_mode == "rank":
        rows["training_target"] = labels.groupby(rows["feature_date"]).rank(pct=True) - 0.5
    elif target_mode == "raw":
        rows["training_target"] = labels
    else:
        raise ValueError("target_mode must be rank or raw")
    return rows


def _as_qlib_frame(
    rows: pd.DataFrame,
    feature_columns: Sequence[str],
    include_label: bool = True,
) -> pd.DataFrame:
    """Convert flat samples to Qlib's two-level feature/label columns."""

    if rows.empty:
        raise ValueError("cannot create a Qlib dataset from an empty frame")
    required = {"feature_date", "symbol", *feature_columns}
    missing = sorted(required - set(rows.columns))
    if missing:
        raise ValueError("training rows missing columns: " + ", ".join(missing))
    index = pd.MultiIndex.from_arrays(
        [pd.to_datetime(rows["feature_date"]), rows["symbol"].astype(str)],
        names=["datetime", "instrument"],
    )
    frame = pd.DataFrame(
        rows[list(feature_columns)].to_numpy(dtype=float),
        index=index,
        columns=pd.MultiIndex.from_product([["feature"], list(feature_columns)]),
    )
    if include_label:
        frame["label", "label"] = pd.to_numeric(rows["training_target"], errors="coerce").to_numpy()
    return frame.sort_index()


def _dataset(frame: pd.DataFrame, segments: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]]):
    from qlib.data.dataset import DatasetH
    from qlib.data.dataset.handler import DataHandlerLP

    handler = DataHandlerLP(
        data_loader=StaticDataLoader(frame),
        infer_processors=[],
        learn_processors=[],
        shared_processors=[],
    )
    return DatasetH(handler=handler, segments=segments)


class QlibLightGBMModel(Model):
    """Qlib-compatible LightGBM model without Qlib recorder side effects."""

    def __init__(
        self,
        feature_columns: Sequence[str] = QLIB_FEATURE_COLUMNS,
        params: Optional[Dict[str, object]] = None,
        num_boost_round: int = 80,
        early_stopping_rounds: int = 0,
    ):
        self.feature_columns = list(feature_columns)
        self.params = {
            "objective": "regression",
            "metric": "l2",
            "verbosity": -1,
            "learning_rate": 0.03,
            "num_leaves": 7,
            "max_depth": -1,
            "min_data_in_leaf": 20,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "lambda_l1": 0.10,
            "lambda_l2": 2.0,
            "seed": 7,
            "feature_fraction_seed": 7,
            "bagging_seed": 7,
            "deterministic": True,
            "force_col_wise": True,
        }
        if params:
            self.params.update(params)
        self.num_boost_round = int(num_boost_round)
        self.early_stopping_rounds = int(early_stopping_rounds)
        self.booster = None
        self.best_iteration = None

    def fit(self, dataset, reweighter=None):
        from qlib.data.dataset.handler import DataHandlerLP

        lgb = _load_lightgbm()
        train = dataset.prepare("train", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
        if train.empty:
            raise ValueError("Qlib train segment is empty")
        x_train = train["feature"].to_numpy(dtype=float)
        y_train = np.asarray(train["label"].to_numpy(dtype=float)).reshape(-1)
        train_set = lgb.Dataset(x_train, label=y_train, feature_name=self.feature_columns)
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks = []
        if "valid" in dataset.segments:
            valid = dataset.prepare("valid", col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
            if not valid.empty:
                x_valid = valid["feature"].to_numpy(dtype=float)
                y_valid = np.asarray(valid["label"].to_numpy(dtype=float)).reshape(-1)
                valid_sets.append(
                    lgb.Dataset(
                        x_valid,
                        label=y_valid,
                        reference=train_set,
                        feature_name=self.feature_columns,
                    )
                )
                valid_names.append("valid")
                if self.early_stopping_rounds > 0:
                    callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        self.booster = lgb.train(
            self.params,
            train_set,
            num_boost_round=self.num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self.best_iteration = int(self.booster.current_iteration())
        return self

    def predict(self, dataset, segment: str = "test") -> pd.Series:
        from qlib.data.dataset.handler import DataHandlerLP

        if self.booster is None:
            raise ValueError("model is not fitted")
        features = dataset.prepare(segment, col_set="feature", data_key=DataHandlerLP.DK_I)
        values = self.booster.predict(features.to_numpy(dtype=float))
        return pd.Series(values, index=features.index, name="score")


def _purged_split(train: pd.DataFrame, valid_fraction: float = 0.2):
    """Split by date and purge any training label reaching into validation."""

    dates = sorted(pd.to_datetime(train["feature_date"]).unique())
    if len(dates) < 3:
        return train.copy(), pd.DataFrame(columns=train.columns)
    cut = max(1, int(len(dates) * (1.0 - valid_fraction)))
    valid_start = pd.Timestamp(dates[cut])
    valid = train[train["feature_date"] >= valid_start].copy()
    core = train[
        (train["feature_date"] < valid_start)
        & (pd.to_datetime(train["label_end"]) < valid_start)
    ].copy()
    return core, valid


def _metrics(predictions: pd.DataFrame) -> Dict[str, object]:
    if predictions.empty:
        return {
            "walk_forward_events": 0,
            "mean_selected_excess_return": None,
            "hit_rate_vs_benchmark": None,
            "cumulative_selected_return": None,
            "cumulative_benchmark_return": None,
        }
    return {
        "walk_forward_events": int(len(predictions)),
        "mean_selected_excess_return": float(predictions["selected_excess_return"].mean()),
        "hit_rate_vs_benchmark": float((predictions["selected_excess_return"] > 0).mean()),
        "cumulative_selected_return": float((1.0 + predictions["selected_return"]).prod() - 1.0),
        "cumulative_benchmark_return": float((1.0 + predictions["benchmark_return"]).prod() - 1.0),
    }


def _scores_by_symbol(scores: pd.Series) -> Dict[str, float]:
    if scores.empty:
        return {}
    return scores.groupby(level="instrument").first().astype(float).to_dict()


def _date_segment(frame: pd.DataFrame) -> Tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.to_datetime(frame["feature_date"])
    return pd.Timestamp(dates.min()), pd.Timestamp(dates.max())


def train_qlib_walk_forward(
    samples: pd.DataFrame,
    latest_features: pd.DataFrame,
    output_dir: str,
    config: Optional[TrainingConfig] = None,
    model_params: Optional[Dict[str, object]] = None,
    feature_columns: Optional[Sequence[str]] = None,
    target_mode: str = "rank",
    cross_sectional_rank: bool = True,
    num_boost_round: int = 80,
    early_stopping_rounds: int = 0,
    purge_validation: bool = True,
) -> Dict[str, object]:
    """Train and export point-in-time Qlib scores with robust walk-forward splits."""

    config = config or TrainingConfig()
    feature_columns = list(feature_columns or QLIB_FEATURE_COLUMNS)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    samples = samples.copy()
    samples["feature_date"] = pd.to_datetime(samples["feature_date"])
    samples["label_end"] = pd.to_datetime(samples["label_end"])
    latest_features = latest_features.copy()
    latest_features["feature_date"] = pd.to_datetime(latest_features["feature_date"])
    predictions = []
    historical_scores = []
    iteration_counts = []
    evaluation_dates = sorted(samples["feature_date"].unique())[:: config.evaluation_stride]

    for date in evaluation_dates:
        date = pd.Timestamp(date)
        train = samples[(samples["label_end"] < date) & (samples["feature_date"] < date)]
        if config.train_years:
            train = train[train["feature_date"] >= date - pd.Timedelta(days=365 * config.train_years)]
        test = samples[samples["feature_date"] == date].copy()
        if test.empty:
            continue
        core, valid = _purged_split(train) if purge_validation else (train.copy(), pd.DataFrame())
        if len(core) < config.minimum_train_samples:
            continue
        fit_rows = pd.concat([core, valid, test], ignore_index=True)
        processed = _prepare_rows(
            fit_rows,
            feature_columns=feature_columns,
            target_mode=target_mode,
            cross_sectional_rank=cross_sectional_rank,
        )
        frame = _as_qlib_frame(processed, feature_columns=feature_columns)
        segments = {"train": _date_segment(core), "test": (date, date)}
        if not valid.empty:
            segments["valid"] = _date_segment(valid)
        dataset = _dataset(frame, segments)
        model = QlibLightGBMModel(
            feature_columns=feature_columns,
            params=model_params,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
        )
        model.fit(dataset)
        iteration_counts.append(model.best_iteration)
        scored = test.copy()
        scores = model.predict(dataset, "test")
        scored["predicted_score"] = scored["symbol"].map(_scores_by_symbol(scores))
        historical_scores.append(
            scored[["symbol", "predicted_score"]]
            .rename(columns={"predicted_score": "score"})
            .assign(available_at=date)
        )
        eligible = scored[scored["eligible"]].sort_values("predicted_score", ascending=False).head(config.top_k)
        if eligible.empty:
            continue
        predictions.append(
            {
                "date": date,
                "selected_count": int(len(eligible)),
                "selected_return": float(eligible["asset_forward_return"].mean()),
                "benchmark_return": float(eligible["benchmark_forward_return"].iloc[0]),
                "selected_excess_return": float(eligible["excess_return"].mean()),
                "selected_symbols": ",".join(eligible["symbol"].tolist()),
            }
        )

    prediction_frame = pd.DataFrame(predictions)
    if not prediction_frame.empty:
        prediction_frame["date"] = pd.to_datetime(prediction_frame["date"])
    prediction_frame.to_csv(output / "qlib_walk_forward_predictions.csv", index=False)
    historical_score_frame = (
        pd.concat(historical_scores, ignore_index=True)
        if historical_scores
        else pd.DataFrame(columns=["symbol", "score", "available_at"])
    )
    historical_score_frame.to_csv(output / "qlib_walk_forward_scores.csv", index=False)

    asof = latest_features["feature_date"].max()
    train_final = samples[(samples["label_end"] < asof) & (samples["feature_date"] < asof)]
    if config.train_years:
        train_final = train_final[train_final["feature_date"] >= asof - pd.Timedelta(days=365 * config.train_years)]
    if len(train_final) < config.minimum_train_samples:
        raise RuntimeError("not enough point-in-time samples for final Qlib model: " + str(len(train_final)))
    final_processed = _prepare_rows(
        pd.concat([train_final, latest_features], ignore_index=True),
        feature_columns=feature_columns,
        target_mode=target_mode,
        cross_sectional_rank=cross_sectional_rank,
    )
    frame = _as_qlib_frame(final_processed, feature_columns=feature_columns)
    segments = {"train": _date_segment(train_final), "test": (asof, asof)}
    dataset = _dataset(frame, segments)
    model = QlibLightGBMModel(
        feature_columns=feature_columns,
        params=model_params,
        num_boost_round=num_boost_round,
        early_stopping_rounds=0,
    )
    model.fit(dataset)
    latest_features["score"] = latest_features["symbol"].map(_scores_by_symbol(model.predict(dataset, "test")))
    latest_features["available_at"] = latest_features["feature_date"]
    latest_features[["symbol", "score", "available_at"]].to_csv(output / "qlib_scores.csv", index=False)
    with open(output / "qlib_model.pkl", "wb") as handle:
        pickle.dump(model, handle)
    pd.DataFrame({"iteration": iteration_counts}).to_csv(output / "qlib_walk_forward_iterations.csv", index=False)
    processed_train = _prepare_rows(
        train_final,
        feature_columns=feature_columns,
        target_mode=target_mode,
        cross_sectional_rank=cross_sectional_rank,
    )
    processed_train.to_csv(output / "processed_training_samples.csv", index=False)

    summary = {
        "model": "Qlib DatasetH/DataHandlerLP + LightGBM",
        "qlib_version": _qlib_version(),
        "feature_columns": list(feature_columns),
        "target_mode": target_mode,
        "cross_sectional_rank": bool(cross_sectional_rank),
        "purge_validation": bool(purge_validation),
        "num_boost_round": int(num_boost_round),
        "early_stopping_rounds": int(early_stopping_rounds),
        "training_samples": int(len(train_final)),
        "training_symbols": int(train_final["symbol"].nunique()),
        "walk_forward_score_rows": int(len(historical_score_frame)),
        "walk_forward_events_attempted": int(len(evaluation_dates)),
        "walk_forward_iterations_median": float(np.median(iteration_counts)) if iteration_counts else None,
        "walk_forward_iterations_mean": float(np.mean(iteration_counts)) if iteration_counts else None,
        "top_k": int(config.top_k),
        "asof": str(asof),
        "data_contract": "features are as-of feature_date; labels satisfy label_end < evaluation date; validation labels are purged across the split",
        **_metrics(prediction_frame),
    }
    (output / "qlib_training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def _qlib_version() -> str:
    import qlib

    return str(getattr(qlib, "__version__", "unknown"))
