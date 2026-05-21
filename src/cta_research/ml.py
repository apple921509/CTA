from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cta_research.backtest import BacktestResult, run_backtest
from cta_research.data import MarketData


ModelName = Literal["linear", "ridge", "random_forest"]


@dataclass(frozen=True)
class SupervisedDataset:
    features: pd.DataFrame
    target: pd.Series


@dataclass(frozen=True)
class MLBacktestResult:
    predictions: pd.DataFrame
    positions: pd.DataFrame
    metrics: pd.DataFrame
    backtest: BacktestResult


def build_supervised_dataset(
    factors: dict[str, pd.DataFrame],
    close: pd.DataFrame,
    horizon: int = 1,
    feature_names: list[str] | None = None,
) -> SupervisedDataset:
    if feature_names is None:
        feature_names = list(factors)
    future_returns = close.pct_change(horizon).shift(-horizon)

    feature_columns = {}
    for name in feature_names:
        if name not in factors:
            raise ValueError(f"Unknown factor: {name}")
        feature_columns[name] = factors[name].stack(future_stack=True)

    features = pd.DataFrame(feature_columns)
    features.index = features.index.set_names(["timestamp", "symbol"])
    target = future_returns.stack(future_stack=True).rename("target")
    target.index = target.index.set_names(["timestamp", "symbol"])

    aligned = features.join(target, how="inner").replace([np.inf, -np.inf], np.nan).dropna()
    return SupervisedDataset(
        features=aligned.loc[:, feature_names],
        target=aligned["target"],
    )


def make_model(
    model_name: ModelName,
    random_state: int = 42,
    **params,
):
    if model_name == "linear":
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LinearRegression(**params)),
            ]
        )
    if model_name == "ridge":
        alpha = params.pop("alpha", 1.0)
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha, **params)),
            ]
        )
    if model_name == "random_forest":
        defaults = {
            "n_estimators": 100,
            "max_depth": 4,
            "min_samples_leaf": 5,
            "random_state": random_state,
        }
        defaults.update(params)
        return RandomForestRegressor(**defaults)
    raise ValueError("model_name must be one of: linear, ridge, random_forest")


def expanding_window_predictions(
    dataset: SupervisedDataset,
    model_name: ModelName = "ridge",
    train_size: int = 365,
    test_size: int = 30,
    step_size: int | None = None,
    random_state: int = 42,
    model_params: dict | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    if step_size is None:
        step_size = test_size
    if train_size <= 0 or test_size <= 0 or step_size <= 0:
        raise ValueError("train_size, test_size, and step_size must be positive")
    if model_params is None:
        model_params = {}

    timestamps = pd.DatetimeIndex(dataset.features.index.get_level_values("timestamp").unique()).sort_values()
    predictions = []
    metrics = []
    start = train_size
    while start < len(timestamps):
        train_end = timestamps[start - 1]
        test_start = timestamps[start]
        test_end = timestamps[min(start + test_size, len(timestamps)) - 1]

        train_mask = dataset.features.index.get_level_values("timestamp") <= train_end
        test_dates = timestamps[start : min(start + test_size, len(timestamps))]
        test_mask = dataset.features.index.get_level_values("timestamp").isin(test_dates)
        if not test_mask.any():
            break

        x_train = dataset.features.loc[train_mask]
        y_train = dataset.target.loc[train_mask]
        x_test = dataset.features.loc[test_mask]
        y_test = dataset.target.loc[test_mask]

        model = make_model(model_name, random_state=random_state, **model_params)
        model.fit(x_train, y_train)
        y_pred = pd.Series(model.predict(x_test), index=x_test.index, name="prediction")
        predictions.append(y_pred)

        directional_accuracy = accuracy_score((y_test > 0).astype(int), (y_pred > 0).astype(int))
        ic = y_pred.corr(y_test, method="spearman")
        metrics.append(
            {
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
                "model": model_name,
                "observation_count": int(len(y_test)),
                "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5),
                "directional_accuracy": float(directional_accuracy),
                "rank_ic": 0.0 if pd.isna(ic) else float(ic),
            }
        )
        start += step_size

    if not predictions:
        return pd.Series(dtype=float, name="prediction"), pd.DataFrame.from_records(metrics)

    return pd.concat(predictions).sort_index(), pd.DataFrame.from_records(metrics)


def predictions_to_signal(
    predictions: pd.Series,
    index: pd.Index,
    columns: pd.Index | list[str],
    gross_exposure: float = 1.0,
    long_short: bool = True,
) -> pd.DataFrame:
    if predictions.empty:
        return pd.DataFrame(0.0, index=index, columns=columns)

    matrix = predictions.unstack("symbol").reindex(index=index, columns=columns)
    if long_short:
        demeaned = matrix.sub(matrix.mean(axis=1), axis=0)
    else:
        demeaned = matrix.clip(lower=0.0)
    gross = demeaned.abs().sum(axis=1).replace(0, np.nan)
    return demeaned.divide(gross, axis=0).mul(gross_exposure).fillna(0.0)


def run_ml_signal_backtest(
    data: MarketData,
    factors: dict[str, pd.DataFrame],
    model_name: ModelName = "ridge",
    horizon: int = 1,
    feature_names: list[str] | None = None,
    train_size: int = 365,
    test_size: int = 30,
    initial_capital: float = 100000,
    fee_bps: float = 6,
    slippage_bps: float = 2,
    gross_exposure: float = 1.0,
    random_state: int = 42,
    model_params: dict | None = None,
) -> MLBacktestResult:
    dataset = build_supervised_dataset(
        factors=factors,
        close=data.close,
        horizon=horizon,
        feature_names=feature_names,
    )
    predictions, metrics = expanding_window_predictions(
        dataset,
        model_name=model_name,
        train_size=train_size,
        test_size=test_size,
        random_state=random_state,
        model_params=model_params,
    )
    positions = predictions_to_signal(
        predictions,
        index=data.close.index,
        columns=data.close.columns,
        gross_exposure=gross_exposure,
    )
    prediction_panel = pd.DataFrame(0.0, index=data.close.index, columns=data.close.columns)
    if not predictions.empty:
        prediction_panel = predictions.unstack("symbol").reindex(
            index=data.close.index,
            columns=data.close.columns,
        )
    result = run_backtest(
        data=data,
        target_positions=positions,
        initial_capital=initial_capital,
        fee_bps=fee_bps,
        slippage_bps=slippage_bps,
    )
    return MLBacktestResult(
        predictions=prediction_panel,
        positions=positions,
        metrics=metrics,
        backtest=result,
    )
