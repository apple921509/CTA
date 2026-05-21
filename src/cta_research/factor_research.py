from __future__ import annotations

import numpy as np
import pandas as pd


def winsorize_cross_section(
    factor: pd.DataFrame,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
) -> pd.DataFrame:
    lower = factor.quantile(lower_quantile, axis=1)
    upper = factor.quantile(upper_quantile, axis=1)
    return factor.clip(lower=lower, upper=upper, axis=0)


def zscore_cross_section(factor: pd.DataFrame) -> pd.DataFrame:
    mean = factor.mean(axis=1)
    std = factor.std(axis=1, ddof=0)
    return factor.sub(mean, axis=0).divide(std.mask(std == 0), axis=0)


def rank_cross_section(factor: pd.DataFrame, center: bool = True) -> pd.DataFrame:
    ranked = factor.rank(axis=1, pct=True)
    if not center:
        return ranked
    return ranked.sub(ranked.mean(axis=1), axis=0)


def preprocess_factor(
    factor: pd.DataFrame,
    winsorize: bool = True,
    lower_quantile: float = 0.05,
    upper_quantile: float = 0.95,
    method: str = "zscore",
) -> pd.DataFrame:
    cleaned = factor.replace([np.inf, -np.inf], np.nan)
    if winsorize:
        cleaned = winsorize_cross_section(cleaned, lower_quantile, upper_quantile)
    if method == "zscore":
        return zscore_cross_section(cleaned)
    if method == "rank":
        return rank_cross_section(cleaned)
    if method == "raw":
        return cleaned
    raise ValueError("method must be one of: zscore, rank, raw")


def preprocess_factor_set(
    factors: dict[str, pd.DataFrame],
    method: str = "zscore",
    winsorize: bool = True,
) -> dict[str, pd.DataFrame]:
    return {
        name: preprocess_factor(frame, method=method, winsorize=winsorize)
        for name, frame in factors.items()
    }


def factor_ic_table(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: list[int],
    method: str = "spearman",
) -> pd.DataFrame:
    records = []
    for horizon in horizons:
        future_returns = close.pct_change(horizon).shift(-horizon)
        aligned_factor, aligned_returns = factor.align(future_returns, join="inner")
        for timestamp in aligned_factor.index:
            pair = pd.concat(
                [aligned_factor.loc[timestamp], aligned_returns.loc[timestamp]],
                axis=1,
                keys=["factor", "future_return"],
            ).dropna()
            if len(pair) < 2:
                continue
            if pair["factor"].nunique() < 2 or pair["future_return"].nunique() < 2:
                continue
            records.append(
                {
                    "timestamp": timestamp,
                    "horizon": horizon,
                    "ic": pair["factor"].corr(pair["future_return"], method=method),
                }
            )
    return pd.DataFrame.from_records(records, columns=["timestamp", "horizon", "ic"])


def factor_ic_decay(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizons: list[int],
    method: str = "spearman",
) -> pd.DataFrame:
    table = factor_ic_table(factor, close, horizons, method=method)
    if table.empty:
        return pd.DataFrame(
            columns=[
                "horizon",
                "ic_mean",
                "ic_std",
                "information_ratio",
                "positive_rate",
                "observation_count",
            ]
        )
    grouped = table.groupby("horizon")["ic"]
    result = grouped.agg(ic_mean="mean", ic_std="std", observation_count="count")
    result["information_ratio"] = result["ic_mean"] / result["ic_std"].replace(0, np.nan)
    result["positive_rate"] = grouped.apply(lambda values: float((values > 0).mean()))
    return result.reset_index().loc[
        :,
        [
            "horizon",
            "ic_mean",
            "ic_std",
            "information_ratio",
            "positive_rate",
            "observation_count",
        ],
    ]


def factor_correlation_matrix(
    factors: dict[str, pd.DataFrame],
    method: str = "spearman",
) -> pd.DataFrame:
    stacked = {
        name: frame.stack(future_stack=True)
        for name, frame in factors.items()
    }
    aligned = pd.DataFrame(stacked).dropna(how="all")
    return aligned.corr(method=method)


def orthogonalize_factor(
    target: pd.DataFrame,
    controls: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if not controls:
        return target.copy()

    records: list[pd.Series] = []
    for timestamp in target.index:
        y = target.loc[timestamp]
        x = pd.DataFrame({name: frame.loc[timestamp] for name, frame in controls.items()})
        sample = pd.concat([y.rename("target"), x], axis=1).dropna()
        residual = pd.Series(np.nan, index=target.columns, name=timestamp)
        if len(sample) <= len(controls):
            records.append(residual)
            continue
        design = np.column_stack([np.ones(len(sample)), sample[list(controls)].to_numpy()])
        beta, *_ = np.linalg.lstsq(design, sample["target"].to_numpy(), rcond=None)
        fitted = design @ beta
        residual.loc[sample.index] = sample["target"].to_numpy() - fitted
        records.append(residual)

    return pd.DataFrame(records, index=target.index, columns=target.columns)


def quantile_forward_returns(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 1,
    quantiles: int = 5,
) -> pd.DataFrame:
    future_returns = close.pct_change(horizon).shift(-horizon)
    aligned_factor, aligned_returns = factor.align(future_returns, join="inner")
    records = []
    for timestamp in aligned_factor.index:
        pair = pd.concat(
            [aligned_factor.loc[timestamp], aligned_returns.loc[timestamp]],
            axis=1,
            keys=["factor", "forward_return"],
        ).dropna()
        if len(pair) < quantiles:
            continue
        try:
            pair["quantile"] = pd.qcut(
                pair["factor"],
                q=quantiles,
                labels=False,
                duplicates="drop",
            ) + 1
        except ValueError:
            continue
        for quantile, group in pair.groupby("quantile"):
            records.append(
                {
                    "timestamp": timestamp,
                    "horizon": horizon,
                    "quantile": int(quantile),
                    "mean_forward_return": float(group["forward_return"].mean()),
                    "count": int(len(group)),
                }
            )

    return pd.DataFrame.from_records(
        records,
        columns=["timestamp", "horizon", "quantile", "mean_forward_return", "count"],
    )


def quantile_return_summary(
    factor: pd.DataFrame,
    close: pd.DataFrame,
    horizon: int = 1,
    quantiles: int = 5,
) -> pd.DataFrame:
    returns = quantile_forward_returns(factor, close, horizon=horizon, quantiles=quantiles)
    if returns.empty:
        return pd.DataFrame(
            columns=["horizon", "quantile", "mean_return", "std_return", "observation_count"]
        )
    return (
        returns.groupby(["horizon", "quantile"])["mean_forward_return"]
        .agg(mean_return="mean", std_return="std", observation_count="count")
        .reset_index()
    )
