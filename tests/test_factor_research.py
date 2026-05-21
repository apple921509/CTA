import numpy as np
import pandas as pd

from cta_research.factor_research import (
    factor_correlation_matrix,
    factor_ic_decay,
    factor_ic_table,
    orthogonalize_factor,
    preprocess_factor,
    quantile_forward_returns,
    quantile_return_summary,
    rank_cross_section,
    winsorize_cross_section,
    zscore_cross_section,
)


def _sample_factor_and_close() -> tuple[pd.DataFrame, pd.DataFrame]:
    index = pd.date_range("2024-01-01", periods=5)
    columns = ["A", "B", "C", "D"]
    factor = pd.DataFrame(
        [
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
            [4.0, 3.0, 2.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        index=index,
        columns=columns,
    )
    close = pd.DataFrame(
        [
            [100.0, 100.0, 100.0, 100.0],
            [101.0, 102.0, 103.0, 104.0],
            [105.0, 104.0, 103.0, 102.0],
            [106.0, 105.0, 104.0, 103.0],
            [107.0, 106.0, 105.0, 104.0],
        ],
        index=index,
        columns=columns,
    )
    return factor, close


def test_winsorize_cross_section_clips_row_extremes() -> None:
    factor = pd.DataFrame([[1.0, 2.0, 100.0]], columns=["A", "B", "C"])

    result = winsorize_cross_section(factor, lower_quantile=0.0, upper_quantile=0.5)

    assert result.loc[0, "C"] == 2.0


def test_zscore_cross_section_centers_each_row() -> None:
    factor = pd.DataFrame([[1.0, 2.0, 3.0], [5.0, 5.0, 5.0]], columns=["A", "B", "C"])

    result = zscore_cross_section(factor)

    assert round(result.iloc[0].mean(), 12) == 0.0
    assert np.isnan(result.iloc[1]).all()


def test_rank_cross_section_can_center_percentile_ranks() -> None:
    factor = pd.DataFrame([[10.0, 20.0, 30.0]], columns=["A", "B", "C"])

    result = rank_cross_section(factor)

    assert result.loc[0, "A"] < 0
    assert result.loc[0, "C"] > 0
    assert round(result.iloc[0].sum(), 12) == 0.0


def test_preprocess_factor_replaces_infinite_values_and_ranks() -> None:
    factor = pd.DataFrame([[1.0, np.inf, 3.0]], columns=["A", "B", "C"])

    result = preprocess_factor(factor, winsorize=False, method="rank")

    assert pd.isna(result.loc[0, "B"])
    assert result.loc[0, "A"] < result.loc[0, "C"]


def test_factor_ic_table_and_decay_report_horizons() -> None:
    factor, close = _sample_factor_and_close()

    table = factor_ic_table(factor, close, horizons=[1, 2])
    decay = factor_ic_decay(factor, close, horizons=[1, 2])

    assert set(table["horizon"]) == {1, 2}
    assert set(decay["horizon"]) == {1, 2}
    assert {"ic_mean", "information_ratio", "positive_rate"}.issubset(decay.columns)


def test_factor_correlation_matrix_aligns_stacked_factors() -> None:
    factor, _ = _sample_factor_and_close()
    factors = {"raw": factor, "double": factor * 2.0}

    corr = factor_correlation_matrix(factors)

    assert corr.loc["raw", "double"] == 1.0


def test_orthogonalize_factor_removes_linear_control_exposure() -> None:
    index = pd.date_range("2024-01-01", periods=2)
    columns = ["A", "B", "C", "D"]
    control = pd.DataFrame([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]], index=index, columns=columns)
    target = control * 2.0 + 1.0

    residual = orthogonalize_factor(target, {"control": control})

    assert residual.abs().max().max() < 1e-12


def test_quantile_forward_returns_and_summary_bucket_assets() -> None:
    factor, close = _sample_factor_and_close()

    returns = quantile_forward_returns(factor, close, horizon=1, quantiles=2)
    summary = quantile_return_summary(factor, close, horizon=1, quantiles=2)

    assert set(returns["quantile"].unique()) == {1, 2}
    assert {"mean_return", "std_return", "observation_count"}.issubset(summary.columns)
