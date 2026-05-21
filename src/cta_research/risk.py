from __future__ import annotations

import pandas as pd


def cap_exposure(
    positions: pd.DataFrame,
    max_symbol_weight: float,
    max_gross_exposure: float,
) -> pd.DataFrame:
    capped = positions.clip(lower=-max_symbol_weight, upper=max_symbol_weight)
    gross_exposure = capped.abs().sum(axis=1)
    scale = (max_gross_exposure / gross_exposure.replace(0, pd.NA)).clip(upper=1.0)
    return capped.mul(scale.fillna(1.0), axis=0)


def drawdown_series(equity: pd.Series) -> pd.Series:
    return equity / equity.cummax() - 1.0


def apply_drawdown_derisking(
    positions: pd.DataFrame,
    equity: pd.Series,
    drawdown_warning: float,
    drawdown_hard: float,
    min_exposure_multiplier: float,
) -> pd.DataFrame:
    drawdown = drawdown_series(equity).abs()
    multiplier = pd.Series(1.0, index=positions.index)

    in_reduction_band = (drawdown > drawdown_warning) & (drawdown < drawdown_hard)
    multiplier.loc[in_reduction_band] = 1.0 - (
        (drawdown.loc[in_reduction_band] - drawdown_warning)
        / (drawdown_hard - drawdown_warning)
    ) * (1.0 - min_exposure_multiplier)
    multiplier.loc[drawdown >= drawdown_hard] = min_exposure_multiplier

    return positions.mul(multiplier, axis=0)
