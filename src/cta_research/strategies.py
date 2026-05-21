from __future__ import annotations

import numpy as np
import pandas as pd


def _clip_signal(signal: pd.DataFrame) -> pd.DataFrame:
    return signal.replace([np.inf, -np.inf], np.nan).clip(-1.0, 1.0)


def _directional_score(frame: pd.DataFrame) -> pd.DataFrame:
    if len(frame.columns) == 1:
        return np.sign(frame)

    rank = frame.rank(axis=1, pct=True)
    centered = (rank - rank.mean(axis=1).to_frame().to_numpy()) * 2.0
    return centered


def trend_signal(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    raw = (
        0.45 * _directional_score(factors["momentum"])
        + 0.35 * _directional_score(factors["ma_slope"])
        + 0.20 * factors["donchian"]
    )
    return _clip_signal(raw)


def mean_reversion_signal(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rsi_component = -((factors["rsi"] - 50.0) / 50.0)
    bollinger_component = -factors["bollinger_zscore"] / 2.0
    raw = 0.55 * rsi_component + 0.45 * bollinger_component
    trend_filter = 1.0 - factors["donchian"].abs().clip(0.0, 0.7)
    return _clip_signal(raw * trend_filter)


def swing_signal(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    trend_direction = np.sign(factors["ma_slope"])
    pullback = -factors["bollinger_zscore"].clip(-2.0, 2.0) / 2.0
    raw = trend_direction * pullback.where(pullback.abs() > 0.25, 0.0)
    return _clip_signal(raw)


def alternative_signal(factors: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if "alternative" not in factors:
        template = next(iter(factors.values()))
        return pd.DataFrame(0.0, index=template.index, columns=template.columns)
    return _clip_signal(factors["alternative"])


def calculate_strategy_signals(factors: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    return {
        "trend": trend_signal(factors),
        "mean_reversion": mean_reversion_signal(factors),
        "swing": swing_signal(factors),
        "alternative": alternative_signal(factors),
    }
