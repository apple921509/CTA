from __future__ import annotations

import pandas as pd

from cta_research.data import MarketData


def _zero_like(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=frame.index, columns=frame.columns)


def blend_strategy_signals(
    signals: dict[str, pd.DataFrame],
    weights: dict[str, float],
) -> pd.DataFrame:
    active = {
        name: frame
        for name, frame in signals.items()
        if name in weights and frame.fillna(0.0).abs().sum().sum() > 0
    }
    if not active:
        template = next(iter(signals.values()))
        return _zero_like(template)

    active_frames = list(active.values())
    union_index = active_frames[0].index
    union_columns = active_frames[0].columns
    for frame in active_frames[1:]:
        union_index = union_index.union(frame.index)
        union_columns = union_columns.union(frame.columns)

    total_weight = sum(weights[name] for name in active)
    blended = pd.DataFrame(0.0, index=union_index, columns=union_columns)
    for name, frame in active.items():
        aligned = frame.reindex(index=union_index, columns=union_columns).fillna(0.0)
        blended = blended + aligned * (weights[name] / total_weight)
    return blended.clip(-1.0, 1.0)


def size_by_volatility(
    signal: pd.DataFrame,
    volatility: pd.DataFrame,
    volatility_target: float,
) -> pd.DataFrame:
    safe_volatility = volatility.mask(volatility == 0).ffill().bfill()
    scalar = (volatility_target / safe_volatility).clip(upper=1.0)
    return (signal.fillna(0.0) * scalar).fillna(0.0)


def strategy_return_attribution(
    data: MarketData,
    strategy_signals: dict[str, pd.DataFrame],
    weights: dict[str, float],
    volatility: pd.DataFrame,
    volatility_target: float,
    final_positions: pd.DataFrame,
) -> pd.DataFrame:
    active = {
        name: frame
        for name, frame in strategy_signals.items()
        if name in weights and frame.fillna(0.0).abs().sum().sum() > 0
    }
    if not active:
        return pd.DataFrame(index=data.close.index)

    total_weight = sum(weights[name] for name in active)
    weighted_positions = {}
    for name, signal in active.items():
        weighted_signal = signal * (weights[name] / total_weight)
        weighted_positions[name] = size_by_volatility(
            weighted_signal,
            volatility,
            volatility_target,
        ).reindex(index=final_positions.index, columns=final_positions.columns).fillna(0.0)

    pre_cap_sum = sum(weighted_positions.values())
    scaling = final_positions.divide(pre_cap_sum.mask(pre_cap_sum == 0))
    scaling = scaling.replace([float("inf"), float("-inf")], pd.NA).fillna(0.0)

    asset_returns = data.close.pct_change().fillna(0.0)
    records = {}
    for name, positions in weighted_positions.items():
        attributed_positions = positions * scaling
        records[name] = (attributed_positions.shift(1).fillna(0.0) * asset_returns).sum(axis=1)

    return pd.DataFrame(records, index=data.close.index)
