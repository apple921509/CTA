from __future__ import annotations

import pandas as pd


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

    union_index = pd.Index([])
    union_columns = pd.Index([])
    for frame in active.values():
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
    safe_volatility = volatility.replace(0, pd.NA).ffill().bfill()
    scalar = (volatility_target / safe_volatility).clip(upper=1.0)
    return (signal.fillna(0.0) * scalar).fillna(0.0)
