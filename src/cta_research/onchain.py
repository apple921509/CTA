from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cta_research.factor_research import zscore_cross_section


FEATURE_COLUMNS = {
    "sopr",
    "nupl",
    "mvrv",
    "exchange_inflow",
    "exchange_outflow",
    "exchange_netflow",
    "whale_transaction_count",
    "active_addresses",
}


@dataclass(frozen=True)
class OnchainData:
    sopr: pd.DataFrame | None = None
    nupl: pd.DataFrame | None = None
    mvrv: pd.DataFrame | None = None
    exchange_inflow: pd.DataFrame | None = None
    exchange_outflow: pd.DataFrame | None = None
    exchange_netflow: pd.DataFrame | None = None
    whale_transaction_count: pd.DataFrame | None = None
    active_addresses: pd.DataFrame | None = None

    @property
    def symbols(self) -> list[str]:
        for frame in self.frames().values():
            if frame is not None:
                return list(frame.columns)
        return []

    def frames(self) -> dict[str, pd.DataFrame | None]:
        return {
            "sopr": self.sopr,
            "nupl": self.nupl,
            "mvrv": self.mvrv,
            "exchange_inflow": self.exchange_inflow,
            "exchange_outflow": self.exchange_outflow,
            "exchange_netflow": self.exchange_netflow,
            "whale_transaction_count": self.whale_transaction_count,
            "active_addresses": self.active_addresses,
        }


def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
    return numerator.divide(denominator.mask(denominator == 0))


def _rolling_zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    minimum_periods = max(3, window // 3)
    mean = frame.rolling(window, min_periods=minimum_periods).mean()
    std = frame.rolling(window, min_periods=minimum_periods).std(ddof=0)
    return frame.sub(mean).divide(std.mask(std == 0))


def _normalize_timestamp(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce", utc=True)
    if normalized["timestamp"].isna().any():
        raise ValueError("On-chain feature data contains invalid timestamps")
    normalized["timestamp"] = normalized["timestamp"].dt.tz_convert(None)
    return normalized


def load_onchain_feature_directory(
    directory: Path | str,
    symbols: list[str],
) -> OnchainData:
    directory_path = Path(directory)
    frames: dict[str, pd.DataFrame] = {}
    for feature in FEATURE_COLUMNS:
        by_symbol = []
        for symbol in symbols:
            path = directory_path / feature / f"{symbol}.csv"
            if not path.exists():
                continue
            raw = _normalize_timestamp(pd.read_csv(path))
            if "value" not in raw.columns:
                raise ValueError(f"On-chain feature file missing value column: {path}")
            values = pd.to_numeric(raw["value"], errors="coerce")
            if values.isna().any():
                raise ValueError(f"On-chain feature file contains non-numeric values: {path}")
            by_symbol.append(pd.Series(values.to_numpy(), index=raw["timestamp"], name=symbol))
        if by_symbol:
            frames[feature] = pd.concat(by_symbol, axis=1).sort_index()

    return OnchainData(**frames)


def sopr_factor(sopr: pd.DataFrame, neutral_level: float = 1.0) -> pd.DataFrame:
    return (sopr - neutral_level).clip(-1.0, 1.0)


def nupl_factor(nupl: pd.DataFrame) -> pd.DataFrame:
    return nupl.clip(-1.0, 1.0)


def mvrv_factor(mvrv: pd.DataFrame, fair_value_level: float = 1.0) -> pd.DataFrame:
    return np.log(_safe_divide(mvrv, pd.DataFrame(fair_value_level, index=mvrv.index, columns=mvrv.columns)))


def exchange_netflow_factor(
    netflow: pd.DataFrame | None = None,
    inflow: pd.DataFrame | None = None,
    outflow: pd.DataFrame | None = None,
    window: int = 30,
) -> pd.DataFrame:
    if netflow is None:
        if inflow is None or outflow is None:
            raise ValueError("exchange_netflow_factor requires netflow or inflow and outflow")
        netflow = inflow - outflow
    return -_rolling_zscore(netflow, window=window)


def whale_activity_factor(
    whale_transaction_count: pd.DataFrame,
    window: int = 30,
) -> pd.DataFrame:
    return _rolling_zscore(whale_transaction_count, window=window)


def active_address_factor(active_addresses: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    return _rolling_zscore(active_addresses.pct_change(), window=window)


def align_onchain_factors(
    factors: dict[str, pd.DataFrame],
    index: pd.Index,
    columns: pd.Index | list[str],
) -> dict[str, pd.DataFrame]:
    return {
        name: frame.reindex(index=index, columns=columns).ffill()
        for name, frame in factors.items()
    }


def composite_onchain_factor(
    factors: dict[str, pd.DataFrame],
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    active = {
        name: frame
        for name, frame in factors.items()
        if frame.fillna(0.0).abs().sum().sum() > 0
    }
    if not active:
        raise ValueError("At least one non-empty on-chain factor is required")

    if weights is None:
        weights = {name: 1.0 for name in active}

    total_weight = sum(weights.get(name, 0.0) for name in active)
    if total_weight == 0:
        raise ValueError("At least one active on-chain factor must have positive weight")

    template = next(iter(active.values()))
    composite = pd.DataFrame(0.0, index=template.index, columns=template.columns)
    for name, frame in active.items():
        normalized = zscore_cross_section(frame).fillna(0.0)
        composite = composite + normalized * (weights.get(name, 0.0) / total_weight)
    return composite.clip(-3.0, 3.0)


def calculate_onchain_factor_set(
    data: OnchainData,
    index: pd.Index | None = None,
    columns: pd.Index | list[str] | None = None,
    window: int = 30,
) -> dict[str, pd.DataFrame]:
    factors: dict[str, pd.DataFrame] = {}
    if data.sopr is not None:
        factors["sopr"] = sopr_factor(data.sopr)
    if data.nupl is not None:
        factors["nupl"] = nupl_factor(data.nupl)
    if data.mvrv is not None:
        factors["mvrv"] = mvrv_factor(data.mvrv)
    if data.exchange_netflow is not None or (
        data.exchange_inflow is not None and data.exchange_outflow is not None
    ):
        factors["exchange_netflow"] = exchange_netflow_factor(
            netflow=data.exchange_netflow,
            inflow=data.exchange_inflow,
            outflow=data.exchange_outflow,
            window=window,
        )
    if data.whale_transaction_count is not None:
        factors["whale_activity"] = whale_activity_factor(
            data.whale_transaction_count,
            window=window,
        )
    if data.active_addresses is not None:
        factors["active_addresses"] = active_address_factor(
            data.active_addresses,
            window=window,
        )

    if index is not None and columns is not None:
        factors = align_onchain_factors(factors, index=index, columns=columns)
    if factors:
        factors["alternative"] = composite_onchain_factor(factors)
    return factors
