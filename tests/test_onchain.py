from pathlib import Path

import pandas as pd

from cta_research.data import load_ohlcv_directory
from cta_research.factors import calculate_factor_set
from cta_research.onchain import (
    OnchainData,
    active_address_factor,
    calculate_onchain_factor_set,
    composite_onchain_factor,
    exchange_netflow_factor,
    load_onchain_feature_directory,
    mvrv_factor,
    nupl_factor,
    sopr_factor,
    whale_activity_factor,
)
from cta_research.strategies import alternative_signal


def _feature_frame() -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=8)
    return pd.DataFrame(
        {
            "BTCUSDT": [1.0, 1.1, 1.3, 1.6, 1.5, 1.4, 1.8, 2.0],
            "ETHUSDT": [2.0, 1.9, 1.7, 1.4, 1.5, 1.6, 1.2, 1.0],
        },
        index=index,
    )


def test_load_onchain_feature_directory_reads_value_files(tmp_path: Path) -> None:
    for symbol, value in {"BTCUSDT": 1.2, "ETHUSDT": 0.9}.items():
        path = tmp_path / "sopr" / f"{symbol}.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            f"timestamp,value\n2024-01-01T08:00:00+08:00,{value}\n",
            encoding="utf-8",
        )

    data = load_onchain_feature_directory(tmp_path, ["BTCUSDT", "ETHUSDT"])

    assert data.sopr is not None
    assert list(data.sopr.columns) == ["BTCUSDT", "ETHUSDT"]
    assert data.sopr.index[0] == pd.Timestamp("2024-01-01")


def test_core_onchain_factors_are_directional_and_bounded() -> None:
    frame = _feature_frame()

    assert sopr_factor(frame).loc["2024-01-01", "BTCUSDT"] == 0.0
    assert nupl_factor(frame - 1.5).max().max() <= 1.0
    assert mvrv_factor(frame).loc["2024-01-01", "BTCUSDT"] == 0.0


def test_flow_whale_and_address_factors_use_rolling_zscores() -> None:
    frame = _feature_frame()
    inflow = frame * 2.0
    outflow = frame

    netflow = exchange_netflow_factor(inflow=inflow, outflow=outflow, window=4)
    whales = whale_activity_factor(frame, window=4)
    addresses = active_address_factor(frame, window=4)

    assert netflow.iloc[:2].isna().all().all()
    assert whales.iloc[3:].notna().all().all()
    assert addresses.iloc[3:].notna().all().all()


def test_composite_onchain_factor_combines_cross_sectional_scores() -> None:
    frame = _feature_frame()
    factors = {
        "sopr": sopr_factor(frame),
        "nupl": nupl_factor(frame - 1.5),
    }

    composite = composite_onchain_factor(factors)

    assert list(composite.columns) == ["BTCUSDT", "ETHUSDT"]
    assert composite.abs().max().max() <= 3.0


def test_calculate_onchain_factor_set_returns_aligned_alternative_factor() -> None:
    frame = _feature_frame()
    target_index = pd.date_range("2024-01-01", periods=10)
    data = OnchainData(
        sopr=frame,
        nupl=frame - 1.5,
        exchange_netflow=frame,
        whale_transaction_count=frame,
        active_addresses=frame,
    )

    factors = calculate_onchain_factor_set(
        data,
        index=target_index,
        columns=["BTCUSDT", "ETHUSDT"],
        window=4,
    )

    assert "alternative" in factors
    assert factors["alternative"].shape == (10, 2)
    assert factors["alternative"].index[-1] == pd.Timestamp("2024-01-10")


def test_calculate_factor_set_with_onchain_data_feeds_alternative_signal() -> None:
    market_data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])
    frame = pd.DataFrame(
        {
            "BTCUSDT": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
            "ETHUSDT": [1.9, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2, 1.1, 1.0],
        },
        index=market_data.close.index,
    )

    factors = calculate_factor_set(market_data, onchain_data=OnchainData(sopr=frame))
    signal = alternative_signal(factors)

    assert "alternative" in factors
    assert signal.fillna(0.0).abs().sum().sum() > 0
