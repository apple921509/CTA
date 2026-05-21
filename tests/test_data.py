from pathlib import Path

import pandas as pd
import pytest

from cta_research.data import MarketData, load_ohlcv_directory


def test_load_ohlcv_directory_aligns_symbols() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])

    assert isinstance(data, MarketData)
    assert list(data.close.columns) == ["BTCUSDT", "ETHUSDT"]
    assert data.close.index[0] == pd.Timestamp("2024-01-01")
    assert data.close.loc[pd.Timestamp("2024-01-10"), "ETHUSDT"] == 64


def test_load_ohlcv_directory_rejects_missing_required_columns(tmp_path: Path) -> None:
    bad = tmp_path / "BTCUSDT.csv"
    bad.write_text("timestamp,open,high,low,close\n2024-01-01,1,2,0,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing columns"):
        load_ohlcv_directory(tmp_path, ["BTCUSDT"])


def test_load_ohlcv_directory_normalizes_offset_timestamps_to_utc_naive(tmp_path: Path) -> None:
    csv = tmp_path / "BTCUSDT.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T08:00:00+08:00,1,2,0,1,100\n",
        encoding="utf-8",
    )

    data = load_ohlcv_directory(tmp_path, ["BTCUSDT"])

    assert data.close.index[0] == pd.Timestamp("2024-01-01T00:00:00")
    assert data.close.index.tz is None


def test_load_ohlcv_directory_rejects_duplicate_timestamps_after_normalization(tmp_path: Path) -> None:
    csv = tmp_path / "BTCUSDT.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01T00:00:00Z,1,2,0,1,100\n"
        "2024-01-01T08:00:00+08:00,2,3,1,2,200\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BTCUSDT.*duplicate"):
        load_ohlcv_directory(tmp_path, ["BTCUSDT"])


def test_load_ohlcv_directory_reports_missing_aligned_symbol_and_timestamp(tmp_path: Path) -> None:
    (tmp_path / "BTCUSDT.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01,1,2,0,1,100\n"
        "2024-01-02,2,3,1,2,200\n",
        encoding="utf-8",
    )
    (tmp_path / "ETHUSDT.csv").write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01,10,20,0,10,1000\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing bars.*ETHUSDT.*2024-01-02"):
        load_ohlcv_directory(tmp_path, ["BTCUSDT", "ETHUSDT"])


def test_load_ohlcv_directory_rejects_non_numeric_ohlcv_values(tmp_path: Path) -> None:
    csv = tmp_path / "BTCUSDT.csv"
    csv.write_text(
        "timestamp,open,high,low,close,volume\n"
        "2024-01-01,1,not-a-number,0,1,100\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="BTCUSDT.*non-numeric"):
        load_ohlcv_directory(tmp_path, ["BTCUSDT"])


def test_load_ohlcv_directory_rejects_empty_symbols_list() -> None:
    with pytest.raises(ValueError, match="symbols"):
        load_ohlcv_directory(Path("tests/fixtures/ohlcv"), [])


def test_load_ohlcv_directory_rejects_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="symbols"):
        load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "BTCUSDT"])
