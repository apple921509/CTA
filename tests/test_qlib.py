from pathlib import Path

from cta_research.data import load_ohlcv_directory
from cta_research.qlib import (
    build_alpha360_like_features,
    market_data_to_qlib_frame,
    write_alpha360_like_csv,
    write_qlib_csv,
)


def test_market_data_to_qlib_frame_creates_long_table() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])

    frame = market_data_to_qlib_frame(data)

    assert len(frame) == len(data.close.index) * 2
    assert {"datetime", "instrument", "open", "high", "low", "close", "volume"}.issubset(frame.columns)
    assert set(frame["instrument"]) == {"BTCUSDT", "ETHUSDT"}


def test_build_alpha360_like_features_adds_rolling_fields() -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])

    frame = build_alpha360_like_features(data, windows=[2, 3])

    assert {"return_1", "intraday_return", "high_low_range"}.issubset(frame.columns)
    assert {"return_2", "ma_ratio_3", "volatility_3", "volume_ratio_2"}.issubset(frame.columns)


def test_write_qlib_csv_outputs_expected_files(tmp_path: Path) -> None:
    data = load_ohlcv_directory(Path("tests/fixtures/ohlcv"), ["BTCUSDT", "ETHUSDT"])

    qlib_path = write_qlib_csv(data, tmp_path / "qlib" / "crypto.csv")
    alpha_path = write_alpha360_like_csv(data, tmp_path / "qlib" / "alpha360.csv", windows=[2])

    assert qlib_path.exists()
    assert alpha_path.exists()
    assert qlib_path.read_text(encoding="utf-8").startswith("datetime,instrument")
