"""OHLCV market data loading utilities."""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {"timestamp", "open", "high", "low", "close", "volume"}
OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass(frozen=True)
class MarketData:
    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)


def _read_symbol_csv(path: Path, symbol: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"OHLCV file not found for {symbol}: {path}")

    frame = pd.read_csv(path)
    missing_columns = REQUIRED_COLUMNS.difference(frame.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"OHLCV file for {symbol} missing columns: {missing}")

    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    if frame["timestamp"].isna().any():
        raise ValueError(f"OHLCV file for {symbol} contains invalid timestamps")

    frame["timestamp"] = frame["timestamp"].dt.tz_convert(None)
    if frame["timestamp"].duplicated().any():
        duplicates = frame.loc[frame["timestamp"].duplicated(keep=False), "timestamp"]
        duplicate_values = ", ".join(
            timestamp.isoformat() for timestamp in duplicates.drop_duplicates()
        )
        raise ValueError(
            f"OHLCV file for {symbol} contains duplicate timestamps: {duplicate_values}"
        )

    frame = frame.sort_values("timestamp")
    frame = frame.set_index("timestamp")

    numeric = frame.loc[:, OHLCV_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        raise ValueError(f"OHLCV file for {symbol} contains non-numeric or NaN values")

    return numeric


def _validate_symbols(symbols: list[str]) -> None:
    if not symbols:
        raise ValueError("symbols must include at least one symbol")

    if len(set(symbols)) != len(symbols):
        raise ValueError("symbols must not contain duplicates")


def _format_missing_bars(frame: pd.DataFrame) -> str:
    missing = frame.isna()
    details = []
    for symbol in frame.columns[missing.any(axis=0)]:
        timestamps = [
            timestamp.isoformat()
            for timestamp in frame.index[missing[symbol]]
        ]
        details.append(f"{symbol}: {', '.join(timestamps)}")
    return "; ".join(details)


def load_ohlcv_directory(directory: Path | str, symbols: list[str]) -> MarketData:
    _validate_symbols(symbols)

    directory_path = Path(directory)
    by_symbol = {
        symbol: _read_symbol_csv(directory_path / f"{symbol}.csv", symbol)
        for symbol in symbols
    }

    fields = {
        field: pd.concat(
            [by_symbol[symbol][field].rename(symbol) for symbol in symbols],
            axis=1,
        ).sort_index()
        for field in OHLCV_COLUMNS
    }

    for field, frame in fields.items():
        if frame.isna().any().any():
            missing_bars = _format_missing_bars(frame)
            raise ValueError(
                f"OHLCV data has missing bars in {field} for {missing_bars}"
            )

    return MarketData(
        open=fields["open"],
        high=fields["high"],
        low=fields["low"],
        close=fields["close"],
        volume=fields["volume"],
    )
