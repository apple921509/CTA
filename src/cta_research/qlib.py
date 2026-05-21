from __future__ import annotations

from pathlib import Path

import pandas as pd

from cta_research.data import MarketData


QLIB_FIELD_COLUMNS = ["datetime", "instrument", "open", "high", "low", "close", "volume"]


def market_data_to_qlib_frame(data: MarketData) -> pd.DataFrame:
    records = []
    for symbol in data.symbols:
        frame = pd.DataFrame(
            {
                "datetime": data.close.index,
                "instrument": symbol,
                "open": data.open[symbol].to_numpy(),
                "high": data.high[symbol].to_numpy(),
                "low": data.low[symbol].to_numpy(),
                "close": data.close[symbol].to_numpy(),
                "volume": data.volume[symbol].to_numpy(),
            }
        )
        records.append(frame)
    if not records:
        return pd.DataFrame(columns=QLIB_FIELD_COLUMNS)
    return pd.concat(records, ignore_index=True).sort_values(
        ["instrument", "datetime"]
    ).reset_index(drop=True)


def write_qlib_csv(data: MarketData, output_path: Path | str) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    market_data_to_qlib_frame(data).to_csv(path, index=False)
    return path


def build_alpha360_like_features(
    data: MarketData,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    if windows is None:
        windows = [5, 10, 20, 30, 60]

    base = market_data_to_qlib_frame(data)
    if base.empty:
        return base

    frames = []
    for symbol, group in base.groupby("instrument", sort=False):
        group = group.sort_values("datetime").copy()
        close = group["close"]
        volume = group["volume"]
        group["return_1"] = close.pct_change()
        group["intraday_return"] = group["close"] / group["open"] - 1.0
        group["high_low_range"] = group["high"] / group["low"] - 1.0
        for window in windows:
            group[f"return_{window}"] = close.pct_change(window)
            group[f"ma_ratio_{window}"] = close / close.rolling(window).mean() - 1.0
            group[f"volatility_{window}"] = close.pct_change().rolling(window).std()
            group[f"volume_ratio_{window}"] = volume / volume.rolling(window).mean() - 1.0
        frames.append(group)

    return pd.concat(frames, ignore_index=True).sort_values(
        ["instrument", "datetime"]
    ).reset_index(drop=True)


def write_alpha360_like_csv(
    data: MarketData,
    output_path: Path | str,
    windows: list[int] | None = None,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    build_alpha360_like_features(data, windows=windows).to_csv(path, index=False)
    return path
