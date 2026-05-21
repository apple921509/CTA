from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


JsonPayload = list[Any] | dict[str, Any]
HttpGet = Callable[[str, dict[str, str | int | float | None]], JsonPayload]


BINANCE_USDM_BASE_URL = "https://fapi.binance.com"
BINANCE_FUTURES_DATA_BASE_URL = "https://fapi.binance.com"
OKX_BASE_URL = "https://www.okx.com"

BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trade_count",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

OKX_CANDLE_COLUMNS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "volume_ccy",
    "volume_quote",
    "confirm",
]

OKX_TIMEFRAME_MAP = {
    "1m": "1m",
    "3m": "3m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "12h": "12H",
    "1d": "1Dutc",
}


@dataclass(frozen=True)
class DownloadWindow:
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None


def _timestamp_ms(value: pd.Timestamp | str | int | float | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return int(timestamp.timestamp() * 1000)


def _utc_naive_from_ms(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype("int64"), unit="ms", utc=True).dt.tz_convert(None)


def _default_http_get(
    base_url: str,
) -> HttpGet:
    def get(path: str, params: dict[str, str | int | float | None]) -> JsonPayload:
        filtered = {key: value for key, value in params.items() if value is not None}
        query = urlencode(filtered)
        url = f"{base_url}{path}"
        if query:
            url = f"{url}?{query}"
        request = Request(url, headers={"User-Agent": "cta-research/0.1"})
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))

    return get


def _normalize_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    normalized = frame.copy()
    normalized[columns] = normalized[columns].apply(pd.to_numeric, errors="coerce")
    if normalized[columns].isna().any().any():
        raise ValueError(f"Downloaded data contains non-numeric values in {columns}")
    return normalized


def _finalize_time_series(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.drop_duplicates(subset=["timestamp"])
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def write_ohlcv_cache(frame: pd.DataFrame, directory: Path | str, symbol: str) -> Path:
    path = Path(directory) / f"{symbol}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]].to_csv(
        path,
        index=False,
    )
    return path


def write_feature_cache(
    frame: pd.DataFrame,
    directory: Path | str,
    exchange: str,
    dataset: str,
    symbol: str,
) -> Path:
    path = Path(directory) / exchange / dataset / f"{symbol}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


class BinanceUsdmClient:
    def __init__(self, http_get: HttpGet | None = None) -> None:
        self._http_get = http_get or _default_http_get(BINANCE_USDM_BASE_URL)

    def fetch_ohlcv(
        self,
        symbol: str,
        interval: str,
        window: DownloadWindow = DownloadWindow(),
        limit: int = 1500,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": _timestamp_ms(window.start),
            "endTime": _timestamp_ms(window.end),
            "limit": limit,
        }
        payload = self._http_get("/fapi/v1/klines", params)
        frame = pd.DataFrame(payload, columns=BINANCE_KLINE_COLUMNS)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = _utc_naive_from_ms(frame["open_time"])
        frame = _normalize_numeric(frame, ["open", "high", "low", "close", "volume"])
        return _finalize_time_series(frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]])

    def fetch_funding_rates(
        self,
        symbol: str,
        window: DownloadWindow = DownloadWindow(),
        limit: int = 1000,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "startTime": _timestamp_ms(window.start),
            "endTime": _timestamp_ms(window.end),
            "limit": limit,
        }
        payload = self._http_get("/fapi/v1/fundingRate", params)
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "symbol", "funding_rate", "mark_price"])
        frame["timestamp"] = _utc_naive_from_ms(frame["fundingTime"])
        frame = frame.rename(columns={"fundingRate": "funding_rate", "markPrice": "mark_price"})
        frame = _normalize_numeric(frame, ["funding_rate", "mark_price"])
        return _finalize_time_series(frame.loc[:, ["timestamp", "symbol", "funding_rate", "mark_price"]])

    def fetch_open_interest(
        self,
        symbol: str,
        period: str,
        window: DownloadWindow = DownloadWindow(),
        limit: int = 500,
    ) -> pd.DataFrame:
        params = {
            "symbol": symbol,
            "period": period,
            "startTime": _timestamp_ms(window.start),
            "endTime": _timestamp_ms(window.end),
            "limit": limit,
        }
        payload = self._http_get("/futures/data/openInterestHist", params)
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "open_interest", "open_interest_value"]
            )
        frame["timestamp"] = _utc_naive_from_ms(frame["timestamp"])
        frame = frame.rename(
            columns={
                "sumOpenInterest": "open_interest",
                "sumOpenInterestValue": "open_interest_value",
            }
        )
        frame = _normalize_numeric(frame, ["open_interest", "open_interest_value"])
        return _finalize_time_series(
            frame.loc[:, ["timestamp", "symbol", "open_interest", "open_interest_value"]]
        )


class OkxClient:
    def __init__(self, http_get: HttpGet | None = None) -> None:
        self._http_get = http_get or _default_http_get(OKX_BASE_URL)

    def fetch_ohlcv(
        self,
        inst_id: str,
        timeframe: str,
        window: DownloadWindow = DownloadWindow(),
        limit: int = 300,
        history: bool = True,
    ) -> pd.DataFrame:
        path = "/api/v5/market/history-candles" if history else "/api/v5/market/candles"
        params = {
            "instId": inst_id,
            "bar": OKX_TIMEFRAME_MAP.get(timeframe, timeframe),
            "before": _timestamp_ms(window.start),
            "after": _timestamp_ms(window.end),
            "limit": limit,
        }
        payload = self._http_get(path, params)
        data = payload.get("data", []) if isinstance(payload, dict) else payload
        frame = pd.DataFrame(data, columns=OKX_CANDLE_COLUMNS)
        if frame.empty:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        frame["timestamp"] = _utc_naive_from_ms(frame["timestamp"])
        frame = _normalize_numeric(frame, ["open", "high", "low", "close", "volume"])
        return _finalize_time_series(frame.loc[:, ["timestamp", "open", "high", "low", "close", "volume"]])

    def fetch_funding_rates(
        self,
        inst_id: str,
        window: DownloadWindow = DownloadWindow(),
        limit: int = 400,
    ) -> pd.DataFrame:
        params = {
            "instId": inst_id,
            "before": _timestamp_ms(window.start),
            "after": _timestamp_ms(window.end),
            "limit": limit,
        }
        payload = self._http_get("/api/v5/public/funding-rate-history", params)
        data = payload.get("data", []) if isinstance(payload, dict) else payload
        frame = pd.DataFrame(data)
        if frame.empty:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "funding_rate", "realized_rate"]
            )
        frame["timestamp"] = _utc_naive_from_ms(frame["fundingTime"])
        frame = frame.rename(
            columns={
                "instId": "symbol",
                "fundingRate": "funding_rate",
                "realizedRate": "realized_rate",
            }
        )
        frame = _normalize_numeric(frame, ["funding_rate", "realized_rate"])
        return _finalize_time_series(
            frame.loc[:, ["timestamp", "symbol", "funding_rate", "realized_rate"]]
        )

    def fetch_open_interest(
        self,
        inst_type: str = "SWAP",
        inst_id: str | None = None,
    ) -> pd.DataFrame:
        params = {"instType": inst_type, "instId": inst_id}
        payload = self._http_get("/api/v5/public/open-interest", params)
        data = payload.get("data", []) if isinstance(payload, dict) else payload
        frame = pd.DataFrame(data)
        if frame.empty:
            return pd.DataFrame(
                columns=["timestamp", "symbol", "open_interest", "open_interest_ccy", "open_interest_usd"]
            )
        frame["timestamp"] = _utc_naive_from_ms(frame["ts"])
        frame = frame.rename(
            columns={
                "instId": "symbol",
                "oi": "open_interest",
                "oiCcy": "open_interest_ccy",
                "oiUsd": "open_interest_usd",
            }
        )
        frame = _normalize_numeric(
            frame,
            ["open_interest", "open_interest_ccy", "open_interest_usd"],
        )
        return _finalize_time_series(
            frame.loc[
                :,
                [
                    "timestamp",
                    "symbol",
                    "open_interest",
                    "open_interest_ccy",
                    "open_interest_usd",
                ],
            ]
        )
