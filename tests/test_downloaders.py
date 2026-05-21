from pathlib import Path
from typing import Any

import pandas as pd

from cta_research.downloaders import (
    BinanceUsdmClient,
    DownloadWindow,
    OkxClient,
    write_feature_cache,
    write_ohlcv_cache,
)


class StubHttpGet:
    def __init__(self, payloads: dict[str, Any]) -> None:
        self.payloads = payloads
        self.calls: list[tuple[str, dict[str, str | int | float | None]]] = []

    def __call__(self, path: str, params: dict[str, str | int | float | None]) -> Any:
        self.calls.append((path, params))
        return self.payloads[path]


def test_binance_fetch_ohlcv_normalizes_kline_payload() -> None:
    http = StubHttpGet(
        {
            "/fapi/v1/klines": [
                [
                    1704067200000,
                    "42000.0",
                    "43000.0",
                    "41000.0",
                    "42500.0",
                    "123.4",
                    1704153599999,
                    "0",
                    1,
                    "0",
                    "0",
                    "0",
                ]
            ]
        }
    )
    client = BinanceUsdmClient(http_get=http)

    frame = client.fetch_ohlcv(
        "BTCUSDT",
        "1d",
        DownloadWindow(start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-01-02")),
    )

    assert http.calls[0][0] == "/fapi/v1/klines"
    assert http.calls[0][1]["symbol"] == "BTCUSDT"
    assert http.calls[0][1]["interval"] == "1d"
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2024-01-01")
    assert frame.loc[0, "close"] == 42500.0


def test_binance_fetch_funding_rates_normalizes_payload() -> None:
    http = StubHttpGet(
        {
            "/fapi/v1/fundingRate": [
                {
                    "symbol": "BTCUSDT",
                    "fundingRate": "0.0001",
                    "fundingTime": 1704067200000,
                    "markPrice": "42000.5",
                }
            ]
        }
    )

    frame = BinanceUsdmClient(http_get=http).fetch_funding_rates("BTCUSDT")

    assert list(frame.columns) == ["timestamp", "symbol", "funding_rate", "mark_price"]
    assert frame.loc[0, "funding_rate"] == 0.0001
    assert frame.loc[0, "mark_price"] == 42000.5


def test_binance_fetch_open_interest_normalizes_payload() -> None:
    http = StubHttpGet(
        {
            "/futures/data/openInterestHist": [
                {
                    "symbol": "BTCUSDT",
                    "sumOpenInterest": "20403.637",
                    "sumOpenInterestValue": "150570784.078",
                    "timestamp": "1704067200000",
                }
            ]
        }
    )

    frame = BinanceUsdmClient(http_get=http).fetch_open_interest("BTCUSDT", "1d")

    assert http.calls[0][1]["period"] == "1d"
    assert frame.loc[0, "open_interest"] == 20403.637
    assert frame.loc[0, "open_interest_value"] == 150570784.078


def test_okx_fetch_ohlcv_uses_history_candles_and_sorts_ascending() -> None:
    http = StubHttpGet(
        {
            "/api/v5/market/history-candles": {
                "code": "0",
                "msg": "",
                "data": [
                    ["1704153600000", "2", "3", "1", "2.5", "20", "20", "50", "1"],
                    ["1704067200000", "1", "2", "0.5", "1.5", "10", "10", "15", "1"],
                ],
            }
        }
    )

    frame = OkxClient(http_get=http).fetch_ohlcv("BTC-USDT-SWAP", "1d")

    assert http.calls[0][0] == "/api/v5/market/history-candles"
    assert http.calls[0][1]["bar"] == "1Dutc"
    assert frame.loc[0, "timestamp"] == pd.Timestamp("2024-01-01")
    assert frame.loc[1, "close"] == 2.5


def test_okx_fetch_funding_rates_normalizes_payload() -> None:
    http = StubHttpGet(
        {
            "/api/v5/public/funding-rate-history": {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instId": "BTC-USDT-SWAP",
                        "fundingRate": "0.0002",
                        "realizedRate": "0.00019",
                        "fundingTime": "1704067200000",
                    }
                ],
            }
        }
    )

    frame = OkxClient(http_get=http).fetch_funding_rates("BTC-USDT-SWAP")

    assert list(frame.columns) == ["timestamp", "symbol", "funding_rate", "realized_rate"]
    assert frame.loc[0, "symbol"] == "BTC-USDT-SWAP"
    assert frame.loc[0, "realized_rate"] == 0.00019


def test_okx_fetch_open_interest_normalizes_snapshot_payload() -> None:
    http = StubHttpGet(
        {
            "/api/v5/public/open-interest": {
                "code": "0",
                "msg": "",
                "data": [
                    {
                        "instType": "SWAP",
                        "instId": "BTC-USDT-SWAP",
                        "oi": "5000",
                        "oiCcy": "555.55",
                        "oiUsd": "50000",
                        "ts": "1704067200000",
                    }
                ],
            }
        }
    )

    frame = OkxClient(http_get=http).fetch_open_interest(inst_id="BTC-USDT-SWAP")

    assert http.calls[0][1]["instType"] == "SWAP"
    assert frame.loc[0, "open_interest_usd"] == 50000.0


def test_cache_writers_create_expected_paths(tmp_path: Path) -> None:
    ohlcv = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "open": [1.0],
            "high": [2.0],
            "low": [0.5],
            "close": [1.5],
            "volume": [10.0],
        }
    )
    feature = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-01")],
            "symbol": ["BTCUSDT"],
            "funding_rate": [0.0001],
        }
    )

    ohlcv_path = write_ohlcv_cache(ohlcv, tmp_path / "ohlcv", "BTCUSDT")
    feature_path = write_feature_cache(
        feature,
        tmp_path / "features",
        "binance",
        "funding_rates",
        "BTCUSDT",
    )

    assert ohlcv_path == tmp_path / "ohlcv" / "BTCUSDT.csv"
    assert feature_path == tmp_path / "features" / "binance" / "funding_rates" / "BTCUSDT.csv"
    assert ohlcv_path.exists()
    assert feature_path.exists()
