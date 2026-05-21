"""YAML configuration loading for CTA Research."""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class DataConfig:
    directory: Path
    timeframe: str
    symbols: list[str]


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float
    fee_bps: float
    slippage_bps: float
    execution: str


@dataclass(frozen=True)
class PortfolioConfig:
    strategy_weights: dict[str, float]
    max_symbol_weight: float
    max_gross_exposure: float
    volatility_target: float


@dataclass(frozen=True)
class RiskConfig:
    drawdown_warning: float
    drawdown_hard: float
    min_exposure_multiplier: float


@dataclass(frozen=True)
class AppConfig:
    data: DataConfig
    backtest: BacktestConfig
    portfolio: PortfolioConfig
    risk: RiskConfig


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required configuration key: {key}")
    return mapping[key]


def _require_section(config: dict[str, Any], key: str) -> dict[str, Any]:
    section = _require(config, key)
    if not isinstance(section, Mapping):
        raise ValueError(f"Config section '{key}' must be a mapping")
    return dict(section)


def _parse_symbols(value: Any) -> list[str]:
    if not isinstance(value, (list, tuple)):
        raise ValueError("data.symbols must be a list of non-empty strings")

    symbols = list(value)
    if not symbols:
        raise ValueError("Configuration must include at least one symbol in data.symbols")

    if not all(isinstance(symbol, str) and symbol for symbol in symbols):
        raise ValueError("data.symbols must contain only non-empty strings")

    return symbols


def _resolve_directory(config_path: Path, directory: Any) -> Path:
    directory_path = Path(directory)
    if directory_path.is_absolute():
        return directory_path
    return (config_path.parent / directory_path).resolve()


def load_config(path: Path | str) -> AppConfig:
    """Load and validate an application config from YAML."""
    config_path = Path(path).resolve()
    raw_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise ValueError("Configuration file must contain a YAML mapping")

    data_raw = _require_section(raw_config, "data")
    backtest_raw = _require_section(raw_config, "backtest")
    portfolio_raw = _require_section(raw_config, "portfolio")
    risk_raw = _require_section(raw_config, "risk")

    data = DataConfig(
        directory=_resolve_directory(config_path, _require(data_raw, "directory")),
        timeframe=str(_require(data_raw, "timeframe")),
        symbols=_parse_symbols(_require(data_raw, "symbols")),
    )

    backtest = BacktestConfig(
        initial_capital=float(_require(backtest_raw, "initial_capital")),
        fee_bps=float(_require(backtest_raw, "fee_bps")),
        slippage_bps=float(_require(backtest_raw, "slippage_bps")),
        execution=str(_require(backtest_raw, "execution")),
    )
    if backtest.execution != "next_open":
        raise ValueError("Only next_open execution is supported")

    portfolio = PortfolioConfig(
        strategy_weights={
            str(name): float(weight)
            for name, weight in _require(portfolio_raw, "strategy_weights").items()
        },
        max_symbol_weight=float(_require(portfolio_raw, "max_symbol_weight")),
        max_gross_exposure=float(_require(portfolio_raw, "max_gross_exposure")),
        volatility_target=float(_require(portfolio_raw, "volatility_target")),
    )

    risk = RiskConfig(
        drawdown_warning=float(_require(risk_raw, "drawdown_warning")),
        drawdown_hard=float(_require(risk_raw, "drawdown_hard")),
        min_exposure_multiplier=float(_require(risk_raw, "min_exposure_multiplier")),
    )
    if risk.drawdown_warning >= risk.drawdown_hard:
        raise ValueError("drawdown_warning must be less than drawdown_hard")

    return AppConfig(
        data=data,
        backtest=backtest,
        portfolio=portfolio,
        risk=risk,
    )
