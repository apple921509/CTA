from pathlib import Path

import pytest

from cta_research.config import load_config


def _write_valid_config(config_path: Path, data_directory: str = "data/ohlcv") -> None:
    config_path.write_text(
        f"""
data:
  directory: {data_directory}
  timeframe: 1d
  symbols: [BTCUSDT, ETHUSDT]
backtest:
  initial_capital: 100000
  fee_bps: 6
  slippage_bps: 2
  execution: next_open
portfolio:
  strategy_weights:
    trend: 0.45
    mean_reversion: 0.20
    swing: 0.25
    alternative: 0.10
  max_symbol_weight: 0.25
  max_gross_exposure: 1.5
  volatility_target: 0.25
risk:
  drawdown_warning: 0.15
  drawdown_hard: 0.30
  min_exposure_multiplier: 0.25
""",
        encoding="utf-8",
    )


def test_load_config_reads_required_sections(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    _write_valid_config(config_path)

    config = load_config(config_path)

    assert config.data.symbols == ["BTCUSDT", "ETHUSDT"]
    assert config.backtest.fee_bps == 6
    assert config.portfolio.strategy_weights["trend"] == 0.45
    assert config.risk.drawdown_hard == 0.30


def test_load_config_rejects_missing_symbols(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  directory: data/ohlcv
  timeframe: 1d
  symbols: []
backtest:
  initial_capital: 100000
  fee_bps: 6
  slippage_bps: 2
  execution: next_open
portfolio:
  strategy_weights:
    trend: 1.0
  max_symbol_weight: 0.25
  max_gross_exposure: 1.5
  volatility_target: 0.25
risk:
  drawdown_warning: 0.15
  drawdown_hard: 0.30
  min_exposure_multiplier: 0.25
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="at least one symbol"):
        load_config(config_path)


def test_load_config_rejects_scalar_symbols(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  directory: data/ohlcv
  timeframe: 1d
  symbols: BTCUSDT
backtest:
  initial_capital: 100000
  fee_bps: 6
  slippage_bps: 2
  execution: next_open
portfolio:
  strategy_weights:
    trend: 1.0
  max_symbol_weight: 0.25
  max_gross_exposure: 1.5
  volatility_target: 0.25
risk:
  drawdown_warning: 0.15
  drawdown_hard: 0.30
  min_exposure_multiplier: 0.25
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.symbols"):
        load_config(config_path)


def test_load_config_rejects_non_string_symbols(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data:
  directory: data/ohlcv
  timeframe: 1d
  symbols: [BTCUSDT, 123]
backtest:
  initial_capital: 100000
  fee_bps: 6
  slippage_bps: 2
  execution: next_open
portfolio:
  strategy_weights:
    trend: 1.0
  max_symbol_weight: 0.25
  max_gross_exposure: 1.5
  volatility_target: 0.25
risk:
  drawdown_warning: 0.15
  drawdown_hard: 0.30
  min_exposure_multiplier: 0.25
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="data.symbols"):
        load_config(config_path)


def test_load_config_rejects_non_mapping_section(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
data: not-a-mapping
backtest:
  initial_capital: 100000
  fee_bps: 6
  slippage_bps: 2
  execution: next_open
portfolio:
  strategy_weights:
    trend: 1.0
  max_symbol_weight: 0.25
  max_gross_exposure: 1.5
  volatility_target: 0.25
risk:
  drawdown_warning: 0.15
  drawdown_hard: 0.30
  min_exposure_multiplier: 0.25
""",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Config section 'data' must be a mapping"):
        load_config(config_path)


def test_load_config_resolves_relative_data_directory_from_config_parent(
    tmp_path: Path,
) -> None:
    config_dir = tmp_path / "configs"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    _write_valid_config(config_path, data_directory="../tests/fixtures/ohlcv")

    config = load_config(config_path)

    assert config.data.directory == (
        config_path.parent / "../tests/fixtures/ohlcv"
    ).resolve()
