# CTA Research

Research-only composite CTA backtest framework inspired by a multi-factor, multi-strategy CTA process.

This project does not place live trades.

## Install

```powershell
python -m pip install -e ".[dev]"
```

## Run Example

```powershell
cta-research configs/example.yaml --output-dir runs
```

The command writes:

- `config.yaml`
- `equity_curve.csv`
- `positions.csv`
- `trades.csv`
- `strategy_returns.csv`
- `factor_ic.csv`
- `metrics.json`
- `report.html`

## Input Data

Put one CSV per symbol in the configured data directory. Required columns:

```text
timestamp, open, high, low, close, volume
```

Relative data paths are resolved from the YAML config file location.

## Version 1 Scope

Version 1 is for research backtesting only. Live trading, paper trading, exchange API execution, and high-frequency market making are out of scope.

## Current Progress

- Completed the research backtest framework skeleton as a Python package.
- Added YAML configuration loading with validation.
- Added strict multi-symbol OHLCV CSV loading and timestamp normalization.
- Added factor calculations for momentum, trend, volatility, volume, VWAP, and price-volume divergence.
- Added trend, mean-reversion, swing, and alternative-data strategy signal modules.
- Added portfolio blending, volatility sizing, exposure caps, and drawdown de-risking helpers.
- Added backtest accounting with fees, slippage, positions, trades, equity curve, metrics, and factor IC.
- Added CLI/reporting pipeline that writes CSV, JSON, and HTML run outputs.
- Current verification: `pytest -v` passes with 35 tests.

## Roadmap

- Add Binance/OKX historical data downloaders.
- Add 4h timeframe examples and multi-timeframe configuration.
- Add funding rate, open interest, and on-chain alternative factor inputs.
- Add richer HTML charts for equity, drawdown, monthly returns, and attribution.
- Add walk-forward analysis and parameter stability reports.
- Add paper trading only after the research backtest workflow is stable.
