from __future__ import annotations

import argparse
from pathlib import Path

from cta_research.analysis import calculate_metrics, factor_ic
from cta_research.backtest import run_backtest
from cta_research.config import load_config
from cta_research.data import load_ohlcv_directory
from cta_research.factor_research import (
    factor_correlation_matrix,
    factor_ic_decay,
    quantile_return_summary,
)
from cta_research.factor_mining import (
    factor_quantile_report,
    mine_factors,
    single_factor_backtest_report,
)
from cta_research.factors import calculate_factor_set
from cta_research.portfolio import (
    blend_strategy_signals,
    size_by_volatility,
    strategy_return_attribution,
)
from cta_research.qlib import build_alpha360_like_features, market_data_to_qlib_frame
from cta_research.reporting import write_run_outputs
from cta_research.risk import apply_drawdown_derisking, cap_exposure
from cta_research.strategies import calculate_strategy_signals


def _next_run_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(path for path in output_dir.iterdir() if path.is_dir())
    return output_dir / f"run_{len(existing) + 1:03d}"


def run_from_config(config_path: str | Path, output_dir: str | Path = "runs") -> Path:
    config_path = Path(config_path)
    config = load_config(config_path)
    data = load_ohlcv_directory(config.data.directory, config.data.symbols)
    factors = calculate_factor_set(data)
    strategy_signals = calculate_strategy_signals(factors)
    combined_signal = blend_strategy_signals(
        strategy_signals,
        config.portfolio.strategy_weights,
    )
    sized = size_by_volatility(
        combined_signal,
        factors["historical_volatility"],
        config.portfolio.volatility_target,
    )
    positions = cap_exposure(
        sized,
        max_symbol_weight=config.portfolio.max_symbol_weight,
        max_gross_exposure=config.portfolio.max_gross_exposure,
    )
    result = run_backtest(
        data=data,
        target_positions=positions,
        initial_capital=config.backtest.initial_capital,
        fee_bps=config.backtest.fee_bps,
        slippage_bps=config.backtest.slippage_bps,
    )
    positions = apply_drawdown_derisking(
        positions,
        result.equity,
        drawdown_warning=config.risk.drawdown_warning,
        drawdown_hard=config.risk.drawdown_hard,
        min_exposure_multiplier=config.risk.min_exposure_multiplier,
    )
    result = run_backtest(
        data=data,
        target_positions=positions,
        initial_capital=config.backtest.initial_capital,
        fee_bps=config.backtest.fee_bps,
        slippage_bps=config.backtest.slippage_bps,
    )
    metrics = calculate_metrics(result.equity, result.returns, result.trades)
    ic = factor_ic(factors["momentum"], data.close, horizons=[1, 3, 5, 10])
    research_factors = {
        name: frame
        for name, frame in factors.items()
        if name
        in {
            "momentum",
            "ma_slope",
            "donchian",
            "rsi",
            "bollinger_zscore",
            "volume_anomaly",
            "vwap_deviation",
            "price_volume_divergence",
            "alternative",
        }
    }
    research_outputs = {
        "factor_scorecard": mine_factors(
            research_factors,
            data,
            horizon=1,
            quantiles=min(5, len(data.symbols)),
            initial_capital=config.backtest.initial_capital,
            fee_bps=config.backtest.fee_bps,
            slippage_bps=config.backtest.slippage_bps,
            gross_exposure=config.portfolio.max_gross_exposure,
        ),
        "factor_quantile_returns": factor_quantile_report(
            research_factors,
            data.close,
            horizon=1,
            quantiles=min(5, len(data.symbols)),
        ),
        "factor_single_backtests": single_factor_backtest_report(
            research_factors,
            data,
            initial_capital=config.backtest.initial_capital,
            fee_bps=config.backtest.fee_bps,
            slippage_bps=config.backtest.slippage_bps,
            gross_exposure=config.portfolio.max_gross_exposure,
        ),
        "factor_ic_summary": factor_ic_decay(
            factors["momentum"],
            data.close,
            horizons=[1, 3, 5, 10],
        ),
        "factor_correlation": factor_correlation_matrix(research_factors).reset_index(
            names="factor"
        ),
        "momentum_quantile_returns": quantile_return_summary(
            factors["momentum"],
            data.close,
            horizon=1,
            quantiles=min(5, len(data.symbols)),
        ),
        "qlib_ohlcv": market_data_to_qlib_frame(data),
        "qlib_alpha360_like": build_alpha360_like_features(data, windows=[3, 5]),
    }
    strategy_returns = strategy_return_attribution(
        data=data,
        strategy_signals=strategy_signals,
        weights=config.portfolio.strategy_weights,
        volatility=factors["historical_volatility"],
        volatility_target=config.portfolio.volatility_target,
        final_positions=result.positions,
    )
    run_dir = _next_run_dir(Path(output_dir))
    write_run_outputs(
        run_dir=run_dir,
        equity=result.equity,
        positions=result.positions,
        trades=result.trades,
        metrics=metrics,
        factor_ic=ic,
        strategy_returns=strategy_returns,
        research_outputs=research_outputs,
        config_text=config_path.read_text(encoding="utf-8"),
    )
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CTA research backtest")
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    run_dir = run_from_config(args.config, args.output_dir)
    print(f"Wrote run outputs to {run_dir}")


if __name__ == "__main__":
    main()
