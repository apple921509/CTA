from pathlib import Path
import subprocess
import sys

import pandas as pd

from cta_research.cli import run_from_config


def test_run_from_config_writes_report_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"
    run_dir = run_from_config(Path("configs/example.yaml"), output_dir=output_dir)

    assert (run_dir / "config.yaml").exists()
    assert (run_dir / "equity_curve.csv").exists()
    assert (run_dir / "positions.csv").exists()
    assert (run_dir / "trades.csv").exists()
    assert (run_dir / "strategy_returns.csv").exists()
    assert (run_dir / "factor_ic.csv").exists()
    assert (run_dir / "factor_ic_summary.csv").exists()
    assert (run_dir / "factor_correlation.csv").exists()
    assert (run_dir / "momentum_quantile_returns.csv").exists()
    assert (run_dir / "qlib_ohlcv.csv").exists()
    assert (run_dir / "qlib_alpha360_like.csv").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.html").exists()
    assert (run_dir / "charts" / "equity_curve.png").exists()
    assert (run_dir / "charts" / "drawdown.png").exists()
    assert (run_dir / "charts" / "strategy_returns.png").exists()
    assert (run_dir / "charts" / "factor_ic.png").exists()

    strategy_returns = pd.read_csv(run_dir / "strategy_returns.csv")
    factor_summary = pd.read_csv(run_dir / "factor_ic_summary.csv")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert {"trend", "mean_reversion", "swing"}.issubset(strategy_returns.columns)
    assert {"horizon", "ic_mean", "positive_rate"}.issubset(factor_summary.columns)
    assert "charts/equity_curve.png" in html


def test_cli_module_entrypoint_writes_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "runs"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cta_research.cli",
            "configs/example.yaml",
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "Wrote run outputs to" in completed.stdout
    assert (output_dir / "run_001" / "report.html").exists()
    assert (output_dir / "run_001" / "qlib_ohlcv.csv").exists()
    assert (output_dir / "run_001" / "charts" / "equity_curve.png").exists()
