from pathlib import Path

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
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.html").exists()
