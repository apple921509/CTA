from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from jinja2 import Template


REPORT_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>CTA Research Report</title>
  <style>
    body { font-family: Georgia, serif; margin: 2rem; color: #1f2933; }
    img { display: block; max-width: 920px; width: 100%; margin: 1rem 0 2rem; }
    table { border-collapse: collapse; margin: 1rem 0; }
    td, th { border: 1px solid #d8dee9; padding: 0.4rem 0.6rem; }
    th { background: #f1f5f9; }
    pre { background: #f8fafc; padding: 1rem; overflow-x: auto; }
  </style>
</head>
<body>
  <h1>CTA Research Report</h1>
  <h2>Metrics</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    {% for key, value in metrics.items() %}
    <tr>
      <td>{{ key }}</td>
      <td>{{ "%.6f"|format(value) if value is number else value }}</td>
    </tr>
    {% endfor %}
  </table>
  <h2>Charts</h2>
  {% for chart in charts %}
  <h3>{{ chart.title }}</h3>
  <img src="{{ chart.path }}" alt="{{ chart.title }}">
  {% endfor %}
  <h2>Equity Curve Tail</h2>
  <pre>{{ equity_tail }}</pre>
  <h2>Trades</h2>
  <p>Total trades: {{ trade_count }}</p>
  <h2>Research Attachments</h2>
  <ul>
    {% for attachment in attachments %}
    <li>{{ attachment }}</li>
    {% endfor %}
  </ul>
</body>
</html>
"""


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _write_equity_chart(charts_dir: Path, equity: pd.Series) -> str | None:
    if equity.empty:
        return None
    plt.figure(figsize=(10, 4))
    equity.plot(color="#2563eb", linewidth=1.8)
    plt.title("Equity Curve")
    plt.xlabel("Date")
    plt.ylabel("Equity")
    plt.grid(alpha=0.25)
    path = charts_dir / "equity_curve.png"
    _save_figure(path)
    return "charts/equity_curve.png"


def _write_drawdown_chart(charts_dir: Path, equity: pd.Series) -> str | None:
    if equity.empty:
        return None
    drawdown = equity / equity.cummax() - 1.0
    plt.figure(figsize=(10, 3.5))
    drawdown.plot(color="#dc2626", linewidth=1.5)
    plt.title("Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.grid(alpha=0.25)
    path = charts_dir / "drawdown.png"
    _save_figure(path)
    return "charts/drawdown.png"


def _write_strategy_returns_chart(
    charts_dir: Path,
    strategy_returns: pd.DataFrame,
) -> str | None:
    if strategy_returns.empty or len(strategy_returns.columns) == 0:
        return None
    cumulative = (1.0 + strategy_returns.fillna(0.0)).cumprod() - 1.0
    plt.figure(figsize=(10, 4))
    cumulative.plot(ax=plt.gca(), linewidth=1.5)
    plt.title("Strategy Return Attribution")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.grid(alpha=0.25)
    path = charts_dir / "strategy_returns.png"
    _save_figure(path)
    return "charts/strategy_returns.png"


def _write_factor_ic_chart(charts_dir: Path, factor_ic: pd.DataFrame) -> str | None:
    if factor_ic.empty or not {"horizon", "rank_ic"}.issubset(factor_ic.columns):
        return None
    summary = factor_ic.groupby("horizon")["rank_ic"].mean()
    if summary.empty:
        return None
    plt.figure(figsize=(8, 3.5))
    summary.plot(kind="bar", color="#16a34a")
    plt.title("Mean Rank IC by Horizon")
    plt.xlabel("Horizon")
    plt.ylabel("Mean Rank IC")
    plt.grid(axis="y", alpha=0.25)
    path = charts_dir / "factor_ic.png"
    _save_figure(path)
    return "charts/factor_ic.png"


def _write_charts(
    run_dir: Path,
    equity: pd.Series,
    strategy_returns: pd.DataFrame,
    factor_ic: pd.DataFrame,
) -> list[dict[str, str]]:
    charts_dir = run_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        ("Equity Curve", _write_equity_chart(charts_dir, equity)),
        ("Drawdown", _write_drawdown_chart(charts_dir, equity)),
        (
            "Strategy Return Attribution",
            _write_strategy_returns_chart(charts_dir, strategy_returns),
        ),
        ("Mean Rank IC by Horizon", _write_factor_ic_chart(charts_dir, factor_ic)),
    ]
    return [
        {"title": title, "path": path}
        for title, path in specs
        if path is not None
    ]


def write_run_outputs(
    run_dir: Path,
    equity: pd.Series,
    positions: pd.DataFrame,
    trades: pd.DataFrame,
    metrics: dict[str, float | int],
    factor_ic: pd.DataFrame,
    strategy_returns: pd.DataFrame | None = None,
    research_outputs: dict[str, pd.DataFrame] | None = None,
    config_text: str | None = None,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    equity.rename("equity").to_csv(run_dir / "equity_curve.csv")
    positions.to_csv(run_dir / "positions.csv")
    trades.to_csv(run_dir / "trades.csv", index=False)
    factor_ic.to_csv(run_dir / "factor_ic.csv", index=False)

    if strategy_returns is None:
        strategy_returns = pd.DataFrame(index=equity.index)
    strategy_returns.to_csv(run_dir / "strategy_returns.csv")
    charts = _write_charts(run_dir, equity, strategy_returns, factor_ic)

    attachments = [
        "config.yaml",
        "equity_curve.csv",
        "positions.csv",
        "trades.csv",
        "strategy_returns.csv",
        "factor_ic.csv",
        "metrics.json",
    ]
    if research_outputs is not None:
        for name, frame in research_outputs.items():
            filename = f"{name}.csv"
            frame.to_csv(run_dir / filename, index=False)
            attachments.append(filename)

    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    if config_text is not None:
        (run_dir / "config.yaml").write_text(config_text, encoding="utf-8")

    html = Template(REPORT_TEMPLATE).render(
        metrics=metrics,
        equity_tail=equity.tail().to_string(),
        trade_count=len(trades),
        attachments=attachments,
        charts=charts,
    )
    (run_dir / "report.html").write_text(html, encoding="utf-8")
