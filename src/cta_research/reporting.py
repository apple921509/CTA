from __future__ import annotations

import json
from pathlib import Path

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
    )
    (run_dir / "report.html").write_text(html, encoding="utf-8")
