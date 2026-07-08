# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Reusable alpha/model report generation utilities.

This module contains the Plotly HTML report helpers used by the Alpha158
experiment pipeline.  The inputs are deliberately plain pandas objects so the
same report can be reused by other factor/model experiments:

- ``metrics``: one-row or multi-row metric table with a ``model`` column.
- ``returns``: daily frame with ``long`` and ``benchmark`` columns.
- ``ic_frame``: daily frame with ``IC`` and ``Rank IC`` columns.
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objs as go
from plotly.colors import qualitative
from plotly.io import to_html
from plotly.subplots import make_subplots

from qlib.contrib.report.analysis_position.report import _calculate_report_data
from qlib.contrib.report.graph import ScatterGraph


GENERATED_HTML_FILES = {
    "metrics.html",
    "metrics_table.html",
    "strategy_cumulative_return.html",
    "yearly_ic_rankic.html",
}
TOP_LEVEL_OUTPUT_FILES = {
    "daily_returns.pkl",
    "metrics.csv",
    "metrics.html",
    "split.json",
}


def figure_fragment(fig: go.Figure, include_plotlyjs: bool = False) -> str:
    """Return an embeddable Plotly HTML fragment."""
    return to_html(fig, include_plotlyjs="cdn" if include_plotlyjs else False, full_html=False)


def cleanup_top_level_outputs(out_dir: str | Path, market: str) -> None:
    """Remove report files written by earlier runs in an output directory."""
    out_dir = Path(out_dir)
    names = TOP_LEVEL_OUTPUT_FILES | GENERATED_HTML_FILES | {f"{market.lower()}_excess_return.html"}
    for name in names:
        path = out_dir / name
        if path.exists():
            path.unlink()


def cleanup_model_html(model_dir: str | Path, market: str) -> None:
    """Remove generated HTML report files for one model directory."""
    model_dir = Path(model_dir)
    names = GENERATED_HTML_FILES | {f"{market.lower()}_excess_return.html"}
    for name in names:
        path = model_dir / name
        if path.exists():
            path.unlink()


def write_metrics_html(
    sections: list[tuple[str, list[go.Figure]]],
    path: str | Path,
    title: str = "Alpha Metrics",
) -> None:
    """Write a standalone HTML page containing one or more Plotly sections."""
    body: list[str] = []
    first = True
    for section_title, figures in sections:
        body.append(f"<section><h2>{escape(section_title)}</h2>")
        for fig in figures:
            body.append(figure_fragment(fig, include_plotlyjs=first))
            first = False
        body.append("</section>")
    html = (
        """<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{title}</title>
  <style>
    body {{ margin: 0; padding: 24px; font-family: Arial, sans-serif; background: #f7f7f7; color: #222; }}
    h1 {{ margin: 0 0 24px; font-size: 28px; }}
    h2 {{ margin: 32px 0 12px; font-size: 22px; }}
    section {{ max-width: 1280px; margin: 0 auto 32px; padding: 18px; background: #fff; border: 1px solid #ddd; }}
  </style>
</head>
<body>
  <h1>{title}</h1>
"""
        .format(title=escape(title))
        + "\n".join(body)
        + """
</body>
</html>
"""
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def to_report_frame(ret: pd.DataFrame) -> pd.DataFrame:
    """Convert long/benchmark return data to qlib report graph input."""
    missing = {"long", "benchmark"} - set(ret.columns)
    if missing:
        raise ValueError(f"returns is missing required columns: {sorted(missing)}")
    report = pd.DataFrame(
        {
            "return": ret["long"],
            "bench": ret["benchmark"],
            "cost": ret["cost"] if "cost" in ret.columns else 0.0,
            "turnover": ret["turnover"] if "turnover" in ret.columns else 0.0,
        },
        index=ret.index,
    )
    report.index.name = "date"
    return report


def model_report_data(
    returns: dict[str, pd.DataFrame],
    column: str,
    market: str,
    include_benchmark: bool = False,
    strategy_suffix: str = "Alpha158",
) -> pd.DataFrame:
    """Build cumulative return data for one or more models."""
    data = {}
    for name, ret in returns.items():
        report_df = _calculate_report_data(to_report_frame(ret).copy())
        if include_benchmark and not data:
            data[f"{market.upper()} benchmark"] = report_df["cum_bench"]
        label = f"{name} + {strategy_suffix}" if strategy_suffix else name
        data[label] = report_df[column]
    return pd.DataFrame(data)


def cumulative_figure(
    returns: dict[str, pd.DataFrame],
    market: str,
    strategy_suffix: str = "Alpha158",
) -> go.Figure:
    """Return cumulative strategy and benchmark return figure."""
    df = model_report_data(
        returns,
        "cum_return_wo_cost",
        market,
        include_benchmark=True,
        strategy_suffix=strategy_suffix,
    )
    return ScatterGraph(
        df,
        layout=dict(title="Strategy Performance", yaxis=dict(title="Cumulative Return")),
        graph_kwargs={"mode": "lines"},
    ).figure


def excess_figure(
    returns: dict[str, pd.DataFrame],
    market: str,
    strategy_suffix: str = "Alpha158",
) -> go.Figure:
    """Return cumulative excess return figure."""
    df = model_report_data(returns, "cum_ex_return_wo_cost", market, strategy_suffix=strategy_suffix)
    fig = ScatterGraph(
        df,
        layout=dict(title=f"{market.upper()} Excess Return", yaxis=dict(title="Cumulative Excess Return")),
        graph_kwargs={"mode": "lines"},
    ).figure
    fig.add_hline(y=0, line_dash="dash", line_color="black")
    return fig


def yearly_ic_figure(ic_frames: dict[str, pd.DataFrame]) -> go.Figure:
    """Return yearly IC and RankIC comparison figure."""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Yearly IC", "Yearly RankIC"],
        horizontal_spacing=0.08,
    )
    palette = qualitative.Plotly
    for idx, (name, frame) in enumerate(ic_frames.items()):
        missing = {"IC", "Rank IC"} - set(frame.columns)
        if missing:
            raise ValueError(f"ic frame for {name} is missing required columns: {sorted(missing)}")
        yearly = frame.groupby(frame.index.year).mean()
        years = yearly.index.astype(str)
        color = palette[idx % len(palette)]
        fig.add_trace(
            go.Scatter(
                x=years,
                y=yearly["IC"],
                mode="lines+markers",
                name=name,
                legendgroup=name,
                line=dict(color=color),
                marker=dict(color=color),
                hovertemplate="Model=%{fullData.name}<br>Year=%{x}<br>IC=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=years,
                y=yearly["Rank IC"],
                mode="lines+markers",
                name=name,
                legendgroup=name,
                showlegend=False,
                line=dict(color=color),
                marker=dict(color=color),
                hovertemplate="Model=%{fullData.name}<br>Year=%{x}<br>RankIC=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    fig.update_layout(
        title="Yearly Factor Predictive Power",
        legend=dict(title="Model", orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5),
        hovermode="x unified",
        width=1200,
        height=520,
    )
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_xaxes(title_text="Year", row=1, col=2)
    fig.update_yaxes(title_text="IC", zeroline=True, zerolinecolor="#888", row=1, col=1)
    fig.update_yaxes(title_text="RankIC", zeroline=True, zerolinecolor="#888", row=1, col=2)
    return fig


def metrics_table_figure(metrics: pd.DataFrame, title: str = "Alpha Model Comparison") -> go.Figure:
    """Return metric table figure."""
    display = metrics.copy()
    for col in display.columns:
        if col != "model":
            display[col] = display[col].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    fig = go.Figure(
        data=[
            go.Table(
                header=dict(values=list(display.columns), align="center", fill_color="#f2f2f2"),
                cells=dict(values=[display[col].tolist() for col in display.columns], align="center"),
            )
        ]
    )
    fig.update_layout(title=title, width=1200, height=max(260, 80 + 32 * len(display)))
    return fig


def write_model_report(
    model_name: str,
    metrics: pd.DataFrame,
    returns: pd.DataFrame,
    ic_frame: pd.DataFrame,
    model_dir: str | Path,
    market: str,
    strategy_suffix: str = "Alpha158",
    title: str = "Alpha158 Metrics",
) -> None:
    """Write ``metrics.html`` for one model."""
    model_dir = Path(model_dir)
    cleanup_model_html(model_dir, market)
    model_returns = {model_name: returns}
    model_ic = {model_name: ic_frame}
    sections: list[tuple[str, list[go.Figure]]] = [
        (
            model_name,
            [
                metrics_table_figure(metrics, title=f"{strategy_suffix} Model Comparison"),
                cumulative_figure(model_returns, market, strategy_suffix=strategy_suffix),
                excess_figure(model_returns, market, strategy_suffix=strategy_suffix),
                yearly_ic_figure(model_ic),
            ],
        )
    ]
    write_metrics_html(sections, model_dir / "metrics.html", title=title)


__all__ = [
    "GENERATED_HTML_FILES",
    "TOP_LEVEL_OUTPUT_FILES",
    "cleanup_model_html",
    "cleanup_top_level_outputs",
    "cumulative_figure",
    "excess_figure",
    "figure_fragment",
    "metrics_table_figure",
    "model_report_data",
    "to_report_frame",
    "write_metrics_html",
    "write_model_report",
    "yearly_ic_figure",
]
