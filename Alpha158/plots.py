"""Compatibility wrapper for Alpha158 report helpers.

The implementation lives in ``qlib.contrib.report.alpha`` so other qlib-based
experiments can reuse the same HTML report generation utilities.
"""

from __future__ import annotations

from qlib.contrib.report.alpha import (  # noqa: F401
    GENERATED_HTML_FILES,
    TOP_LEVEL_OUTPUT_FILES,
    cleanup_model_html,
    cleanup_top_level_outputs,
    cumulative_figure,
    excess_figure,
    figure_fragment,
    metrics_table_figure,
    model_report_data,
    to_report_frame,
    write_metrics_html,
    write_model_report,
    yearly_ic_figure,
)


_figure_fragment = figure_fragment
_cleanup_model_html = cleanup_model_html
_write_metrics_html = write_metrics_html
_to_report_frame = to_report_frame
_model_report_data = model_report_data
