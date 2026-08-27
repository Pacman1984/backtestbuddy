"""Plotly charts for sports-betting backtest results."""

from typing import Any, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtestbuddy.metrics.sport_metrics import (
    ECE_BINS,
    _equity_drawdown_stats,
    _filter_placed_bets,
    _reliability_points,
    calculate_all_metrics,
    calculate_ece,
)


def _format_metric_value(value: Any) -> str:
    """Format one metric for the side table.

    Args:
        value: Raw metric from ``calculate_all_metrics``.

    Returns:
        Display string. Non-finite floats become an em dash.

    Example:
        ``float("nan")`` → ``"—"``; ``12.345`` → ``"12.35"``.
    """
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "—"
        return f"{float(value):.2f}"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if hasattr(value, "strftime"):
        try:
            return value.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            pass
    text = str(value)
    if text.endswith(" 00:00:00"):
        return text[: -len(" 00:00:00")]
    return text


def _metric_table_cells(metrics: dict) -> Tuple[List[str], List[str]]:
    """Build one Metric/Value column pair from ``calculate_all_metrics``.

    Args:
        metrics: Mapping from ``calculate_all_metrics``.

    Returns:
        ``(names, values)`` in dictionary order.
    """
    names = list(metrics.keys())
    values = [_format_metric_value(v) for v in metrics.values()]
    return names, values


def _panel_x(frame: pd.DataFrame, x_axis: str) -> pd.Series:
    """X coordinates for a results frame.

    Args:
        frame: Rows with ``bt_date_column``.
        x_axis: ``"bet"`` (1-based row number) or ``"date"``.

    Returns:
        Series aligned to ``frame.index``.

    Raises:
        ValueError: If ``x_axis`` is not ``"bet"`` or ``"date"``.
    """
    if x_axis == "date":
        return pd.to_datetime(frame["bt_date_column"])
    if x_axis == "bet":
        return pd.Series(np.arange(1, len(frame) + 1), index=frame.index)
    raise ValueError("x_axis must be 'bet' or 'date'.")


def _underwater_pct(equity: np.ndarray) -> np.ndarray:
    """Percent below running peak: ``(equity / cummax - 1) * 100``.

    Args:
        equity: Sequential bankroll values.

    Returns:
        Underwater series. Empty input stays empty; non-positive peaks
        become 0.

    Example:
        ``[100, 110, 99]`` → ``[0, 0, -10]``.
    """
    if len(equity) == 0:
        return np.asarray(equity, dtype=float)
    values = np.asarray(equity, dtype=float)
    peak = np.maximum.accumulate(values)
    with np.errstate(divide="ignore", invalid="ignore"):
        underwater = (values / peak - 1.0) * 100.0
    return np.nan_to_num(underwater, nan=0.0, posinf=0.0, neginf=0.0)


def plot_backtest(
    backtest: Any,
    x_axis: Literal["bet", "date"] = "bet",
    show_bookie: bool = True,
) -> go.Figure:
    """Plot bankroll, drawdown, ROI, and stake % for placed bets.

    Four stacked panels plus a metrics table: bankroll (max-drawdown
    window from the last peak to the trough), underwater drawdown,
    per-bet ROI, stake as a percentage of starting bankroll, and a
    right-hand table of ``calculate_all_metrics``.

    Args:
        backtest: A backtest instance with ``detailed_results``.
        x_axis: ``"bet"`` uses a compact placed-bet index. ``"date"``
            uses ``bt_date_column``. Date mode is the fair overlay when
            the strategy skips bets that the bookie still takes.
        show_bookie: If True and ``bookie_results`` is present, overlay
            the bookie bankroll on the first panel. KellyCriterion and
            ValueBet stake 0 on the bookie path, so that line is flat.

    Returns:
        Plotly figure with the four panels and a metrics table.

    Example:
        >>> fig = plot_backtest(backtest, x_axis="date")
        >>> fig.show()
    """
    main_results = backtest.detailed_results
    bet_placed = _filter_placed_bets(main_results).copy()
    x_main = _panel_x(bet_placed, x_axis)
    date_strings = bet_placed["bt_date_column"].astype(str)
    equity = bet_placed["bt_ending_bankroll"].to_numpy()

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        row_heights=[0.38, 0.18, 0.22, 0.22],
        subplot_titles=(
            "Bankroll Over Time",
            "Drawdown",
            "ROI",
            "Stake Percentage",
        ),
    )

    fig.add_trace(
        go.Scatter(
            x=x_main,
            y=bet_placed["bt_ending_bankroll"],
            name="Main Strategy",
            line=dict(color="#2563eb", width=2),
            hovertemplate=(
                "Game: %{x}<br>"
                "Date: %{customdata[0]}<br>"
                "Starting Bankroll: $%{customdata[1]:.2f}<br>"
                "Ending Bankroll: $%{y:.2f}<br>"
                "Stake: $%{customdata[2]:.2f}<br>"
                "Stake Percentage: %{customdata[3]:.2f}%"
                "<extra></extra>"
            ),
            customdata=np.column_stack(
                (
                    date_strings,
                    bet_placed["bt_starting_bankroll"],
                    bet_placed["bt_stake"],
                    bet_placed["bt_stake"]
                    / bet_placed["bt_starting_bankroll"]
                    * 100,
                )
            ),
        ),
        row=1,
        col=1,
    )

    bookie = getattr(backtest, "bookie_results", None)
    if (
        show_bookie
        and isinstance(bookie, pd.DataFrame)
        and not bookie.empty
    ):
        fig.add_trace(
            go.Scatter(
                x=_panel_x(bookie, x_axis),
                y=bookie["bt_ending_bankroll"],
                name="Bookie",
                line=dict(color="#64748b", width=2, dash="dot"),
            ),
            row=1,
            col=1,
        )

    fig.add_trace(
        go.Scatter(
            x=x_main,
            y=_underwater_pct(equity),
            name="Drawdown",
            fill="tozeroy",
            line=dict(color="#dc2626", width=1.5),
            fillcolor="rgba(220, 38, 38, 0.18)",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=x_main,
            y=bet_placed["bt_roi"],
            mode="markers",
            name="ROI",
            marker=dict(size=7, opacity=0.7, color="#e11d48"),
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    wins = bet_placed[bet_placed["bt_win"] == True]
    losses = bet_placed[bet_placed["bt_win"] == False]

    fig.add_trace(
        go.Scatter(
            x=x_main.loc[wins.index],
            y=wins["bt_ending_bankroll"],
            mode="markers",
            marker=dict(color="#16a34a", symbol="triangle-up", size=9),
            name="Wins",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_main.loc[losses.index],
            y=losses["bt_ending_bankroll"],
            mode="markers",
            marker=dict(color="#dc2626", symbol="triangle-down", size=9),
            name="Losses",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=x_main,
            y=(
                bet_placed["bt_stake"]
                / bet_placed["bt_starting_bankroll"]
                * 100
            ),
            name="Stake Percentage",
            marker_color="rgba(13, 148, 136, 0.75)",
            showlegend=False,
        ),
        row=4,
        col=1,
    )

    window = _equity_drawdown_stats(equity)
    if window.magnitude > 0 and len(bet_placed) > 0:
        fig.add_vrect(
            x0=x_main.iloc[window.start],
            x1=x_main.iloc[window.end],
            fillcolor="rgba(220, 38, 38, 0.12)",
            opacity=1.0,
            layer="below",
            line_width=0,
            annotation_text=(
                f"Max Drawdown: {window.magnitude:.2%}<br>"
                f"Length: {window.duration} bets"
            ),
            annotation_position="top left",
            row=1,
            col=1,
        )

    x_title = "Date" if x_axis == "date" else "Bet Number"
    fig.update_layout(
        title=dict(
            text="Backtest Results (Bets Placed Only)",
            x=0.0,
            xanchor="left",
        ),
        height=1400,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="left",
            x=0.0,
            bgcolor="rgba(255, 255, 255, 0.9)",
            borderwidth=0,
            itemsizing="constant",
        ),
        hovermode="x unified",
        template="plotly_white",
        margin=dict(l=64, r=16, t=150, b=52),
        hoverlabel=dict(bgcolor="white", font_size=12),
        bargap=0.25,
    )

    fig.update_yaxes(title_text="Bankroll", title_standoff=8, row=1, col=1)
    fig.update_yaxes(title_text="DD %", title_standoff=8, row=2, col=1)
    fig.update_yaxes(title_text="ROI %", title_standoff=8, row=3, col=1)
    fig.update_yaxes(title_text="Stake %", title_standoff=8, row=4, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=3, col=1)
    fig.update_xaxes(
        title_text="",
        domain=[0.0, 0.64],
        showticklabels=False,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text="",
        domain=[0.0, 0.64],
        showticklabels=False,
        row=2,
        col=1,
    )
    fig.update_xaxes(
        title_text="",
        domain=[0.0, 0.64],
        showticklabels=False,
        row=3,
        col=1,
    )
    fig.update_xaxes(
        title_text=x_title, domain=[0.0, 0.64], row=4, col=1
    )
    chart_titles = (
        "Bankroll Over Time",
        "Drawdown",
        "ROI",
        "Stake Percentage",
    )
    for ann in fig.layout.annotations or []:
        if ann.text in chart_titles:
            ann.x = 0.32
            ann.xanchor = "center"
            ann.yshift = 22

    names, values = _metric_table_cells(calculate_all_metrics(main_results))
    n_rows = len(names)
    stripe = ["#f4f6f8" if i % 2 else "#ffffff" for i in range(n_rows)]
    fig.update_layout(height=max(1400, 180 + n_rows * 26))
    fig.add_trace(
        go.Table(
            header=dict(
                values=["Metric", "Value"],
                fill_color="#1e293b",
                font=dict(color="white", size=11),
                align=["right", "left"],
                height=28,
            ),
            cells=dict(
                values=[names, values],
                fill_color=[stripe, stripe],
                align=["right", "left"],
                font=dict(size=10),
                height=24,
            ),
            columnwidth=[2.4, 0.9],
            domain=dict(x=[0.68, 1.0], y=[0.0, 1.0]),
        )
    )

    return fig


def plot_calibration(
    backtest: Any,
    n_bins: Optional[int] = None,
) -> go.Figure:
    """Plot a reliability diagram for selected-outcome model probabilities.

    Equal-width bins on ``[0, 1]``, same as ``calculate_ece``. Each
    marker is the mean predicted probability vs the observed win rate;
    marker size scales with the number of bets in the bin. The diagonal
    is perfect calibration. No probabilities yields the diagonal only.

    Args:
        backtest: A backtest instance with ``detailed_results``.
        n_bins: Number of equal-width bins. Defaults to ``ECE_BINS``.

    Returns:
        Plotly figure with predicted p on x and observed frequency on y.

    Example:
        All p=0.6 and half win → one marker at ``(0.6, 0.5)``.
    """
    if n_bins is None:
        n_bins = ECE_BINS
    results = backtest.detailed_results
    points = _reliability_points(results, n_bins=n_bins)
    ece = calculate_ece(results, n_bins=n_bins)
    title = "Reliability Diagram"
    if np.isfinite(ece):
        title = f"Reliability Diagram (ECE = {ece:.3f})"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0.0, 1.0],
            y=[0.0, 1.0],
            mode="lines",
            name="Perfect calibration",
            line=dict(color="#94a3b8", width=1.5, dash="dash"),
        )
    )
    if not points.empty:
        max_count = max(float(points["count"].max()), 1.0)
        fig.add_trace(
            go.Scatter(
                x=points["mean_p"],
                y=points["win_rate"],
                mode="markers",
                name="Observed",
                marker=dict(
                    size=points["count"],
                    sizemode="area",
                    sizeref=2.0 * max_count / (18.0 ** 2),
                    sizemin=8,
                    color="#2563eb",
                    line=dict(width=1, color="#1e3a8a"),
                ),
                customdata=points["count"],
                hovertemplate=(
                    "Predicted: %{x:.3f}<br>"
                    "Observed: %{y:.3f}<br>"
                    "Bets: %{customdata}<extra></extra>"
                ),
            )
        )
    else:
        fig.add_annotation(
            text="No model probabilities on placed bets",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=13, color="#64748b"),
        )

    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left"),
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        margin=dict(l=64, r=32, t=80, b=52),
        width=720,
        height=720,
    )
    fig.update_xaxes(
        title_text="Predicted probability",
        range=[-0.02, 1.02],
        constrain="domain",
    )
    fig.update_yaxes(
        title_text="Observed win rate",
        range=[-0.02, 1.02],
        scaleanchor="x",
        scaleratio=1,
    )
    return fig


def plot_odds_histogram(
    backtest: Any, num_bins: Optional[int] = None
) -> go.Figure:
    """
    Create a histogram plot of the odds distribution for the main strategy,
    splitting each bin into winning and losing bets, and adding dotted
    lines for break-even win rates.

    Args:
        backtest (Any): The backtest object containing detailed results.
        num_bins (Optional[int]): The number of bins to use for the
            histogram. If None, auto-binning is used.

    Returns:
        go.Figure: A Plotly figure object containing the odds histogram.
    """
    detailed_results = _filter_placed_bets(backtest.detailed_results)

    odds_column = next(
        (
            col
            for col in detailed_results.columns
            if "odds" in col.lower()
        ),
        None,
    )
    if odds_column is None:
        raise ValueError(
            "Could not find an odds column in the detailed results."
        )

    odds = detailed_results[odds_column]
    wins = detailed_results["bt_win"] > 0

    valid_mask = (
        ~(odds.isna() | wins.isna()) & (detailed_results["bt_bet_on"] != -1)
    )
    odds = odds[valid_mask]
    wins = wins[valid_mask]

    if num_bins is None:
        num_bins = int(np.sqrt(len(odds)))
    bin_edges = np.logspace(
        np.log10(odds.min()), np.log10(odds.max()), num_bins + 1
    )

    win_hist, _ = np.histogram(odds[wins], bins=bin_edges)
    lose_hist, _ = np.histogram(odds[~wins], bins=bin_edges)

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    break_even_win_rates = 1 / bin_centers

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=win_hist,
            name="Winning Bets",
            marker_color="green",
            opacity=0.7,
        )
    )

    fig.add_trace(
        go.Bar(
            x=bin_centers,
            y=lose_hist,
            name="Losing Bets",
            marker_color="red",
            opacity=0.7,
        )
    )

    for i in range(len(bin_centers)):
        total_bets = win_hist[i] + lose_hist[i]
        if total_bets > 0:
            break_even_height = total_bets * break_even_win_rates[i]
            fig.add_shape(
                type="line",
                x0=bin_edges[i],
                y0=break_even_height,
                x1=bin_edges[i + 1],
                y1=break_even_height,
                line=dict(color="blue", width=2, dash="dot"),
            )

    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="blue", width=2, dash="dot"),
            name="Break-Even Win Rate",
        )
    )

    fig.update_layout(
        title="Distribution of played Odds and Break-Even Win Rates",
        xaxis_title="Odds",
        yaxis_title="Frequency",
        barmode="stack",
        bargap=0.1,
        xaxis=dict(
            tickmode="array",
            tickvals=bin_edges,
            ticktext=[f"{x:.2f}" for x in bin_edges],
            tickangle=45,
        ),
    )

    avg_odds = odds.mean()
    fig.add_vline(
        x=avg_odds,
        line_dash="dash",
        line_color="blue",
        annotation_text=f"Avg: {avg_odds:.2f}",
        annotation_position="top left",
    )

    median_odds = odds.median()
    fig.add_vline(
        x=median_odds,
        line_dash="dot",
        line_color="purple",
        annotation_text=f"Median: {median_odds:.2f}",
        annotation_position="bottom right",
    )

    y_max = max(win_hist.max(), lose_hist.max())
    fig.update_yaxes(range=[0, y_max * 1.2])
    fig.update_layout(margin=dict(t=100))

    return fig
