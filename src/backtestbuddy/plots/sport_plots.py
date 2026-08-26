"""Plotly charts for sports-betting backtest results."""

from typing import Any, List, Optional, Tuple

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtestbuddy.metrics.sport_metrics import (
    _equity_drawdown_stats,
    _filter_placed_bets,
    calculate_all_metrics,
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


def plot_backtest(backtest: Any) -> go.Figure:
    """Plot bankroll, ROI, and stake % for placed bets of the main strategy.

    Three stacked panels plus a metrics table: bankroll over bet number
    (max-drawdown window highlighted using the last peak before the
    trough), per-bet ROI, stake as a percentage of starting bankroll, and
    a right-hand table of ``calculate_all_metrics``. Bookie results are
    not plotted.

    Args:
        backtest: A backtest instance with ``detailed_results``.

    Returns:
        Plotly figure with the three panels and a metrics table.

    Example:
        >>> fig = plot_backtest(backtest)
        >>> fig.show()
    """
    main_results = backtest.detailed_results
    bet_placed = _filter_placed_bets(main_results).copy()

    bet_placed.loc[:, "game_index"] = range(1, len(bet_placed) + 1)
    date_strings = bet_placed["bt_date_column"].astype(str)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.18,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=("Bankroll Over Time", "ROI", "Stake Percentage"),
    )

    fig.add_trace(
        go.Scatter(
            x=bet_placed["game_index"],
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

    fig.add_trace(
        go.Scatter(
            x=bet_placed["game_index"],
            y=bet_placed["bt_roi"],
            mode="markers",
            name="ROI",
            marker=dict(size=7, opacity=0.7, color="#e11d48"),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    wins = bet_placed[bet_placed["bt_win"] == True]
    losses = bet_placed[bet_placed["bt_win"] == False]

    fig.add_trace(
        go.Scatter(
            x=wins["game_index"],
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
            x=losses["game_index"],
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
            x=bet_placed["game_index"],
            y=(
                bet_placed["bt_stake"]
                / bet_placed["bt_starting_bankroll"]
                * 100
            ),
            name="Stake Percentage",
            marker_color="rgba(13, 148, 136, 0.75)",
            showlegend=False,
        ),
        row=3,
        col=1,
    )

    window = _equity_drawdown_stats(
        bet_placed["bt_ending_bankroll"].to_numpy()
    )
    if window.magnitude > 0:
        fig.add_vrect(
            x0=bet_placed["game_index"].iloc[window.start],
            x1=bet_placed["game_index"].iloc[window.end],
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

    fig.update_layout(
        title=dict(
            text="Backtest Results (Bets Placed Only)",
            x=0.0,
            xanchor="left",
        ),
        height=1200,
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
    fig.update_yaxes(title_text="ROI %", title_standoff=8, row=2, col=1)
    fig.update_yaxes(title_text="Stake %", title_standoff=8, row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", row=2, col=1)
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
        title_text="Bet Number", domain=[0.0, 0.64], row=3, col=1
    )
    for ann in fig.layout.annotations or []:
        if ann.text in ("Bankroll Over Time", "ROI", "Stake Percentage"):
            ann.x = 0.32
            ann.xanchor = "center"
            ann.yshift = 22

    names, values = _metric_table_cells(calculate_all_metrics(main_results))
    n_rows = len(names)
    stripe = ["#f4f6f8" if i % 2 else "#ffffff" for i in range(n_rows)]
    fig.update_layout(height=max(1200, 180 + n_rows * 26))
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
