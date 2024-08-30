from typing import Any, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.metrics.sport_metrics import calculate_all_metrics


def plot_backtest(backtest: Any) -> go.Figure:
    """
    Create a plot of the backtest results, including the bookie strategy and Max Drawdown.

    This function generates a plot with three subplots:
    1. Bankroll over time for both the main strategy and the Bookie strategy, with Max Drawdown highlighted.
    2. ROI for each bet for the main strategy.
    3. ROI for each bet for the bookie strategy.

    Args:
        backtest (Any): An instance of a Backtest class containing the results.

    Returns:
        go.Figure: A Plotly figure object containing the backtest results plot.

    Example:
        >>> backtest = YourBacktestClass(...)
        >>> backtest.run()
        >>> fig = plot_backtest(backtest)
        >>> fig.show()  # Display the plot
    """
    # Create subplots: one for bankroll, two for ROI (main and bookie)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.1, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Bankroll Over Time", "Main Strategy ROI", "Bookie Strategy ROI"))

    # Get results
    main_results = backtest.get_detailed_results()
    bookie_results = backtest.get_bookie_results()

    # Plot bankroll over time for main strategy
    fig.add_trace(go.Scatter(x=main_results['bt_date_column'], 
                             y=main_results['bt_ending_bankroll'], 
                             mode='lines', name='Main Strategy Bankroll'), row=1, col=1)

    # Plot bankroll over time for bookie strategy
    fig.add_trace(go.Scatter(x=bookie_results['bt_date_column'], 
                             y=bookie_results['bt_ending_bankroll'], 
                             mode='lines', name='Bookie Strategy Bankroll'), row=1, col=1)

    # Calculate and highlight Max Drawdown for main strategy
    cummax = np.maximum.accumulate(main_results['bt_ending_bankroll'])
    drawdown = (cummax - main_results['bt_ending_bankroll']) / cummax
    max_drawdown = np.max(drawdown)
    max_drawdown_end = np.argmax(drawdown)
    max_drawdown_start = np.argmax(main_results['bt_ending_bankroll'][:max_drawdown_end])

    fig.add_vrect(
        x0=main_results['bt_date_column'].iloc[max_drawdown_start],
        x1=main_results['bt_date_column'].iloc[max_drawdown_end],
        fillcolor="rgba(255, 0, 0, 0.2)", opacity=0.5,
        layer="below", line_width=0,
        annotation_text=f"Max Drawdown: {max_drawdown:.2%}",
        annotation_position="top left",
        row=1, col=1
    )

    # Plot ROI for each bet for main strategy
    fig.add_trace(go.Scatter(x=main_results['bt_date_column'], 
                             y=main_results['bt_roi'], 
                             mode='markers', name='Main Strategy ROI', 
                             marker=dict(size=5, opacity=0.5)), row=2, col=1)

    # Plot ROI for each bet for bookie strategy
    fig.add_trace(go.Scatter(x=bookie_results['bt_date_column'], 
                             y=bookie_results['bt_roi'], 
                             mode='markers', name='Bookie Strategy ROI', 
                             marker=dict(size=5, opacity=0.5)), row=3, col=1)

    # Add win/loss markers for main strategy
    wins = main_results[main_results['bt_win']]
    losses = main_results[~main_results['bt_win']]

    fig.add_trace(go.Scatter(x=wins['bt_date_column'], y=wins['bt_ending_bankroll'],
                             mode='markers', marker=dict(color='green', symbol='triangle-up', size=8),
                             name='Main Strategy Wins'), row=1, col=1)
    fig.add_trace(go.Scatter(x=losses['bt_date_column'], y=losses['bt_ending_bankroll'],
                             mode='markers', marker=dict(color='red', symbol='triangle-down', size=8),
                             name='Main Strategy Losses'), row=1, col=1)

    # Calculate metrics
    metrics = calculate_all_metrics(main_results)

    # Create annotations for all metrics
    annotations = []
    for i, (key, value) in enumerate(metrics.items()):
        if isinstance(value, float):
            text = f"{key}: {value:.2f}"
        elif isinstance(value, (int, np.integer)):
            text = f"{key}: {value}"
        else:
            text = f"{key}: {value}"
        
        annotations.append(dict(
            xref='paper', yref='paper',
            x=1.02, y=0.02 + i*0.025,  # Start from bottom and go up
            xanchor='left', yanchor='bottom',
            text=text,
            font=dict(size=8),
            showarrow=False
        ))

    # Update layout to include annotations
    fig.update_layout(
        annotations=annotations,
        margin=dict(r=250, b=100)  # Increase bottom margin to accommodate metrics
    )

    # Update layout
    fig.update_layout(
        title='Backtest Results: Main Strategy vs Bookie Strategy',
        xaxis_title='Date',
        height=1000,
        showlegend=True
    )

    fig.update_yaxes(title_text="Bankroll", row=1, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=2, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=3, col=1)

    # Add zero lines to ROI plots
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    return fig

def plot_odds_histogram(backtest: Any, num_bins: Optional[int] = None) -> go.Figure:
    """
    Create a histogram plot of the odds distribution for the main strategy,
    splitting each bin into winning and losing bets, and adding dotted lines for break-even win rates.
    
    Args:
        backtest (Any): The backtest object containing detailed results.
        num_bins (Optional[int]): The number of bins to use for the histogram. If None, auto-binning is used.
    
    Returns:
        go.Figure: A Plotly figure object containing the odds histogram.
    """
    # Extract odds and outcomes from the main strategy
    odds = backtest.detailed_results['bt_odds']
    wins = backtest.detailed_results['bt_win']

    # Determine bin edges
    if num_bins is None:
        num_bins = int(np.sqrt(len(odds)))  # Square root rule for number of bins
    bin_edges = np.logspace(np.log10(odds.min()), np.log10(odds.max()), num_bins + 1)

    # Create histograms for winning and losing bets
    win_hist, _ = np.histogram(odds[wins], bins=bin_edges)
    lose_hist, _ = np.histogram(odds[~wins], bins=bin_edges)

    # Calculate bin centers for x-axis
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Calculate break-even win rates for each bin
    break_even_win_rates = 1 / bin_centers

    # Create the figure
    fig = go.Figure()

    # Add winning bets histogram
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=win_hist,
        name='Winning Bets',
        marker_color='green',
        opacity=0.7
    ))

    # Add losing bets histogram
    fig.add_trace(go.Bar(
        x=bin_centers,
        y=lose_hist,
        name='Losing Bets',
        marker_color='red',
        opacity=0.7
    ))

    # Add break-even win rate lines for each bin
    for i in range(len(bin_centers)):
        total_bets = win_hist[i] + lose_hist[i]
        if total_bets > 0:
            break_even_height = total_bets * break_even_win_rates[i]
            fig.add_shape(
                type="line",
                x0=bin_edges[i],
                y0=break_even_height,
                x1=bin_edges[i+1],
                y1=break_even_height,
                line=dict(color="blue", width=2, dash="dot"),
            )

    # Add a dummy trace for the legend
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='lines',
        line=dict(color='blue', width=2, dash='dot'),
        name='Break-Even Win Rate'
    ))

    # Update layout
    fig.update_layout(
        title='Distribution of played Odds and Break-Even Win Rates',
        xaxis_title='Odds',
        yaxis_title='Frequency',
        barmode='stack',
        bargap=0.1,
        xaxis=dict(
            tickmode='array',
            tickvals=bin_edges,
            ticktext=[f'{x:.2f}' for x in bin_edges],
            tickangle=45
        )
    )

    # Add a vertical line for the average odds
    avg_odds = odds.mean()
    fig.add_vline(x=avg_odds, line_dash="dash", line_color="blue", 
                  annotation_text=f"Avg: {avg_odds:.2f}", 
                  annotation_position="top right")

    # Add median line
    median_odds = odds.median()
    fig.add_vline(x=median_odds, line_dash="dot", line_color="purple", 
                  annotation_text=f"Median: {median_odds:.2f}", 
                  annotation_position="top left")

    return fig
