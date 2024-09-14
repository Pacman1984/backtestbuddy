from typing import Any, Optional

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtestbuddy.metrics.sport_metrics import calculate_all_metrics


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
    # Filter main_results to only include games where a bet was placed
    main_results = backtest.detailed_results
    bet_placed = main_results[(main_results['bt_stake'] > 0) & (main_results['bt_bet_on'] != -1)]

    # Create a game index for bet_placed
    bet_placed['game_index'] = range(1, len(bet_placed) + 1)

    # Convert date column to string to avoid dtype issues
    date_strings = bet_placed['bt_date_column'].astype(str)

    # Create subplots: one for bankroll, one for ROI, one for stake percentage
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.05, row_heights=[0.5, 0.25, 0.25],
                        subplot_titles=("Bankroll Over Time", "ROI", "Stake Percentage"))

    # Plot bankroll over time for main strategy
    fig.add_trace(go.Scatter(
        x=bet_placed['game_index'], 
        y=bet_placed['bt_ending_bankroll'], 
        name='Main Strategy',
        line=dict(color='blue'),
        hovertemplate='Game: %{x}<br>' +
                      'Date: %{customdata[0]}<br>' +
                      'Starting Bankroll: $%{customdata[1]:.2f}<br>' +
                      'Ending Bankroll: $%{y:.2f}<br>' +
                      'Stake: $%{customdata[2]:.2f}<br>' +
                      'Stake Percentage: %{customdata[3]:.2f}%' +
                      '<extra></extra>',
        customdata=np.column_stack((date_strings, bet_placed['bt_starting_bankroll'], 
                                    bet_placed['bt_stake'], bet_placed['bt_stake'] / bet_placed['bt_starting_bankroll'] * 100))
    ), row=1, col=1)

    # Plot ROI for each bet for main strategy
    fig.add_trace(go.Scatter(x=bet_placed['game_index'], 
                             y=bet_placed['bt_roi'], 
                             mode='markers', name='ROI', 
                             marker=dict(size=5, opacity=0.5)), row=2, col=1)

    # Add win/loss markers for main strategy
    wins = bet_placed[bet_placed['bt_win'] == True]
    losses = bet_placed[bet_placed['bt_win'] == False]

    fig.add_trace(go.Scatter(x=wins['game_index'], y=wins['bt_ending_bankroll'],
                             mode='markers', marker=dict(color='green', symbol='triangle-up', size=8),
                             name='Wins'), row=1, col=1)
    fig.add_trace(go.Scatter(x=losses['game_index'], y=losses['bt_ending_bankroll'],
                             mode='markers', marker=dict(color='red', symbol='triangle-down', size=8),
                             name='Losses'), row=1, col=1)

    # Plot stake percentage as bars
    fig.add_trace(go.Bar(x=bet_placed['game_index'], 
                         y=bet_placed['bt_stake'] / bet_placed['bt_starting_bankroll'] * 100,
                         name='Stake Percentage',
                         marker_color='rgba(0, 128, 128, 0.7)'), row=3, col=1)

    # Update layout
    fig.update_layout(
        title='Backtest Results (Bets Placed Only)',
        xaxis_title='Bet Number',
        height=1000,
        showlegend=True,
        hovermode='x unified',
        margin=dict(r=100, t=100, b=100, l=100),
        hoverlabel=dict(bgcolor="white", font_size=12),
    )

    # Calculate and highlight Max Drawdown for main strategy
    cummax = np.maximum.accumulate(bet_placed['bt_ending_bankroll'])
    drawdown = (cummax - bet_placed['bt_ending_bankroll']) / cummax
    max_drawdown = np.max(drawdown)
    max_drawdown_end = np.argmax(drawdown)
    max_drawdown_start = np.argmax(bet_placed['bt_ending_bankroll'][:max_drawdown_end])
    max_drawdown_length = max_drawdown_end - max_drawdown_start + 1  # +1 to include both start and end

    fig.add_vrect(
        x0=bet_placed['game_index'].iloc[max_drawdown_start],
        x1=bet_placed['game_index'].iloc[max_drawdown_end],
        fillcolor="rgba(255, 0, 0, 0.2)", opacity=0.5,
        layer="below", line_width=0,
        annotation_text=f"Max Drawdown: {max_drawdown:.2%}<br>Length: {max_drawdown_length} bets",
        annotation_position="top left",
        row=1, col=1
    )

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
    fig.update_layout(annotations=annotations)

    # Update y-axis labels
    fig.update_yaxes(title_text="Bankroll", row=1, col=1)
    fig.update_yaxes(title_text="ROI", row=2, col=1)
    fig.update_yaxes(title_text="Stake %", row=3, col=1)

    # Add zero line to ROI plot
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # Update x-axis to show bet numbers
    fig.update_xaxes(title_text="Bet Number", row=1, col=1)
    fig.update_xaxes(title_text="Bet Number", row=2, col=1)
    fig.update_xaxes(title_text="Bet Number", row=3, col=1)

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
    detailed_results = backtest.detailed_results
    
    # Find the odds column (it might be named differently)
    odds_column = next((col for col in detailed_results.columns if 'odds' in col.lower()), None)
    if odds_column is None:
        raise ValueError("Could not find an odds column in the detailed results.")
    
    odds = detailed_results[odds_column]
    wins = detailed_results['bt_win'] > 0  # Assume positive values indicate wins

    # Remove NaN values and ensure we only consider placed bets
    valid_mask = ~(odds.isna() | wins.isna()) & (detailed_results['bt_bet_on'] != -1)
    odds = odds[valid_mask]
    wins = wins[valid_mask]

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
                  annotation_position="top left")

    # Add median line
    median_odds = odds.median()
    fig.add_vline(x=median_odds, line_dash="dot", line_color="purple", 
                  annotation_text=f"Median: {median_odds:.2f}", 
                  annotation_position="bottom right")

    # Adjust the y-axis range to create more space for annotations
    y_max = max(win_hist.max(), lose_hist.max())
    fig.update_yaxes(range=[0, y_max * 1.2])

    # Add some padding to the top of the plot
    fig.update_layout(margin=dict(t=100))

    return fig
