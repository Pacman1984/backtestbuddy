import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Any
import numpy as np

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

    # Update layout
    fig.update_layout(title='Backtest Results: Main Strategy vs Bookie Strategy',
                      xaxis_title='Date',
                      height=1000,
                      showlegend=True)

    fig.update_yaxes(title_text="Bankroll", row=1, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=2, col=1)
    fig.update_yaxes(title_text="ROI (%)", row=3, col=1)

    # Add zero lines to ROI plots
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=3, col=1)

    return fig
