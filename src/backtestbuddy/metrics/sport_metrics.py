import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

def calculate_roi(detailed_results: pd.DataFrame) -> float:
    """
    Calculate Return on Investment (ROI).
    """
    initial_bankroll = detailed_results['bt_starting_bankroll'].iloc[0]
    final_bankroll = detailed_results['bt_ending_bankroll'].iloc[-1]
    return (final_bankroll - initial_bankroll) / initial_bankroll

def calculate_sharpe_ratio(detailed_results: pd.DataFrame, return_period: int = 1, output_period: int = 252) -> float:
    """
    Calculate the Sharpe Ratio for sports betting.
    Assumes returns are daily by default and output is annualized.
    
    The Sharpe Ratio measures the risk-adjusted return of an investment. It is calculated as the 
    ratio of the average excess return (return over the risk-free rate) to the standard deviation 
    of the excess return. A higher Sharpe Ratio indicates better risk-adjusted performance.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
        return_period (int): The period over which returns are calculated (default is 1 for daily).
        output_period (int): The period over which the Sharpe Ratio is annualized (default is 252 for yearly).
    
    Returns:
        float: The Sharpe Ratio.
    
    Examples:
        >>> data = {
        ...     'bt_date_column': ['2023-01-01', '2023-01-02', '2023-01-03'],
        ...     'bt_profit': [100, -50, 200],
        ...     'bt_starting_bankroll': [1000, 1000, 1000]
        ... }
        >>> df = pd.DataFrame(data)
        >>> calculate_sharpe_ratio(df)
        12.24744871391589
        
        >>> calculate_sharpe_ratio(df, return_period=7, output_period=52)
        3.4641016151377544
    """
    # Convert the date column to datetime and set it as the index
    detailed_results['bt_date_column'] = pd.to_datetime(detailed_results['bt_date_column'])
    detailed_results = detailed_results.set_index('bt_date_column')
    
    # Calculate returns based on the profit and starting bankroll
    returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']
    
    # Resample returns based on the return_period (e.g., daily, weekly)
    returns = returns.resample(f'{return_period}D').sum()
    
    # Annualize the mean return by multiplying with output_period
    annualized_mean_return = returns.mean() * output_period
    
    # Annualize the standard deviation by multiplying with sqrt(output_period)
    annualized_std_return = returns.std() * np.sqrt(output_period)
    
    # Add check for zero standard deviation
    if annualized_std_return == 0:
        return 0.0  # or float('inf') if you prefer
    
    # Calculate and return the Sharpe Ratio
    return annualized_mean_return / annualized_std_return

def calculate_max_drawdown(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the Maximum Drawdown.
    """
    equity_curve = detailed_results['bt_ending_bankroll']
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    return drawdown.min()

def calculate_win_rate(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the win rate, considering only rows where a bet was placed.
    """
    bet_placed = detailed_results[(detailed_results['bt_stake'] > 0) | (detailed_results['bt_bet_on'] != -1)]
    total_bets = len(bet_placed)
    winning_bets = bet_placed['bt_win'].sum()
    return winning_bets / total_bets if total_bets else 0

def calculate_average_odds(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the average odds.
    """
    return detailed_results['bt_odds'].mean()

def calculate_total_profit(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the total profit.
    """
    return detailed_results['bt_profit'].sum()

def calculate_average_stake(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the average stake, considering only non-zero stakes.
    """
    non_zero_stakes = detailed_results['bt_stake'][detailed_results['bt_stake'] > 0]
    return non_zero_stakes.mean() if len(non_zero_stakes) > 0 else 0

def calculate_sortino_ratio(detailed_results: pd.DataFrame, return_period: int = 1, output_period: int = 252) -> float:
    """
    Calculate the Sortino Ratio for sports betting.
    Assumes returns are daily by default and output is annualized.
    
    The Sortino Ratio is a variation of the Sharpe Ratio that differentiates harmful volatility 
    from total overall volatility by using the standard deviation of negative asset returns, 
    called downside deviation. A higher Sortino Ratio indicates better risk-adjusted performance 
    with a focus on downside risk.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
        return_period (int): The period over which returns are calculated (default is 1 for daily).
        output_period (int): The period over which the Sortino Ratio is annualized (default is 252 for yearly).
    
    Returns:
        float: The Sortino Ratio.
    
    Examples:
        >>> data = {
        ...     'bt_date_column': ['2023-01-01', '2023-01-02', '2023-01-03'],
        ...     'bt_profit': [100, -50, 200],
        ...     'bt_starting_bankroll': [1000, 1000, 1000]
        ... }
        >>> df = pd.DataFrame(data)
        >>> calculate_sortino_ratio(df)
        18.0
        
        >>> calculate_sortino_ratio(df, return_period=7, output_period=52)
        5.196152422706632
    """
    # Convert the date column to datetime and set it as the index
    detailed_results['bt_date_column'] = pd.to_datetime(detailed_results['bt_date_column'])
    detailed_results = detailed_results.set_index('bt_date_column')
    
    # Calculate returns based on the profit and starting bankroll
    returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']
    
    # Resample returns based on the return_period (e.g., daily, weekly)
    returns = returns.resample(f'{return_period}D').sum()
    
    # Calculate downside deviation (only negative returns)
    downside_deviation = returns[returns < 0].std() * np.sqrt(output_period)
    
    # Annualize the mean return by multiplying with output_period
    annualized_mean_return = returns.mean() * output_period
    
    # Calculate and return the Sortino Ratio
    return annualized_mean_return / downside_deviation

def calculate_calmar_ratio(detailed_results: pd.DataFrame, return_period: int = 1, output_period: int = 252) -> float:
    """
    Calculate the Calmar Ratio for sports betting.
    Assumes returns are daily by default and output is annualized.
    
    The Calmar Ratio measures the risk-adjusted return of an investment by comparing the average 
    annual compounded rate of return and the maximum drawdown risk. A higher Calmar Ratio indicates 
    better risk-adjusted performance with a focus on drawdown risk.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
        return_period (int): The period over which returns are calculated (default is 1 for daily).
        output_period (int): The period over which the Calmar Ratio is annualized (default is 252 for yearly).
    
    Returns:
        float: The Calmar Ratio.
    
    Examples:
        >>> data = {
        ...     'bt_date_column': ['2023-01-01', '2023-01-02', '2023-01-03'],
        ...     'bt_profit': [100, -50, 200],
        ...     'bt_starting_bankroll': [1000, 1000, 1000]
        ... }
        >>> df = pd.DataFrame(data)
        >>> calculate_calmar_ratio(df)
        12.24744871391589
        
        >>> calculate_calmar_ratio(df, return_period=7, output_period=52)
        3.4641016151377544
    """
    # Convert the date column to datetime and set it as the index
    detailed_results['bt_date_column'] = pd.to_datetime(detailed_results['bt_date_column'])
    detailed_results = detailed_results.set_index('bt_date_column')
    
    # Calculate returns based on the profit and starting bankroll
    returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']
    
    # Resample returns based on the return_period (e.g., daily, weekly)
    returns = returns.resample(f'{return_period}D').sum()
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()
    
    # Add check for zero maximum drawdown
    if max_drawdown == 0:
        return 0.0  # or float('inf') if you prefer
    
    # Annualize the mean return by multiplying with output_period
    annualized_mean_return = returns.mean() * output_period
    
    # Calculate and return the Calmar Ratio
    return annualized_mean_return / abs(max_drawdown)

def calculate_drawdowns(detailed_results: pd.DataFrame) -> Tuple[float, int]:
    """
    Calculate drawdowns and their durations.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
    
    Returns:
        Tuple[float, float]: 
            Maximum drawdown in percentage,
            Maximum drawdown duration in bets
    """
    equity_curve = detailed_results['bt_ending_bankroll'].values
    if len(equity_curve) == 0:
        return 0.0, 0

    cummax = np.maximum.accumulate(equity_curve)
    drawdown = (cummax - equity_curve) / cummax
    max_drawdown = np.max(drawdown)

    if max_drawdown == 0:
        return 0.0, 0

    # Find the end of the maximum drawdown period
    max_drawdown_end = np.argmax(drawdown)
    # Find the start of the maximum drawdown period
    max_drawdown_start = np.argmax(equity_curve[:max_drawdown_end])
    max_duration = max_drawdown_end - max_drawdown_start + 1  # +1 to include both start and end

    return max_drawdown, max_duration

def calculate_best_worst_bets(detailed_results: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate the best and worst bets in terms of profit.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
    
    Returns:
        Tuple[float, float]: Best bet profit, Worst bet profit
    """
    best_bet = detailed_results['bt_profit'].max()
    worst_bet = detailed_results['bt_profit'].min()
    return best_bet, worst_bet

def calculate_highest_odds(detailed_results: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculate the highest winning odds and highest losing odds.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
    
    Returns:
        Tuple[float, float]: Highest winning odds, Highest losing odds
    """
    winning_bets = detailed_results[detailed_results['bt_win'] == True]
    losing_bets = detailed_results[detailed_results['bt_win'] == False]
    
    highest_winning_odds = winning_bets['bt_odds'].max() if not winning_bets.empty else 0
    highest_losing_odds = losing_bets['bt_odds'].max() if not losing_bets.empty else 0
    
    return highest_winning_odds, highest_losing_odds

def calculate_avg_roi_per_bet(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the average ROI per bet.
    Only considers rows where a bet was placed.
    """
    bet_placed = detailed_results[(detailed_results['bt_stake'] > 0) | (detailed_results['bt_bet_on'] != -1)]
    roi_per_bet = bet_placed['bt_profit'] / bet_placed['bt_stake'] * 100
    return roi_per_bet.mean() if len(roi_per_bet) > 0 else 0

def calculate_all_metrics(detailed_results: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all metrics and return them in a dictionary.
    """
    # Filter out rows where no bet was placed
    bet_placed = detailed_results[(detailed_results['bt_stake'] > 0) | (detailed_results['bt_bet_on'] != -1)]

    # Calculate backtesting period information
    start_date = detailed_results['bt_date_column'].min()
    end_date = detailed_results['bt_date_column'].max()
    duration = end_date - start_date

    # Calculate Bankroll Final, Peak, and Valley
    bankroll_final = detailed_results['bt_ending_bankroll'].iloc[-1]
    bankroll_peak = detailed_results['bt_ending_bankroll'].max()
    bankroll_valley = detailed_results['bt_ending_bankroll'].min()

    # Calculate drawdowns
    max_drawdown, max_drawdown_duration = calculate_drawdowns(bet_placed)

    # Calculate best and worst bets
    best_bet, worst_bet = calculate_best_worst_bets(bet_placed)

    # Calculate highest winning and losing odds
    highest_winning_odds, highest_losing_odds = calculate_highest_odds(bet_placed)

    metrics = {
        # Backtest Period Information
        'Backtest Start Date': start_date,
        'Backtest End Date': end_date,
        'Backtest Duration': duration,

        # Overall Performance
        'ROI [%]': calculate_roi(detailed_results) * 100,
        'Avg. ROI per Bet [%]': calculate_avg_roi_per_bet(detailed_results),
        'Total Profit [$]': calculate_total_profit(detailed_results),
        'Bankroll Final [$]': bankroll_final,
        'Bankroll Peak [$]': bankroll_peak,
        'Bankroll Valley [$]': bankroll_valley,

        # Risk-Adjusted Performance
        'Sharpe Ratio [-]': calculate_sharpe_ratio(detailed_results),
        'Sortino Ratio [-]': calculate_sortino_ratio(detailed_results),
        'Calmar Ratio [-]': calculate_calmar_ratio(detailed_results),

        # Drawdown Analysis
        'Max Drawdown [%]': max_drawdown * 100,
        'Max. Drawdown Duration [bets]': max_drawdown_duration,

        # Betting Performance
        'Win Rate [%]': calculate_win_rate(bet_placed) * 100,
        'Average Odds [-]': calculate_average_odds(bet_placed),
        'Highest Winning Odds [-]': highest_winning_odds,
        'Highest Losing Odds [-]': highest_losing_odds,
        'Average Stake [$]': calculate_average_stake(bet_placed),
        'Best Bet [$]': best_bet,
        'Worst Bet [$]': worst_bet,

        # Additional Information
        'Total Bets': len(bet_placed),
        'Total Opportunities': len(detailed_results),
        'Bet Frequency [%]': (len(bet_placed) / len(detailed_results)) * 100,
    }

    return metrics