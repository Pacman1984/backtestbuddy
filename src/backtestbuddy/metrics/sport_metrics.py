import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple

def calculate_roi(detailed_results: pd.DataFrame) -> float:
    """
    Calculate Return on Investment (ROI).
    Returns 0.0 for empty DataFrames.
    """
    if len(detailed_results) == 0:
        return 0.0
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

def calculate_avg_roi_per_bet_micro(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the micro-average ROI per bet (averaging individual bet ROIs).
    
    For each bet, calculates ROI as: (profit / stake) * 100 to get percentage return.
    Then takes the mean across all bets to get the average ROI per bet.
    This gives equal weight to each bet's individual ROI.
    
    Only considers rows where either:
    1. A stake was placed (bt_stake > 0), or 
    2. A bet was made (bt_bet_on != -1)
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest
                                       with columns bt_stake, bt_profit, bt_bet_on
    
    Returns:
        float: The micro-average percentage ROI per bet. Returns 0 if no bets were placed.
        
    Example:
        If stakes were $100 each with profits of $20, -$50, $30:
        ROIs would be: 20%, -50%, 30%
        Micro-average ROI per bet = 0%
    """
    bet_placed = detailed_results[(detailed_results['bt_stake'] > 0) | (detailed_results['bt_bet_on'] != -1)].copy()
    if bet_placed.empty:
        return 0
    # Calculate ROI for each bet: (profit / stake) * 100
    roi_per_bet = bet_placed['bt_profit'] / bet_placed['bt_stake'] * 100
    return roi_per_bet.mean() if len(roi_per_bet) > 0 else 0

def calculate_avg_roi_per_bet_macro(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the macro-average ROI per bet (total ROI divided by number of bets).
    
    Unlike calculate_avg_roi_per_bet_micro which averages individual bet ROIs,
    this calculates total ROI first and then divides by number of bets.
    This gives equal weight to the overall return rather than individual bet returns.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest
                                       with columns bt_stake, bt_profit, bt_bet_on
    
    Returns:
        float: Total ROI divided by number of bets. Returns 0 if no bets were placed.
        
    Example:
        If initial bankroll was $1000, final bankroll is $1200 (20% total ROI),
        and 10 bets were placed, macro-average ROI per bet would be 2% (20% / 10 bets)
    """
    bet_placed = detailed_results[(detailed_results['bt_stake'] > 0) | (detailed_results['bt_bet_on'] != -1)]
    if bet_placed.empty:
        return 0
    
    total_roi = calculate_roi(detailed_results)
    num_bets = len(bet_placed)
    
    return (total_roi / num_bets) * 100 if num_bets > 0 else 0

def calculate_avg_roi_per_year_micro(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the micro-average ROI per year by averaging individual yearly ROIs.
    
    This function:
    1. Groups data by year
    2. Calculates ROI for each year individually
    3. Takes the average of these yearly ROIs
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
    
    Returns:
        float: The micro-average annual ROI as a percentage (average of individual yearly ROIs).
        Returns 0.0 for empty DataFrames or when no bets are placed.
    """
    if len(detailed_results) == 0:
        return 0.0
        
    # Convert date column to datetime if it's not already
    dates = pd.to_datetime(detailed_results['bt_date_column'])
    
    # Add year column for grouping
    detailed_results = detailed_results.copy()
    detailed_results['year'] = dates.dt.year
    
    # Group by year and calculate ROI for each year
    yearly_rois = []
    for year, group in detailed_results.groupby('year'):
        initial_bankroll = group['bt_starting_bankroll'].iloc[0]
        final_bankroll = group['bt_ending_bankroll'].iloc[-1]
        yearly_roi = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100
        yearly_rois.append(yearly_roi)
    
    # Return average of yearly ROIs
    return np.mean(yearly_rois) if yearly_rois else 0.0

def calculate_avg_roi_per_year_macro(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the macro-average ROI per year by dividing total ROI by number of years.
    
    This function:
    1. Calculates total ROI for the entire period
    2. Divides by the number of years to get average annual ROI
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
    
    Returns:
        float: The macro-average annual ROI as a percentage (total ROI divided by years).
        Returns 0.0 for empty DataFrames or when duration is 0.
    """
    if len(detailed_results) == 0:
        return 0.0
        
    # Calculate total ROI for the entire period
    total_roi = calculate_roi(detailed_results)
    
    # Convert date column to datetime if it's not already
    dates = pd.to_datetime(detailed_results['bt_date_column'])
    
    # Calculate the number of years (including partial years)
    years = (dates.max() - dates.min()).days / 365.25
    
    # Avoid division by zero
    if years == 0:
        return 0.0
    
    # Return the annual ROI as a percentage
    return (total_roi / years) * 100

def calculate_risk_adjusted_annual_roi(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the Risk-Adjusted Annual ROI.
    This metric divides the average yearly ROI (macro) by the maximum drawdown,
    providing a measure of return per unit of downside risk.
    A higher value indicates better risk-adjusted annual performance.
    
    Returns:
        float: Risk-Adjusted Annual ROI or 0 if max_drawdown is 0
    """
    avg_yearly_roi = calculate_avg_roi_per_year_macro(detailed_results)
    max_drawdown = calculate_max_drawdown(detailed_results)
    
    # Avoid division by zero and handle edge cases
    if max_drawdown == 0:
        return avg_yearly_roi  # Return the ROI itself if there's no drawdown
    
    return avg_yearly_roi / abs(max_drawdown)

def calculate_cagr(detailed_results: pd.DataFrame) -> float:
    """
    Calculate the Compound Annual Growth Rate (CAGR).
    
    CAGR is the geometric mean of annual returns, calculated as:
    CAGR = (End Value / Start Value)^(1/years) - 1
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
    
    Returns:
        float: The CAGR as a percentage. Returns 0.0 for empty DataFrames or when duration is 0.
    """
    if len(detailed_results) == 0:
        return 0.0
        
    # Get initial and final values
    initial_value = detailed_results['bt_starting_bankroll'].iloc[0]
    final_value = detailed_results['bt_ending_bankroll'].iloc[-1]
    
    # Calculate time period in years
    dates = pd.to_datetime(detailed_results['bt_date_column'])
    years = (dates.max() - dates.min()).days / 365.25
    
    # Avoid division by zero
    if years == 0 or initial_value == 0:
        return 0.0
    
    # Calculate CAGR
    cagr = (pow(final_value / initial_value, 1/years) - 1) * 100
    return cagr

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
        'Avg. ROI per Bet [%] (micro)': calculate_avg_roi_per_bet_micro(detailed_results),
        'Avg. ROI per Bet [%] (macro)': calculate_avg_roi_per_bet_macro(detailed_results),
        'Avg. ROI per Year [%] (micro)': calculate_avg_roi_per_year_micro(detailed_results),
        'Avg. ROI per Year [%] (macro)': calculate_avg_roi_per_year_macro(detailed_results),
        'CAGR [%]': calculate_cagr(detailed_results),
        'Risk-Adjusted Annual ROI [-]': calculate_risk_adjusted_annual_roi(detailed_results),
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