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

def calculate_sharpe_ratio(detailed_results: pd.DataFrame, risk_free_rate: float = 0.02, periods: int = 252) -> float:
    """
    Calculate the Sharpe Ratio.
    Assumes returns are daily and risk-free rate is annual.
    """
    returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']
    excess_returns = returns - risk_free_rate / periods
    return np.sqrt(periods) * excess_returns.mean() / excess_returns.std()

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
    Calculate the win rate.
    """
    total_bets = len(detailed_results)
    winning_bets = detailed_results['bt_win'].sum()
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
    Calculate the average stake.
    """
    return detailed_results['bt_stake'].mean()

def calculate_sortino_ratio(detailed_results: pd.DataFrame, risk_free_rate: float = 0.02, periods: int = 252) -> float:
    """
    Calculate the Sortino Ratio.
    Assumes returns are daily and risk-free rate is annual.
    """
    returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']
    excess_returns = returns - risk_free_rate / periods
    downside_returns = excess_returns[excess_returns < 0]
    return np.sqrt(periods) * excess_returns.mean() / downside_returns.std()

def calculate_calmar_ratio(detailed_results: pd.DataFrame, periods: int = 252) -> float:
    """
    Calculate the Calmar Ratio.
    Assumes returns are daily.
    """
    returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']
    annual_return = returns.mean() * periods
    max_drawdown = calculate_max_drawdown(detailed_results)
    return annual_return / abs(max_drawdown)

def calculate_drawdowns(detailed_results: pd.DataFrame) -> Tuple[float, float, float, int, float]:
    """
    Calculate drawdowns and their durations.
    
    Args:
        detailed_results (pd.DataFrame): DataFrame containing detailed results of the backtest.
    
    Returns:
        Tuple[float, float, float, int, float]: 
            Average drawdown, 
            Average drawdown duration,
            Maximum drawdown,
            Maximum drawdown duration,
            Median drawdown duration
    """
    equity_curve = detailed_results['bt_ending_bankroll']
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    
    # Find drawdown periods
    is_drawdown = drawdown < 0
    drawdown_start = is_drawdown.ne(is_drawdown.shift()).cumsum()
    
    drawdowns = []
    durations = []
    
    for _, group in drawdown.groupby(drawdown_start):
        if (group < 0).any():
            max_drawdown = group.min()
            duration = len(group)
            drawdowns.append(abs(max_drawdown))
            durations.append(duration)
    
    avg_drawdown = np.mean(drawdowns) if drawdowns else 0
    avg_duration = np.mean(durations) if durations else 0
    max_drawdown = max(drawdowns) if drawdowns else 0
    max_duration = max(durations) if durations else 0
    median_duration = np.median(durations) if durations else 0
    
    return avg_drawdown, avg_duration, max_drawdown, max_duration, median_duration

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

def calculate_all_metrics(detailed_results: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate all metrics and return them in a dictionary.
    """
    # Calculate backtesting period information
    start_date = detailed_results['bt_date_column'].min()
    end_date = detailed_results['bt_date_column'].max()
    duration = end_date - start_date

    # Calculate Bankroll Final, Peak, and Valley
    bankroll_final = detailed_results['bt_ending_bankroll'].iloc[-1]
    bankroll_peak = detailed_results['bt_ending_bankroll'].max()
    bankroll_valley = detailed_results['bt_ending_bankroll'].min()

    # Calculate drawdowns
    avg_drawdown, avg_drawdown_duration, max_drawdown, max_drawdown_duration, median_drawdown_duration = calculate_drawdowns(detailed_results)

    # Calculate best and worst bets
    best_bet, worst_bet = calculate_best_worst_bets(detailed_results)

    # Calculate highest winning and losing odds
    highest_winning_odds, highest_losing_odds = calculate_highest_odds(detailed_results)

    metrics = {
        # Backtest Period Information
        'Backtest Start Date': start_date,
        'Backtest End Date': end_date,
        'Backtest Duration': duration,

        # Overall Performance
        'ROI [%]': calculate_roi(detailed_results) * 100,
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
        'Avg. Drawdown [%]': avg_drawdown * 100,
        'Max. Drawdown Duration [bets]': max_drawdown_duration,
        'Avg. Drawdown Duration [bets]': avg_drawdown_duration,
        'Median Drawdown Duration [bets]': median_drawdown_duration,

        # Betting Performance
        'Win Rate [%]': calculate_win_rate(detailed_results) * 100,
        'Average Odds [-]': calculate_average_odds(detailed_results),
        'Highest Winning Odds [-]': highest_winning_odds,
        'Highest Losing Odds [-]': highest_losing_odds,
        'Average Stake [$]': calculate_average_stake(detailed_results),
        'Best Bet [$]': best_bet,
        'Worst Bet [$]': worst_bet,

        # Additional Information
        'Total Bets': len(detailed_results),
    }

    return metrics