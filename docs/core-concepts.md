# Core Concepts

BacktestBuddy is designed to be a flexible and extensible framework for backtesting various strategies. Here are the core concepts:

## Backtest

The `backtest` module contains the main classes for running backtests. This includes:

- `BaseBacktest`: The base class for all backtests.
- `ModelBacktest`: A class for backtesting machine learning models.
- `PredictionBacktest`: A class for backtesting based on precomputed predictions.

## Metrics

The `metrics` module provides functionality for calculating various performance metrics after a backtest.

- ROI (Return on Investment): Measures the profitability of your strategy relative to the initial investment. (Percentage)
- Total Profit: The total amount of money gained or lost during the backtest period. (Currency)
- Bankroll Final: The final value of your bankroll at the end of the backtest period. (Currency)
- Bankroll Peak: The highest value your bankroll reached during the backtest period. (Currency)
- Bankroll Valley: The lowest value your bankroll reached during the backtest period. (Currency)
- Sharpe Ratio: Measures the risk-adjusted return of the betting strategy. (Ratio)
- Sortino Ratio: Similar to Sharpe Ratio, but only considers downside risk (negative returns). (Ratio)
- Calmar Ratio: Measures the risk-adjusted return relative to maximum drawdown. (Ratio)
- Max Drawdown: The largest peak-to-trough decline in the bankroll. (Percentage)
- Average Drawdown: The average of all drawdowns during the backtest period. (Percentage)
- Max Drawdown Duration: The longest period (in number of bets) that the bankroll was in a drawdown state. (Number of bets)
- Average Drawdown Duration: The average duration of all drawdowns during the backtest period. (Number of bets)
- Median Drawdown Duration: The median duration of all drawdowns during the backtest period. (Number of bets)
- Win Rate: The percentage of bets that resulted in a profit. (Percentage)
- Average Odds: The average odds of all bets placed. (Decimal odds)
- Highest Winning Odds: The highest odds of a winning bet. (Decimal odds)
- Highest Losing Odds: The highest odds of a losing bet. (Decimal odds)
- Average Stake: The average amount staked per bet. (Currency)
- Best Bet: The highest profit achieved from a single bet. (Currency)
- Worst Bet: The largest loss incurred from a single bet. (Currency)
- Total Bets: The total number of bets placed during the backtest period. (Count)
- Total Opportunities: The total number of betting opportunities during the backtest period. (Count)
- Bet Frequency: The percentage of bets placed out of the total opportunities. (Percentage)
- Backtest Duration: The time period covered by the backtest. (Time period, e.g., days, months, years)

## Strategies

The `strategies` module contains the base class for defining betting strategies and any specific strategies you implement.

- `BaseStrategy`: The base class for all strategies.
- `FixedStake`: A strategy that bets a fixed amount per bet.
- `KellyCriterion`: A strategy that bets a fraction of the bankroll based on the Kelly Criterion.

## Plots

The `plots` module contains the base class for plotting backtest results and any specific plots you implement.

- `plot_backtest`: Creates a plot of the backtest results, including the bankroll over time, ROI for each bet, and stake percentage.
- `plot_odds_histogram`: Creates a histogram plot of the odds distribution, splitting each bin into winning and losing bets, and adding dotted lines for break-even win rates.
