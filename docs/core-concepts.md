# Core Concepts

BacktestBuddy is designed to be a flexible and extensible framework for backtesting various strategies. Here are the core concepts:

## Sport Backtest

The `backtest` module consists right now of the `sport_backtest.py` file, which contains the main classes for running sports betting backtests. This includes:

- `BaseBacktest`: The base class for all sports backtests.
- `ModelBacktest`: A class for backtesting machine learning models in sports.
- `PredictionBacktest`: A class for backtesting based on precomputed sports predictions.

## Sport Metrics

The `metrics` module currently only contains the `sport_metrics.py` file, which provides functionality for calculating various performance metrics after a backtest.

- ROI (Return on Investment): Measures the profitability of your strategy relative to the initial investment. (Percentage)
- Total Profit: The total amount of money gained or lost during the backtest period. (Currency)
- Bankroll Final: The final value of your bankroll at the end of the backtest period. (Currency)
- Bankroll Peak: The highest value your bankroll reached during the backtest period. (Currency)
- Bankroll Valley: The lowest value your bankroll reached during the backtest period. (Currency)
- Sharpe Ratio: Annualized mean of compounded period returns over sample std (`ddof=1`); risk-free rate is 0. Keys name the scale: `(365.25)`, `(252)`, `(obs/year)`. Function default `P` is 365.25. (Ratio)
- Sortino Ratio: Same period returns and the same three named `P` scales as Sharpe. Volatility is downside deviation below target 0 (zeros included for upside periods). `inf` if no downside and positive mean excess. (Ratio)
- Calmar Ratio: Geometric annual return / |max drawdown on the compounded return curve|. Not the same drawdown as Max Drawdown. (Ratio)
- Max Drawdown: Largest peak-to-trough decline on the bankroll, reported as a positive percent. Duration is last peak to trough in placed bets. (Percentage)
- Max Drawdown Duration: Inclusive bet count from the last peak before the trough to the trough. (Number of bets)
- Win Rate: Winning placed bets / placed bets. A placed bet has `bt_stake > 0` and `bt_bet_on != -1`. (Percentage)
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

## Sport Strategies

The `strategies` module currently only contains the `sport_strategies.py` file, which provides functionality for defining betting strategies and any specific strategies you implement.

- `BaseStrategy`: The base class for all strategies.
- `FixedStake`: A strategy that bets a fixed dollar amount, or a fixed percentage of the **current** bankroll (`stake < 1`).
- `KellyCriterion`: A strategy that bets a fraction of the bankroll based on the Kelly Criterion.

## Sport Plots

The `plots` module currently only contains the `sport_plots.py` file, which provides functionality for plotting backtest results and any specific plots you implement.

- `plot_backtest`: Creates a plot of the backtest results, including the bankroll over time, ROI for each bet, and stake percentage.
- `plot_odds_histogram`: Creates a histogram plot of the odds distribution, splitting each bin into winning and losing bets, and adding dotted lines for break-even win rates.
