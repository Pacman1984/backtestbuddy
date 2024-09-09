# BacktestBuddy Documentation

Welcome to the official documentation for BacktestBuddy!

> **Note:** The following features are currently under development and not yet implemented:

BacktestBuddy is an open-source Python package for backtesting various trading and betting strategies, including:

- Sports betting (✅ Done)
- Stocks trading (🏗️ In progress)
- Cryptocurrencies trading (🔮 Planned)

## Features

- Flexible backtesting framework for different types of strategies
- Support for sports betting, stock trading, and cryptocurrency trading
- Easy-to-use API for implementing custom strategies
- Comprehensive performance metrics and visualization tools

## Table of Contents

- [Core Concepts](core-concepts.md)
- [Backtest Module](backtest-module.md)
- [Metrics Module](metrics-module.md)
- [Strategies Module](strategies-module.md)
- [Plots Module](plots-module.md)
- [Examples](examples.md)

## Installation

To install BacktestBuddy, run the following command:

```bash
pip install backtestbuddy
```

## Quick Example

Here's a simple example of how to use BacktestBuddy for a sports betting backtest:

```python
from src.strategies.sport_strategies import FixedStake
from src.backtest.sport_backtest import PredictionBacktest

# Prepare your data

data = ... # Your historical betting data

# Define your strategy

strategy = FixedStake(stake=10)

# Create a backtest object
backtest = PredictionBacktest(
    data=data,
    date_column='date',
    odds_columns=['odds_team_a', 'odds_team_b'],
    outcome_column='actual_winner',
    prediction_column='model_predictions',
    initial_bankroll=1000,
    strategy=strategy,
)

# Run the backtest
backtest.run()

# Show the detailed match results
backtest.detailed_results
# or
bookie_results = backtest.get_detailed_results()
bookie_results

# Show Bookie Results
bookie_results = backtest.get_bookie_results()
bookie_results

# Calculate Performance Metrics
backtest.calculate_metrics()

# Visualize the Backtest Results
backtest.plot()

# Visualize the Odds Distribution etc...
backtest.plot_odds_distribution()

```
