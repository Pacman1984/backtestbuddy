# BacktestBuddy Documentation

Welcome to the official documentation for BacktestBuddy!

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
from backtestbuddy.strategies.sport_strategies import FixedStake
from backtestbuddy.backtest.sport_backtest import PredictionBacktest

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
    model_prob_columns=['prob_team_a', 'prob_team_b'] # Needed for PredictionBacktest in combination with Kelly Strategy. Not needed for ModelBacktest in combination with Kelly Strategy, because the model probabiliies will be calculated by the model or if the Strategy does not require model probabilities, like Fixed Stake.
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

## Changelog

### Version 0.1.10 (2025-01-23)
- Fixed Risk-Adjusted Annual ROI calculation to use macro ROI and handle negative values correctly
- Kelly fractions are now always returned in bet details regardless of min_kelly and min_prob filters
- Added Compound Annual Growth Rate (CAGR) metric 'CAGR [%]'

### Version 0.1.9 (2025-01-19)
- Added micro/macro perspectives for annual ROI:
  - 'Avg. ROI per Year [%]' → 'Avg. ROI per Year [%] (micro)' - averages individual yearly ROIs
  - Added 'Avg. ROI per Year [%] (macro)' - total ROI divided by number of years

### Version 0.1.8 (2025-01-19)
- Improved ROI metrics clarity:
  - Renamed ROI metrics for better understanding:
    - 'Avg. ROI per Bet [%]' → 'Avg. ROI per Bet [%] (micro)' - averages individual bet ROIs
    - Added 'Avg. ROI per Bet [%] (macro)' - total ROI divided by number of bets

### Version 0.1.7 (2025-01-19)
- Fixed ROI calculations:
  - Modified `calculate_avg_roi_per_year` to calculate annual ROI by dividing total ROI by number of years
  - Improved handling of empty DataFrames in ROI calculations
  - Added better handling of edge cases (same day, zero years) in ROI calculations
  - Added proper docstrings and test cases for ROI calculations

### Version 0.1.6 (2025-01-19)
- Fixed ROI calculations:
  - Average ROI per Bet now correctly calculates per-bet returns
  - Average ROI per Year now uses total profits and stakes per year
  - Risk-Adjusted Annual ROI improved to handle edge cases

### Version 0.1.5 (2025-01-19)
- Fixed pandas SettingWithCopyWarning in metrics calculation

### Version 0.1.4 (2025-01-19)
- Added new metrics:
  - Average ROI per Year [%]
  - Risk-Adjusted Annual ROI [-] (Avg. ROI per Year / Max Drawdown)

### Version 0.1.3 (2025-01-18)
- Added new metric:
  - Average ROI per Bet [%]

### Version 0.1.2 (2023-11-28)
- Added bookie simulation functionality
- Enhanced plotting capabilities
- Improved documentation

### Version 0.1.1 (2023-11-18)
- Initial release with sports betting functionality
- Basic metrics and visualization tools
- Core backtesting framework
