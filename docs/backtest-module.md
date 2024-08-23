# Backtest Module

The `backtest` module is the core of the BacktestBuddy framework. It contains the main classes for running backtests.

## Classes

### `BaseBacktest`

The `BaseBacktest` class is the base class for all backtests. It provides common functionality that other backtest classes inherit.

### `ModelBacktest`

The `ModelBacktest` class is used for backtesting machine learning models. It takes a model, data, and a strategy as input and performs the backtest.

### `PredictionBacktest`

The `PredictionBacktest` class is used for backtesting based on precomputed predictions. It takes data with predictions and a strategy as input and performs the backtest.

## Example Usage

```python
from src.strategies.base import FixedStake
from src.backtest.backtest import PredictionBacktest
from sklearn.model_selection import TimeSeriesSplit


# Load your data
data = ...

# Define your strategy
strategy = FixedStake(stake=10)

# Create a backtest object
backtest = PredictionBacktest(
    data=data,
    odds_columns=['odds_team_a', 'odds_team_b'],
    outcome_column='actual_winner',
    prediction_column='model_predictions',
    initial_bankroll=1000,
    strategy=strategy,
    cv_schema=TimeSeriesSplit(n_splits=5)
)

# Run the backtest
backtest.run()

# Show the results
detailed_results = backtest.get_detailed_results()
detailed_results
```