# Backtest Module

The `backtest` module is the core of the BacktestBuddy framework. It contains the main classes for running backtests on sports betting strategies.

## Classes

### `BaseBacktest`

The `BaseBacktest` class is an abstract base class that provides a framework for implementing different backtesting approaches. It should be subclassed to create specific backtesting strategies.

#### Key Methods

- `__init__(self, data, odds_columns, outcome_column, date_column, initial_bankroll, model, strategy, cv_schema)`: Initializes the backtest with the given parameters.
- `run(self)`: Abstract method that should be implemented by subclasses to perform the actual backtesting logic.
- `get_detailed_results(self)`: Returns a DataFrame containing detailed results for each bet.
  - `detailed_results` is a DataFrame with the following columns:
    - 'bt_index': Index of the current data point.
    - 'bt_fold': Current fold number.
    - 'bt_predicted_outcome': Predicted outcome index.
    - 'bt_actual_outcome': Actual outcome index.
    - 'bt_starting_bankroll': Bankroll before the bet.
    - 'bt_ending_bankroll': Bankroll after the bet.
    - 'bt_stake': Amount staked on the bet.
    - 'bt_potential_return': Potential return if the bet wins.
    - 'bt_win': Boolean indicating if the bet won.
    - 'bt_profit': Profit (positive) or loss (negative) from the bet.
    - 'bt_roi': Return on Investment as a percentage.
    - 'bt_odds': The odds for the predicted outcome.
    - 'bt_date_column': The date of the bet.
    - Additional columns from the original dataset.
- `get_bookie_results(self)`: Returns a DataFrame containing the results of the bookie strategy simulation.
- `calculate_metrics(self)`: Calculates performance metrics based on the backtest results.
- `plot(self)`: Generates and displays a plot of the backtest results.
- `plot_odds_distribution(self, num_bins)`: Generates a histogram plot of the odds distribution.

### `ModelBacktest`

The `ModelBacktest` class is used for backtesting strategies that use a predictive model. It implements the backtesting logic for strategies where a model is used to make predictions before applying the betting strategy.

#### Key Methods (ModelBacktest)

- `__init__(self, data, odds_columns, outcome_column, date_column, model, initial_bankroll, strategy, cv_schema)`: Initializes the ModelBacktest with the given parameters, including a predictive model.
- `run(self)`: Implements the backtesting logic for a model-based strategy. It uses the model to make predictions, applies the betting strategy, and populates `self.detailed_results` and `self.bookie_results` with the outcomes.

### `PredictionBacktest`

The `PredictionBacktest` class is used for backtesting strategies that use pre-computed predictions. It implements the backtesting logic for strategies where predictions are already available in the dataset, and only the betting strategy needs to be applied.

#### Key Methods (PredictionBacktest)

- `__init__(self, data, odds_columns, outcome_column, date_column, prediction_column, initial_bankroll, strategy, model_prob_columns)`: Initializes the PredictionBacktest with the given parameters, including a column for pre-computed predictions and optional model probability columns.
- `run(self)`: Implements the backtesting logic for a strategy based on pre-computed predictions. It processes the entire dataset sequentially, simulating a betting strategy and populates `self.detailed_results` and `self.bookie_results` with the outcomes.

## Example Usage

```python
from src.strategies.base import FixedStake
from src.backtest.backtest import PredictionBacktest


# Load your data
data = ...

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

# Show the results
detailed_results = backtest.get_detailed_results()
detailed_results
```
