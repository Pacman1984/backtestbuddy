# Backtest Module

The `backtest` module is the core of the BacktestBuddy framework. It contains the main classes for running backtests on sports betting strategies.

## Classes

### `BaseBacktest`

The `BaseBacktest` class is an abstract base class that provides a framework for implementing different backtesting approaches. It should be subclassed to create specific backtesting strategies.

#### Key Methods

- `__init__(self, data, odds_columns, outcome_column, date_column, initial_bankroll, model, strategy, cv_schema, closing_odds_columns=None, odds_haircut=0.0, margin_method="multiplicative")`: Initializes the backtest with the given parameters (see [Data and parameters](#data-and-parameters)).
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
    - 'bt_odd_{i}': Odds of each outcome as bet (after `odds_haircut`).
    - 'bt_closing_odd_{i}': Reference odds of each outcome (only with `closing_odds_columns`).
    - Additional columns from the original dataset.
- `get_bookie_results(self)`: Returns a DataFrame containing the results of the bookie strategy simulation.
- `calculate_metrics(self)`: Calculates performance metrics based on the backtest results.
- `plot(self, x_axis="bet", show_bookie=True)`: Generates and displays a plot of the backtest results.
- `stress_test(self, haircuts=(0, 0.01, 0.02, 0.03), metrics=...)`: Re-runs the backtest with each odds haircut and returns one row of metrics per haircut. The backtest's own results stay unchanged.
- `plot_calibration(self, n_bins, scope="placed")`: Reliability diagram of model probabilities (`"placed"` or `"all"` games).
- `plot_odds_distribution(self, num_bins)`: Generates a histogram plot of the odds distribution.

### `ModelBacktest`

The `ModelBacktest` class is used for backtesting strategies that use a predictive model. It implements the backtesting logic for strategies where a model is used to make predictions before applying the betting strategy.

#### Key Methods (ModelBacktest)

- `__init__(self, data, odds_columns, outcome_column, date_column, model, initial_bankroll, strategy, cv_schema, feature_columns=None, closing_odds_columns=None, odds_haircut=0.0, margin_method="multiplicative")`: Initializes the ModelBacktest with the given parameters, including a predictive model. Set `feature_columns` explicitly to keep IDs, predictions, and closing odds out of the model.
- `run(self)`: Implements the backtesting logic for a model-based strategy. It uses the model to make predictions, applies the betting strategy, and populates `self.detailed_results` and `self.bookie_results` with the outcomes.

### `PredictionBacktest`

The `PredictionBacktest` class is used for backtesting strategies that use pre-computed predictions. It implements the backtesting logic for strategies where predictions are already available in the dataset, and only the betting strategy needs to be applied.

#### Key Methods (PredictionBacktest)

- `__init__(self, data, odds_columns, outcome_column, date_column, prediction_column, initial_bankroll, strategy, model_prob_columns, closing_odds_columns=None, odds_haircut=0.0, margin_method="multiplicative")`: Initializes the PredictionBacktest with the given parameters, including a column for pre-computed predictions and optional model probability columns.
- `run(self)`: Implements the backtesting logic for a strategy based on pre-computed predictions. It processes the entire dataset sequentially, simulating a betting strategy and populates `self.detailed_results` and `self.bookie_results` with the outcomes.

## Data and parameters

BacktestBuddy does not assume a data provider or sport. Every input is a column name you pass in:

| Parameter | Meaning |
|---|---|
| `odds_columns` | Decimal odds you bet into, one column per outcome. Their order defines the outcome indices 0, 1, 2, … |
| `outcome_column` | Actual outcome as an index into `odds_columns` (0-based). |
| `date_column` | Event date or timestamp; rows are sorted by it (stable for ties). |
| `prediction_column` / `model_prob_columns` | Pre-computed predictions and probabilities (`PredictionBacktest`), probabilities aligned with `odds_columns`. |
| `feature_columns` | Model inputs (`ModelBacktest`); defaults to every column except date and outcome. |
| `closing_odds_columns` | Optional reference odds per outcome, same order as `odds_columns`. Usually closing odds, but any later or sharper line works: e.g. opening odds as `odds_columns` and a late live feed as reference. Enables CLV metrics. |
| `odds_haircut` | Lower every bet odd by this fraction, e.g. `0.02`. Strategy and settlement use the lowered odds (floored at 1.0); model features, probabilities, and closing odds do not change. |
| `margin_method` | Margin removal for no-vig CLV and the market baseline: `multiplicative`, `additive`, `power`, `shin` (see the Odds Module). |

Assumptions to check when you bring your own data:

- Odds are **decimal**. Convert American or fractional odds first.
- The outcome index matches the column order of `odds_columns` (and of `predict_proba` for `ModelBacktest`; sklearn sorts classes).
- A bet either wins or loses. Voids and pushes (e.g. tennis retirements at some bookmakers) are not modeled; drop or settle them beforehand.

### Stress test

```python
backtest.run()
backtest.stress_test(haircuts=[0, 0.01, 0.02, 0.03])
```

Returns a table indexed by `odds_haircut` with ROI, Yield, CAGR, Max Drawdown, Sharpe, Total Bets, and final bankroll. Use it when the odds in your data are better than what you can actually get, for example opening odds with low limits.

## Example Usage

```python
from backtestbuddy.strategies.base import FixedStake
from backtestbuddy.backtest.backtest import PredictionBacktest


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
