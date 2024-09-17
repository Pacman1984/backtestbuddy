# Examples of Backtesting with BacktestBuddy

## Basic Usage Example (Fixed Stake Strategy + PredictionBacktest)

Here's an example of how to use BacktestBuddy to backtest a fixed stake strategy using pre-computed predictions:

```python
# Import the backtestbuddy package
import backtestbuddy as btb
import pandas as pd
import numpy as np

# Create a dummy dataset
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
n_samples = len(dates)
data = pd.DataFrame({
    'date': dates,
    'odds_team_a': np.random.uniform(1.5, 3, n_samples),
    'odds_team_b': np.random.uniform(1.5, 3, n_samples),
    'actual_winner': np.random.randint(0, 2, n_samples),
    'model_predictions': np.random.randint(0, 2, n_samples)
})

# Define a fixed stake strategy with a stake of $10
strategy = btb.FixedStake(stake=10)

# Initialize the backtester with the strategy and initial bankroll
backtest = btb.PredictionBacktest(
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

# Get the detailed results
detailed_results = backtest.detailed_results
print(detailed_results)

# Get aggregated metrics
aggregated_metrics = backtest.calculate_metrics()
print(aggregated_metrics)

# Plot the results
backtest.plot()

# Further plotting options
backtest.plot_odds_distribution()
```

## Advanced Usage Example (ModelBacktest + Half-Kelly Strategy)

```python
# Import the backtestbuddy package
import backtestbuddy as btb
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScale

# Create a dummy dataset
np.random.seed(42)
dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
n_samples = len(dates)
data = pd.DataFrame({
    'date': dates,
    'odds_team_a': np.random.uniform(1.5, 3, n_samples),
    'odds_team_b': np.random.uniform(1.5, 3, n_samples),
    'actual_winner': np.random.randint(0, 2, n_samples),
})

# Define Strategy
strategy = btb.KellyCriterion(downscaling=0.5) # Half-Kelly

# Define Model
model = make_pipeline(StandardScaler(), LogisticRegression())

# Initialize the backtester with the strategy and initial bankroll
backtest = btb.ModelBacktest(
    data=data,
    date_column='date',
    odds_columns=['odds_team_a', 'odds_team_b'],
    outcome_column='actual_winner',
    initial_bankroll=1000,
    strategy=strategy,
    cv_schema=TimeSeriesSplit(n_splits=5),
    model=model
)

# Run the backtest
backtest.run()

# Get the detailed results
detailed_results = backtest.detailed_results
print(detailed_results)

# Get aggregated metrics
aggregated_metrics = backtest.calculate_metrics()
print(aggregated_metrics)

# Plot the results
backtest.plot()

# Further plotting options
backtest.plot_odds_distribution()
```
