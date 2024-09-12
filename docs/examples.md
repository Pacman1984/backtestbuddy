# Examples of Backtesting with BacktestBuddy

## Basic Usage Example

Here's an example of how to use BacktestBuddy to backtest a simple moving average crossover strategy:

```python
import backtestbuddy as btb

# Load historical data for the S&P 500 index
data = btb.load_data('sp500.csv')

# Define a moving average crossover strategy
strategy = btb.MovingAverageCrossoverStrategy(data)

# Backtest the strategy
results = strategy.backtest()

# Print the results
print(results)
```

## Advanced Usage Example

Here is an example of how to use BacktestBuddy to backtest a machine learning strategy that uses a random forest classifier to predict whether the price of the S&P 500 index will go up or down:

```python
import backtestbuddy as btb
from sklearn.ensemble import RandomForestClassifier

# Load historical data for the S&P 500 index
data = btb.load_data('sp500.csv')

# Define a random forest classifier
clf = RandomForestClassifier()

# Define a strategy that uses the classifier to make predictions
strategy = btb.MachineLearningStrategy(data, clf)

# Backtest the strategy
results = strategy.backtest()

# Print the results
print(results)
```
