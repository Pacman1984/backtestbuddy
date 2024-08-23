# Core Concepts

BacktestBuddy is designed to be a flexible and extensible framework for backtesting various strategies. Here are the core concepts:

## Backtest

The `backtest` module contains the main classes for running backtests. This includes:

- `BaseBacktest`: The base class for all backtests.
- `ModelBacktest`: A class for backtesting machine learning models.
- `PredictionBacktest`: A class for backtesting based on precomputed predictions.

## Metrics

The `metrics` module provides functionality for calculating various performance metrics after a backtest.

## Strategies

The `strategies` module contains the base class for defining betting strategies and any specific strategies you implement.