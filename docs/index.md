# BacktestBuddy

BacktestBuddy backtests sports-betting strategies in Python. You give it a pandas DataFrame of historical matches with odds, results, and your model output; it simulates the bankroll bet by bet and reports risk metrics and interactive charts.

It works with any data source. Every input is a column name you pass in, so the same code runs on football, tennis, or any other market with decimal odds.

## What you can do

- **Backtest predictions you already have** (`PredictionBacktest`) or **train a model walk-forward** (`ModelBacktest`).
- **Size bets** with fixed stakes, Kelly, value betting, unit plans, or an odds-band filter, and protect the bankroll with risk limits.
- **Measure results**: ROI, Yield, CAGR, Sharpe, Sortino, Calmar, drawdowns, calibration of your probabilities, and a market baseline.
- **Check your prices** with closing line value (CLV) against any later or sharper line.
- **Remove the bookmaker margin** with four methods.
- **Stress-test** the strategy with worse odds.
- **Plot** the bankroll, drawdown, calibration, and odds distribution.

## Where to start

| If you want to… | Read |
|---|---|
| Run a first backtest on your own data | [Getting Started](getting-started.md) |
| Understand how the simulation works | [Core Concepts](core-concepts.md) |
| Look up a class, parameter, or results column | [Backtest Module](backtest-module.md) |
| Choose or write a staking strategy | [Strategies Module](strategies-module.md) |
| Know exactly how a metric is computed | [Metrics Module](metrics-module.md) |
| Remove the margin from odds | [Odds Module](odds-module.md) |
| Customize or export charts | [Plots Module](plots-module.md) |
| Copy a complete script | [Examples](examples.md) |

## Installation

```bash
pip install backtestbuddy
```

Requires Python 3.9 or newer.

## Scope

Sports betting is the supported use case. Stock and crypto backtests are ideas for the future, not features.
