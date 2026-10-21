# BacktestBuddy

[![Tests](https://github.com/Pacman1984/backtestbuddy/actions/workflows/tests.yml/badge.svg)](https://github.com/Pacman1984/backtestbuddy/actions/workflows/tests.yml)

Backtest sports-betting strategies in Python: a sequential bankroll simulation over your historical odds, with staking strategies, risk metrics, closing line value, and interactive Plotly charts.

Works with any data source: you pass a pandas DataFrame and tell BacktestBuddy which columns hold the odds, the outcome, the date, and your model output.

**Python >= 3.9 · MIT license**

## Features

- **Two ways to backtest**: `PredictionBacktest` for predictions you already have, `ModelBacktest` to train an sklearn-style model walk-forward.
- **Strategies**: fixed stake, Kelly criterion, value betting, unit staking, odds-band filter.
- **Metrics**: ROI, Yield, CAGR, Sharpe, Sortino, Calmar, drawdown, calibration (Brier, log loss, ECE), and a market baseline.
- **Closing line value**: compare the odds you took with any later or sharper line.
- **Margin removal**: multiplicative, additive, power, and Shin methods.
- **Odds stress test**: re-run with 1%, 2%, 3% worse odds to see how robust the edge is.
- **Charts**: bankroll and drawdown, reliability diagram, odds distribution.

## Installation

```bash
pip install backtestbuddy
```

## Quick example

```python
import numpy as np
import pandas as pd
import backtestbuddy as btb

# One row per match: date, decimal odds per outcome, actual outcome (0-based), model output
rng = np.random.default_rng(0)
n = 500
p_home = rng.uniform(0.3, 0.7, n)
data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "odds_home": np.round(0.95 / p_home, 2),
    "odds_away": np.round(0.95 / (1 - p_home), 2),
    "winner": (rng.uniform(size=n) > p_home).astype(int),  # 0 = home, 1 = away
})
model_p = np.clip(p_home + rng.normal(0, 0.05, n), 0.01, 0.99)
data["prob_home"] = model_p
data["prob_away"] = 1 - model_p
data["prediction"] = (model_p < 0.5).astype(int)

backtest = btb.PredictionBacktest(
    data=data,
    odds_columns=["odds_home", "odds_away"],
    outcome_column="winner",
    date_column="date",
    prediction_column="prediction",
    model_prob_columns=["prob_home", "prob_away"],
    strategy=btb.KellyCriterion(downscaling=0.5),
    initial_bankroll=1000,
)
backtest.run()

metrics = backtest.calculate_metrics()
print(metrics["ROI [%]"], metrics["Yield [%]"], metrics["Max Drawdown [%]"])
backtest.plot()  # interactive Plotly chart
```

## Documentation

- [Getting Started](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/getting-started.md): data format and a first backtest, step by step
- [Core Concepts](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/core-concepts.md): how the simulation works
- [Backtest Module](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/backtest-module.md): classes, parameters, results columns, stress test
- [Strategies Module](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/strategies-module.md)
- [Metrics Module](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/metrics-module.md): every metric with its formula
- [Odds Module](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/odds-module.md): margin removal
- [Plots Module](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/plots-module.md)
- [Examples](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/examples.md)
- [Changelog](https://github.com/Pacman1984/backtestbuddy/blob/master/docs/changelog.md)

Browse the docs locally with `pip install backtestbuddy[docs]` and `mkdocs serve`.

## Development

The project uses [uv](https://docs.astral.sh/uv/). From a clone of the repository:

```bash
uv run --extra test pytest          # full test suite, including every docs example
uv run --extra docs mkdocs serve    # docs at http://127.0.0.1:8000
```

GitHub Actions runs the tests on Python 3.9 to 3.13 and builds the docs for every pull request.
