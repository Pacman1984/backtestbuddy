# Getting Started

This page takes you from an empty notebook to a first backtest with metrics and a chart.

## 1. Install

```bash
pip install backtestbuddy
```

## 2. Prepare your data

BacktestBuddy reads one pandas DataFrame with **one row per match**. It needs at least:

| Column | Content | Example |
|---|---|---|
| Date | When the match was played | `2023-08-12` |
| Odds, one column per outcome | **Decimal** odds you could bet at | `1.85`, `2.05` |
| Outcome | Index of the outcome that happened, **0-based in the order of the odds columns** | `0` = first odds column won |

Depending on the backtest you also need your model's output:

- `PredictionBacktest`: a prediction column (predicted outcome index) and, for Kelly or value betting, one probability column per outcome.
- `ModelBacktest`: feature columns; the model computes predictions and probabilities itself.

A small two-way example (outcome 0 = player A, 1 = player B):

```python
import pandas as pd

matches = pd.DataFrame({
    "date": pd.to_datetime(["2023-08-12", "2023-08-12", "2023-08-13"]),
    "odds_a": [1.85, 1.40, 2.60],
    "odds_b": [2.05, 3.10, 1.55],
    "winner": [0, 1, 1],          # 0 = A won, 1 = B won
    "prob_a": [0.58, 0.70, 0.35],  # your model's probabilities
    "prob_b": [0.42, 0.30, 0.65],
    "prediction": [0, 0, 1],
})
```

Odds in American or fractional format must be converted to decimal first. Rows are sorted by date; matches on the same date keep their input order.

## 3. Pick a backtest and a strategy

| You have… | Use |
|---|---|
| Predictions and probabilities computed elsewhere | `PredictionBacktest` |
| Features and an sklearn-style model (`fit`, `predict`, `predict_proba`) | `ModelBacktest` (walk-forward with `TimeSeriesSplit` by default) |

The strategy decides which outcome to bet on and how much. Start with one of these:

- `FixedStake(10)`: 10 currency units per bet; `FixedStake(0.02)` is 2% of the current bankroll.
- `KellyCriterion(downscaling=0.5)`: half Kelly from your probabilities.
- `ValueBet(min_ev=0.03, stake=10)`: only bets when `p * odds - 1 > 0.03`.

See the [Strategies Module](strategies-module.md) for all options.

## 4. Run the backtest

The next block builds a larger synthetic dataset so the results are meaningful. Replace it with your own DataFrame.

```python
import numpy as np
import backtestbuddy as btb

rng = np.random.default_rng(0)
n = 500
p_a = rng.uniform(0.3, 0.7, n)
data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "odds_a": np.round(0.95 / p_a, 2),
    "odds_b": np.round(0.95 / (1 - p_a), 2),
    "winner": (rng.uniform(size=n) > p_a).astype(int),
})
model_p = np.clip(p_a + rng.normal(0, 0.05, n), 0.01, 0.99)
data["prob_a"] = model_p
data["prob_b"] = 1 - model_p
data["prediction"] = (model_p < 0.5).astype(int)

backtest = btb.PredictionBacktest(
    data=data,
    odds_columns=["odds_a", "odds_b"],
    outcome_column="winner",
    date_column="date",
    prediction_column="prediction",
    model_prob_columns=["prob_a", "prob_b"],
    strategy=btb.KellyCriterion(downscaling=0.5),
    initial_bankroll=1000,
)
backtest.run()
```

## 5. Read the results

**Per match**: `detailed_results` has one row per match with the stake, the outcome bet on, profit, and the bankroll before and after. Rows with `bt_stake == 0` are matches the strategy skipped.

```python
results = backtest.detailed_results
results[["bt_date_column", "bt_bet_on", "bt_odds", "bt_stake", "bt_profit", "bt_ending_bankroll"]].head()
```

**Summary**: `calculate_metrics()` returns a dictionary of every metric. A few to look at first:

```python
metrics = backtest.calculate_metrics()
for key in ["Total Bets", "ROI [%]", "Yield [%]", "Max Drawdown [%]", "Sharpe Ratio (365.25) [-]"]:
    print(f"{key:28s} {metrics[key]:.2f}")
```

- **ROI** is bankroll growth; **Yield** is profit per unit staked. Yield is the usual way to compare betting strategies.
- **Max Drawdown** is the worst peak-to-trough fall of the bankroll.

The [Metrics Module](metrics-module.md) explains every key.

**Chart**: `plot()` shows bankroll, drawdown, per-bet ROI, and stake size with the metrics table next to it.

```python
backtest.plot()
```

## 6. Next steps

- Compare your model with the market: `Brier Score (all games)` vs `Brier Score (market, all games)` in the metrics.
- Pass `closing_odds_columns=` to get closing line value (see [Backtest Module](backtest-module.md#parameters)).
- Check robustness with `backtest.stress_test(haircuts=[0, 0.01, 0.02, 0.03])`.
- Copy a complete script from [Examples](examples.md).
