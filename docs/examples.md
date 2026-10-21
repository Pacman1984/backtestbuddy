# Examples

Complete scripts you can copy and run. Each one builds a synthetic dataset first; replace it with your own DataFrame.

## 1. Fixed stake on pre-computed predictions

```python
import numpy as np
import pandas as pd
import backtestbuddy as btb

rng = np.random.default_rng(42)
n = 365
p_a = rng.uniform(0.3, 0.7, n)
data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "odds_team_a": np.round(0.95 / p_a, 2),
    "odds_team_b": np.round(0.95 / (1 - p_a), 2),
    "actual_winner": (rng.uniform(size=n) > p_a).astype(int),  # 0 = team A, 1 = team B
    "model_predictions": rng.integers(0, 2, n),
})

backtest = btb.PredictionBacktest(
    data=data,
    date_column="date",
    odds_columns=["odds_team_a", "odds_team_b"],
    outcome_column="actual_winner",
    prediction_column="model_predictions",
    initial_bankroll=1000,
    strategy=btb.FixedStake(amount=10),  # 10 units on the predicted team
)
backtest.run()

print(backtest.detailed_results.head())
metrics = backtest.calculate_metrics()
print(metrics["Yield [%]"], metrics["Win Rate [%]"])

backtest.plot()
backtest.plot_odds_distribution().show()
```

## 2. Train a model walk-forward with half Kelly

`ModelBacktest` trains on each training fold and bets on the following test fold. Set `feature_columns` so the model only sees what you intend.

```python
import numpy as np
import pandas as pd
import backtestbuddy as btb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(42)
n = 600
strength_diff = rng.normal(0, 1, n)                 # a feature your model can learn from
p_a = 1 / (1 + np.exp(-strength_diff))
data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "strength_diff": strength_diff,
    "odds_team_a": np.round(0.95 / np.clip(p_a + rng.normal(0, 0.05, n), 0.05, 0.95), 2),
    "actual_winner": (rng.uniform(size=n) > p_a).astype(int),
})
data["odds_team_b"] = np.round(1 / (1.05 - 1 / data["odds_team_a"]), 2)

backtest = btb.ModelBacktest(
    data=data,
    date_column="date",
    odds_columns=["odds_team_a", "odds_team_b"],
    outcome_column="actual_winner",
    model=make_pipeline(StandardScaler(), LogisticRegression()),
    feature_columns=["strength_diff"],
    cv_schema=TimeSeriesSplit(n_splits=5),
    strategy=btb.KellyCriterion(downscaling=0.5, max_bet=0.05),  # half Kelly, at most 5% per bet
    initial_bankroll=1000,
)
backtest.run()

metrics = backtest.calculate_metrics()
print(metrics["CAGR [%]"], metrics["Max Drawdown [%]"])
print("Model vs market Brier:", metrics["Brier Score (all games) [-]"],
      metrics["Brier Score (market, all games) [-]"])
backtest.plot_calibration(scope="all").show()
```

## 3. Closing line value and a stress test

Bet at the opening odds, compare with the closing odds, and check how much edge survives worse prices.

```python
import numpy as np
import pandas as pd
import backtestbuddy as btb

rng = np.random.default_rng(7)
n = 800
true_p = rng.uniform(0.2, 0.8, n)
opening_p = np.clip(true_p + rng.normal(0, 0.04, n), 0.05, 0.95)  # noisy early line
model_p = np.clip(true_p + rng.normal(0, 0.03, n), 0.02, 0.98)     # model close to the truth

data = pd.DataFrame({
    "date": pd.date_range("2022-01-01", periods=n, freq="D"),
    "open_a": np.round(1 / (opening_p * 1.04), 2),
    "open_b": np.round(1 / ((1 - opening_p) * 1.04), 2),
    "close_a": np.round(1 / (true_p * 1.025), 2),                    # sharp closing line
    "close_b": np.round(1 / ((1 - true_p) * 1.025), 2),
    "winner": (rng.uniform(size=n) > true_p).astype(int),
    "prob_a": model_p,
    "prob_b": 1 - model_p,
    "prediction": (model_p < 0.5).astype(int),
})

backtest = btb.PredictionBacktest(
    data,
    odds_columns=["open_a", "open_b"],
    outcome_column="winner",
    date_column="date",
    prediction_column="prediction",
    model_prob_columns=["prob_a", "prob_b"],
    closing_odds_columns=["close_a", "close_b"],
    margin_method="power",
    strategy=btb.ValueBet(min_ev=0.03, fraction=0.01),  # 1% of the bankroll on value bets
)
backtest.run()

metrics = backtest.calculate_metrics()
for key in ["Total Bets", "Yield [%]", "CLV [%]", "CLV (no-vig) [%]", "Beat Closing Line [%]"]:
    print(f"{key:24s} {metrics[key]:.2f}")

print(backtest.stress_test(haircuts=[0, 0.01, 0.02, 0.03]))
```

## 4. Remove the margin from odds

```python
import backtestbuddy as btb

odds = [[1.25, 4.20], [2.10, 3.40, 3.60]]   # a tennis match and a football match
for market in odds:
    for method in ["multiplicative", "power", "shin"]:
        print(market, method, btb.remove_margin(market, method=method).round(4))
```
