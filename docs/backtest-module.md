# Backtest Module

`backtestbuddy.backtest` runs the simulation. Use `PredictionBacktest` when your predictions already exist as columns, and `ModelBacktest` to train a model walk-forward. Both share the base class `BaseBacktest`. See [Core Concepts](core-concepts.md) for how the loop works.

## Parameters

### Shared by both classes

| Parameter | Default | Meaning |
|---|---|---|
| `data` | required | DataFrame, one row per match. Sorted by `date_column` (stable for ties). |
| `odds_columns` | required | Decimal odds you bet into, one column per outcome. Their order defines outcome indices 0, 1, 2, … |
| `outcome_column` | required | Index of the outcome that happened, 0-based in the order of `odds_columns`. |
| `date_column` | required | Match date or timestamp. |
| `initial_bankroll` | `1000.0` | Starting bankroll. |
| `strategy` | `FixedStake(fraction=0.01)` | Any strategy from the [Strategies Module](strategies-module.md). |
| `closing_odds_columns` | `None` | Reference odds per outcome, same order as `odds_columns`, for CLV. Usually closing odds, but any later or sharper line works, e.g. opening odds as `odds_columns` and a late feed as the reference. |
| `odds_haircut` | `0.0` | Lower every bet odd by this fraction (`0.02` = 2% worse). The strategy decides and bets settle at the lowered odds, floored at 1.0. Model features, probabilities, and closing odds are unchanged. Must be in `[0, 1)`. |
| `margin_method` | `"multiplicative"` | Margin removal for no-vig CLV and the market baseline: `multiplicative`, `additive`, `power`, or `shin`. See the [Odds Module](odds-module.md). |
| `settle_simultaneous` | `False` | Treat rows with the same date value as one slot of simultaneous matches: all bets of the slot are sized from the bankroll at its start, their total stake is capped at that bankroll, and they settle together. See [Simultaneous matches](#simultaneous-matches). |

### `PredictionBacktest` only

| Parameter | Default | Meaning |
|---|---|---|
| `prediction_column` | required | Predicted outcome index. |
| `model_prob_columns` | `None` | Model probability per outcome, same order as `odds_columns`. Required for `KellyCriterion` and `ValueBet`. When given, `FixedStake` and `UnitStake` bet on the highest probability instead of the prediction. |

### `ModelBacktest` only

| Parameter | Default | Meaning |
|---|---|---|
| `model` | required | Object with `fit(X, y)`, `predict(X)`, and `predict_proba(X)`. The `predict_proba` columns must follow the order of `odds_columns` (sklearn sorts the classes). |
| `cv_schema` | `TimeSeriesSplit(n_splits=5)` | Any splitter with `split(data)`. Each test fold is predicted by a model trained on its training fold. |
| `feature_columns` | all except date and outcome | Model inputs. Set this explicitly to keep IDs, prediction columns, and closing odds out of the model. |

### Data assumptions

- Odds are **decimal**. Convert American or fractional odds first.
- A bet either wins or loses. Voids and pushes (e.g. tennis retirements at some bookmakers) are not modeled; remove or settle them beforehand.
- Invalid values raise a `ValueError` at construction: missing columns, a closing-odds list of the wrong length, `odds_haircut` outside `[0, 1)`, an unknown `margin_method`, or a probability-based strategy without `model_prob_columns`.

## Methods

| Method | Returns | What it does |
|---|---|---|
| `run()` | `None` | Runs the simulation and fills `detailed_results` and `bookie_results`. |
| `get_detailed_results()` | DataFrame | One row per match for your strategy. Raises if `run()` was not called. |
| `get_bookie_results()` | DataFrame | Same for the favourite benchmark. |
| `history` (attribute) | `BetHistory` | Ledger of the bets settled in the last `run()`; see [Strategies with memory](strategies-module.md#strategies-with-memory-bethistory). |
| `calculate_metrics()` | dict | All metrics; also stored in `.metrics`. Uses `margin_method`. |
| `evaluate(metrics="Yield [%]", min_bets=None, fallback=nan)` | value or dict | Runs the backtest and returns only the requested metrics, several times faster than `run()` plus all metrics, with identical values. For optimizers; see [Hyperparameter Tuning](tuning.md). Leaves earlier `run()` results untouched. |
| `stress_test(haircuts=(0, 0.01, 0.02, 0.03), metrics=...)` | DataFrame | Re-runs with each haircut; one row per haircut. Leaves this backtest's results unchanged. |
| `plot(x_axis="bet", show_bookie=True)` | `None` | Shows the bankroll chart. Use `plot_backtest(backtest)` to get the figure object. |
| `plot_calibration(n_bins=None, scope="placed")` | Figure | Reliability diagram; `scope="all"` uses every game. |
| `plot_odds_distribution(num_bins=None)` | Figure | Histogram of the odds bet, split into wins and losses. |

## Results columns

`detailed_results` and `bookie_results` contain one row per match. Columns added by BacktestBuddy start with `bt_`; your original columns follow.

| Column | Meaning |
|---|---|
| `bt_index` | Row position in the sorted data |
| `bt_fold` | CV fold (`ModelBacktest`); 0 for `PredictionBacktest` |
| `bt_date_column` | Match date |
| `bt_predicted_outcome` | Predicted outcome index |
| `bt_bet_on` | Outcome bet on; `-1` if no bet |
| `bt_actual_outcome` | Outcome that happened |
| `bt_odds` | Odds of the outcome bet on (after `odds_haircut`); empty if no bet |
| `bt_odd_{i}` | Odds of every outcome as bet (after `odds_haircut`) |
| `bt_closing_odd_{i}` | Reference odds of every outcome (only with `closing_odds_columns`) |
| `bt_model_prob_{i}` | Model probability of every outcome (when available) |
| `bt_stake` | Amount staked; 0 if no bet |
| `bt_potential_return` | `stake * odds` |
| `bt_win` | `True` / `False`; empty if no bet |
| `bt_profit` | Profit of the bet; 0 if no bet |
| `bt_roi` | Profit / stake in percent |
| `bt_starting_bankroll`, `bt_ending_bankroll` | Bankroll before and after the match |
| `bt_sizing_bankroll` | Bankroll the stake was sized from (only with `settle_simultaneous=True`) |
| Strategy extras | e.g. `bt_kelly_fraction_{i}` (Kelly), `bt_ev_{i}` (ValueBet), `bt_plan` and `bt_unit` (UnitStake), `bt_risk_limit` (RiskLimits: the rule that changed or blocked the bet) |

## Stress test

```python
import numpy as np
import pandas as pd
import backtestbuddy as btb

rng = np.random.default_rng(1)
n = 400
p = rng.uniform(0.3, 0.7, n)
data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "odds_a": np.round(0.96 / p, 2),
    "odds_b": np.round(0.96 / (1 - p), 2),
    "winner": (rng.uniform(size=n) > p).astype(int),
})
data["prob_a"] = np.clip(p + rng.normal(0, 0.04, n), 0.01, 0.99)
data["prob_b"] = 1 - data["prob_a"]
data["prediction"] = (data["prob_a"] < 0.5).astype(int)

backtest = btb.PredictionBacktest(
    data, ["odds_a", "odds_b"], "winner", "date", "prediction",
    model_prob_columns=["prob_a", "prob_b"],
    strategy=btb.ValueBet(min_ev=0.02, amount=10),
)
backtest.run()
print(backtest.stress_test(haircuts=[0, 0.01, 0.02, 0.03]))
```

The table is indexed by `odds_haircut` and shows ROI, Yield, CAGR, Max Drawdown, Sharpe, Total Bets, and the final bankroll. Pass `metrics=[...]` with any keys of `calculate_metrics()` to choose other columns. Use it when the odds in your data are better than what you can actually get, for example opening odds with low limits.

## Simultaneous matches

By default rows settle one after another: with percentage staking, a later match on the same day is sized from a bankroll that already includes the result of an earlier one, which a real bettor could not know. With `settle_simultaneous=True`, rows that share the same date value form one **slot**:

1. Every bet of the slot is decided from the bankroll at the start of the slot.
2. The slot's total stake is capped at that bankroll, in row order.
3. The strategy may rescale the slot's stakes (`SimultaneousKelly` caps their sum at `max_total`).
4. The bets settle; their results reach `BetHistory` only after the whole slot.

```python
import pandas as pd
import backtestbuddy as btb

slate = pd.DataFrame({
    "date": pd.to_datetime(["2023-01-01"] * 3 + ["2023-01-02"]),
    "odds_a": [2.0] * 4, "odds_b": [2.0] * 4, "winner": [0, 1, 0, 0], "prediction": [0] * 4,
    "prob_a": [0.6] * 4, "prob_b": [0.4] * 4,
})
backtest = btb.PredictionBacktest(
    slate, ["odds_a", "odds_b"], "winner", "date", "prediction",
    model_prob_columns=["prob_a", "prob_b"],
    strategy=btb.Strategy(btb.HighestKelly(), btb.SimultaneousKelly(fraction=1.0, max_bet=0.2, max_total=0.3)),
    settle_simultaneous=True,
)
backtest.run()
print(backtest.detailed_results[["bt_date_column", "bt_sizing_bankroll", "bt_stake", "bt_profit"]])
# Jan 1: three full-Kelly stakes of 20% would add up to 60%; each is scaled to 10% (total 30%)
```

Slots are defined by exact equality of the date values. With timestamps, matches at different kick-off times are separate slots; round the column (e.g. `df["date"].dt.floor("D")`) to group a whole day. `bt_starting_bankroll` and `bt_ending_bankroll` stay a running ledger within the slot, so bankroll curves, drawdowns, and per-period returns remain consistent.
