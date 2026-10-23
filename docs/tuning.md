# Hyperparameter Tuning

Strategy and stake parameters (Kelly fraction, minimum edge, stake caps, …) can be tuned with an optimizer such as [Optuna](https://optuna.org). Two things matter: each trial must be fast, and the result must not just be luck.

## Fast trials: `evaluate()`

`backtest.evaluate()` runs the same simulation as `run()` but skips the bookie benchmark, the original data columns, and every metric you did not ask for. The numbers are identical to `run()` followed by `calculate_metrics()`. On 10,000 matches a trial takes about 0.13 s with `FixedStake` and 0.25 s with Kelly, against 1.2 s for `run()` plus all metrics in 0.2.0.

```python
import numpy as np
import pandas as pd
import backtestbuddy as btb

# Synthetic matches: the model is a little sharper than the bookmaker's line
rng = np.random.default_rng(11)
n = 2500
true_p = rng.uniform(0.25, 0.75, n)
book_p = np.clip(true_p + rng.normal(0, 0.05, n), 0.05, 0.95)
model_p = np.clip(true_p + rng.normal(0, 0.03, n), 0.02, 0.98)
data = pd.DataFrame({
    "date": pd.Timestamp("2021-01-01") + pd.to_timedelta(np.arange(n) // 3, unit="D"),
    "odds_a": np.round(1 / (book_p * 1.04), 2),
    "odds_b": np.round(1 / ((1 - book_p) * 1.04), 2),
    "winner": (rng.uniform(size=n) > true_p).astype(int),
    "prob_a": model_p,
    "prob_b": 1 - model_p,
    "prediction": (model_p < 0.5).astype(int),
})


def make_backtest(frame, strategy, **kwargs):
    return btb.PredictionBacktest(frame, ["odds_a", "odds_b"], "winner", "date", "prediction",
                                  model_prob_columns=["prob_a", "prob_b"], strategy=strategy, **kwargs)


backtest = make_backtest(data, btb.KellyCriterion(downscaling=0.5))
print(backtest.evaluate("Yield [%]"))
print(backtest.evaluate(["Yield [%]", "Max Drawdown [%]", "Total Bets"]))
```

- `metrics`: one key or a list of keys of `calculate_metrics()` (all keys: `btb.METRIC_NAMES`).
- `min_bets`: below this many bets the result is `fallback` (NaN by default, which Optuna records as a failed trial). Without it, an optimizer happily picks settings that placed ten lucky bets.

## Keep a holdout

Tuning many settings on one history finds settings that happened to work there. Hold back the most recent period, tune on the rest, and look at the holdout **once**, at the end:

```python
tuning_data, holdout_data = btb.split_by_date(data, "date", holdout_fraction=0.25)
print(len(tuning_data), "matches to tune on,", len(holdout_data), "held out")
```

`split_by_date(data, "date", holdout_from="2024-01-01")` splits at a fixed date instead. Matches on the cutoff date go to the holdout.

## An Optuna study

```python
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


def build_strategy(params):
    if params["sizing"] == "kelly":
        inner = btb.Strategy(btb.HighestEV(min_ev=params["min_ev"]),
                             btb.KellySize(fraction=params["kelly_fraction"], max_bet=0.1))
    else:
        inner = btb.EdgeProportionalStake(min_ev=params["min_ev"], multiplier=params["multiplier"], max_fraction=0.05)
    return btb.RiskLimits(inner, max_stake_fraction=params["max_stake"])


def objective(trial):
    params = {
        "sizing": trial.suggest_categorical("sizing", ["kelly", "edge"]),
        "min_ev": trial.suggest_float("min_ev", 0.0, 0.10),
        "max_stake": trial.suggest_float("max_stake", 0.01, 0.10),
    }
    if params["sizing"] == "kelly":
        params["kelly_fraction"] = trial.suggest_float("kelly_fraction", 0.1, 1.0)
    else:
        params["multiplier"] = trial.suggest_float("multiplier", 0.1, 2.0)

    # Robust objective: 2% worse odds than in the data, at least 150 bets,
    # and a penalty for deep drawdowns
    scores = make_backtest(tuning_data, build_strategy(params), odds_haircut=0.02).evaluate(
        ["Yield [%]", "Max Drawdown [%]"], min_bets=150)
    return scores["Yield [%]"] - 0.1 * scores["Max Drawdown [%]"]


study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1))
study.optimize(objective, n_trials=40)
print("best params:", study.best_params)

# One look at the holdout, with the winning settings
best = {**study.best_params}
holdout = make_backtest(holdout_data, build_strategy(best))
print(holdout.evaluate(["Yield [%]", "Max Drawdown [%]", "Total Bets"]))
print(holdout.stress_test(haircuts=[0, 0.01, 0.02]))
```

What makes the objective robust:

- **`odds_haircut=0.02`**: the search optimizes for prices you can actually get, not the best prices in the data.
- **`min_bets=150`**: settings that bet rarely cannot win by luck.
- **Drawdown penalty**: between two settings with a similar yield, the one with the smoother bankroll wins. Choose the weight to match your risk appetite.

If the holdout result is much worse than the tuning result, the search fitted noise. Use fewer parameters, wider ranges, or a longer tuning period.

## Tuning a model and a strategy

With `ModelBacktest`, every trial would retrain the model on every fold, which is slow. Tune in two stages instead:

1. **Tune the model** for good probabilities, e.g. on `Log Loss (all games) [-]` or `Brier Score (all games) [-]`, not on profit. Calibration is far more stable than betting returns.
2. **Freeze its probabilities**: run the tuned `ModelBacktest` once; its `detailed_results` hold the out-of-fold probabilities (`bt_model_prob_*`) and predictions (`bt_predicted_outcome`).
3. **Tune the strategy** on those frozen probabilities with `PredictionBacktest`. Each trial then takes milliseconds.

```py
model_bt = btb.ModelBacktest(data, ["odds_a", "odds_b"], "winner", "date", tuned_model,
                             feature_columns=feature_columns)
model_bt.run()
frozen = model_bt.detailed_results

def objective(trial):
    strategy = btb.KellyCriterion(downscaling=trial.suggest_float("kelly", 0.1, 1.0))
    return btb.PredictionBacktest(
        frozen, ["odds_a", "odds_b"], "winner", "date", "bt_predicted_outcome",
        model_prob_columns=["bt_model_prob_0", "bt_model_prob_1"], strategy=strategy,
    ).evaluate("Yield [%]", min_bets=150)
```

## Using several CPU cores

Python threads do not speed up backtests (the GIL lets one run at a time), so `study.optimize(n_jobs=4)` gains little. Run several processes on one shared study instead, for example with a journal file:

```py
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend

storage = JournalStorage(JournalFileBackend("tuning.log"))
study = optuna.create_study(study_name="kelly", storage=storage, direction="maximize", load_if_exists=True)
study.optimize(objective, n_trials=500)
```

Start that script in as many terminals (or processes) as you have cores; they share the trials through the file.
