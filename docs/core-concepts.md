# Core Concepts

This page explains how a backtest works. For exact signatures and formulas, follow the links to the reference pages.

## The simulation loop

A backtest walks through your matches in date order and, for each one:

1. Hands the odds, the **current** bankroll, and your model's probabilities and prediction to the strategy.
2. The strategy returns a **stake** and the **outcome to bet on** (or skips with stake 0).
3. The bet is settled against the actual outcome: profit `stake * (odds - 1)` on a win, `-stake` on a loss.
4. The bankroll is updated, and the next match uses the new bankroll.

Every match produces one row in `detailed_results`, whether a bet was placed or not.

```text
match 1 ──► strategy ──► stake, outcome ──► settle ──► bankroll ──► match 2 ──► …
```

**Placed bet**: a row with `bt_stake > 0` and `bt_bet_on != -1`. Win rate, odds, stake, drawdown, and risk ratios count placed bets only.

**Sequential bankroll**: percentage stakes (Kelly, `FixedStake(fraction=0.02)`) are sized from the bankroll after all earlier matches. Matches with the same date are settled one after another in input order, so a later match on the same day already "knows" the result of an earlier one. For large same-day slates with percentage staking, pass `settle_simultaneous=True`: same-date matches are then sized from the bankroll at the start of the day and settle together (see [Simultaneous matches](backtest-module.md#simultaneous-matches)).

## Two backtest classes

| | `PredictionBacktest` | `ModelBacktest` |
|---|---|---|
| Input | Your predictions and probabilities as columns | Feature columns and a model with `fit`, `predict`, `predict_proba` |
| Training | None; you are responsible for avoiding look-ahead | Walk-forward: `TimeSeriesSplit(n_splits=5)` by default, trained on past folds only |
| Matches bet on | All rows | Test folds only; the first training block is never bet on |

## The bookie benchmark

Every run also simulates a simple benchmark in `bookie_results`: always bet the favourite (lowest odds), sized by your strategy without model probabilities. It answers "would betting the favourite have done as well?". Kelly and ValueBet need probabilities, so their benchmark line stays flat.

## Odds you bet vs reference odds

- `odds_columns` are the odds you actually bet into.
- `closing_odds_columns` (optional) are a reference line, usually the closing odds, for closing line value (CLV).
- `odds_haircut` lowers the odds you bet into, to simulate getting worse prices. `stress_test()` runs several haircuts at once.

## Metrics at a glance

`calculate_metrics()` returns one dictionary. The keys fall into these groups:

| Group | Examples | Question it answers |
|---|---|---|
| Returns | ROI, Yield, CAGR, Total Profit | Did the strategy make money, relative to bankroll and to stakes? |
| Risk | Max Drawdown, Sharpe, Sortino, Calmar | How bumpy was the ride? |
| Betting | Win Rate, Average Odds, Total Bets, Bet Frequency | What did the strategy actually bet on? |
| Probability | Brier, Log Loss, ECE, Expected Yield | Are the model's probabilities calibrated? |
| Market | Brier (market), CLV, Beat Closing Line, Overround | Does the model beat the bookmaker's line? |

Units are in the key: `[%]`, `[$]`, `[-]` (ratio), or none for counts. Values that cannot be computed, e.g. CLV without closing odds, are `nan`.

See the [Metrics Module](metrics-module.md) for every key and formula.

## Package layout

| Module | Contents |
|---|---|
| `backtestbuddy.backtest` | `PredictionBacktest`, `ModelBacktest`, `BaseBacktest` |
| `backtestbuddy.strategies` | `FixedStake`, `KellyCriterion`, `ValueBet`, `UnitStake`, `OddsFilter`, `BaseStrategy` |
| `backtestbuddy.metrics` | `calculate_all_metrics` and the individual metric functions |
| `backtestbuddy.odds` | `remove_margin` |
| `backtestbuddy.plots` | `plot_backtest`, `plot_calibration`, `plot_odds_histogram` |

The common names are importable from the top level: `import backtestbuddy as btb`, then `btb.PredictionBacktest`, `btb.KellyCriterion`, `btb.remove_margin`, and so on.
