# Changelog

All notable changes to BacktestBuddy. Newest first.

## Version 0.3.1 (2026-10-28)

Bug-fix release. Upgrade notes: no API changes; if you replicate the simultaneous slot flow outside a backtest, add the two `BetHistory` calls described below.

**Fixes:**
- **`RiskLimits(max_daily_fraction)` with `settle_simultaneous=True`** now caps the day's total stake. Before, the bets of a slot were decided before any of them settled, `history.staked_on(date)` returned 0 for every bet of the slot, and the daily limit only acted per bet (e.g. 10 bets and 5x the limit instead of 2 bets). Reported from the tennis repository.

**Changed slot flow (relevant if you replicate it outside a backtest, e.g. for live betting):**
- Inside a simultaneous slot, the backtest now calls `history.add_open_stake(date, stake)` right after each `get_bet_details` that places a bet, then `adjust_simultaneous(stakes, bankroll)` and the bankroll cap as before, and `history.clear_open_stakes()` before recording the settled bets. `adjust_simultaneous` keeps its signature. A live implementation that mirrors the 0.3.0 flow must add these two calls to match backtest results.
- New `BetHistory` methods: `add_open_stake`, `clear_open_stakes`, and the `open_stake` property. `staked_on(date)` counts settled plus open stakes; results, streaks, and drawdown still update only when a bet settles.

**Docs:**
- Simultaneous slots: `pause_after_losses` and `stop_at_drawdown` deliberately see only earlier slots, because a slot's matches have not finished when its bets are placed.

## Version 0.3.0 (2026-10-24)

Upgrade notes: fully backward compatible with 0.2.0. Existing strategies, constructors, attributes, and results are unchanged. Passing a single `stake=` (or a positional number) to `FixedStake`, `ValueBet`, or `OddsFilter` now emits a `DeprecationWarning`; use `amount=` for currency or `fraction=` for a share of the bankroll instead.

**New features:**
- **Explicit stake sizes**: `FixedStake`, `ValueBet`, and `OddsFilter` take `amount=` (currency, values below 1 allowed) or `fraction=` (share of the current bankroll). The single `stake=` argument still works with its old rule (below 1 = fraction) but emits a `DeprecationWarning`, because `0.99` and `1.0` meant very different bets. New `StakeSize` class.
- **`RiskLimits` wrapper**: `max_stake_fraction`, `max_stake`, `max_daily_fraction`, `min_stake`, `stop_at_drawdown`, `pause_after_losses` around any strategy. The rule that fired is reported in `bt_risk_limit`.
- **Strategy toolkit**: `Strategy(select=..., size=...)` combines a selector (`ModelPick`, `HighestProbability`, `Prediction`, `Favourite`, `HighestEV`, `HighestKelly`) with a sizer (`StakeSize`, `KellySize`, `SimultaneousKelly`, `UnitPlan`, `EdgeProportional`). Custom blocks are small `Selector`/`Sizer` subclasses. `FixedStake`, `KellyCriterion`, `ValueBet`, and `UnitStake` are now preset combinations with unchanged constructors, attributes (still reassignable), subclass overrides, and results. `BaseStrategy` moved to `backtestbuddy.strategies.base`; the old import path still works.
- **Edge-proportional sizing**: `EdgeProportionalStake(min_ev, multiplier, max_fraction)` and the `EdgeProportional` sizer stake `min(multiplier × edge, max_fraction)` of the bankroll.
- **Simultaneous matches**: `settle_simultaneous=True` sizes all bets with the same date from the bankroll at the start of that slot, caps their total at that bankroll, and records them in the history only after the slot. `SimultaneousKelly` scales a slot's Kelly stakes to `max_total`. New results column `bt_sizing_bankroll` in that mode.
- **Fast evaluation for hyperparameter search**: `backtest.evaluate(metrics, min_bets=None, fallback=nan)` runs the backtest without the bookie benchmark, the original data columns, and unrequested metrics, and returns exactly the values `run()` plus `calculate_all_metrics()` would. On 10,000 matches a trial takes 0.13 s (FixedStake) / 0.25 s (Kelly) instead of 1.2 s. `calculate_metrics(results, keys)` and `METRIC_NAMES` compute selected metrics; `split_by_date()` makes a holdout split. New docs page *Hyperparameter Tuning* with an Optuna recipe (robust objective, holdout check, two-stage model tuning, multi-process studies). `stress_test()` uses `evaluate()` internally and is faster.
- **`BetHistory` ledger**: every run records settled bets and passes the ledger to the strategy as `history=` with the match `date=`, so strategies can use memory (streaks, drawdown, daily totals) without their own state. Available after a run as `backtest.history`.

**Performance:**
- Backtests build their results column-wise instead of from per-row dicts, and both backtest classes share one betting loop: about 2x faster (20,000 matches: FixedStake 1.17 s → 0.60 s, Kelly 1.96 s → 0.98 s). Results are identical.

**Fixes:**
- A strategy that picks no outcome (`-1`) now always reports stake 0. Before, `FixedStake` with a `-1` prediction kept a non-zero `bt_stake` on a row with no bet, which leaked into `Average Stake`.

**Docs:**
- `BACKLOG.md` describes planned work, starting with the selector/sizer strategy toolkit (fully backward compatible).

## Version 0.2.0 (2026-10-22)

Upgrade notes: metric keys for Brier, Log Loss, and ECE changed, and several metric values changed (see Breaking Changes). Code that reads `calculate_metrics()` by key needs updating.

**New features:**
- **Margin removal**: `remove_margin(odds, method=...)` with `multiplicative`, `additive`, `power`, and `shin` (new `backtestbuddy.odds` module).
- **Closing line value**: Backtests take `closing_odds_columns` (any reference line, same order as `odds_columns`). New metrics `CLV [%]`, `CLV (no-vig) [%]`, `Beat Closing Line [%]`; `nan` without closing odds.
- **Market baseline**: `Brier Score (market, all games) [-]` and `Log Loss (market, all games) [-]` score the margin-free odds, so you can see whether the model beats the line.
- **Odds stress test**: `odds_haircut=` lowers bet odds (floored at 1.0) before staking and settlement; `backtest.stress_test(haircuts=[...])` returns metrics per haircut.
- **`margin_method=`** on backtests and `calculate_all_metrics` controls margin removal for no-vig CLV and the market baseline.

**⚠️ Breaking Changes (metric values and keys):**
- **Brier / Log Loss / ECE**: Keys now name the scope: `Brier Score (placed) [-]` etc. (selected outcome of placed bets, the old values) and new `(all games) [-]` variants over every outcome of every game. `calculate_brier_score`, `calculate_log_loss`, `calculate_ece`, and `plot_calibration` take `scope="placed" | "all"`.
- **Sharpe / Sortino / Calmar**: Period returns use placed bets only. Days with only skipped opportunities no longer enter as 0% periods, so results no longer depend on whether skipped rows are in the data.
- **Sharpe / Sortino (obs/year)**: Periods per year use `span_days + return_period`, so `n` consecutive betting days annualize to 365.25 instead of more.
- **Avg. ROI per Year [%] (micro)**: Weighted by the days each calendar year covers; partial years no longer count as full years.
- **Avg. ROI per Bet [%] (macro)**: Removed from `calculate_all_metrics` (bankroll ROI / bet count has no per-bet meaning with compounding stakes). The function remains; use Yield per bet.
- **Max Drawdown**: Starts from the opening bankroll, so early losses count (from the audit fixes).

**Fixes and speed (audit):**
- Stable same-date ordering, KellyCriterion requires probabilities, stake validation, empty odds histogram, `ModelBacktest(feature_columns=...)`, ~12x faster backtest loops.

**Testing:**
- `tests/integration/test_metrics_reference.py` cross-checks every metric against a naive reference implementation on 20 randomized backtests.
- Every Python example in README and docs runs as a test.
- GitHub Actions runs the suite on Python 3.9 to 3.13, once more with the oldest allowed dependencies, and builds the docs.

**Fixes:**
- Sharpe, Sortino, and Calmar no longer turn NaN on pandas 1.3 when a period contains a total loss (`-inf + 0` summed to NaN there).

**Packaging:**
- Removed the unused direct dependencies `matplotlib` and `scipy`. Installs no longer pull in matplotlib and its dependencies; scipy still arrives through scikit-learn.
- Removed the stale `setup.py`, `setup.cfg`, and `requirements.txt`; `pyproject.toml` is the only package definition.

**Docs:**
- All pages checked against the code and rewritten for readability; six broken examples fixed. New Odds Module page, a step-by-step Getting Started, and this changelog page.

## Version 0.1.13 (2026-08-26)

**⚠️ Breaking Changes:**
- **Sharpe / Sortino / Calmar**: Period returns compound with `prod(1+r)-1` instead of summing simple returns; empty calendar days are no longer filled with zeros. Numeric values change vs 0.1.12.
- **Placed bets**: Metrics count a bet only when `bt_stake > 0` and `bt_bet_on != -1`.
- **Calmar / Risk-Adjusted Annual ROI**: Return `inf` when there is no drawdown and return is positive; `0` when return is also 0.
- **FixedStake.get_bet_details**: Uses `select_bet` (model probabilities override a class prediction). Unused `initial_bankroll` attribute removed.

**Improvements:**
- Bankroll max drawdown uses the last peak before the trough; plot overlay matches that window.
- Kelly fraction is 0 when decimal odds are `<= 1` (no division by zero).
- PyPI metadata: SPDX `MIT` license, Python 3.13 classifier, tighter project description.

**Testing / docs:**
- Independent numeric tests for Sharpe, Sortino, Calmar, same-day compounding, and the bet filter.
- Metrics and strategies docs rewritten to match the implementations.

## Version 0.1.12 (2025-11-07)

**⚠️ Breaking Changes:**
- **Risk-Adjusted Annual ROI**: Fixed critical units mismatch - now returns unitless ratio (e.g., 0.667) instead of percentage-based values (e.g., 66.67). Values are reduced by 100x. Update any code comparing or thresholding this metric.

**Improvements:**
- **Sortino Ratio**: Enhanced calculation with proper downside deviation formula and edge case handling (`inf` when downside deviation is zero and mean excess return is positive; `0.0` when both are zero)
- **Calmar Ratio**: Improved calculation using geometric annual return for better accuracy in risk-adjusted performance metrics
- **Max Drawdown Duration**: Fixed calculation to correctly identify last peak before trough (duration = end - start + 1 from last peak)

**Infrastructure:**
- Migrated to `uv` package manager for faster dependency resolution
- Updated minimum Python version to 3.9 (removed support for 3.6, 3.7, 3.8)
- Modernized dependencies:
  - numpy: 1.13.3 → 1.26.0
  - pandas: 0.25 → 1.3.0
  - matplotlib: 3.0.0 → 3.5.0
  - scipy: 1.0.0 → 1.7.0
  - plotly: 4.14.0 → 5.0.0
  - scikit-learn: 0.24.0 → 1.0.0
  - nbformat: 4.2.0 → 5.0.0
- Added Python 3.12 support
- Fixed package metadata for PyPI compatibility (license field configuration)
- Updated metrics documentation with precise mathematical formulas

**Testing:**
- Added comprehensive test suites for Sortino Ratio, Calmar Ratio, and Risk-Adjusted Annual ROI
- Enhanced test coverage for edge cases and error handling

## Version 0.1.10 (2025-01-23)
- Fixed Risk-Adjusted Annual ROI calculation to use macro ROI and handle negative values correctly
- Kelly fractions are now always returned in bet details regardless of min_kelly and min_prob filters
- Added Compound Annual Growth Rate (CAGR) metric 'CAGR [%]'

## Version 0.1.9 (2025-01-19)
- Added micro/macro perspectives for annual ROI:
  - 'Avg. ROI per Year [%]' → 'Avg. ROI per Year [%] (micro)' - averages individual yearly ROIs
  - Added 'Avg. ROI per Year [%] (macro)' - total ROI divided by number of years

## Version 0.1.8 (2025-01-19)
- Improved ROI metrics clarity:
  - Renamed ROI metrics for better understanding:
    - 'Avg. ROI per Bet [%]' → 'Avg. ROI per Bet [%] (micro)' - averages individual bet ROIs
    - Added 'Avg. ROI per Bet [%] (macro)' - total ROI divided by number of bets

## Version 0.1.7 (2025-01-19)
- Fixed ROI calculations:
  - Modified `calculate_avg_roi_per_year` to calculate annual ROI by dividing total ROI by number of years
  - Improved handling of empty DataFrames in ROI calculations
  - Added better handling of edge cases (same day, zero years) in ROI calculations
  - Added proper docstrings and test cases for ROI calculations

## Version 0.1.6 (2025-01-19)
- Fixed ROI calculations:
  - Average ROI per Bet now correctly calculates per-bet returns
  - Average ROI per Year now uses total profits and stakes per year
  - Risk-Adjusted Annual ROI improved to handle edge cases

## Version 0.1.5 (2025-01-19)
- Fixed pandas SettingWithCopyWarning in metrics calculation

## Version 0.1.4 (2025-01-19)
- Added new metrics:
  - Average ROI per Year [%]
  - Risk-Adjusted Annual ROI [-] (Avg. ROI per Year / Max Drawdown)

## Version 0.1.3 (2025-01-18)
- Added new metric:
  - Average ROI per Bet [%]

## Version 0.1.2 (2023-11-28)
- Added bookie simulation functionality
- Enhanced plotting capabilities
- Improved documentation

## Version 0.1.1 (2023-11-18)
- Initial release with sports betting functionality
- Basic metrics and visualization tools
- Core backtesting framework
