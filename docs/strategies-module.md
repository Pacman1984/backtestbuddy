# Strategies Module

The Strategies module provides a framework for implementing various betting strategies. It defines an abstract base class `BaseStrategy` that serves as a template for all concrete strategy implementations. Right now, the only module that implements these strategies is the `sport_strategies.py` file.

## Sport Strategies

### Understanding Stake in Betting Strategies

In the context of betting strategies, the "stake" refers to the absolute amount of money bet on a single game or event. Here's how it works in BacktestBuddy:

- The output of a strategy's `calculate_stake` method is used as the 'bt_stake' column in the results dataframe.
- This 'bt_stake' represents the actual amount of money wagered on each individual bet.
- For example, if a strategy returns a stake of 10, it means $10 (or 10 units of the chosen currency) will be bet on that particular game.

Understanding the stake is crucial for interpreting the results of your backtests and for designing effective betting strategies.

### BaseStrategy

`BaseStrategy` is an abstract base class that defines the interface for all betting strategies.

#### Methods

##### `calculate_stake`

This abstract method must be implemented by all concrete strategy classes. It calculates the stake for a bet based on the given odds, current bankroll, and other optional parameters.

##### `select_bet`

This abstract method must be implemented by all concrete strategy classes. It selects the outcome to bet on based on the given odds, model probabilities, and other optional parameters.

##### `get_bet_details`

This method combines `calculate_stake` and `select_bet` to provide complete bet details. It returns a tuple containing the stake, the index of the outcome to bet on, and additional information specific to the strategy. Each key of the additional information becomes a `bt_{key}` column in the results.

### Which strategy?

| Strategy | Needs model probabilities | Picks the outcome by | Stake |
|---|---|---|---|
| `FixedStake` | No | Highest probability, else prediction, else favourite | Fixed amount or % of current bankroll |
| `KellyCriterion` | **Yes** | Highest Kelly fraction | Scaled Kelly fraction of current bankroll |
| `ValueBet` | **Yes** | Highest expected value above `min_ev` | Fixed amount or % of current bankroll |
| `UnitStake` | No | Same as `FixedStake` | Unit-loss, unit-win, or unit-impact plan |
| `EdgeProportionalStake` | **Yes** | Highest expected value above `min_ev` | Proportional to the edge, capped |
| `OddsFilter` | Only if the inner strategy does | Inner strategy, among outcomes in the odds band | Inner strategy |
| `RiskLimits` | Only if the inner strategy does | Inner strategy | Inner strategy, capped or paused by the limits |
| `Strategy(select=…, size=…)` | If a part does | Any selector | Any sizer (see [Strategy toolkit](#strategy-toolkit)) |

`PredictionBacktest` raises a `ValueError` if a strategy that needs probabilities gets no `model_prob_columns`.

## Strategy toolkit

A betting decision has two independent parts: **which** outcome to back and **how much** to stake. The toolkit gives you each part as a small building block and lets you combine any selector with any sizer:

```python
import backtestbuddy as btb

# Value bets only, but sized with half Kelly and at most 5% per bet
strategy = btb.Strategy(select=btb.HighestEV(min_ev=0.03), size=btb.KellySize(fraction=0.5, max_bet=0.05))
```

The ready-made classes are preset combinations of the same blocks, with unchanged constructors and results:

| Class | = selector | + sizer |
|---|---|---|
| `FixedStake(amount=10)` | `ModelPick()` | `StakeSize(amount=10)` |
| `KellyCriterion(0.5, 0.1, min_kelly, min_prob)` | `HighestKelly(min_kelly, min_prob)` | `KellySize(0.5, 0.1, min_kelly, min_prob)` |
| `ValueBet(min_ev, amount=10)` | `HighestEV(min_ev)` | `StakeSize(amount=10)` |
| `UnitStake(unit, plan)` | `ModelPick()` | `UnitPlan(unit, plan)` |
| `EdgeProportionalStake(min_ev, multiplier, max_fraction)` | `HighestEV(min_ev)` | `EdgeProportional(multiplier, max_fraction)` |

### Selectors: which outcome

| Selector | Picks | Needs probabilities |
|---|---|---|
| `ModelPick()` | Highest probability, else the prediction, else the favourite | No |
| `HighestProbability()` | Highest model probability | Yes |
| `Prediction()` | The prediction column; skips rows without one | No |
| `Favourite()` | Lowest odds | No |
| `HighestEV(min_ev=0.0)` | Highest `p * odds - 1`, only above `min_ev`; adds `ev_{i}` columns | Yes |
| `HighestKelly(min_kelly=0.0, min_prob=0.0)` | Highest Kelly fraction above `min_kelly`; adds `kelly_fraction_{i}` | Yes |

### Sizers: how much

| Sizer | Stake | Needs probabilities |
|---|---|---|
| `StakeSize(amount=…)` / `StakeSize(fraction=…)` | Fixed amount, or a share of the current bankroll | No |
| `KellySize(fraction=0.5, max_bet=0.1)` | `min(f* × fraction, max_bet) × bankroll` | Yes |
| `SimultaneousKelly(fraction, max_bet, max_total=0.3)` | Like `KellySize`; with `settle_simultaneous=True` the stakes of one slot are scaled down together so their sum stays under `max_total` of the bankroll | Yes |
| `EdgeProportional(multiplier=0.5, max_fraction=0.05)` | `min(multiplier × edge, max_fraction) × bankroll`, edge = `p * odds - 1`; adds an `edge` column | Yes |
| `UnitPlan(unit, plan)` | Unit-loss, unit-win, or unit-impact | No |

### Writing your own block

A selector implements `select(ctx) -> int` (return `-1` to skip); a sizer implements `size(ctx, outcome) -> float`. Both receive a `BetContext` with `odds`, `bankroll`, `model_probs`, `prediction`, `history` (the run's `BetHistory`), and `date`. Optional `info(...)` methods return extra values that become `bt_{key}` result columns.

```python
import backtestbuddy as btb


class Underdog(btb.Selector):
    """Back the outcome with the highest odds."""

    def select(self, ctx):
        return ctx.odds.index(max(ctx.odds))


class SmallerAfterLosses(btb.Sizer):
    """2% of the bankroll, 1% while the last bet lost."""

    def size(self, ctx, outcome):
        losing = ctx.history is not None and ctx.history.losing_streak > 0
        return ctx.bankroll * (0.01 if losing else 0.02)


strategy = btb.Strategy(select=Underdog(), size=SmallerAfterLosses())
```

Toolkit strategies work everywhere the classic ones do: in `OddsFilter`, `RiskLimits`, `stress_test()`, and with `settle_simultaneous=True`. Writing a strategy as one `BaseStrategy` subclass (see [Adding New Strategies](#adding-new-strategies)) remains fully supported.

### Implemented Strategies

#### FixedStake

The `FixedStake` class bets either a fixed amount or a fixed **share of the current bankroll** on every bet. Fraction sizing follows the bankroll as it grows or shrinks; it is not a freeze of the opening bankroll.

```python
import backtestbuddy as btb

btb.FixedStake(amount=10)      # 10 currency units per bet, capped at the bankroll
btb.FixedStake(fraction=0.02)  # 2% of the current bankroll per bet
```

##### Stake sizes: `amount=` or `fraction=`

`FixedStake`, `ValueBet`, and `OddsFilter` (for its default inner strategy) take the size as one of:

- `amount=`: currency units per bet, `>= 0`, capped at the current bankroll. Values below 1 are allowed (`amount=0.5` is half a unit).
- `fraction=`: share of the current bankroll, in `[0, 1]`.

Older code that passes a single `stake=` (or a positional number) keeps working: below 1 is read as a fraction, 1 and above as an amount. It emits a `DeprecationWarning`, because `stake=0.99` (99% of the bankroll) and `stake=1.0` (1 unit) are easy to mix up. Passing more than one of `stake=`, `amount=`, `fraction=` raises a `ValueError`.

##### Attributes (FixedStake)

- `size` (`StakeSize`): the configured `amount` or `fraction`.
- `stake` (float): the configured number, kept for backward compatibility.

##### Methods (FixedStake)

Implements all methods from `BaseStrategy`. Outcome selection: the highest model probability if probabilities are passed, else the prediction, else the favourite (lowest odds). Passing `model_prob_columns` therefore changes which outcome FixedStake bets on.

#### KellyCriterion

The `KellyCriterion` class implements a betting strategy based on the Kelly Criterion. This strategy calculates the optimal fraction of the **current** bankroll to bet from decimal odds: $f^* = (p(odds-1)-(1-p))/(odds-1)$. Odds `<= 1` yield a fraction of 0.

The stake is `min(f* × downscaling, max_bet) × current bankroll`, on the outcome with the highest Kelly fraction.

Bookie simulations call `calculate_stake` without model probabilities, so Kelly always stakes 0 on the bookie path. Use FixedStake if you need a bookie benchmark.

##### Attributes (KellyCriterion)

- `downscaling` (float): Factor to scale down the Kelly fraction (default is 0.5 for "half Kelly"). Must be non-negative.
- `max_bet` (float): Maximum bet size as a fraction of the bankroll (default is 0.1 or 10%). Must be in `[0, 1]`.
- `min_kelly` (float): Minimum Kelly fraction required to place a bet (default is 0).
- `min_prob` (float): Minimum model probability required to place a bet (default is 0).
- `requires_probabilities` (bool): Always `True`. `PredictionBacktest` raises if `model_prob_columns` is omitted.

##### Methods (KellyCriterion)

Implements all methods from `BaseStrategy` with logic specific to Kelly Criterion betting. Additionally includes:

###### `calculate_kelly_fraction`

Calculates the Kelly fraction for a given odds and probability.

###### `get_bet_details` (KellyCriterion)

Returns a tuple containing the stake, the index of the outcome to bet on, and a dictionary of additional information. The additional information includes the Kelly fractions for each possible outcome, stored as `kelly_fraction_0`, `kelly_fraction_1`, etc.

#### ValueBet

The `ValueBet` class bets only when expected value exceeds `min_ev`. EV of an outcome is `p * odds - 1`. Among outcomes with EV strictly greater than `min_ev`, the highest EV is selected (not the highest probability). Stake follows FixedStake rules.

Requires model probabilities. Bookie simulations call `calculate_stake` without probabilities, so ValueBet always stakes 0 on the bookie path.

##### Attributes (ValueBet)

- `min_ev` (float): Minimum expected value to place a bet (default 0). Outcomes with EV at or below this skip.
- `size` (`StakeSize`): `amount=` or `fraction=` as for FixedStake; default `amount=1`.
- `requires_probabilities` (bool): Always `True`. `PredictionBacktest` will raise if `model_prob_columns` is omitted.

##### Methods (ValueBet)

Implements all methods from `BaseStrategy`. `get_bet_details` extra info includes `ev_0`, `ev_1`, … for each outcome.

#### UnitStake

The `UnitStake` class implements Cortés (2020) unit plans. Selection matches FixedStake (max model probability, else prediction, else lowest-odds favorite). The `unit` is always a currency amount, never a bankroll fraction.

Plans:

- `loss`: stake = unit (flat risk if the bet loses).
- `win`: stake = unit / (odds − 1) so the net win equals unit. Odds `<= 1` skip.
- `impact`: stake = unit / odds so the difference between winning and losing is unit. Odds `<= 0` skip.

All three plans cap the stake at the current bankroll. Bookie path works (favorite + unit size).

##### Attributes (UnitStake)

- `unit` (float): Staking constant in currency units (must be positive).
- `plan` (str): `loss`, `win`, or `impact`.

#### EdgeProportionalStake

Bets the highest-EV outcome, like `ValueBet`, but the stake grows with the edge: `min(multiplier × edge, max_fraction) × bankroll`, where edge = `p * odds - 1`.

```python
import backtestbuddy as btb

strategy = btb.EdgeProportionalStake(min_ev=0.02, multiplier=0.5, max_fraction=0.05)
# edge 6%  -> 3% of the bankroll
# edge 20% -> capped at 5%
```

Compared with Kelly, the stake does not shrink for long odds (Kelly divides by `odds - 1`), and the cap keeps the variance low. Results include an `edge` column next to `ev_{i}`.

##### Attributes (EdgeProportionalStake)

- `min_ev` (float): Minimum expected value to bet (default 0).
- `multiplier` (float): Bankroll share per unit of edge (default 0.5).
- `max_fraction` (float): Cap as a bankroll share (default 0.05).
- `requires_probabilities` (bool): Always `True`.

#### OddsFilter

The `OddsFilter` class restricts an inner strategy to outcomes with `min_odds <= odds <= max_odds`. Without `inner`, it uses `FixedStake` with the given `amount=` or `fraction=` (amount 1 by default). If no outcome is in the band, the bet is skipped (`bet_on = -1`, stake 0).

This is “only consider in-band outcomes”, not “place the inner pick then skip if it is out of band”. Wrapping `FixedStake` without probabilities therefore bets the in-band favorite, not the overall favorite.

`requires_probabilities` is true when the inner strategy requires probabilities (for example `ValueBet`).

##### Attributes (OddsFilter)

- `min_odds` (float): Inclusive lower bound (default 1.01).
- `max_odds` (float): Inclusive upper bound (default infinity).
- `inner` (`BaseStrategy`): Wrapped strategy.
- `stake` (float): Used only when `inner` is omitted.

#### RiskLimits

`RiskLimits` wraps any strategy and shrinks or blocks its bets. Every limit is optional; combine as many as you need.

```python
import backtestbuddy as btb

strategy = btb.RiskLimits(
    btb.KellyCriterion(downscaling=0.5),
    max_stake_fraction=0.05,   # at most 5% of the bankroll per bet
    max_daily_fraction=0.15,   # at most 15% of the day's opening bankroll per day
    pause_after_losses=3,      # stop for the day after 3 losses in a row that day
    stop_at_drawdown=0.30,     # stop for good once 30% below the peak
    min_stake=2,               # skip bets below the bookmaker minimum
)
```

| Limit | Effect | Needs the history |
|---|---|---|
| `stop_at_drawdown` | No more bets once the bankroll is this far below its peak. Permanent for the run. | yes |
| `pause_after_losses` | No more bets that calendar day after this many consecutive losses on that day. | yes |
| `max_stake_fraction` | Caps each stake at this share of the current bankroll. | no |
| `max_stake` | Caps each stake at this amount. | no |
| `max_daily_fraction` | Caps the total stake per calendar day at this share of the bankroll before the day's first bet. | yes |
| `min_stake` | Skips bets whose stake is below this amount. | no |

The rules are applied in this order. The rule that changed or blocked a bet is written to the results column `bt_risk_limit`, so `backtest.detailed_results["bt_risk_limit"].value_counts()` shows how often each limit fired.

With `settle_simultaneous=True`, `max_daily_fraction` also caps the total of one slot, because stakes count as open in the history as soon as they are placed. `pause_after_losses` and `stop_at_drawdown` only see results of earlier slots, since the slot's matches have not finished when its bets are placed.

Rules that need the history only work inside a backtest. The bookie benchmark calls the strategy without a history, so only the stateless caps apply there. Wrappers can be nested, e.g. `OddsFilter(min_odds=1.5, inner=RiskLimits(...))`.

## Strategies with memory: `BetHistory`

Every run keeps a ledger of settled bets, a `BetHistory`, and passes it to the strategy on each match together with the match date:

```text
strategy.get_bet_details(odds, bankroll, model_probs, prediction, history=<BetHistory>, date=<match date>)
```

A strategy that needs to remember anything reads the ledger instead of keeping its own state. The backtest creates a fresh ledger in every `run()`, so repeated runs such as `stress_test()` never leak state into each other. After a run, the ledger is available as `backtest.history`.

| Query | Returns |
|---|---|
| `bankroll`, `peak_bankroll`, `drawdown` | Current bankroll, highest so far (including the start), fall below the peak as a fraction |
| `n_bets`, `bets`, `last_bet` | Count, all settled bets, the most recent one (`SettledBet` with `date`, `stake`, `profit`, `bankroll_before`, `bankroll_after`, `won`) |
| `losing_streak`, `losing_streak_on(date)` | Consecutive losses at the end, overall or within one day |
| `staked_on(date)`, `profit_on(date)`, `bankroll_at_start_of(date)` | Per-calendar-day totals; `staked_on` includes stakes that are placed but not settled yet |
| `add_open_stake(date, stake)`, `clear_open_stakes()`, `open_stake` | Register, clear, and sum placed-but-unsettled stakes (used for simultaneous slots; call them yourself if you replicate the slot flow outside a backtest) |

Example: a flat stake that halves after two losses in a row.

```python
import numpy as np
import pandas as pd
import backtestbuddy as btb


class HalveAfterLosses(btb.FixedStake):
    """FixedStake that halves the stake while the last two bets lost."""

    def get_bet_details(self, odds, bankroll, model_probs=None, prediction=None, history=None, **kwargs):
        stake, bet_on, info = super().get_bet_details(odds, bankroll, model_probs, prediction)
        if history is not None and history.losing_streak >= 2:
            stake /= 2
        return stake, bet_on, {"losing_streak": history.losing_streak if history else 0}


rng = np.random.default_rng(5)
n = 200
data = pd.DataFrame({
    "date": pd.date_range("2023-01-01", periods=n, freq="D"),
    "odds_a": np.full(n, 1.9),
    "odds_b": np.full(n, 1.9),
    "winner": rng.integers(0, 2, n),
    "prediction": np.zeros(n, dtype=int),
})
backtest = btb.PredictionBacktest(data, ["odds_a", "odds_b"], "winner", "date", "prediction",
                                  strategy=HalveAfterLosses(amount=20))
backtest.run()
print(backtest.detailed_results[["bt_stake", "bt_win", "bt_losing_streak"]].head(8))
print("Bets:", backtest.history.n_bets, "| peak:", backtest.history.peak_bankroll)
```

## Utility Functions

### `get_default_strategy`

``` python
def get_default_strategy() -> FixedStake:
```

Returns the default betting strategy.

- **Returns:**
  - `FixedStake`: 1% of the **current** bankroll (`FixedStake(fraction=0.01)`).

## Adding New Strategies

The shortest way is usually the [Strategy toolkit](#strategy-toolkit): write a selector or a sizer and combine it. To write a complete strategy as one class instead:

1. Create a new class that inherits from `BaseStrategy`.
2. Implement the `calculate_stake`, `select_bet`, and `get_bet_details` methods.
3. Add any additional methods or attributes specific to the new strategy.
4. Optionally, override the `__str__` method to provide a custom string representation.

Example: bet a fixed amount on the favourite, but only when its odds are at least `min_odds`.

```python
from typing import Any, Dict, List, Optional, Tuple

from backtestbuddy import BaseStrategy


class FavouriteAboveOdds(BaseStrategy):
    """Bet `stake` on the favourite when its odds are at least `min_odds`."""

    def __init__(self, stake: float = 10.0, min_odds: float = 1.3):
        self.stake = stake
        self.min_odds = min_odds

    def select_bet(self, odds: List[float], model_probs: Optional[List[float]] = None,
                   prediction: Optional[int] = None, **kwargs: Any) -> int:
        favourite = odds.index(min(odds))
        return favourite if odds[favourite] >= self.min_odds else -1

    def calculate_stake(self, odds: List[float], bankroll: float,
                        model_probs: Optional[List[float]] = None, **kwargs: Any) -> float:
        if self.select_bet(odds, model_probs) == -1:
            return 0.0
        return min(self.stake, bankroll)

    def get_bet_details(self, odds: List[float], bankroll: float,
                        model_probs: Optional[List[float]] = None,
                        prediction: Optional[int] = None,
                        **kwargs: Any) -> Tuple[float, int, Dict[str, Any]]:
        stake = self.calculate_stake(odds, bankroll, model_probs)
        bet_on = self.select_bet(odds, model_probs, prediction) if stake > 0 else -1
        return stake, bet_on, {"favourite_odds": min(odds)}  # becomes bt_favourite_odds
```

Rules the backtest relies on:

- Return `bet_on = -1` and stake 0 to skip a match.
- `select_bet` must accept `prediction` as the third positional argument.
- Set `requires_probabilities = True` on the class if the strategy cannot work without model probabilities.
- Keep `**kwargs` in `get_bet_details`: the backtest passes `history=` and `date=` (see [Strategies with memory](#strategies-with-memory-bethistory)). Read them if you need memory, ignore them otherwise.
