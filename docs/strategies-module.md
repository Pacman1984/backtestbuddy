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
| `OddsFilter` | Only if the inner strategy does | Inner strategy, among outcomes in the odds band | Inner strategy |

`PredictionBacktest` raises a `ValueError` if a strategy that needs probabilities gets no `model_prob_columns`.

### Implemented Strategies

#### FixedStake

The `FixedStake` class implements a fixed-stake strategy. It bets either a fixed dollar amount (`stake >= 1`, capped at current bankroll) or a fixed **percentage of the current bankroll** (`stake < 1`). This is constant-fraction sizing, not a freeze of the opening bankroll.

##### Attributes (FixedStake)

- `stake` (float): Absolute amount (>= 1) or current-bankroll fraction (< 1). Negative values raise a `ValueError`.

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
- `stake` (float): Absolute amount (>= 1) or current-bankroll fraction (< 1). Negative values raise a `ValueError`.
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

#### OddsFilter

The `OddsFilter` class restricts an inner strategy to outcomes with `min_odds <= odds <= max_odds`. Default inner strategy is `FixedStake(stake)`. If no outcome is in the band, the bet is skipped (`bet_on = -1`, stake 0).

This is “only consider in-band outcomes”, not “place the inner pick then skip if it is out of band”. Wrapping `FixedStake` without probabilities therefore bets the in-band favorite, not the overall favorite.

`requires_probabilities` is true when the inner strategy requires probabilities (for example `ValueBet`).

##### Attributes (OddsFilter)

- `min_odds` (float): Inclusive lower bound (default 1.01).
- `max_odds` (float): Inclusive upper bound (default infinity).
- `inner` (`BaseStrategy`): Wrapped strategy.
- `stake` (float): Used only when `inner` is omitted.

## Utility Functions

### `get_default_strategy`

``` python
def get_default_strategy() -> FixedStake:
```

Returns the default betting strategy.

- **Returns:**
  - `FixedStake`: 1% of the **current** bankroll (`FixedStake(0.01)`).

## Adding New Strategies

To add a new strategy:

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
