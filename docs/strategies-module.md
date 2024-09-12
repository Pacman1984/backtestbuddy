# Strategies Module

The Strategies module provides a framework for implementing various betting strategies. It defines an abstract base class `BaseStrategy` that serves as a template for all concrete strategy implementations.

## Understanding Stake in Betting Strategies

In the context of betting strategies, the "stake" refers to the absolute amount of money bet on a single game or event. Here's how it works in BacktestBuddy:

- The output of a strategy's `calculate_stake` method is used as the 'bt_stake' column in the results dataframe.
- This 'bt_stake' represents the actual amount of money wagered on each individual bet.
- For example, if a strategy returns a stake of 10, it means $10 (or 10 units of the chosen currency) will be bet on that particular game.

Understanding the stake is crucial for interpreting the results of your backtests and for designing effective betting strategies.

## BaseStrategy

`BaseStrategy` is an abstract base class that defines the interface for all betting strategies.

### Methods

#### `calculate_stake`

This abstract method must be implemented by all concrete strategy classes. It calculates the stake for a bet based on the given odds, current bankroll, and other optional parameters.

#### `select_bet`

This abstract method must be implemented by all concrete strategy classes. It selects the outcome to bet on based on the given odds, model probabilities, and other optional parameters.

#### `get_bet_details`

This method combines `calculate_stake` and `select_bet` to provide complete bet details. It returns a tuple containing the stake, the index of the outcome to bet on, and additional information specific to the strategy.

## Implemented Strategies

### FixedStake

The `FixedStake` class implements a fixed stake (flat betting) strategy. This strategy bets either a fixed amount or a fixed percentage of the initial bankroll, depending on the stake value.

#### Attributes (FixedStake)

- `stake` (float): The fixed stake amount to bet.
- `initial_bankroll` (Union[float, None]): The initial bankroll, set on the first bet.

#### Methods (FixedStake)

Implements all methods from `BaseStrategy` with logic specific to fixed stake betting.

### KellyCriterion

The `KellyCriterion` class implements a betting strategy based on the Kelly Criterion. This strategy calculates the optimal fraction of the bankroll to bet based on the perceived edge and the odds offered.

#### Attributes (KellyCriterion)

- `downscaling` (float): Factor to scale down the Kelly fraction (default is 0.5 for "half Kelly").
- `max_bet` (float): Maximum bet size as a fraction of the bankroll (default is 0.1 or 10%).
- `min_kelly` (float): Minimum Kelly fraction required to place a bet (default is 0).
- `min_prob` (float): Minimum model probability required to place a bet (default is 0).

#### Methods (KellyCriterion)

Implements all methods from `BaseStrategy` with logic specific to Kelly Criterion betting. Additionally includes:

##### `calculate_kelly_fraction`

Calculates the Kelly fraction for a given odds and probability.

##### `get_bet_details` (KellyCriterion)

Returns a tuple containing the stake, the index of the outcome to bet on, and a dictionary of additional information. The additional information includes the Kelly fractions for each possible outcome, stored as `kelly_fraction_0`, `kelly_fraction_1`, etc.

## Utility Functions

### `get_default_strategy`

``` python
def get_default_strategy() -> FixedStake:
```

Returns the default betting strategy.

- **Returns:**
  - `FixedStake`: A FixedStake strategy with a 1% stake of the initial bankroll.

## Adding New Strategies

To add a new strategy:

1. Create a new class that inherits from `BaseStrategy`.
2. Implement the `calculate_stake`, `select_bet`, and `get_bet_details` methods.
3. Add any additional methods or attributes specific to the new strategy.
4. Optionally, override the `__str__` method to provide a custom string representation.

Example template for a new strategy:

``` python
class NewStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def calculate_stake(self, odds: List[float], bankroll: float, model_probs: Optional[List[float]] = None, **kwargs: Any) -> float:
        # Your stake calculation logic here
        pass

    def select_bet(self, odds: List[float], model_probs: Optional[List[float]] = None, **kwargs: Any) -> int:
        # Your bet selection logic here
        pass

    def get_bet_details(self, odds: List[float], bankroll: float, model_probs: Optional[List[float]] = None, prediction: Optional[int] = None, **kwargs: Any) -> Tuple[float, int, Dict[str, Any]]:
        stake = self.calculate_stake(odds, bankroll, model_probs, **kwargs)
        bet_on = self.select_bet(odds, model_probs, prediction, **kwargs)
        additional_info = {
            "custom_info_1": some_value,
            "custom_info_2": another_value,
            # Add any other relevant information
        }
        return stake, bet_on, additional_info
```
