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

This abstract method must be implemented by all concrete strategy classes. It calculates the stake for a bet based on the given odds and current bankroll.

- **Parameters:**
  - `odds` (float): The odds for the bet.
  - `bankroll` (float): The current bankroll.
  - `**kwargs`: Additional keyword arguments that might be needed for specific strategies.
- **Returns:**
  - `float`: The calculated stake for the bet.

## Implemented Strategies

### FixedStake

The `FixedStake` class implements a fixed stake (flat betting) strategy. This strategy bets either a fixed amount or a fixed percentage of the initial bankroll, depending on the stake value.

#### Attributes

- `stake` (Union[float, int]): The fixed stake amount or percentage to bet.
- `initial_bankroll` (Union[float, None]): The initial bankroll, set on the first bet.

#### Methods

##### `__init__`

``` python
def __init__(self, stake: Union[float, int]):
```

Initializes the FixedStake strategy.

- **Parameters:**
  - `stake` (Union[float, int]): The fixed stake amount or percentage to bet.
- **Raises:**
  - `ValueError`: If stake is >= 1 and not an integer.

##### `calculate_stake`

``` python
def calculate_stake(self, odds: float, bankroll: float, **kwargs: Any) -> float:
```

Calculates the stake for a bet using the fixed stake strategy.

- **Parameters:**
  - `odds` (float): The odds for the bet (not used in this strategy, but required by the interface).
  - `bankroll` (float): The current bankroll.
  - `**kwargs`: Additional keyword arguments (not used in this strategy, but might be used in subclasses).
- **Returns:**
  - `float`: The calculated stake for the bet.

##### `__str__`

``` python
def __str__(self) -> str:
```

Returns a string representation of the strategy.

- **Returns:**
  - `str`: A string describing the strategy.

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
2. Implement the `calculate_stake` method.
3. Add any additional methods or attributes specific to the new strategy.
4. Optionally, override the `__str__` method to provide a custom string representation.

Example template for a new strategy:

``` python
class NewStrategy(BaseStrategy):
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def calculate_stake(self, odds: float, bankroll: float, **kwargs: Any) -> float:
        # Implement the stake calculation logic
        pass

    def __str__(self) -> str:
        return f"NewStrategy(param1={self.param1}, param2={self.param2})"
```

