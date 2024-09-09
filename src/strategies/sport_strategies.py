from abc import ABC, abstractmethod
from typing import Union, Any

class BaseStrategy(ABC):
    """
    Abstract base class for betting strategies.
    """

    @abstractmethod
    def calculate_stake(self, odds: float, bankroll: float, **kwargs: Any) -> float:
        """
        Calculate the stake for a bet.

        Args:
            odds (float): The odds for the bet.
            bankroll (float): The current bankroll.
            **kwargs: Additional keyword arguments that might be needed for specific strategies.

        Returns:
            float: The calculated stake for the bet.
        """
        pass

# What "stake" means in betting strategies:
# The stake is the absolute amount of money bet on a single game or event.
# In the context of BacktestBuddy:
# - The output of a strategy's calculate_stake method is used as the 'bt_stake' column in the results dataframe.
# - This 'bt_stake' represents the actual amount of money wagered on each individual bet.
# - For example, if a strategy returns a stake of 10, it means $10 (or 10 units of the chosen currency) will be bet on that particular game.

class FixedStake(BaseStrategy):
    """
    A fixed stake (flat betting) strategy.

    This strategy bets either a fixed amount or a fixed percentage of the initial bankroll,
    determined by the value of the stake:
    - If stake < 1, it's treated as a percentage of the initial bankroll.
    - If stake >= 1, it's treated as an absolute value (must be an integer).

    Attributes:
        stake (Union[float, int]): The fixed stake amount or percentage to bet.
    """

    def __init__(self, stake: Union[float, int]):
        """
        Initialize the FixedStake strategy.

        Args:
            stake (Union[float, int]): The fixed stake amount or percentage to bet.
        """
        if stake >= 1 and not isinstance(stake, int):
            raise ValueError("Stake must be an integer when >= 1")
        self.stake = stake
        self.initial_bankroll: Union[float, None] = None

    def calculate_stake(self, odds: float, bankroll: float, **kwargs: Any) -> float:
        """
        Calculate the stake for a bet using the fixed stake strategy.

        Args:
            odds (float): The odds for the bet (not used in this strategy, but required by the interface).
            bankroll (float): The current bankroll.
            **kwargs: Additional keyword arguments (not used in this strategy, but might be used in subclasses).

        Returns:
            float: The calculated stake for the bet.
        """
        if self.initial_bankroll is None:
            self.initial_bankroll = bankroll

        if self.stake < 1:
            return self.initial_bankroll * self.stake
        else:
            return min(self.stake, bankroll)  # Ensure we don't bet more than the current bankroll

    def __str__(self) -> str:
        """
        Return a string representation of the strategy.

        Returns:
            str: A string describing the strategy.
        """
        if self.stake < 1:
            return f"Fixed Stake Strategy ({self.stake:.2%} of initial bankroll)"
        else:
            return f"Fixed Stake Strategy (${self.stake:.2f})"

def get_default_strategy() -> FixedStake:
    """
    Get the default betting strategy.

    Returns:
        FixedStake: A FixedStake strategy with a 1% stake of the initial bankroll.
    """
    return FixedStake(0.01)