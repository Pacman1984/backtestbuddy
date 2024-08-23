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

class FixedStake(BaseStrategy):
    """
    A fixed stake (flat betting) strategy.

    This strategy always bets a fixed amount or a fixed percentage of the initial bankroll,
    regardless of the odds or other factors.

    Attributes:
        stake (float): The fixed stake amount or percentage to bet.
        is_percentage (bool): If True, the stake is treated as a percentage of the initial bankroll.
    """

    def __init__(self, stake: float, is_percentage: bool = False):
        """
        Initialize the FixedStake strategy.

        Args:
            stake (float): The fixed stake amount or percentage to bet.
            is_percentage (bool, optional): If True, the stake is treated as a percentage of the initial bankroll. 
                Defaults to False.
        """
        self.stake = stake
        self.is_percentage = is_percentage
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

        if self.is_percentage:
            return self.initial_bankroll * (self.stake / 100)
        else:
            return min(self.stake, bankroll)  # Ensure we don't bet more than the current bankroll

    def __str__(self) -> str:
        """
        Return a string representation of the strategy.

        Returns:
            str: A string describing the strategy.
        """
        if self.is_percentage:
            return f"Fixed Stake Strategy ({self.stake}% of initial bankroll)"
        else:
            return f"Fixed Stake Strategy (${self.stake:.2f})"

def get_default_strategy() -> FixedStake:
    """
    Get the default betting strategy.

    Returns:
        FixedStake: A FixedStake strategy with a 1% stake of the initial bankroll.
    """
    return FixedStake(1, is_percentage=True)