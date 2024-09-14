from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np


class BaseStrategy(ABC):
    """
    Abstract base class for betting strategies.
    """

    @abstractmethod
    def calculate_stake(self, odds: List[float], bankroll: float, **kwargs: Any) -> float:
        """
        Calculate the stake for a bet.

        Args:
            odds (List[float]): The odds for each possible outcome.
            bankroll (float): The current bankroll.
            **kwargs: Additional keyword arguments that might be needed for specific strategies.

        Returns:
            float: The calculated stake for the bet.
        """
        pass

    @abstractmethod
    def select_bet(self, odds: List[float], model_probs: Optional[List[float]] = None, prediction: Optional[int] = None, **kwargs: Any) -> int:
        """
        Select the outcome to bet on.

        Args:
            odds (List[float]): The odds for each possible outcome.
            model_probs (Optional[List[float]]): The model's predicted probabilities for each outcome.
            prediction (Optional[int]): The predicted outcome index.
            **kwargs: Additional keyword arguments that might be needed for specific strategies.

        Returns:
            int: The index of the outcome to bet on.
        """
        pass

    @abstractmethod
    def get_bet_details(self, odds: List[float], bankroll: float, model_probs: Optional[List[float]] = None, prediction: Optional[int] = None, **kwargs: Any) -> Tuple[float, int]:
        """
        Get the details of the bet, including the stake and the outcome to bet on.

        Args:
            odds (List[float]): The odds for each possible outcome.
            bankroll (float): The current bankroll.
            model_probs (Optional[List[float]]): The model's predicted probabilities for each outcome.
            prediction (Optional[int]): The predicted outcome index.
            **kwargs: Additional keyword arguments that might be needed for specific strategies.

        Returns:
            Tuple[float, int]: A tuple containing the calculated stake and the index of the outcome to bet on.
        """
        stake = self.calculate_stake(odds, bankroll, model_probs=model_probs, **kwargs)
        bet_on = self.select_bet(odds, model_probs, prediction, **kwargs)
        return stake, bet_on if stake > 0 else -1


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

    def __init__(self, stake: float):
        """
        Initialize the FixedStake strategy.

        Args:
            stake (float): The fixed stake amount or percentage to bet.
        """
        self.stake = stake
        self.initial_bankroll: Union[float, None] = None

    def calculate_stake(self, odds: List[float], bankroll: float, model_probs: Optional[List[float]] = None, **kwargs: Any) -> float:
        if self.initial_bankroll is None:
            self.initial_bankroll = bankroll

        if self.stake < 1:
            return self.initial_bankroll * self.stake
        else:
            return min(self.stake, bankroll)

    def select_bet(self, odds: List[float], model_probs: Optional[List[float]] = None, prediction: Optional[int] = None, **kwargs: Any) -> int:
        if model_probs:
            return model_probs.index(max(model_probs))
        elif prediction is not None:
            return prediction
        else:
            return odds.index(min(odds))

    def get_bet_details(self, odds: List[float], current_bankroll: float, model_probs: Optional[List[float]] = None, prediction: Optional[int] = None) -> Tuple[float, int, Dict[str, Any]]:
        bet_on = prediction if prediction is not None else odds.index(min(odds))
        return self.stake, bet_on, {}  # Return an empty dictionary as additional_info

    def __str__(self) -> str:
        if self.stake < 1:
            return f"Fixed Stake Strategy ({self.stake:.2%} of initial bankroll)"
        else:
            return f"Fixed Stake Strategy (${self.stake:.2f})"


class KellyCriterion(BaseStrategy):
    """
    A betting strategy based on the Kelly Criterion.

    This strategy calculates the optimal fraction of the bankroll to bet based on the
    perceived edge and the odds offered. It bets on the outcome with the highest Kelly fraction.

    Attributes:
        downscaling (float): Factor to scale down the Kelly fraction (default is 0.5 for "half Kelly").
        max_bet (float): Maximum bet size as a fraction of the bankroll (default is 0.1 or 10%).
        min_kelly (float): Minimum Kelly fraction required to place a bet (default is 0).
        min_prob (float): Minimum model probability required to place a bet (default is 0).
    """

    def __init__(self, downscaling: float = 0.5, max_bet: float = 0.1, min_kelly: float = 0, min_prob: float = 0):
        self.downscaling = downscaling
        self.max_bet = max_bet
        self.min_kelly = min_kelly
        self.min_prob = min_prob

    def calculate_kelly_fraction(self, odds: float, prob: float) -> float:
        if prob > self.min_prob:
            adj_odds = odds - 1
            kelly = (prob * adj_odds - (1 - prob)) / adj_odds
            return max(0, kelly)
        return 0

    def calculate_stake(self, odds: List[float], bankroll: float, model_probs: Optional[List[float]] = None, **kwargs) -> float:
        if model_probs is None:
            return 0

        bet_on = self.select_bet(odds, model_probs, **kwargs)
        if bet_on == -1:
            return 0

        kelly_fraction = self.calculate_kelly_fraction(odds[bet_on], model_probs[bet_on])

        if kelly_fraction > self.min_kelly:
            return min(kelly_fraction * self.downscaling, self.max_bet) * bankroll

        return 0

    def select_bet(self, odds: List[float], model_probs: Optional[List[float]] = None, prediction: Optional[int] = None, **kwargs) -> int:
        if model_probs is None:
            return -1
        kelly_fractions = [self.calculate_kelly_fraction(odd, prob) for odd, prob in zip(odds, model_probs)]
        return kelly_fractions.index(max(kelly_fractions))

    def get_bet_details(self, odds: List[float], bankroll: float, model_probs: Optional[List[float]] = None, prediction: Optional[int] = None, **kwargs: Any) -> Tuple[float, int, Dict[str, Any]]:
        stake, bet_on = super().get_bet_details(odds, bankroll, model_probs, prediction, **kwargs)
        kelly_fractions = []
        if model_probs:
            for odd, prob in zip(odds, model_probs):
                kelly_fractions.append(self.calculate_kelly_fraction(odd, prob))
        
        additional_info = {f"kelly_fraction_{i}": kf for i, kf in enumerate(kelly_fractions)}
        return stake, bet_on, additional_info

    def __str__(self) -> str:
        return (f"Kelly Criterion Strategy (downscaling={self.downscaling}, max_bet={self.max_bet}, "
                f"min_kelly={self.min_kelly}, min_prob={self.min_prob})")


def get_default_strategy() -> FixedStake:
    """
    Get the default betting strategy.

    Returns:
        FixedStake: A FixedStake strategy with a 1% stake of the initial bankroll.
    """
    return FixedStake(0.01)