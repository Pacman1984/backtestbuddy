"""Betting strategies for sports backtests.

FixedStake sizes bets as a constant dollar amount or as a fraction of the
*current* bankroll. KellyCriterion sizes bets from the decimal-odds Kelly
fraction, scaled by ``downscaling`` and capped by ``max_bet``.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class BaseStrategy(ABC):
    """Abstract base class for betting strategies."""

    @abstractmethod
    def calculate_stake(
        self, odds: List[float], bankroll: float, **kwargs: Any
    ) -> float:
        """Calculate the stake for a bet.

        Args:
            odds: Decimal odds for each possible outcome.
            bankroll: Current bankroll.
            **kwargs: Strategy-specific options (e.g. model_probs).

        Returns:
            Stake in currency units.
        """
        pass

    @abstractmethod
    def select_bet(
        self,
        odds: List[float],
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Select the outcome index to bet on.

        Args:
            odds: Decimal odds for each possible outcome.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Predicted outcome index.
            **kwargs: Strategy-specific options.

        Returns:
            Outcome index, or ``-1`` to skip the bet.
        """
        pass

    @abstractmethod
    def get_bet_details(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Return stake, selected outcome, and strategy metadata.

        Args:
            odds: Decimal odds for each possible outcome.
            bankroll: Current bankroll.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Predicted outcome index.
            **kwargs: Strategy-specific options.

        Returns:
            ``(stake, bet_on, additional_info)``. ``bet_on`` is ``-1`` when
            stake is 0.
        """
        stake = self.calculate_stake(
            odds, bankroll, model_probs=model_probs, **kwargs
        )
        bet_on = self.select_bet(odds, model_probs, prediction, **kwargs)
        return stake, bet_on if stake > 0 else -1, {}


class FixedStake(BaseStrategy):
    """Flat dollar stake or constant fraction of the current bankroll.

    - If ``stake < 1``, it is a fraction of the *current* bankroll
      (e.g. ``0.01`` = 1% of current bankroll on each bet).
    - If ``stake >= 1``, it is an absolute currency amount, capped at the
      current bankroll.

    Attributes:
        stake: Absolute amount (if >= 1) or bankroll fraction (if < 1).
    """

    def __init__(self, stake: float):
        """Initialize the strategy.

        Args:
            stake: Absolute amount (>= 1) or current-bankroll fraction (< 1).
        """
        self.stake = stake

    def calculate_stake(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> float:
        """Calculate the stake from current bankroll.

        Args:
            odds: Unused; present for the strategy interface.
            bankroll: Current bankroll.
            model_probs: Unused.
            **kwargs: Unused.

        Returns:
            Fraction of current bankroll, or min(absolute stake, bankroll).
        """
        if self.stake < 1:
            return bankroll * self.stake
        return min(self.stake, bankroll)

    def select_bet(
        self,
        odds: List[float],
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Select an outcome: model probs, else prediction, else favorite.

        Args:
            odds: Decimal odds; favorite is the lowest-odds outcome.
            model_probs: If provided, bet on the highest probability.
            prediction: Used when model probabilities are not provided.
            **kwargs: Unused.

        Returns:
            Outcome index.
        """
        if model_probs:
            return model_probs.index(max(model_probs))
        if prediction is not None:
            return prediction
        return odds.index(min(odds))

    def get_bet_details(
        self,
        odds: List[float],
        current_bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Return stake, selected outcome, and empty metadata.

        Args:
            odds: Decimal odds for each possible outcome.
            current_bankroll: Current bankroll.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Predicted outcome index.
            **kwargs: Unused.

        Returns:
            ``(stake, bet_on, {})``.
        """
        stake = self.calculate_stake(
            odds, current_bankroll, model_probs=model_probs
        )
        bet_on = self.select_bet(odds, model_probs, prediction)
        if stake == 0:
            bet_on = -1
        return stake, bet_on, {}

    def __str__(self) -> str:
        if self.stake < 1:
            return (
                f"Fixed Stake Strategy "
                f"({self.stake:.2%} of current bankroll)"
            )
        return f"Fixed Stake Strategy (${self.stake:.2f})"


class KellyCriterion(BaseStrategy):
    """Bet the (scaled, capped) Kelly fraction of current bankroll.

    Kelly fraction for decimal odds: ``f* = (p*(odds-1) - (1-p)) / (odds-1)``.
    Negative or undefined fractions (including ``odds <= 1``) are treated as
    0. The stake is ``min(f* * downscaling, max_bet) * bankroll``.

    Bookie simulations call ``calculate_stake`` without model probabilities;
    this strategy then stakes 0. Kelly is not a valid bookie benchmark.

    Attributes:
        downscaling: Multiplier on the Kelly fraction (0.5 = half Kelly).
        max_bet: Cap as a fraction of bankroll.
        min_kelly: Minimum Kelly fraction required to bet.
        min_prob: Minimum model probability required to bet.
    """

    def __init__(
        self,
        downscaling: float = 0.5,
        max_bet: float = 0.1,
        min_kelly: float = 0,
        min_prob: float = 0,
    ):
        """Initialize Kelly parameters.

        Args:
            downscaling: Fraction of full Kelly to stake.
            max_bet: Maximum stake as a fraction of bankroll.
            min_kelly: Skip bets with Kelly at or below this value.
            min_prob: Skip bets with model probability below this value.
        """
        self.downscaling = downscaling
        self.max_bet = max_bet
        self.min_kelly = min_kelly
        self.min_prob = min_prob

    def calculate_kelly_fraction(self, odds: float, prob: float) -> float:
        """Calculate the full Kelly fraction for one outcome.

        Args:
            odds: Decimal odds. Values ``<= 1`` yield 0.
            prob: Estimated win probability in ``[0, 1]``.

        Returns:
            Kelly fraction floored at 0.

        Example:
            odds=2.0, prob=0.6 → ``(0.6*1 - 0.4)/1 = 0.2``.
        """
        adj_odds = odds - 1
        if adj_odds <= 0 or not np.isfinite(adj_odds) or not np.isfinite(prob):
            return 0.0
        kelly = (prob * adj_odds - (1 - prob)) / adj_odds
        return max(0.0, kelly)

    def calculate_stake(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> float:
        """Calculate the Kelly stake for the selected outcome.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Required; stake is 0 if missing.
            **kwargs: Passed to ``select_bet``.

        Returns:
            Stake in currency units, or 0 if no bet.
        """
        if model_probs is None:
            return 0.0

        bet_on = self.select_bet(odds, model_probs, **kwargs)
        if bet_on == -1:
            return 0.0

        kelly_fraction = self.calculate_kelly_fraction(
            odds[bet_on], model_probs[bet_on]
        )

        if model_probs[bet_on] < self.min_prob:
            return 0.0

        if kelly_fraction > self.min_kelly:
            return (
                min(kelly_fraction * self.downscaling, self.max_bet) * bankroll
            )

        return 0.0

    def select_bet(
        self,
        odds: List[float],
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Select the outcome with the highest Kelly fraction.

        Args:
            odds: Decimal odds for each outcome.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Unused; Kelly uses probabilities, not a class label.
            **kwargs: Unused.

        Returns:
            Outcome index, or ``-1`` if no valid bet.
        """
        if model_probs is None:
            return -1
        kelly_fractions = [
            self.calculate_kelly_fraction(odd, prob)
            for odd, prob in zip(odds, model_probs)
        ]
        max_kelly = max(kelly_fractions)
        if max_kelly <= self.min_kelly:
            return -1
        best_idx = kelly_fractions.index(max_kelly)
        if model_probs[best_idx] < self.min_prob:
            return -1
        return best_idx

    def get_bet_details(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Return stake, selected outcome, and per-outcome Kelly fractions.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Unused by Kelly selection.
            **kwargs: Passed to stake calculation.

        Returns:
            ``(stake, bet_on, {kelly_fraction_i: ...})``.
        """
        kelly_fractions = []
        if model_probs:
            kelly_fractions = [
                self.calculate_kelly_fraction(odd, prob)
                for odd, prob in zip(odds, model_probs)
            ]

        stake = self.calculate_stake(
            odds, bankroll, model_probs=model_probs, **kwargs
        )
        bet_on = self.select_bet(odds, model_probs, prediction, **kwargs)
        if stake == 0:
            bet_on = -1

        additional_info = {
            f"kelly_fraction_{i}": kf for i, kf in enumerate(kelly_fractions)
        }
        return stake, bet_on, additional_info

    def __str__(self) -> str:
        return (
            f"Kelly Criterion Strategy (downscaling={self.downscaling}, "
            f"max_bet={self.max_bet}, min_kelly={self.min_kelly}, "
            f"min_prob={self.min_prob})"
        )


def get_default_strategy() -> FixedStake:
    """Return the default strategy: 1% of current bankroll.

    Returns:
        ``FixedStake(0.01)``.
    """
    return FixedStake(0.01)
