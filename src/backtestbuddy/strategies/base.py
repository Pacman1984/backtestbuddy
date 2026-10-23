"""Base class every betting strategy implements."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


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

    def adjust_simultaneous(self, stakes: List[float], bankroll: float) -> List[float]:
        """Rescale the stakes of matches that settle at the same time.

        Called once per slot when a backtest runs with
        ``settle_simultaneous=True``, after every bet of the slot was
        decided. The default leaves the stakes unchanged.

        Args:
            stakes: Proposed stake per match of the slot (0 for no bet).
            bankroll: Bankroll at the start of the slot.

        Returns:
            Stakes in the same order.
        """
        return list(stakes)
