"""Betting strategies for sports backtests.

FixedStake sizes bets as a constant dollar amount or as a fraction of the
*current* bankroll. KellyCriterion sizes bets from the decimal-odds Kelly
fraction, scaled by ``downscaling`` and capped by ``max_bet``. ValueBet
selects the highest expected-value outcome above ``min_ev``. UnitStake
implements unit-loss, unit-win, and unit-impact plans. OddsFilter restricts
an inner strategy to an odds band.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

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


def _expected_value(odds: float, prob: float) -> float:
    """Return ``p * odds - 1`` for one outcome.

    Args:
        odds: Decimal odds.
        prob: Win probability.

    Returns:
        Expected value of a 1-unit stake. ``-inf`` if inputs are invalid.

    Example:
        odds=2.0, prob=0.6 → ``0.2``.
    """
    if (
        odds <= 0
        or not np.isfinite(odds)
        or not np.isfinite(prob)
    ):
        return float("-inf")
    return float(prob * odds - 1.0)


def _cap_stake(stake: float, bankroll: float) -> float:
    """Return a non-negative stake that does not exceed bankroll."""
    if stake <= 0 or bankroll <= 0:
        return 0.0
    return float(min(stake, bankroll))


class ValueBet(BaseStrategy):
    """Bet only when expected value exceeds ``min_ev``.

    EV of an outcome is ``p * odds - 1``. Among outcomes with EV strictly
    greater than ``min_ev``, the highest EV is selected. Stake follows
    FixedStake rules (absolute if ``stake >= 1``, else a fraction of
    current bankroll).

    Requires ``model_probs``. Without them the stake is 0, so this is not
    a valid bookie benchmark.

    Example:
        ``ValueBet(min_ev=0.05, stake=10)`` stakes $10 on the highest-EV
        outcome when ``p * odds - 1 > 0.05``.

    Attributes:
        min_ev: Minimum expected value to place a bet.
        stake: Absolute amount (>= 1) or bankroll fraction (< 1).
        requires_probabilities: Always ``True``.
    """

    requires_probabilities = True

    def __init__(self, min_ev: float = 0.0, stake: float = 1.0):
        """Initialize the value filter and stake size.

        Args:
            min_ev: Skip outcomes with EV at or below this threshold.
            stake: Absolute amount (>= 1) or current-bankroll fraction (< 1).
        """
        self.min_ev = min_ev
        self.stake = stake

    def calculate_stake(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> float:
        """Return the FixedStake-style size if a value bet exists.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Required; stake is 0 if missing.
            **kwargs: Passed to ``select_bet``.

        Returns:
            Stake in currency units, or 0 if no bet.
        """
        if self.select_bet(odds, model_probs, **kwargs) == -1:
            return 0.0
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
        """Select the highest-EV outcome above ``min_ev``.

        Args:
            odds: Decimal odds for each outcome.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Unused; selection is by EV, not a class label.
            **kwargs: Unused.

        Returns:
            Outcome index, or ``-1`` if no outcome clears ``min_ev``.
        """
        if model_probs is None or len(model_probs) != len(odds):
            return -1
        values = [
            _expected_value(odd, prob)
            for odd, prob in zip(odds, model_probs)
        ]
        best = max(values)
        if best <= self.min_ev:
            return -1
        return values.index(best)

    def get_bet_details(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Return stake, selected outcome, and per-outcome EV.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Unused.
            **kwargs: Passed to stake calculation.

        Returns:
            ``(stake, bet_on, {ev_i: ...})``.
        """
        evs: List[float] = []
        if model_probs and len(model_probs) == len(odds):
            evs = [
                _expected_value(odd, prob)
                for odd, prob in zip(odds, model_probs)
            ]
        stake = self.calculate_stake(
            odds,
            bankroll,
            model_probs=model_probs,
            prediction=prediction,
            **kwargs,
        )
        bet_on = self.select_bet(odds, model_probs, prediction, **kwargs)
        if stake == 0:
            bet_on = -1
        additional_info = {f"ev_{i}": ev for i, ev in enumerate(evs)}
        return stake, bet_on, additional_info

    def __str__(self) -> str:
        return (
            f"ValueBet Strategy (min_ev={self.min_ev}, stake={self.stake})"
        )


class UnitStake(BaseStrategy):
    """Cortés unit-loss, unit-win, or unit-impact staking.

    Selection matches FixedStake (max prob, else prediction, else favorite).

    Plans (``unit`` is the constant, capped at current bankroll):

    - ``loss``: stake = unit (flat risk if the bet loses).
    - ``win``: stake = unit / (odds - 1) (net win equals unit). Odds
      ``<= 1`` skip.
    - ``impact``: stake = unit / odds (win vs loss difference is unit).
      Odds ``<= 0`` skip.

    Example:
        ``UnitStake(unit=90, plan="win")`` at odds 3.0 stakes 45 so the
        net win is 90.

    Attributes:
        unit: Staking constant in currency units.
        plan: ``loss``, ``win``, or ``impact``.
    """

    def __init__(
        self,
        unit: float = 1.0,
        plan: Literal["loss", "win", "impact"] = "loss",
    ):
        """Initialize the unit plan.

        Args:
            unit: Staking constant (must be positive).
            plan: ``loss``, ``win``, or ``impact``.

        Raises:
            ValueError: If ``plan`` is unknown or ``unit`` is not positive.
        """
        if plan not in ("loss", "win", "impact"):
            raise ValueError(
                f"plan must be 'loss', 'win', or 'impact', got {plan!r}"
            )
        if unit <= 0:
            raise ValueError("unit must be positive")
        self.unit = unit
        self.plan = plan

    def calculate_stake(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> float:
        """Calculate the unit-plan stake for the selected outcome.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Optional; used only for selection.
            **kwargs: Passed to ``select_bet``.

        Returns:
            Stake in currency units, or 0 if no bet.
        """
        bet_on = self.select_bet(odds, model_probs, **kwargs)
        if bet_on == -1:
            return 0.0
        selected = odds[bet_on]
        if self.plan == "loss":
            return _cap_stake(self.unit, bankroll)
        if self.plan == "win":
            if selected <= 1:
                return 0.0
            return _cap_stake(self.unit / (selected - 1), bankroll)
        if selected <= 0:
            return 0.0
        return _cap_stake(self.unit / selected, bankroll)

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
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Return stake, selected outcome, and plan metadata.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Model probabilities aligned with ``odds``.
            prediction: Predicted outcome index.
            **kwargs: Passed to stake calculation.

        Returns:
            ``(stake, bet_on, {plan, unit})``.
        """
        stake = self.calculate_stake(
            odds,
            bankroll,
            model_probs=model_probs,
            prediction=prediction,
            **kwargs,
        )
        bet_on = self.select_bet(odds, model_probs, prediction, **kwargs)
        if stake == 0:
            bet_on = -1
        return stake, bet_on, {"plan": self.plan, "unit": self.unit}

    def __str__(self) -> str:
        return f"UnitStake Strategy (unit={self.unit}, plan={self.plan})"


class OddsFilter(BaseStrategy):
    """Restrict an inner strategy to outcomes inside an odds band.

    Only outcomes with ``min_odds <= odds <= max_odds`` are passed to
    ``inner``. Default inner strategy is ``FixedStake(stake)``.

    Example:
        ``OddsFilter(min_odds=1.5, inner=UnitStake(90, "loss"))`` skips
        heavy favorites below 1.5 and flat-stakes the rest.

    Attributes:
        min_odds: Inclusive lower bound on decimal odds.
        max_odds: Inclusive upper bound on decimal odds.
        inner: Strategy used for selection and sizing on the filtered set.
    """

    def __init__(
        self,
        min_odds: float = 1.01,
        max_odds: float = float("inf"),
        inner: Optional[BaseStrategy] = None,
        stake: float = 1.0,
    ):
        """Initialize the odds band and inner strategy.

        Args:
            min_odds: Inclusive minimum decimal odds.
            max_odds: Inclusive maximum decimal odds.
            inner: Wrapped strategy. Defaults to ``FixedStake(stake)``.
            stake: Used only when ``inner`` is omitted.
        """
        self.min_odds = min_odds
        self.max_odds = max_odds
        self.inner = inner if inner is not None else FixedStake(stake)

    @property
    def requires_probabilities(self) -> bool:
        """True when the inner strategy requires model probabilities."""
        return bool(getattr(self.inner, "requires_probabilities", False))

    def _filtered_view(
        self,
        odds: List[float],
        model_probs: Optional[List[float]],
        prediction: Optional[int],
    ) -> Optional[
        Tuple[List[int], List[float], Optional[List[float]], Optional[int]]
    ]:
        """Map the full market onto in-band odds, probs, and prediction.

        Args:
            odds: Decimal odds for each outcome.
            model_probs: Optional probabilities aligned with ``odds``.
            prediction: Predicted outcome index in the original list.

        Returns:
            ``(eligible, sub_odds, sub_probs, sub_pred)``, or ``None`` if
            no outcome is in the band.
        """
        eligible = [
            i
            for i, odd in enumerate(odds)
            if self.min_odds <= odd <= self.max_odds
        ]
        if not eligible:
            return None
        sub_odds = [odds[i] for i in eligible]
        sub_probs = (
            [model_probs[i] for i in eligible] if model_probs else None
        )
        sub_pred = None
        if prediction is not None and prediction in eligible:
            sub_pred = eligible.index(prediction)
        return eligible, sub_odds, sub_probs, sub_pred

    def calculate_stake(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> float:
        """Size the bet using the inner strategy on in-band outcomes.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Optional probabilities aligned with ``odds``.
            **kwargs: Passed to the inner strategy (``prediction`` remapped).

        Returns:
            Inner stake, or 0 if no outcome is in the band.
        """
        prediction = kwargs.pop("prediction", None)
        view = self._filtered_view(odds, model_probs, prediction)
        if view is None:
            return 0.0
        _, sub_odds, sub_probs, sub_pred = view
        return self.inner.calculate_stake(
            sub_odds,
            bankroll,
            model_probs=sub_probs,
            prediction=sub_pred,
            **kwargs,
        )

    def select_bet(
        self,
        odds: List[float],
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Select with the inner strategy among in-band outcomes.

        Args:
            odds: Decimal odds for each outcome.
            model_probs: Optional probabilities aligned with ``odds``.
            prediction: Mapped into the in-band list when eligible.
            **kwargs: Passed to the inner strategy.

        Returns:
            Original outcome index, or ``-1`` if none are in band.
        """
        view = self._filtered_view(odds, model_probs, prediction)
        if view is None:
            return -1
        eligible, sub_odds, sub_probs, sub_pred = view
        inner_idx = self.inner.select_bet(
            sub_odds, sub_probs, sub_pred, **kwargs
        )
        if inner_idx < 0:
            return -1
        return eligible[inner_idx]

    def get_bet_details(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Return inner bet details with the outcome index remapped.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Optional probabilities aligned with ``odds``.
            prediction: Predicted outcome index in the original list.
            **kwargs: Passed to the inner strategy.

        Returns:
            ``(stake, bet_on, inner_info)``.
        """
        view = self._filtered_view(odds, model_probs, prediction)
        if view is None:
            return 0.0, -1, {}
        eligible, sub_odds, sub_probs, sub_pred = view
        stake, inner_on, info = self.inner.get_bet_details(
            sub_odds, bankroll, sub_probs, sub_pred, **kwargs
        )
        if inner_on < 0 or stake == 0:
            return 0.0, -1, info
        return stake, eligible[inner_on], info

    def __str__(self) -> str:
        return (
            f"OddsFilter Strategy (min_odds={self.min_odds}, "
            f"max_odds={self.max_odds}, inner={self.inner})"
        )


def get_default_strategy() -> FixedStake:
    """Return the default strategy: 1% of current bankroll.

    Returns:
        ``FixedStake(0.01)``.
    """
    return FixedStake(0.01)
