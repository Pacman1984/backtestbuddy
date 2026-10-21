"""Betting strategies for sports backtests.

FixedStake sizes bets as a constant amount or as a fraction of the
*current* bankroll. KellyCriterion sizes bets from the decimal-odds Kelly
fraction, scaled by ``downscaling`` and capped by ``max_bet``. ValueBet
selects the highest expected-value outcome above ``min_ev``. UnitStake
implements unit-loss, unit-win, and unit-impact plans. OddsFilter restricts
an inner strategy to an odds band. RiskLimits caps and pauses an inner
strategy using the run's ``BetHistory``.

Stake sizes are explicit: ``amount=`` is currency, ``fraction=`` is a share
of the current bankroll. The older single ``stake=`` argument still works
(below 1 = fraction, 1 and above = amount) but warns, because ``0.99``
and ``1.0`` mean very different bets.
"""

import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np

from backtestbuddy.strategies.history import BetHistory


class StakeSize:
    """How much to stake: a fixed amount or a fraction of the bankroll.

    Exactly one of ``amount`` and ``fraction`` is set.

    Attributes:
        amount: Currency amount per bet, capped at the current bankroll.
        fraction: Share of the current bankroll per bet, in ``[0, 1]``.

    Example:
        >>> StakeSize(fraction=0.02).resolve(1000)
        20.0
        >>> StakeSize(amount=0.5).resolve(1000)  # half a unit, not 50%
        0.5
    """

    def __init__(
        self, amount: Optional[float] = None, fraction: Optional[float] = None
    ):
        """Create a stake size.

        Args:
            amount: Currency amount, ``>= 0``.
            fraction: Bankroll fraction in ``[0, 1]``.

        Raises:
            ValueError: If both or neither are given, or a value is out of
                range.
        """
        if (amount is None) == (fraction is None):
            raise ValueError("Pass exactly one of amount= or fraction=")
        if amount is not None and amount < 0:
            raise ValueError("amount must be non-negative")
        if fraction is not None and not 0 <= fraction <= 1:
            raise ValueError("fraction must be in [0, 1]")
        self.amount = amount
        self.fraction = fraction

    @classmethod
    def from_arguments(
        cls,
        owner: str,
        stake: Optional[float] = None,
        amount: Optional[float] = None,
        fraction: Optional[float] = None,
        default_amount: Optional[float] = None,
        stacklevel: int = 3,
    ) -> "StakeSize":
        """Build a stake size from a strategy's constructor arguments.

        Args:
            owner: Strategy name for messages.
            stake: Legacy argument: below 1 is a fraction, else an amount.
                Emits a ``DeprecationWarning``.
            amount: Explicit amount.
            fraction: Explicit fraction.
            default_amount: Used when nothing is given; ``None`` makes a
                size mandatory.
            stacklevel: Warning stack level so it points at user code.

        Returns:
            The resolved stake size.

        Raises:
            ValueError: If more than one argument is given, a mandatory
                size is missing, or a value is out of range.
        """
        if sum(value is not None for value in (stake, amount, fraction)) > 1:
            raise ValueError(f"{owner}: pass only one of stake=, amount=, fraction=")
        if stake is not None:
            if stake < 0:
                raise ValueError("stake must be non-negative")
            warnings.warn(
                f"{owner}(stake={stake!r}) is ambiguous: values below 1 are a "
                "bankroll fraction, 1 and above a currency amount. Use "
                "amount=... or fraction=... instead.",
                DeprecationWarning,
                stacklevel=stacklevel,
            )
            return cls(fraction=stake) if stake < 1 else cls(amount=stake)
        if amount is None and fraction is None:
            if default_amount is None:
                raise ValueError(f"{owner} needs amount=... or fraction=...")
            return cls(amount=default_amount)
        return cls(amount=amount, fraction=fraction)

    def resolve(self, bankroll: float) -> float:
        """Stake for the given current bankroll."""
        if self.fraction is not None:
            return bankroll * self.fraction
        return min(self.amount, bankroll)

    @property
    def value(self) -> float:
        """The configured number: the amount or the fraction."""
        return self.amount if self.amount is not None else self.fraction

    def __str__(self) -> str:
        if self.fraction is not None:
            return f"{self.fraction:.2%} of current bankroll"
        return f"${self.amount:.2f}"


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
    """Flat amount or constant fraction of the current bankroll.

    - ``FixedStake(amount=10)``: 10 currency units per bet, capped at the
      current bankroll.
    - ``FixedStake(fraction=0.02)``: 2% of the *current* bankroll per bet.
    - ``FixedStake(10)`` / ``FixedStake(0.02)``: legacy form, still
      supported; below 1 is a fraction, 1 and above an amount. Warns.

    Attributes:
        size: The ``StakeSize``.
        stake: The configured number (amount or fraction), kept for
            backward compatibility.
    """

    def __init__(
        self,
        stake: Optional[float] = None,
        *,
        amount: Optional[float] = None,
        fraction: Optional[float] = None,
    ):
        """Initialize the strategy.

        Args:
            stake: Legacy size: fraction if below 1, else amount.
            amount: Currency amount per bet.
            fraction: Share of the current bankroll per bet.

        Raises:
            ValueError: If no size or more than one is given, or a value
                is negative or out of range.
        """
        self.size = StakeSize.from_arguments(
            "FixedStake", stake=stake, amount=amount, fraction=fraction
        )
        self.stake = self.size.value

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
            Fraction of current bankroll, or min(amount, bankroll).
        """
        return self.size.resolve(bankroll)

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
        return f"Fixed Stake Strategy ({self.size})"


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
        requires_probabilities: Always ``True``.
    """

    requires_probabilities = True

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

        Raises:
            ValueError: If ``downscaling`` is negative or ``max_bet`` is
                outside ``[0, 1]``.
        """
        if downscaling < 0:
            raise ValueError("downscaling must be non-negative")
        if not 0 <= max_bet <= 1:
            raise ValueError("max_bet must be in [0, 1]")
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
    greater than ``min_ev``, the highest EV is selected. The stake is an
    ``amount=`` or a ``fraction=`` of the current bankroll, as in
    FixedStake; the default is an amount of 1.

    Requires ``model_probs``. Without them the stake is 0, so this is not
    a valid bookie benchmark.

    Example:
        ``ValueBet(min_ev=0.05, amount=10)`` stakes 10 on the highest-EV
        outcome when ``p * odds - 1 > 0.05``.

    Attributes:
        min_ev: Minimum expected value to place a bet.
        size: The ``StakeSize``.
        stake: The configured number (amount or fraction), kept for
            backward compatibility.
        requires_probabilities: Always ``True``.
    """

    requires_probabilities = True

    def __init__(
        self,
        min_ev: float = 0.0,
        stake: Optional[float] = None,
        *,
        amount: Optional[float] = None,
        fraction: Optional[float] = None,
    ):
        """Initialize the value filter and stake size.

        Args:
            min_ev: Skip outcomes with EV at or below this threshold.
            stake: Legacy size: fraction if below 1, else amount. Warns.
            amount: Currency amount per bet.
            fraction: Share of the current bankroll per bet.

        Raises:
            ValueError: If more than one size is given or a value is out
                of range.
        """
        self.size = StakeSize.from_arguments(
            "ValueBet", stake=stake, amount=amount, fraction=fraction,
            default_amount=1.0,
        )
        self.min_ev = min_ev
        self.stake = self.size.value

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
        return self.size.resolve(bankroll)

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
    ``inner``. Without ``inner``, a ``FixedStake`` with the given size is
    used (amount 1 by default).

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
        stake: Optional[float] = None,
        *,
        amount: Optional[float] = None,
        fraction: Optional[float] = None,
    ):
        """Initialize the odds band and inner strategy.

        Args:
            min_odds: Inclusive minimum decimal odds.
            max_odds: Inclusive maximum decimal odds.
            inner: Wrapped strategy. Defaults to a ``FixedStake``.
            stake: Legacy size for the default inner strategy. Warns.
            amount: Amount for the default inner strategy.
            fraction: Bankroll fraction for the default inner strategy.
        """
        self.min_odds = min_odds
        self.max_odds = max_odds
        if inner is None:
            size = StakeSize.from_arguments(
                "OddsFilter", stake=stake, amount=amount, fraction=fraction,
                default_amount=1.0,
            )
            inner = FixedStake(amount=size.amount, fraction=size.fraction)
        self.inner = inner

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


class RiskLimits(BaseStrategy):
    """Cap and pause an inner strategy using the run's ``BetHistory``.

    The inner strategy picks the outcome and a stake; the limits then
    shrink or block it. Rules, in order:

    1. ``stop_at_drawdown``: no more bets once the bankroll is this far
       below its peak (``0.3`` = 30%). Permanent for the run: without bets
       the bankroll cannot recover.
    2. ``pause_after_losses``: no more bets for the rest of the calendar
       day after this many consecutive losses on that day.
    3. ``max_stake_fraction``: stake at most this share of the current
       bankroll.
    4. ``max_stake``: stake at most this amount.
    5. ``max_daily_fraction``: total stake per calendar day at most this
       share of the bankroll before the day's first bet.
    6. ``min_stake``: skip bets whose stake is below this amount, e.g. a
       bookmaker minimum.

    The rule that changed or blocked a bet is stored as ``risk_limit``
    (results column ``bt_risk_limit``; empty when no rule applied). Rules
    1, 2, and 5 need the history and date the backtest passes in; when
    called without them (e.g. the bookie benchmark) only the stateless
    rules 3, 4, and 6 apply.

    Example:
        ``RiskLimits(KellyCriterion(), max_stake_fraction=0.05,
        stop_at_drawdown=0.3)`` bets Kelly, at most 5% per bet, and stops
        after a 30% drawdown.

    Attributes:
        inner: Wrapped strategy.
    """

    def __init__(
        self,
        inner: BaseStrategy,
        *,
        max_stake_fraction: Optional[float] = None,
        max_stake: Optional[float] = None,
        max_daily_fraction: Optional[float] = None,
        min_stake: Optional[float] = None,
        stop_at_drawdown: Optional[float] = None,
        pause_after_losses: Optional[int] = None,
    ):
        """Wrap a strategy with risk limits; every limit is optional.

        Args:
            inner: Strategy that selects and sizes the bet.
            max_stake_fraction: Per-bet cap as a bankroll share, ``(0, 1]``.
            max_stake: Per-bet cap as an amount, ``> 0``.
            max_daily_fraction: Daily stake cap as a share of the bankroll
                at the start of the day, ``(0, 1]``.
            min_stake: Minimum amount; smaller bets are skipped, ``> 0``.
            stop_at_drawdown: Drawdown that ends betting, ``(0, 1]``.
            pause_after_losses: Same-day losing streak that pauses betting
                until the next day, ``>= 1``.

        Raises:
            ValueError: If a limit is out of range.
        """
        for name, value in (
            ("max_stake_fraction", max_stake_fraction),
            ("max_daily_fraction", max_daily_fraction),
            ("stop_at_drawdown", stop_at_drawdown),
        ):
            if value is not None and not 0 < value <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name, value in (("max_stake", max_stake), ("min_stake", min_stake)):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if pause_after_losses is not None and pause_after_losses < 1:
            raise ValueError("pause_after_losses must be at least 1")
        self.inner = inner
        self.max_stake_fraction = max_stake_fraction
        self.max_stake = max_stake
        self.max_daily_fraction = max_daily_fraction
        self.min_stake = min_stake
        self.stop_at_drawdown = stop_at_drawdown
        self.pause_after_losses = pause_after_losses

    @property
    def requires_probabilities(self) -> bool:
        """True when the inner strategy requires model probabilities."""
        return bool(getattr(self.inner, "requires_probabilities", False))

    def apply_limits(
        self,
        stake: float,
        bankroll: float,
        history: Optional[BetHistory] = None,
        date: Any = None,
    ) -> Tuple[float, Optional[str]]:
        """Shrink or block a proposed stake.

        Args:
            stake: Stake proposed by the inner strategy.
            bankroll: Current bankroll.
            history: Settled bets of this run, if available.
            date: Date of the current match, if available.

        Returns:
            ``(stake, rule)``: the allowed stake (0 to skip) and the name of
            the last rule that changed it, or None.

        Example:
            ``max_stake=50``, proposed 80 → ``(50, "max_stake")``.
        """
        if history is not None:
            if (
                self.stop_at_drawdown is not None
                and history.drawdown >= self.stop_at_drawdown
            ):
                return 0.0, "stop_at_drawdown"
            if (
                self.pause_after_losses is not None
                and date is not None
                and history.losing_streak_on(date) >= self.pause_after_losses
            ):
                return 0.0, "pause_after_losses"

        rule = None
        if (
            self.max_stake_fraction is not None
            and stake > bankroll * self.max_stake_fraction
        ):
            stake, rule = bankroll * self.max_stake_fraction, "max_stake_fraction"
        if self.max_stake is not None and stake > self.max_stake:
            stake, rule = self.max_stake, "max_stake"
        if (
            self.max_daily_fraction is not None
            and history is not None
            and date is not None
        ):
            allowed = (
                history.bankroll_at_start_of(date) * self.max_daily_fraction
                - history.staked_on(date)
            )
            if stake > allowed:
                stake, rule = max(allowed, 0.0), "max_daily_fraction"
        if self.min_stake is not None and stake < self.min_stake:
            return 0.0, "min_stake"
        if stake <= 0:
            return 0.0, rule
        return stake, rule

    def calculate_stake(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> float:
        """Inner stake after the limits.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Passed to the inner strategy.
            **kwargs: ``history`` and ``date`` are used here and passed on.

        Returns:
            Limited stake, 0 to skip.
        """
        stake = self.inner.calculate_stake(
            odds, bankroll, model_probs=model_probs, **kwargs
        )
        if stake <= 0:
            return 0.0
        limited, _ = self.apply_limits(
            stake, bankroll, kwargs.get("history"), kwargs.get("date")
        )
        return limited

    def select_bet(
        self,
        odds: List[float],
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Outcome chosen by the inner strategy."""
        return self.inner.select_bet(odds, model_probs, prediction, **kwargs)

    def get_bet_details(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Inner bet details with the limits applied.

        Args:
            odds: Decimal odds for each outcome.
            bankroll: Current bankroll.
            model_probs: Passed to the inner strategy.
            prediction: Passed to the inner strategy.
            **kwargs: ``history`` and ``date`` are used here and passed on,
                so wrappers can be nested.

        Returns:
            ``(stake, bet_on, info)`` with ``info["risk_limit"]`` set.
        """
        stake, bet_on, info = self.inner.get_bet_details(
            odds, bankroll, model_probs, prediction, **kwargs
        )
        info = {**info, "risk_limit": None}
        if bet_on == -1 or stake <= 0:
            return 0.0, -1, info
        limited, rule = self.apply_limits(
            stake, bankroll, kwargs.get("history"), kwargs.get("date")
        )
        info["risk_limit"] = rule
        if limited <= 0:
            return 0.0, -1, info
        return limited, bet_on, info

    def __str__(self) -> str:
        limits = {
            "max_stake_fraction": self.max_stake_fraction,
            "max_stake": self.max_stake,
            "max_daily_fraction": self.max_daily_fraction,
            "min_stake": self.min_stake,
            "stop_at_drawdown": self.stop_at_drawdown,
            "pause_after_losses": self.pause_after_losses,
        }
        active = ", ".join(f"{k}={v}" for k, v in limits.items() if v is not None)
        return f"RiskLimits ({active or 'no limits'}, inner={self.inner})"


def get_default_strategy() -> FixedStake:
    """Return the default strategy: 1% of current bankroll.

    Returns:
        ``FixedStake(fraction=0.01)``.
    """
    return FixedStake(fraction=0.01)
