"""Ready-made betting strategies for sports backtests.

FixedStake sizes bets as a constant amount or as a fraction of the
*current* bankroll. KellyCriterion sizes bets from the decimal-odds Kelly
fraction, scaled by ``downscaling`` and capped by ``max_bet``. ValueBet
selects the highest expected-value outcome above ``min_ev``. UnitStake
implements unit-loss, unit-win, and unit-impact plans.
EdgeProportionalStake stakes in proportion to the edge. OddsFilter
restricts an inner strategy to an odds band. RiskLimits caps and pauses an
inner strategy using the run's ``BetHistory``.

FixedStake, KellyCriterion, ValueBet, UnitStake, and EdgeProportionalStake
are preset combinations of the toolkit in
``backtestbuddy.strategies.toolkit`` (a selector plus a sizer). Their
constructors, attributes, methods, and results are unchanged; attributes
such as ``downscaling`` can still be reassigned after construction.

Stake sizes are explicit: ``amount=`` is currency, ``fraction=`` is a share
of the current bankroll. The older single ``stake=`` argument still works
(below 1 = fraction, 1 and above = amount) but warns, because ``0.99``
and ``1.0`` mean very different bets.
"""

from typing import Any, Dict, List, Literal, Optional, Tuple

from backtestbuddy.strategies.base import BaseStrategy
from backtestbuddy.strategies.history import BetHistory
from backtestbuddy.strategies.toolkit import (
    EdgeProportional,
    HighestEV,
    HighestKelly,
    KellySize,
    ModelPick,
    StakeSize,
    Strategy,
    UnitPlan,
    cap_stake,
    expected_value,
    kelly_fraction,
)

# Former private helpers, kept importable for existing code
_expected_value = expected_value
_cap_stake = cap_stake

__all__ = [
    "BaseStrategy",
    "StakeSize",
    "FixedStake",
    "KellyCriterion",
    "ValueBet",
    "UnitStake",
    "EdgeProportionalStake",
    "OddsFilter",
    "RiskLimits",
    "get_default_strategy",
]


class FixedStake(Strategy):
    """Flat amount or constant fraction of the current bankroll.

    - ``FixedStake(amount=10)``: 10 currency units per bet, capped at the
      current bankroll.
    - ``FixedStake(fraction=0.02)``: 2% of the *current* bankroll per bet.
    - ``FixedStake(10)`` / ``FixedStake(0.02)``: legacy form, still
      supported; below 1 is a fraction, 1 and above an amount. Warns.

    Outcome: highest model probability, else the prediction, else the
    favourite (``ModelPick`` + ``StakeSize``).

    Attributes:
        size: The ``StakeSize``.
        stake: The configured number (amount or fraction). Assigning a new
            number applies the legacy rule.
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
        size = StakeSize.from_arguments(
            "FixedStake", stake=stake, amount=amount, fraction=fraction
        )
        super().__init__(select=ModelPick(), size=size)

    @property
    def size(self) -> StakeSize:
        return self.sizer

    @property
    def stake(self) -> float:
        return self.sizer.value

    @stake.setter
    def stake(self, value: float) -> None:
        self.sizer = StakeSize.from_arguments("FixedStake", stake=value)

    def _legacy_bet(self, odds, bankroll, model_probs, prediction, kwargs):
        stake = self.calculate_stake(odds, bankroll, model_probs=model_probs)
        return stake, self.select_bet(odds, model_probs, prediction)

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
            **kwargs: ``history`` and ``date`` from the backtest; unused.

        Returns:
            ``(stake, bet_on, {})``.
        """
        return super().get_bet_details(odds, current_bankroll, model_probs, prediction, **kwargs)

    def __str__(self) -> str:
        return f"Fixed Stake Strategy ({self.size})"


class KellyCriterion(Strategy):
    """Bet the (scaled, capped) Kelly fraction of current bankroll.

    Kelly fraction for decimal odds: ``f* = (p*(odds-1) - (1-p)) / (odds-1)``.
    Negative or undefined fractions (including ``odds <= 1``) are treated as
    0. The stake is ``min(f* * downscaling, max_bet) * bankroll`` on the
    outcome with the highest Kelly fraction (``HighestKelly`` +
    ``KellySize``).

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
        # Bound method: a subclass override of calculate_kelly_fraction is honored
        super().__init__(
            select=HighestKelly(min_kelly, min_prob, kelly_fn=self.calculate_kelly_fraction),
            size=KellySize(downscaling, max_bet, min_kelly, min_prob, kelly_fn=self.calculate_kelly_fraction),
        )

    @property
    def downscaling(self) -> float:
        return self.sizer.fraction

    @downscaling.setter
    def downscaling(self, value: float) -> None:
        self.sizer.fraction = value

    @property
    def max_bet(self) -> float:
        return self.sizer.max_bet

    @max_bet.setter
    def max_bet(self, value: float) -> None:
        self.sizer.max_bet = value

    @property
    def min_kelly(self) -> float:
        return self.selector.min_kelly

    @min_kelly.setter
    def min_kelly(self, value: float) -> None:
        self.selector.min_kelly = self.sizer.min_kelly = value

    @property
    def min_prob(self) -> float:
        return self.selector.min_prob

    @min_prob.setter
    def min_prob(self, value: float) -> None:
        self.selector.min_prob = self.sizer.min_prob = value

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
        return kelly_fraction(odds, prob)

    def __str__(self) -> str:
        return (
            f"Kelly Criterion Strategy (downscaling={self.downscaling}, "
            f"max_bet={self.max_bet}, min_kelly={self.min_kelly}, "
            f"min_prob={self.min_prob})"
        )


class ValueBet(Strategy):
    """Bet only when expected value exceeds ``min_ev``.

    EV of an outcome is ``p * odds - 1``. Among outcomes with EV strictly
    greater than ``min_ev``, the highest EV is selected. The stake is an
    ``amount=`` or a ``fraction=`` of the current bankroll, as in
    FixedStake; the default is an amount of 1 (``HighestEV`` +
    ``StakeSize``).

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
        size = StakeSize.from_arguments(
            "ValueBet", stake=stake, amount=amount, fraction=fraction,
            default_amount=1.0,
        )
        super().__init__(select=HighestEV(min_ev), size=size)

    @property
    def min_ev(self) -> float:
        return self.selector.min_ev

    @min_ev.setter
    def min_ev(self, value: float) -> None:
        self.selector.min_ev = value

    @property
    def size(self) -> StakeSize:
        return self.sizer

    @property
    def stake(self) -> float:
        return self.sizer.value

    @stake.setter
    def stake(self, value: float) -> None:
        self.sizer = StakeSize.from_arguments("ValueBet", stake=value)

    def _legacy_bet(self, odds, bankroll, model_probs, prediction, kwargs):
        stake = self.calculate_stake(odds, bankroll, model_probs=model_probs, prediction=prediction, **kwargs)
        return stake, self.select_bet(odds, model_probs, prediction, **kwargs)

    def __str__(self) -> str:
        return f"ValueBet Strategy (min_ev={self.min_ev}, stake={self.stake})"


class UnitStake(Strategy):
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
        super().__init__(select=ModelPick(), size=UnitPlan(unit, plan))

    @property
    def unit(self) -> float:
        return self.sizer.unit

    @unit.setter
    def unit(self, value: float) -> None:
        self.sizer.unit = value

    @property
    def plan(self) -> str:
        return self.sizer.plan

    @plan.setter
    def plan(self, value: str) -> None:
        self.sizer.plan = value

    def _legacy_bet(self, odds, bankroll, model_probs, prediction, kwargs):
        stake = self.calculate_stake(odds, bankroll, model_probs=model_probs, prediction=prediction, **kwargs)
        return stake, self.select_bet(odds, model_probs, prediction, **kwargs)

    def __str__(self) -> str:
        return f"UnitStake Strategy (unit={self.unit}, plan={self.plan})"


class EdgeProportionalStake(Strategy):
    """Bet the highest-EV outcome with a stake proportional to its edge.

    Selects like ValueBet (highest ``p * odds - 1`` above ``min_ev``) and
    stakes ``min(multiplier * edge, max_fraction) * bankroll``
    (``HighestEV`` + ``EdgeProportional``). Larger edges get larger bets,
    with less variance than Kelly.

    Example:
        ``EdgeProportionalStake(min_ev=0.02, multiplier=0.5, max_fraction=0.05)``:
        edge 6% → 3% of the bankroll; edge 20% → capped at 5%.

    Attributes:
        min_ev: Minimum expected value to place a bet.
        multiplier: Bankroll share per unit of edge.
        max_fraction: Cap as a bankroll share.
        requires_probabilities: Always ``True``.
    """

    requires_probabilities = True

    def __init__(self, min_ev: float = 0.0, multiplier: float = 0.5, max_fraction: float = 0.05):
        """Initialize the selection threshold and sizing.

        Args:
            min_ev: Skip outcomes with EV at or below this threshold.
            multiplier: Bankroll share per unit of edge, ``> 0``.
            max_fraction: Cap as a bankroll share, ``(0, 1]``.

        Raises:
            ValueError: If ``multiplier`` or ``max_fraction`` is out of range.
        """
        super().__init__(select=HighestEV(min_ev), size=EdgeProportional(multiplier, max_fraction))

    @property
    def min_ev(self) -> float:
        return self.selector.min_ev

    @min_ev.setter
    def min_ev(self, value: float) -> None:
        self.selector.min_ev = value

    @property
    def multiplier(self) -> float:
        return self.sizer.multiplier

    @multiplier.setter
    def multiplier(self, value: float) -> None:
        self.sizer.multiplier = value

    @property
    def max_fraction(self) -> float:
        return self.sizer.max_fraction

    @max_fraction.setter
    def max_fraction(self, value: float) -> None:
        self.sizer.max_fraction = value

    def __str__(self) -> str:
        return (
            f"EdgeProportionalStake Strategy (min_ev={self.min_ev}, "
            f"multiplier={self.multiplier}, max_fraction={self.max_fraction})"
        )


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

    def adjust_simultaneous(self, stakes: List[float], bankroll: float) -> List[float]:
        """Delegates to the inner strategy."""
        return self.inner.adjust_simultaneous(stakes, bankroll)

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

    def adjust_simultaneous(self, stakes: List[float], bankroll: float) -> List[float]:
        """Let the inner strategy rescale, then drop stakes below ``min_stake``.

        Rescaling only shrinks stakes, so the other caps still hold.
        """
        adjusted = self.inner.adjust_simultaneous(stakes, bankroll)
        if self.min_stake is None:
            return adjusted
        return [0.0 if 0 < stake < self.min_stake else stake for stake in adjusted]

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
