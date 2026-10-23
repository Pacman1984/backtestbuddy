"""Strategy toolkit: combine an outcome selector with a stake sizer.

A betting decision has two independent parts: *which* outcome to back and
*how much* to stake on it. A ``Selector`` answers the first, a ``Sizer``
the second, and ``Strategy`` combines any selector with any sizer:

    >>> import backtestbuddy as btb
    >>> strategy = btb.Strategy(select=btb.HighestEV(min_ev=0.03),
    ...                         size=btb.KellySize(fraction=0.5, max_bet=0.05))

Both parts receive a ``BetContext`` with everything known about the match.
Writing a strategy as a single ``BaseStrategy`` subclass still works; the
toolkit is an additional, shorter path. The ready-made classes such as
``KellyCriterion`` and ``ValueBet`` are preset combinations of these parts.
"""

import functools
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import numpy as np

from backtestbuddy.strategies.base import BaseStrategy
from backtestbuddy.strategies.history import BetHistory


def kelly_fraction(odds: float, prob: float) -> float:
    """Full Kelly fraction for decimal odds, floored at 0.

    ``f* = (p * (odds - 1) - (1 - p)) / (odds - 1)``. Odds ``<= 1`` or
    non-finite inputs give 0.

    Example:
        odds=2.0, prob=0.6 → ``0.2``.
    """
    adj_odds = odds - 1
    if adj_odds <= 0 or not np.isfinite(adj_odds) or not np.isfinite(prob):
        return 0.0
    kelly = (prob * adj_odds - (1 - prob)) / adj_odds
    return max(0.0, kelly)


def expected_value(odds: float, prob: float) -> float:
    """Expected value of a 1-unit stake: ``p * odds - 1``; ``-inf`` if invalid.

    Example:
        odds=2.0, prob=0.6 → ``0.2``.
    """
    if odds <= 0 or not np.isfinite(odds) or not np.isfinite(prob):
        return float("-inf")
    return float(prob * odds - 1.0)


def cap_stake(stake: float, bankroll: float) -> float:
    """Non-negative stake that does not exceed the bankroll."""
    if stake <= 0 or bankroll <= 0:
        return 0.0
    return float(min(stake, bankroll))


class BetContext:
    """Everything a selector or sizer may use for one match.

    Attributes:
        odds: Decimal odds per outcome.
        bankroll: Bankroll the stake is sized from.
        model_probs: Model probabilities per outcome, if available.
        prediction: Predicted outcome index, if available.
        history: Settled bets of the current run (``BetHistory``), if any.
        date: Match date, if known.
        cache: Scratch space so a sizer can reuse what the selector computed.
    """

    __slots__ = ("odds", "bankroll", "model_probs", "prediction", "history", "date", "cache")

    def __init__(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        history: Optional[BetHistory] = None,
        date: Any = None,
    ):
        self.odds = odds
        self.bankroll = bankroll
        self.model_probs = model_probs
        self.prediction = prediction
        self.history = history
        self.date = date
        self.cache: Dict[str, Any] = {}


class Selector(ABC):
    """Chooses the outcome to bet on.

    Attributes:
        requires_probabilities: True if ``select`` needs ``model_probs``.
    """

    requires_probabilities = False

    @abstractmethod
    def select(self, ctx: BetContext) -> int:
        """Return the outcome index, or ``-1`` to skip the match."""

    def info(self, ctx: BetContext) -> Dict[str, Any]:
        """Extra values stored as ``bt_{key}`` result columns."""
        return {}


class Sizer(ABC):
    """Decides the stake on the selected outcome.

    Attributes:
        requires_probabilities: True if ``size`` needs ``model_probs``.
    """

    requires_probabilities = False

    @abstractmethod
    def size(self, ctx: BetContext, outcome: int) -> float:
        """Return the stake for ``outcome`` (0 for no bet)."""

    def info(self, ctx: BetContext, outcome: int) -> Dict[str, Any]:
        """Extra values stored as ``bt_{key}`` result columns."""
        return {}

    def adjust_simultaneous(self, stakes: List[float], bankroll: float) -> List[float]:
        """Rescale the stakes of one simultaneous slot; unchanged by default."""
        return list(stakes)


# ---------------------------------------------------------------- selectors


class ModelPick(Selector):
    """Highest model probability, else the prediction, else the favourite."""

    def select(self, ctx: BetContext) -> int:
        if ctx.model_probs:
            return ctx.model_probs.index(max(ctx.model_probs))
        if ctx.prediction is not None:
            return ctx.prediction
        return ctx.odds.index(min(ctx.odds))


class HighestProbability(Selector):
    """Outcome with the highest model probability."""

    requires_probabilities = True

    def select(self, ctx: BetContext) -> int:
        if not ctx.model_probs:
            return -1
        return ctx.model_probs.index(max(ctx.model_probs))


class Prediction(Selector):
    """The predicted outcome; skips matches without a prediction."""

    def select(self, ctx: BetContext) -> int:
        return -1 if ctx.prediction is None else ctx.prediction


class Favourite(Selector):
    """The outcome with the lowest odds."""

    def select(self, ctx: BetContext) -> int:
        return ctx.odds.index(min(ctx.odds))


class HighestEV(Selector):
    """Highest expected value ``p * odds - 1``, only if above ``min_ev``.

    Attributes:
        min_ev: Skip matches whose best EV is at or below this value.
    """

    requires_probabilities = True

    def __init__(self, min_ev: float = 0.0):
        self.min_ev = min_ev

    def _values(self, ctx: BetContext) -> Optional[List[float]]:
        if ctx.model_probs is None or len(ctx.model_probs) != len(ctx.odds):
            return None
        values = ctx.cache.get("ev")
        if values is None:
            values = ctx.cache["ev"] = [
                expected_value(odd, prob) for odd, prob in zip(ctx.odds, ctx.model_probs)
            ]
        return values

    def select(self, ctx: BetContext) -> int:
        values = self._values(ctx)
        if values is None:
            return -1
        best = max(values)
        if best <= self.min_ev:
            return -1
        return values.index(best)

    def info(self, ctx: BetContext) -> Dict[str, Any]:
        values = self._values(ctx) if ctx.model_probs else None
        return {f"ev_{i}": ev for i, ev in enumerate(values or [])}


class HighestKelly(Selector):
    """Highest Kelly fraction, only if above ``min_kelly`` and ``min_prob``.

    Attributes:
        min_kelly: Skip matches whose best Kelly fraction is at or below this.
        min_prob: Skip if the chosen outcome's probability is below this.
        kelly_fn: Function ``(odds, prob) -> fraction``; ``kelly_fraction``
            by default.
    """

    requires_probabilities = True

    def __init__(
        self,
        min_kelly: float = 0.0,
        min_prob: float = 0.0,
        kelly_fn: Optional[Callable[[float, float], float]] = None,
    ):
        self.min_kelly = min_kelly
        self.min_prob = min_prob
        self.kelly_fn = kelly_fn or kelly_fraction

    def fractions(self, ctx: BetContext) -> List[float]:
        """Kelly fraction of every outcome (cached on the context)."""
        values = ctx.cache.get("kelly")
        if values is None:
            values = ctx.cache["kelly"] = [
                self.kelly_fn(odd, prob) for odd, prob in zip(ctx.odds, ctx.model_probs)
            ]
        return values

    def select(self, ctx: BetContext) -> int:
        if ctx.model_probs is None:
            return -1
        values = self.fractions(ctx)
        best = max(values)
        if best <= self.min_kelly:
            return -1
        index = values.index(best)
        if ctx.model_probs[index] < self.min_prob:
            return -1
        return index

    def info(self, ctx: BetContext) -> Dict[str, Any]:
        if not ctx.model_probs:
            return {}
        return {f"kelly_fraction_{i}": kf for i, kf in enumerate(self.fractions(ctx))}


# ------------------------------------------------------------------- sizers


class StakeSize(Sizer):
    """A fixed amount or a fraction of the bankroll.

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

    def __init__(self, amount: Optional[float] = None, fraction: Optional[float] = None):
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

    def size(self, ctx: BetContext, outcome: int) -> float:
        return self.resolve(ctx.bankroll)

    @property
    def value(self) -> float:
        """The configured number: the amount or the fraction."""
        return self.amount if self.amount is not None else self.fraction

    def __str__(self) -> str:
        if self.fraction is not None:
            return f"{self.fraction:.2%} of current bankroll"
        return f"${self.amount:.2f}"


class KellySize(Sizer):
    """Scaled, capped Kelly stake: ``min(f* * fraction, max_bet) * bankroll``.

    Attributes:
        fraction: Share of full Kelly (0.5 = half Kelly), ``>= 0``.
        max_bet: Cap as a bankroll share, ``[0, 1]``.
        min_kelly: Stake 0 unless the outcome's Kelly fraction exceeds this.
        min_prob: Stake 0 if the outcome's probability is below this.
        kelly_fn: Function ``(odds, prob) -> fraction``.
    """

    requires_probabilities = True

    def __init__(
        self,
        fraction: float = 0.5,
        max_bet: float = 0.1,
        min_kelly: float = 0.0,
        min_prob: float = 0.0,
        kelly_fn: Optional[Callable[[float, float], float]] = None,
    ):
        if fraction < 0:
            raise ValueError("fraction must be non-negative")
        if not 0 <= max_bet <= 1:
            raise ValueError("max_bet must be in [0, 1]")
        self.fraction = fraction
        self.max_bet = max_bet
        self.min_kelly = min_kelly
        self.min_prob = min_prob
        self.kelly_fn = kelly_fn or kelly_fraction

    def size(self, ctx: BetContext, outcome: int) -> float:
        if ctx.model_probs is None:
            return 0.0
        cached = ctx.cache.get("kelly")
        kf = cached[outcome] if cached is not None else self.kelly_fn(ctx.odds[outcome], ctx.model_probs[outcome])
        if ctx.model_probs[outcome] < self.min_prob:
            return 0.0
        if kf > self.min_kelly:
            return min(kf * self.fraction, self.max_bet) * ctx.bankroll
        return 0.0


class SimultaneousKelly(KellySize):
    """Kelly sizing that also caps the total stake of simultaneous matches.

    Each bet is sized like ``KellySize``. In a backtest with
    ``settle_simultaneous=True``, if the stakes of one slot add up to more
    than ``max_total`` of the slot's opening bankroll, all of them are
    scaled down by the same factor.

    Attributes:
        max_total: Cap on the summed stake of one slot as a bankroll share.

    Example:
        Three bets of 8% each with ``max_total=0.15`` → each scaled to 5%.
    """

    def __init__(
        self,
        fraction: float = 0.5,
        max_bet: float = 0.1,
        max_total: float = 0.3,
        min_kelly: float = 0.0,
        min_prob: float = 0.0,
        kelly_fn: Optional[Callable[[float, float], float]] = None,
    ):
        super().__init__(fraction, max_bet, min_kelly, min_prob, kelly_fn)
        if not 0 < max_total <= 1:
            raise ValueError("max_total must be in (0, 1]")
        self.max_total = max_total

    def adjust_simultaneous(self, stakes: List[float], bankroll: float) -> List[float]:
        total = sum(stakes)
        limit = self.max_total * bankroll
        if total <= limit or total <= 0:
            return list(stakes)
        scale = limit / total
        return [stake * scale for stake in stakes]


class UnitPlan(Sizer):
    """Cortés unit plans; ``unit`` is a currency amount.

    - ``loss``: stake = unit (risk a unit).
    - ``win``: stake = unit / (odds - 1) (win a unit); odds ``<= 1`` skip.
    - ``impact``: stake = unit / odds (win and loss differ by a unit).

    Stakes are capped at the bankroll.
    """

    def __init__(self, unit: float = 1.0, plan: Literal["loss", "win", "impact"] = "loss"):
        if plan not in ("loss", "win", "impact"):
            raise ValueError(f"plan must be 'loss', 'win', or 'impact', got {plan!r}")
        if unit <= 0:
            raise ValueError("unit must be positive")
        self.unit = unit
        self.plan = plan

    def size(self, ctx: BetContext, outcome: int) -> float:
        selected = ctx.odds[outcome]
        if self.plan == "loss":
            return cap_stake(self.unit, ctx.bankroll)
        if self.plan == "win":
            if selected <= 1:
                return 0.0
            return cap_stake(self.unit / (selected - 1), ctx.bankroll)
        if selected <= 0:
            return 0.0
        return cap_stake(self.unit / selected, ctx.bankroll)

    def info(self, ctx: BetContext, outcome: int) -> Dict[str, Any]:
        return {"plan": self.plan, "unit": self.unit}


class EdgeProportional(Sizer):
    """Stake grows with the edge: ``min(multiplier * edge, max_fraction) * bankroll``.

    ``edge = p * odds - 1`` of the selected outcome; no bet if the edge is
    not positive. A simpler, lower-variance alternative to Kelly that does
    not scale with ``odds - 1``.

    Attributes:
        multiplier: Bankroll share per unit of edge, ``> 0``.
        max_fraction: Cap as a bankroll share, ``(0, 1]``.

    Example:
        Edge 0.06, multiplier 0.5 → 3% of the bankroll.
    """

    requires_probabilities = True

    def __init__(self, multiplier: float = 0.5, max_fraction: float = 0.05):
        if multiplier <= 0:
            raise ValueError("multiplier must be positive")
        if not 0 < max_fraction <= 1:
            raise ValueError("max_fraction must be in (0, 1]")
        self.multiplier = multiplier
        self.max_fraction = max_fraction

    def size(self, ctx: BetContext, outcome: int) -> float:
        if not ctx.model_probs:
            return 0.0
        edge = expected_value(ctx.odds[outcome], ctx.model_probs[outcome])
        if not edge > 0:
            return 0.0
        return min(self.multiplier * edge, self.max_fraction) * ctx.bankroll

    def info(self, ctx: BetContext, outcome: int) -> Dict[str, Any]:
        if not ctx.model_probs or outcome < 0:
            return {}
        return {"edge": expected_value(ctx.odds[outcome], ctx.model_probs[outcome])}


# ----------------------------------------------------------------- strategy


@functools.lru_cache(maxsize=None)
def _overrides_legacy_methods(cls: type) -> bool:
    """True if ``cls`` overrides ``calculate_stake`` or ``select_bet`` (cached per class)."""
    return cls.calculate_stake is not Strategy.calculate_stake or cls.select_bet is not Strategy.select_bet


class Strategy(BaseStrategy):
    """A betting strategy built from a selector and a sizer.

    Args:
        select: Chooses the outcome.
        size: Decides the stake.

    Subclasses may still override ``calculate_stake`` or ``select_bet`` as
    with any ``BaseStrategy``; ``get_bet_details`` then calls the
    overrides, exactly as before the toolkit existed.

    Example:
        >>> Strategy(select=HighestEV(min_ev=0.02), size=EdgeProportional(0.5, 0.05))
    """

    def __init__(self, select: Selector, size: Sizer):
        self.selector = select
        self.sizer = size

    @property
    def requires_probabilities(self) -> bool:
        """True if the selector or the sizer needs model probabilities."""
        return bool(self.selector.requires_probabilities or self.sizer.requires_probabilities)

    @staticmethod
    def _context(odds, bankroll, model_probs, prediction, kwargs) -> BetContext:
        return BetContext(odds, bankroll, model_probs, prediction, kwargs.get("history"), kwargs.get("date"))

    def select_bet(
        self,
        odds: List[float],
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> int:
        """Outcome chosen by the selector, or ``-1``."""
        return self.selector.select(self._context(odds, 0.0, model_probs, prediction, kwargs))

    def calculate_stake(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        **kwargs: Any,
    ) -> float:
        """Stake on the selected outcome, 0 if the selector skips."""
        ctx = self._context(odds, bankroll, model_probs, kwargs.get("prediction"), kwargs)
        if type(self).select_bet is not Strategy.select_bet:
            # A subclass override of select_bet decides, as in the pre-toolkit classes
            outcome = self.select_bet(odds, model_probs, **kwargs)
        else:
            outcome = self.selector.select(ctx)
        if outcome == -1:
            return 0.0
        return self.sizer.size(ctx, outcome)

    def _overrides_legacy_methods(self) -> bool:
        return _overrides_legacy_methods(type(self))

    def _legacy_bet(self, odds, bankroll, model_probs, prediction, kwargs) -> Tuple[float, int]:
        """Call the (overridden) methods the way ``BaseStrategy`` does.

        Presets override this to keep the exact call pattern of their
        pre-toolkit ``get_bet_details``, so subclass overrides with narrow
        signatures keep working.
        """
        stake = self.calculate_stake(odds, bankroll, model_probs=model_probs, **kwargs)
        outcome = self.select_bet(odds, model_probs, prediction, **kwargs)
        return stake, outcome

    def get_bet_details(
        self,
        odds: List[float],
        bankroll: float,
        model_probs: Optional[List[float]] = None,
        prediction: Optional[int] = None,
        **kwargs: Any,
    ) -> Tuple[float, int, Dict[str, Any]]:
        """Stake, outcome, and selector/sizer info for one match.

        The selector runs once and the sizer reuses its work, unless a
        subclass overrides ``calculate_stake`` or ``select_bet``; then those
        are called as in a plain ``BaseStrategy``.
        """
        ctx = BetContext(odds, bankroll, model_probs, prediction, kwargs.get("history"), kwargs.get("date"))
        if _overrides_legacy_methods(type(self)):
            stake, outcome = self._legacy_bet(odds, bankroll, model_probs, prediction, kwargs)
        else:
            outcome = self.selector.select(ctx)
            stake = 0.0 if outcome == -1 else self.sizer.size(ctx, outcome)
        info = self.selector.info(ctx)
        sizer_info = self.sizer.info(ctx, outcome)
        if sizer_info:
            info = {**info, **sizer_info}
        if stake == 0:
            outcome = -1
        return stake, outcome, info

    def adjust_simultaneous(self, stakes: List[float], bankroll: float) -> List[float]:
        """Delegates to the sizer (e.g. ``SimultaneousKelly``)."""
        return self.sizer.adjust_simultaneous(stakes, bankroll)

    def __str__(self) -> str:
        return f"Strategy (select={type(self.selector).__name__}, size={type(self.sizer).__name__})"
