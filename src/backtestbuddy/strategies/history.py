"""Betting history that strategies can read while a backtest runs.

The backtest owns one ``BetHistory`` per run, records every placed bet
right after it settles, and passes the ledger to the strategy as
``history=`` together with the current match ``date=``. A strategy that
needs memory (risk limits, progressions, pauses after losses) reads the
ledger instead of keeping its own state. Because the ledger is created
fresh in every ``run()``, repeated runs such as ``stress_test()`` never
leak state into each other.

Queries read running totals that ``record`` keeps up to date, so they do
not slow down as the history grows (``losing_streak_on`` walks back only
over the current streak).

With ``settle_simultaneous=True`` the bets of one slot are all placed
before any of them settles. The backtest registers each placed stake with
``add_open_stake`` right after the decision, so ``staked_on`` already
counts the day's open exposure (as a live bettor would), and clears the
open stakes once the slot settles and its bets are recorded. Results
(profit, streaks, drawdown) stay unknown until then.

Day keys: a "day" is the calendar day of the date by default. A bet can
also carry a hashable ``key`` (the backtest passes the row's
``day_column`` value); its day is then the pair (calendar day, key), so
copies of one calendar day keep separate day totals. Queries take the
same optional ``key``. During a backtest with ``day_column``, a query
without ``key`` uses the key of the row being decided, so strategies
need no changes. Drawdown, bankroll, and ``losing_streak`` stay global.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


def _check_day_key(key: Any) -> None:
    """Raise if ``key`` cannot serve as a day key.

    Raises:
        TypeError: If ``key`` is not hashable.
        ValueError: If ``key`` is NaN (NaN never equals itself, so its
            bets could not be found again).
    """
    hash(key)
    if key != key:
        raise ValueError(f"A day key must not be NaN, got {key!r}")


@dataclass(frozen=True)
class SettledBet:
    """One placed and settled bet.

    Attributes:
        date: Match date.
        stake: Amount staked.
        profit: Net profit (negative on a loss).
        bankroll_before: Bankroll before the bet.
        bankroll_after: Bankroll after settlement.
        won: True on a win, False on a loss.
    """

    date: pd.Timestamp
    stake: float
    profit: float
    bankroll_before: float
    bankroll_after: float
    won: bool


class BetHistory:
    """Ledger of settled bets with running aggregates.

    Example:
        >>> history = BetHistory(initial_bankroll=1000)
        >>> history.record("2023-01-01", stake=100, profit=-100,
        ...                bankroll_before=1000, bankroll_after=900)
        >>> history.drawdown
        0.1
    """

    def __init__(self, initial_bankroll: float):
        """Start an empty ledger.

        Args:
            initial_bankroll: Bankroll before the first bet; the first peak.
        """
        self.initial_bankroll = float(initial_bankroll)
        self._bets: List[SettledBet] = []
        self._bankroll = self.initial_bankroll
        self._peak = self.initial_bankroll
        self._losing_streak = 0
        self._staked_by_day: Dict[pd.Timestamp, float] = {}
        self._profit_by_day: Dict[pd.Timestamp, float] = {}
        self._day_start_bankroll: Dict[pd.Timestamp, float] = {}
        self._open_by_day: Dict[pd.Timestamp, float] = {}
        self._last_date: Any = None
        self._last_day: Optional[pd.Timestamp] = None
        # Day keys (see module docstring): key per bet (None without one), per-bucket streaks
        self._bet_keys: List[Any] = []
        self._streak_by_day: Dict[Any, int] = {}
        self._row_key: Any = None

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Restore a pickled ledger, also one saved before day keys existed."""
        self.__dict__.update(state)
        self.__dict__.setdefault("_bet_keys", [None] * len(self._bets))
        self.__dict__.setdefault("_streak_by_day", {})
        self.__dict__.setdefault("_row_key", None)

    def _day(self, date: Any) -> pd.Timestamp:
        """Calendar day of ``date``; consecutive calls with one date are cached."""
        if date is self._last_date or (self._last_day is not None and date == self._last_date):
            return self._last_day
        day = pd.Timestamp(date).normalize()
        self._last_date, self._last_day = date, day
        return day

    def _bucket(self, date: Any, key: Any) -> Any:
        """Day bucket of a booking: the calendar day, or (calendar day, key)."""
        day = self._day(date)
        return day if key is None else (day, key)

    def _query_bucket(self, date: Any, key: Any) -> Any:
        """Day bucket of a query; without ``key`` the current row's key applies."""
        return self._bucket(date, self._row_key if key is None else key)

    def record(
        self,
        date: Any,
        stake: float,
        profit: float,
        bankroll_before: float,
        bankroll_after: float,
        won: Optional[bool] = None,
        *,
        key: Any = None,
        day_start_bankroll: Optional[float] = None,
    ) -> None:
        """Add one settled bet and update the running totals.

        Args:
            date: Match date (anything ``pd.Timestamp`` accepts).
            stake: Amount staked.
            profit: Net profit.
            bankroll_before: Bankroll before the bet.
            bankroll_after: Bankroll after settlement.
            won: Result; inferred from ``profit > 0`` when omitted.
            key: Optional day key; the bet's day is then (calendar day, key).
            day_start_bankroll: Bankroll that starts the bet's day if this is
                its first bet; defaults to ``bankroll_before``. The backtest
                passes the slot's opening bankroll for simultaneous slots, so
                the day start of a key does not depend on the row order
                inside the slot.
        """
        if key is not None:
            _check_day_key(key)
        won = bool(profit > 0) if won is None else bool(won)
        bet = SettledBet(
            pd.Timestamp(date), float(stake), float(profit),
            float(bankroll_before), float(bankroll_after), won,
        )
        self._bets.append(bet)
        self._bankroll = bet.bankroll_after
        self._peak = max(self._peak, bet.bankroll_after)
        self._losing_streak = 0 if won else self._losing_streak + 1
        day = self._bucket(date, key)
        start = bet.bankroll_before if day_start_bankroll is None else float(day_start_bankroll)
        self._day_start_bankroll.setdefault(day, start)
        self._staked_by_day[day] = self._staked_by_day.get(day, 0.0) + bet.stake
        self._profit_by_day[day] = self._profit_by_day.get(day, 0.0) + bet.profit
        self._bet_keys.append(key)
        if key is not None:
            self._streak_by_day[day] = 0 if won else self._streak_by_day.get(day, 0) + 1

    @property
    def bets(self) -> List[SettledBet]:
        """All settled bets in order (a copy)."""
        return list(self._bets)

    @property
    def n_bets(self) -> int:
        """Number of settled bets."""
        return len(self._bets)

    @property
    def last_bet(self) -> Optional[SettledBet]:
        """Most recent settled bet, or None before the first bet."""
        return self._bets[-1] if self._bets else None

    @property
    def bankroll(self) -> float:
        """Bankroll after the most recent bet."""
        return self._bankroll

    @property
    def peak_bankroll(self) -> float:
        """Highest bankroll so far, including the initial bankroll."""
        return self._peak

    @property
    def drawdown(self) -> float:
        """Current fall below the peak as a fraction (0.25 = 25%)."""
        if self._peak <= 0:
            return 0.0
        return 1.0 - self._bankroll / self._peak

    @property
    def losing_streak(self) -> int:
        """Consecutive losses at the end of the history."""
        return self._losing_streak

    def add_open_stake(self, date: Any, stake: float, *, key: Any = None) -> None:
        """Register a placed stake whose match has not settled yet.

        Used for simultaneous slots: the stake counts in ``staked_on`` until
        ``clear_open_stakes`` is called, so daily limits see the exposure of
        bets placed earlier in the same slot.

        Args:
            date: Match date.
            stake: Amount staked; non-positive stakes are ignored.
            key: Optional day key, as in ``record``.
        """
        if stake > 0:
            if key is not None:
                _check_day_key(key)
            day = self._bucket(date, key)
            self._open_by_day[day] = self._open_by_day.get(day, 0.0) + float(stake)

    def clear_open_stakes(self) -> None:
        """Forget open stakes, e.g. after their slot settled and was recorded."""
        self._open_by_day.clear()

    @property
    def open_stake(self) -> float:
        """Total of registered stakes that have not settled yet."""
        return sum(self._open_by_day.values())

    def staked_on(self, date: Any, key: Any = None) -> float:
        """Total stake on the calendar day of ``date``: settled plus still open.

        With ``key`` (or, inside a backtest with ``day_column``, the current
        row's key) only the stakes booked under that key on that day count.
        """
        day = self._query_bucket(date, key)
        return self._staked_by_day.get(day, 0.0) + self._open_by_day.get(day, 0.0)

    def profit_on(self, date: Any, key: Any = None) -> float:
        """Net profit of bets settled on the calendar day of ``date`` (and ``key``)."""
        return self._profit_by_day.get(self._query_bucket(date, key), 0.0)

    def bankroll_at_start_of(self, date: Any, key: Any = None) -> float:
        """Bankroll before the first bet of that day (and ``key``), else the current one."""
        return self._day_start_bankroll.get(self._query_bucket(date, key), self._bankroll)

    def losing_streak_on(self, date: Any, key: Any = None) -> int:
        """Consecutive losses at the end of the history, counted within that day.

        With a key: consecutive losses at the end of the bets of that
        (calendar day, key), even if bets of other keys settled in between.
        Without a key only bets booked without a key count.
        """
        key = self._row_key if key is None else key
        if key is not None:
            return self._streak_by_day.get(self._bucket(date, key), 0)
        day = self._day(date)
        streak = 0
        for bet, bet_key in zip(reversed(self._bets), reversed(self._bet_keys)):
            if bet_key is not None or self._day(bet.date) != day or bet.won:
                break
            streak += 1
        return streak
