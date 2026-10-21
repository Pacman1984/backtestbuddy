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
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd


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

    @staticmethod
    def _day(date: Any) -> pd.Timestamp:
        return pd.Timestamp(date).normalize()

    def record(
        self,
        date: Any,
        stake: float,
        profit: float,
        bankroll_before: float,
        bankroll_after: float,
        won: Optional[bool] = None,
    ) -> None:
        """Add one settled bet and update the running totals.

        Args:
            date: Match date (anything ``pd.Timestamp`` accepts).
            stake: Amount staked.
            profit: Net profit.
            bankroll_before: Bankroll before the bet.
            bankroll_after: Bankroll after settlement.
            won: Result; inferred from ``profit > 0`` when omitted.
        """
        won = bool(profit > 0) if won is None else bool(won)
        bet = SettledBet(
            pd.Timestamp(date), float(stake), float(profit),
            float(bankroll_before), float(bankroll_after), won,
        )
        self._bets.append(bet)
        self._bankroll = bet.bankroll_after
        self._peak = max(self._peak, bet.bankroll_after)
        self._losing_streak = 0 if won else self._losing_streak + 1
        day = self._day(date)
        self._day_start_bankroll.setdefault(day, bet.bankroll_before)
        self._staked_by_day[day] = self._staked_by_day.get(day, 0.0) + bet.stake
        self._profit_by_day[day] = self._profit_by_day.get(day, 0.0) + bet.profit

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

    def staked_on(self, date: Any) -> float:
        """Total stake of bets settled on the calendar day of ``date``."""
        return self._staked_by_day.get(self._day(date), 0.0)

    def profit_on(self, date: Any) -> float:
        """Net profit of bets settled on the calendar day of ``date``."""
        return self._profit_by_day.get(self._day(date), 0.0)

    def bankroll_at_start_of(self, date: Any) -> float:
        """Bankroll before the first bet of that day, else the current one."""
        return self._day_start_bankroll.get(self._day(date), self._bankroll)

    def losing_streak_on(self, date: Any) -> int:
        """Consecutive losses at the end of the history, counted within that day."""
        day = self._day(date)
        streak = 0
        for bet in reversed(self._bets):
            if self._day(bet.date) != day or bet.won:
                break
            streak += 1
        return streak
