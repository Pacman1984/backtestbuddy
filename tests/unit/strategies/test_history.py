"""
Unit tests for BetHistory, the per-run ledger strategies read.

Checks running bankroll, peak and drawdown, losing streaks overall and
per day, and per-day stake, profit, and opening bankroll.
"""
import pytest

from backtestbuddy.strategies.history import BetHistory


@pytest.fixture
def history():
    """Win, loss, loss on day 1; loss on day 2."""
    h = BetHistory(initial_bankroll=1000)
    h.record("2023-01-01 12:00", stake=100, profit=100, bankroll_before=1000, bankroll_after=1100)
    h.record("2023-01-01 15:00", stake=100, profit=-100, bankroll_before=1100, bankroll_after=1000)
    h.record("2023-01-01 18:00", stake=50, profit=-50, bankroll_before=1000, bankroll_after=950)
    h.record("2023-01-02 12:00", stake=50, profit=-50, bankroll_before=950, bankroll_after=900)
    return h


class TestBankroll:
    def test_empty(self):
        h = BetHistory(500)
        assert (h.n_bets, h.bankroll, h.peak_bankroll, h.drawdown, h.last_bet) == (0, 500, 500, 0.0, None)

    def test_running_values(self, history):
        assert history.n_bets == 4
        assert history.bankroll == 900
        assert history.peak_bankroll == 1100
        assert history.drawdown == pytest.approx(1 - 900 / 1100)

    def test_initial_bankroll_is_first_peak(self):
        h = BetHistory(1000)
        h.record("2023-01-01", stake=100, profit=-100, bankroll_before=1000, bankroll_after=900)
        assert h.drawdown == pytest.approx(0.1)


class TestStreaks:
    def test_losing_streak_spans_days(self, history):
        assert history.losing_streak == 3

    def test_losing_streak_on_day(self, history):
        assert history.losing_streak_on("2023-01-02") == 1
        assert history.losing_streak_on("2023-01-03") == 0

    def test_win_resets(self, history):
        history.record("2023-01-02", stake=10, profit=10, bankroll_before=900, bankroll_after=910)
        assert history.losing_streak == 0

    def test_won_inferred_from_profit(self):
        h = BetHistory(100)
        h.record("2023-01-01", stake=10, profit=-10, bankroll_before=100, bankroll_after=90)
        assert h.last_bet.won is False


class TestPerDay:
    def test_staked_and_profit(self, history):
        assert history.staked_on("2023-01-01 23:59") == 250
        assert history.profit_on("2023-01-01") == -50
        assert history.staked_on("2023-01-05") == 0

    def test_bankroll_at_start_of_day(self, history):
        assert history.bankroll_at_start_of("2023-01-01") == 1000
        assert history.bankroll_at_start_of("2023-01-02") == 950
        assert history.bankroll_at_start_of("2023-01-03") == 900  # no bets yet: current

    def test_bets_is_a_copy(self, history):
        history.bets.clear()
        assert history.n_bets == 4
