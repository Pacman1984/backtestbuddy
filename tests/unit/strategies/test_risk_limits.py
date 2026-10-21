"""
Unit and backtest tests for the RiskLimits wrapper.

Each rule is checked on its own with a hand-built history, then inside
real backtests: the bt_risk_limit column, a drawdown stop, a fresh
history per run, nesting with OddsFilter, and the bookie path.
"""
import pandas as pd
import pytest

from backtestbuddy.backtest.sport_backtest import PredictionBacktest
from backtestbuddy.strategies.history import BetHistory
from backtestbuddy.strategies.sport_strategies import (
    FixedStake,
    KellyCriterion,
    OddsFilter,
    RiskLimits,
)

ODDS = [2.0, 2.0]


def _history_with_losses(n_losses, date="2023-01-01", start=1000.0, stake=100.0):
    history = BetHistory(start)
    bankroll = start
    for _ in range(n_losses):
        history.record(date, stake=stake, profit=-stake, bankroll_before=bankroll,
                       bankroll_after=bankroll - stake)
        bankroll -= stake
    return history


class TestRules:
    """apply_limits with one rule at a time."""

    def test_no_limits_passes_through(self):
        assert RiskLimits(FixedStake(amount=80)).apply_limits(80, 1000) == (80, None)

    def test_max_stake_fraction(self):
        limits = RiskLimits(FixedStake(amount=80), max_stake_fraction=0.05)
        assert limits.apply_limits(80, 1000) == (50, "max_stake_fraction")

    def test_max_stake(self):
        limits = RiskLimits(FixedStake(amount=80), max_stake=30)
        assert limits.apply_limits(80, 1000) == (30, "max_stake")

    def test_min_stake_skips(self):
        limits = RiskLimits(FixedStake(amount=1), min_stake=2)
        assert limits.apply_limits(1, 1000) == (0.0, "min_stake")

    def test_max_daily_fraction(self):
        history = _history_with_losses(1, stake=50)  # day start 1000, 50 staked
        limits = RiskLimits(FixedStake(amount=80), max_daily_fraction=0.1)
        assert limits.apply_limits(80, 950, history, "2023-01-01") == (50, "max_daily_fraction")

    def test_max_daily_fraction_resets_next_day(self):
        history = _history_with_losses(1, stake=100)
        limits = RiskLimits(FixedStake(amount=80), max_daily_fraction=0.1)
        assert limits.apply_limits(80, 900, history, "2023-01-02") == (80, None)

    def test_stop_at_drawdown(self):
        history = _history_with_losses(3)  # 1000 -> 700
        limits = RiskLimits(FixedStake(amount=10), stop_at_drawdown=0.3)
        assert limits.apply_limits(10, 700, history, "2023-01-02") == (0.0, "stop_at_drawdown")

    def test_pause_after_losses_same_day_only(self):
        history = _history_with_losses(2)
        limits = RiskLimits(FixedStake(amount=10), pause_after_losses=2)
        assert limits.apply_limits(10, 800, history, "2023-01-01") == (0.0, "pause_after_losses")
        assert limits.apply_limits(10, 800, history, "2023-01-02") == (10, None)

    def test_stateful_rules_skipped_without_history(self):
        limits = RiskLimits(FixedStake(amount=10), stop_at_drawdown=0.1, max_daily_fraction=0.01)
        assert limits.apply_limits(10, 1000) == (10, None)

    @pytest.mark.parametrize("kwargs", [
        {"max_stake_fraction": 0}, {"max_daily_fraction": 1.5}, {"stop_at_drawdown": -0.1},
        {"max_stake": 0}, {"min_stake": -1}, {"pause_after_losses": 0},
    ])
    def test_invalid_limits(self, kwargs):
        with pytest.raises(ValueError):
            RiskLimits(FixedStake(amount=10), **kwargs)


class TestBetDetails:
    """get_bet_details reports the rule and skips blocked bets."""

    def test_info_reports_rule(self):
        stake, bet_on, info = RiskLimits(FixedStake(amount=80), max_stake=30).get_bet_details(ODDS, 1000)
        assert (stake, bet_on, info["risk_limit"]) == (30, 0, "max_stake")

    def test_blocked_bet_is_skipped(self):
        history = _history_with_losses(3)
        limits = RiskLimits(FixedStake(amount=10), stop_at_drawdown=0.3)
        stake, bet_on, info = limits.get_bet_details(ODDS, 700, history=history, date="2023-01-02")
        assert (stake, bet_on, info["risk_limit"]) == (0.0, -1, "stop_at_drawdown")

    def test_keeps_inner_info(self):
        limits = RiskLimits(KellyCriterion(), max_stake=5)
        _, _, info = limits.get_bet_details([2.2, 1.7], 1000, model_probs=[0.55, 0.45])
        assert "kelly_fraction_0" in info and info["risk_limit"] == "max_stake"

    def test_requires_probabilities_follows_inner(self):
        assert RiskLimits(KellyCriterion()).requires_probabilities is True
        assert RiskLimits(FixedStake(amount=1)).requires_probabilities is False


@pytest.fixture
def losing_games():
    """Twelve games over six days; the backed favourite always loses."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01"] * 2 + ["2023-01-02"] * 2 + ["2023-01-03"] * 2
                               + ["2023-01-04"] * 2 + ["2023-01-05"] * 2 + ["2023-01-06"] * 2),
        "odds_a": [1.8] * 12,
        "odds_b": [2.1] * 12,
        "outcome": [1] * 12,
        "pred": [0] * 12,
    })


def _run(games, strategy):
    backtest = PredictionBacktest(games, ["odds_a", "odds_b"], "outcome", "date", "pred",
                                  strategy=strategy, initial_bankroll=1000)
    backtest.run()
    return backtest


class TestInBacktest:
    """RiskLimits driven by the history the backtest keeps."""

    def test_drawdown_stop_ends_betting(self, losing_games):
        backtest = _run(losing_games, RiskLimits(FixedStake(amount=100), stop_at_drawdown=0.3))
        results = backtest.detailed_results
        assert (results["bt_stake"] > 0).sum() == 3  # 1000 -> 700, then stop
        assert results["bt_risk_limit"].iloc[3:].eq("stop_at_drawdown").all()
        assert backtest.history.n_bets == 3

    def test_pause_after_losses_resumes_next_day(self, losing_games):
        backtest = _run(losing_games, RiskLimits(FixedStake(amount=10), pause_after_losses=1))
        placed = backtest.detailed_results["bt_stake"] > 0
        assert placed.tolist() == [True, False] * 6

    def test_fresh_history_per_run(self, losing_games):
        backtest = _run(losing_games, RiskLimits(FixedStake(amount=100), stop_at_drawdown=0.3))
        first = backtest.detailed_results.copy()
        backtest.run()
        pd.testing.assert_frame_equal(first, backtest.detailed_results)

    def test_stress_test_runs_are_independent(self, losing_games):
        backtest = _run(losing_games, RiskLimits(FixedStake(amount=100), stop_at_drawdown=0.3))
        # Every bet loses either way, so each run must stop after the same 3 bets
        table = backtest.stress_test(haircuts=[0.0, 0.01], metrics=["Total Bets"])
        assert table["Total Bets"].tolist() == [3, 3]

    def test_nested_with_odds_filter(self, losing_games):
        strategy = OddsFilter(min_odds=1.5, inner=RiskLimits(FixedStake(amount=100), stop_at_drawdown=0.3))
        backtest = _run(losing_games, strategy)
        assert (backtest.detailed_results["bt_stake"] > 0).sum() == 3

    def test_bookie_path_uses_stateless_rules(self, losing_games):
        backtest = _run(losing_games, RiskLimits(FixedStake(amount=100), max_stake=40, stop_at_drawdown=0.3))
        assert (backtest.bookie_results["bt_stake"] == 40).all()
