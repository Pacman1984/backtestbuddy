"""
Backward-compatibility tests for strategies rebuilt on the toolkit.

Existing user code must keep working: import paths, reassigning
attributes, subclasses that override the classic methods, positional
and keyword calls, and one-class strategies on BaseStrategy.
"""
from typing import Any, List

import pytest

import backtestbuddy as btb
from backtestbuddy.strategies.sport_strategies import (  # old import paths
    BaseStrategy,
    FixedStake,
    KellyCriterion,
    StakeSize,
    UnitStake,
    ValueBet,
    _cap_stake,
    _expected_value,
)

ODDS = [2.0, 2.0]
PROBS = [0.6, 0.4]


class TestImportsAndTypes:
    def test_old_paths_are_the_same_objects(self):
        assert BaseStrategy is btb.BaseStrategy
        assert StakeSize is btb.StakeSize
        assert _expected_value(2.0, 0.6) == pytest.approx(0.2)
        assert _cap_stake(50, 30) == 30

    def test_presets_are_strategies(self):
        for strategy in (FixedStake(amount=1), KellyCriterion(), ValueBet(), UnitStake()):
            assert isinstance(strategy, BaseStrategy)
            assert isinstance(strategy, btb.Strategy)


class TestReassignAttributes:
    """Attributes still change behavior after construction (e.g. tuning loops)."""

    def test_kelly(self):
        strategy = KellyCriterion(downscaling=1.0, max_bet=1.0)
        assert strategy.calculate_stake(ODDS, 1000, model_probs=PROBS) == pytest.approx(200)
        strategy.downscaling = 0.5
        assert strategy.calculate_stake(ODDS, 1000, model_probs=PROBS) == pytest.approx(100)
        strategy.max_bet = 0.05
        assert strategy.calculate_stake(ODDS, 1000, model_probs=PROBS) == pytest.approx(50)
        strategy.min_kelly = 0.3
        assert strategy.get_bet_details(ODDS, 1000, PROBS)[:2] == (0.0, -1)

    def test_value_bet_and_unit_stake(self):
        value = ValueBet(min_ev=0.0, amount=10)
        value.min_ev = 0.5
        assert value.calculate_stake(ODDS, 1000, model_probs=PROBS) == 0.0
        unit = UnitStake(10, "loss")
        unit.unit, unit.plan = 30, "win"
        # No probabilities: bets the favourite (odds 3.0) -> 30 / (3.0 - 1)
        assert unit.calculate_stake([3.0, 4.0], 1000) == pytest.approx(15)

    def test_fixed_stake_legacy_setter(self):
        strategy = FixedStake(amount=10)
        with pytest.warns(DeprecationWarning):
            strategy.stake = 0.5
        assert strategy.calculate_stake(ODDS, 1000) == pytest.approx(500)


class TestSubclassOverrides:
    """Overriding the classic methods keeps working as before."""

    def test_calculate_stake_override_is_used(self):
        class Double(FixedStake):
            def calculate_stake(self, odds, bankroll, model_probs=None):  # narrow signature
                return 2 * super().calculate_stake(odds, bankroll, model_probs=model_probs)

        assert Double(amount=10).get_bet_details(ODDS, 1000, history=None, date=None)[0] == 20

    def test_select_bet_override_drives_kelly_stake(self):
        class AlwaysSecond(KellyCriterion):
            def select_bet(self, odds, model_probs=None, prediction=None, **kwargs):
                return 1

        stake, bet_on, _ = AlwaysSecond(downscaling=1.0, max_bet=1.0).get_bet_details([2.0, 3.0], 1000, [0.4, 0.5])
        assert bet_on == 1
        assert stake == pytest.approx(250)  # Kelly of outcome 1: (0.5*2 - 0.5) / 2

    def test_kelly_fraction_override_is_used(self):
        class Shrunk(KellyCriterion):
            def calculate_kelly_fraction(self, odds, prob):
                return super().calculate_kelly_fraction(odds, prob) / 2

        stake, _, info = Shrunk(downscaling=1.0, max_bet=1.0).get_bet_details(ODDS, 1000, PROBS)
        assert stake == pytest.approx(100)
        assert info["kelly_fraction_0"] == pytest.approx(0.1)

    def test_get_bet_details_keyword_current_bankroll(self):
        assert FixedStake(amount=10).get_bet_details(ODDS, current_bankroll=1000)[0] == 10


class TestOneClassStrategy:
    """A strategy can still be a single BaseStrategy subclass."""

    def test_backtest_runs_one_class_strategy(self):
        import pandas as pd

        class Favourite(BaseStrategy):
            def calculate_stake(self, odds: List[float], bankroll: float, **kwargs: Any) -> float:
                return 10.0

            def select_bet(self, odds, model_probs=None, prediction=None, **kwargs) -> int:
                return odds.index(min(odds))

            def get_bet_details(self, odds, bankroll, model_probs=None, prediction=None, **kwargs):
                return 10.0, self.select_bet(odds), {}

        data = pd.DataFrame({"date": pd.date_range("2023-01-01", periods=3), "a": [1.5, 2.5, 1.8],
                             "b": [2.5, 1.5, 2.0], "y": [0, 1, 1], "pred": [0, 0, 0]})
        bt = btb.PredictionBacktest(data, ["a", "b"], "y", "date", "pred", strategy=Favourite())
        bt.run()
        assert bt.detailed_results["bt_bet_on"].tolist() == [0, 1, 0]
