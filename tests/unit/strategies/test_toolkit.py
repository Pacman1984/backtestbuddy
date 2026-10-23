"""
Unit tests for the strategy toolkit: selectors, sizers, and Strategy.

Each selector and sizer is checked on hand-built contexts; Strategy is
checked for combining them, merging their info, the probability flag,
and for matching the preset classes it replaces internally.
"""
import numpy as np
import pandas as pd
import pytest

import backtestbuddy as btb
from backtestbuddy.strategies.toolkit import BetContext


def ctx(odds, probs=None, prediction=None, bankroll=1000.0):
    return BetContext(odds, bankroll, probs, prediction)


class TestSelectors:
    def test_model_pick_order(self):
        assert btb.ModelPick().select(ctx([2.0, 1.5], probs=[0.3, 0.7], prediction=0)) == 1
        assert btb.ModelPick().select(ctx([2.0, 1.5], prediction=0)) == 0
        assert btb.ModelPick().select(ctx([2.0, 1.5])) == 1  # favourite

    def test_highest_probability_needs_probs(self):
        assert btb.HighestProbability().select(ctx([2.0, 1.5])) == -1
        assert btb.HighestProbability().select(ctx([2.0, 1.5], probs=[0.6, 0.4])) == 0

    def test_prediction_and_favourite(self):
        assert btb.Prediction().select(ctx([2.0, 1.5])) == -1
        assert btb.Prediction().select(ctx([2.0, 1.5], prediction=0)) == 0
        assert btb.Favourite().select(ctx([2.0, 1.5, 4.0])) == 1

    def test_highest_ev(self):
        selector = btb.HighestEV(min_ev=0.05)
        assert selector.select(ctx([2.2, 1.7], probs=[0.5, 0.5])) == 0  # EV 0.10
        assert selector.select(ctx([2.08, 1.7], probs=[0.5, 0.5])) == -1  # EV 0.04
        assert selector.info(ctx([2.2, 1.7], probs=[0.5, 0.5])) == {"ev_0": pytest.approx(0.1), "ev_1": pytest.approx(-0.15)}

    def test_highest_kelly(self):
        selector = btb.HighestKelly(min_kelly=0.05)
        assert selector.select(ctx([2.0, 2.0], probs=[0.6, 0.4])) == 0  # Kelly 0.2
        assert selector.select(ctx([2.0, 2.0], probs=[0.52, 0.48])) == -1  # Kelly 0.04
        assert "kelly_fraction_1" in selector.info(ctx([2.0, 2.0], probs=[0.6, 0.4]))


class TestSizers:
    def test_stake_size(self):
        assert btb.StakeSize(fraction=0.1).size(ctx([2.0, 2.0]), 0) == pytest.approx(100)
        assert btb.StakeSize(amount=30).size(ctx([2.0, 2.0], bankroll=20), 0) == 20

    def test_kelly_size(self):
        sizer = btb.KellySize(fraction=0.5, max_bet=0.05)
        assert sizer.size(ctx([2.0, 2.0], probs=[0.6, 0.4]), 0) == pytest.approx(50)  # min(0.1, 0.05)
        assert sizer.size(ctx([2.0, 2.0]), 0) == 0.0

    def test_unit_plan(self):
        assert btb.UnitPlan(90, "win").size(ctx([3.0, 1.5]), 0) == pytest.approx(45)
        assert btb.UnitPlan(90, "impact").size(ctx([3.0, 1.5]), 0) == pytest.approx(30)
        assert btb.UnitPlan(90, "loss").info(ctx([3.0, 1.5]), 0) == {"plan": "loss", "unit": 90}

    def test_edge_proportional(self):
        sizer = btb.EdgeProportional(multiplier=0.5, max_fraction=0.05)
        assert sizer.size(ctx([2.12, 1.8], probs=[0.5, 0.5]), 0) == pytest.approx(30)  # edge 0.06
        assert sizer.size(ctx([4.0, 1.3], probs=[0.5, 0.5]), 0) == pytest.approx(50)  # capped
        assert sizer.size(ctx([1.9, 1.9], probs=[0.5, 0.5]), 0) == 0.0  # no edge
        assert sizer.info(ctx([2.12, 1.8], probs=[0.5, 0.5]), 0) == {"edge": pytest.approx(0.06)}

    def test_simultaneous_kelly_scales_slot(self):
        sizer = btb.SimultaneousKelly(fraction=1.0, max_bet=0.2, max_total=0.3)
        np.testing.assert_allclose(sizer.adjust_simultaneous([200, 200, 200], 1000), [100, 100, 100])
        assert sizer.adjust_simultaneous([100, 50], 1000) == [100, 50]

    @pytest.mark.parametrize("factory", [
        lambda: btb.EdgeProportional(multiplier=0),
        lambda: btb.EdgeProportional(max_fraction=1.5),
        lambda: btb.KellySize(fraction=-1),
        lambda: btb.KellySize(max_bet=2),
        lambda: btb.SimultaneousKelly(max_total=0),
        lambda: btb.UnitPlan(unit=0),
        lambda: btb.UnitPlan(plan="double"),
    ])
    def test_invalid_parameters(self, factory):
        with pytest.raises(ValueError):
            factory()


class TestStrategy:
    def test_combines_selector_and_sizer(self):
        strategy = btb.Strategy(select=btb.HighestEV(0.02), size=btb.EdgeProportional(0.5, 0.05))
        stake, bet_on, info = strategy.get_bet_details([2.12, 1.8], 1000, model_probs=[0.5, 0.5])
        assert (bet_on, stake) == (0, pytest.approx(30))
        assert set(info) == {"ev_0", "ev_1", "edge"}

    def test_skip_gives_zero_stake(self):
        strategy = btb.Strategy(select=btb.HighestEV(0.5), size=btb.StakeSize(amount=10))
        assert strategy.get_bet_details([2.0, 2.0], 1000, model_probs=[0.5, 0.5])[:2] == (0.0, -1)
        assert strategy.calculate_stake([2.0, 2.0], 1000, model_probs=[0.5, 0.5]) == 0.0

    def test_requires_probabilities(self):
        assert btb.Strategy(btb.HighestEV(), btb.StakeSize(amount=1)).requires_probabilities
        assert btb.Strategy(btb.Favourite(), btb.KellySize()).requires_probabilities
        assert not btb.Strategy(btb.Favourite(), btb.StakeSize(amount=1)).requires_probabilities

    def test_custom_selector_and_sizer(self):
        class Underdog(btb.Selector):
            def select(self, ctx):
                return ctx.odds.index(max(ctx.odds))

        class HalfAfterLoss(btb.Sizer):
            def size(self, ctx, outcome):
                lost = ctx.history is not None and ctx.history.losing_streak > 0
                return 5.0 if lost else 10.0

        strategy = btb.Strategy(select=Underdog(), size=HalfAfterLoss())
        history = btb.BetHistory(1000)
        history.record("2023-01-01", stake=10, profit=-10, bankroll_before=1000, bankroll_after=990)
        assert strategy.get_bet_details([1.5, 3.0], 990, history=history)[:2] == (5.0, 1)

    def test_toolkit_matches_kelly_preset(self):
        rng = np.random.default_rng(3)
        n = 300
        p = rng.uniform(0.3, 0.7, n)
        data = pd.DataFrame({
            "date": pd.date_range("2023-01-01", periods=n), "a": np.round(0.96 / p, 2),
            "b": np.round(0.96 / (1 - p), 2), "y": (rng.uniform(size=n) > p).astype(int),
            "pa": np.clip(p + rng.normal(0, 0.05, n), 0.02, 0.98), "pred": 0,
        })
        data["pb"] = 1 - data["pa"]

        def run(strategy):
            bt = btb.PredictionBacktest(data, ["a", "b"], "y", "date", "pred", strategy=strategy,
                                        model_prob_columns=["pa", "pb"])
            bt.run()
            return bt.detailed_results

        preset = run(btb.KellyCriterion(downscaling=0.5, max_bet=0.05, min_kelly=0.01))
        built = run(btb.Strategy(btb.HighestKelly(min_kelly=0.01), btb.KellySize(0.5, 0.05, min_kelly=0.01)))
        pd.testing.assert_frame_equal(preset, built)
