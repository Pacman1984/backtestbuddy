"""
Tests for settle_simultaneous: same-date matches as one settlement slot.

In a slot every bet is sized from the slot's opening bankroll, the total
stake is capped at that bankroll, strategies may rescale the slot, and
results reach the history only after the slot settles.
"""
import numpy as np
import pandas as pd
import pytest

import backtestbuddy as btb
from backtestbuddy.metrics.sport_metrics import _compound_period_returns


@pytest.fixture
def slate():
    """Three matches on Jan 1 (win, loss, win), one on Jan 2 (win)."""
    return pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01"] * 3 + ["2023-01-02"]),
        "a": [2.0] * 4, "b": [2.0] * 4, "y": [0, 1, 0, 0], "pred": [0] * 4,
        "pa": [0.6] * 4, "pb": [0.4] * 4,
    })


def run(slate, strategy, simultaneous=True, **kwargs):
    bt = btb.PredictionBacktest(slate, ["a", "b"], "y", "date", "pred", strategy=strategy,
                                model_prob_columns=["pa", "pb"], settle_simultaneous=simultaneous, **kwargs)
    bt.run()
    return bt


class TestSlotSizing:
    def test_sized_from_slot_opening_bankroll(self, slate):
        results = run(slate, btb.FixedStake(fraction=0.2)).detailed_results
        assert results["bt_stake"].tolist() == pytest.approx([200, 200, 200, 240])
        assert results["bt_sizing_bankroll"].tolist() == pytest.approx([1000, 1000, 1000, 1200])

    def test_sequential_default_uses_running_bankroll(self, slate):
        results = run(slate, btb.FixedStake(fraction=0.2), simultaneous=False).detailed_results
        assert results["bt_stake"].tolist() == pytest.approx([200, 240, 192, 230.4])
        assert "bt_sizing_bankroll" not in results

    def test_total_stake_capped_at_slot_bankroll(self, slate):
        results = run(slate, btb.FixedStake(fraction=0.4)).detailed_results
        assert results["bt_stake"].tolist()[:3] == pytest.approx([400, 400, 200])

    def test_bookkeeping_is_consistent(self, slate):
        results = run(slate, btb.FixedStake(fraction=0.2)).detailed_results
        np.testing.assert_allclose(results["bt_ending_bankroll"],
                                   1000 + results["bt_profit"].cumsum())
        # Compounding per-bet returns over the day equals the slot's true return
        day_one = _compound_period_returns(results).iloc[0]
        assert day_one == pytest.approx(200 / 1000)


class TestSimultaneousKelly:
    def test_scales_slot_to_max_total(self, slate):
        strategy = btb.Strategy(btb.HighestKelly(), btb.SimultaneousKelly(fraction=1.0, max_bet=0.2, max_total=0.3))
        results = run(slate, strategy).detailed_results
        assert results["bt_stake"].tolist() == pytest.approx([100, 100, 100, 220])

    def test_without_simultaneous_mode_behaves_like_kelly(self, slate):
        strategy = btb.Strategy(btb.HighestKelly(), btb.SimultaneousKelly(fraction=1.0, max_bet=0.2, max_total=0.3))
        plain = btb.Strategy(btb.HighestKelly(), btb.KellySize(fraction=1.0, max_bet=0.2))
        pd.testing.assert_frame_equal(run(slate, strategy, simultaneous=False).detailed_results,
                                      run(slate, plain, simultaneous=False).detailed_results)

    def test_wrappers_forward_adjustment(self, slate):
        inner = btb.Strategy(btb.HighestKelly(), btb.SimultaneousKelly(fraction=1.0, max_bet=0.2, max_total=0.3))
        results = run(slate, btb.OddsFilter(1.5, 3.0, btb.RiskLimits(inner, min_stake=150))).detailed_results
        assert results["bt_stake"].tolist() == pytest.approx([0, 0, 0, 200])


class TestHistoryInSlots:
    def test_results_visible_only_after_slot(self, slate):
        bt = run(slate, btb.RiskLimits(btb.FixedStake(amount=10), pause_after_losses=1))
        # The Jan 1 loss settles with its slot, so no other Jan 1 bet is paused
        assert (bt.detailed_results["bt_stake"] > 0).tolist() == [True, True, True, True]
        assert bt.history.n_bets == 4

    def test_bookie_uses_slot_sizing(self, slate):
        bt = run(slate, btb.FixedStake(fraction=0.4))
        assert bt.bookie_results["bt_stake"].tolist()[:3] == pytest.approx([400, 400, 200])


def test_model_backtest_supports_simultaneous(slate):
    from sklearn.dummy import DummyClassifier
    from sklearn.model_selection import TimeSeriesSplit

    data = pd.concat([slate] * 4, ignore_index=True)
    data["date"] = pd.to_datetime(["2023-01-01"] * 4 + ["2023-01-02"] * 4 + ["2023-01-03"] * 4 + ["2023-01-04"] * 4)
    bt = btb.ModelBacktest(data, ["a", "b"], "y", "date", DummyClassifier(strategy="prior"),
                           strategy=btb.FixedStake(fraction=0.3), cv_schema=TimeSeriesSplit(n_splits=3),
                           feature_columns=["a", "b"], settle_simultaneous=True)
    bt.run()
    results = bt.detailed_results
    for _, day in results.groupby("bt_date_column"):
        assert day["bt_stake"].sum() <= day["bt_sizing_bankroll"].iloc[0] + 1e-9
