"""
Tests for backtest.evaluate(), calculate_metrics(), and split_by_date().

evaluate() must return exactly what run() plus calculate_all_metrics()
returns, for every metric and every engine mode, while leaving results
of an earlier run() untouched.
"""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import backtestbuddy as btb


def _same(a, b):
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


@pytest.fixture
def matches():
    rng = np.random.default_rng(5)
    n = 400
    p = rng.uniform(0.3, 0.7, n)
    data = pd.DataFrame({
        "date": pd.Timestamp("2022-01-01") + pd.to_timedelta(np.arange(n) // 3, unit="D"),
        "a": np.round(0.96 / p, 2), "b": np.round(0.96 / (1 - p), 2),
        "y": (rng.uniform(size=n) > p).astype(int), "league": rng.choice(["ATP", "WTA"], n),
        "pa": np.clip(p + rng.normal(0, 0.05, n), 0.02, 0.98), "feat": rng.normal(size=n),
    })
    data["pb"] = 1 - data["pa"]
    data["pred"] = (data["pa"] < 0.5).astype(int)
    data["ca"] = np.round(data["a"] * np.exp(rng.normal(0, 0.03, n)), 2)
    data["cb"] = np.round(data["b"] * np.exp(rng.normal(0, 0.03, n)), 2)
    return data


STRATEGIES = {
    "fixed": lambda: btb.FixedStake(fraction=0.02),
    "kelly": lambda: btb.KellyCriterion(0.5, 0.1),
    "edge": lambda: btb.EdgeProportionalStake(0.01, 0.5, 0.05),
    "risk": lambda: btb.RiskLimits(btb.KellyCriterion(1.0, 0.2), max_daily_fraction=0.1, pause_after_losses=2),
    "toolkit": lambda: btb.Strategy(btb.HighestEV(0.02), btb.SimultaneousKelly(0.5, 0.1, 0.2)),
}


class TestEvaluateMatchesRun:
    @pytest.mark.parametrize("simultaneous", [False, True])
    @pytest.mark.parametrize("name", list(STRATEGIES))
    def test_all_metrics_identical(self, matches, name, simultaneous):
        bt = btb.PredictionBacktest(matches, ["a", "b"], "y", "date", "pred", strategy=STRATEGIES[name](),
                                    model_prob_columns=["pa", "pb"], closing_odds_columns=["ca", "cb"],
                                    odds_haircut=0.01, settle_simultaneous=simultaneous)
        bt.run()
        full = bt.calculate_metrics()
        lean = bt.evaluate(list(btb.METRIC_NAMES))
        assert [k for k in btb.METRIC_NAMES if not _same(full[k], lean[k])] == []

    def test_model_backtest(self, matches):
        bt = btb.ModelBacktest(matches[["date", "y", "a", "b", "feat"]], ["a", "b"], "y", "date", LogisticRegression(),
                               strategy=btb.KellyCriterion(), feature_columns=["feat", "a", "b"])
        bt.run()
        full = bt.calculate_metrics()
        assert bt.evaluate("Sharpe Ratio (365.25) [-]") == full["Sharpe Ratio (365.25) [-]"]


class TestEvaluateBehavior:
    def test_leaves_run_results_untouched(self, matches):
        bt = btb.PredictionBacktest(matches, ["a", "b"], "y", "date", "pred", strategy=btb.FixedStake(amount=10))
        bt.evaluate("Yield [%]")
        assert bt.detailed_results is None and bt.bookie_results is None and bt.history is None
        bt.run()
        before = bt.detailed_results.copy()
        bt.evaluate("Yield [%]")
        pd.testing.assert_frame_equal(before, bt.detailed_results)

    def test_return_shapes(self, matches):
        bt = btb.PredictionBacktest(matches, ["a", "b"], "y", "date", "pred", strategy=btb.FixedStake(amount=10))
        assert isinstance(bt.evaluate("Yield [%]"), float)
        assert list(bt.evaluate(["Total Bets", "Yield [%]"])) == ["Total Bets", "Yield [%]"]

    def test_min_bets_fallback(self, matches):
        bt = btb.PredictionBacktest(matches, ["a", "b"], "y", "date", "pred", model_prob_columns=["pa", "pb"],
                                    strategy=btb.ValueBet(min_ev=0.5, amount=10))  # almost never bets
        assert math.isnan(bt.evaluate("Yield [%]", min_bets=50))
        assert bt.evaluate(["Yield [%]"], min_bets=50, fallback=-1.0) == {"Yield [%]": -1.0}

    def test_unknown_metric_raises(self, matches):
        bt = btb.PredictionBacktest(matches, ["a", "b"], "y", "date", "pred", strategy=btb.FixedStake(amount=10))
        with pytest.raises(ValueError, match="Unknown metric"):
            bt.evaluate("Profit please")


class TestCalculateMetrics:
    def test_subset_equals_full_dictionary(self, matches):
        bt = btb.PredictionBacktest(matches, ["a", "b"], "y", "date", "pred", strategy=btb.FixedStake(amount=10))
        bt.run()
        full = btb.calculate_all_metrics(bt.detailed_results)
        subset = btb.calculate_metrics(bt.detailed_results, ["Max Drawdown [%]", "ROI [%]"])
        assert subset == {"Max Drawdown [%]": full["Max Drawdown [%]"], "ROI [%]": full["ROI [%]"]}
        assert list(full) == list(btb.METRIC_NAMES)


class TestSplitByDate:
    def test_fraction_keeps_dates_together(self, matches):
        tuning, holdout = btb.split_by_date(matches, "date", holdout_fraction=0.25)
        assert len(tuning) + len(holdout) == len(matches)
        assert tuning["date"].max() < holdout["date"].min()
        assert set(tuning["date"]).isdisjoint(set(holdout["date"]))

    def test_fixed_date(self, matches):
        tuning, holdout = btb.split_by_date(matches, "date", holdout_from="2022-03-01")
        assert holdout["date"].min() >= pd.Timestamp("2022-03-01") > tuning["date"].max()

    @pytest.mark.parametrize("kwargs, match", [
        ({}, "exactly one"),
        ({"holdout_from": "2022-02-01", "holdout_fraction": 0.2}, "exactly one"),
        ({"holdout_fraction": 1.0}, r"\(0, 1\)"),
        ({"holdout_from": "2030-01-01"}, "empty"),
    ])
    def test_invalid(self, matches, kwargs, match):
        with pytest.raises(ValueError, match=match):
            btb.split_by_date(matches, "date", **kwargs)
