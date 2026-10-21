"""
Unit tests for backtest parameters added for CLV and stress testing.

Covers storing closing odds, the odds haircut (floor at 1.0, features
untouched), stress_test output, and parameter validation.
"""
import numpy as np
import pandas as pd
import pytest

from backtestbuddy.backtest.sport_backtest import ModelBacktest, PredictionBacktest
from backtestbuddy.strategies.sport_strategies import FixedStake


@pytest.fixture
def games():
    """Six two-way games with closing odds; the favourite always wins."""
    return pd.DataFrame({
        "date": pd.date_range("2023-01-01", periods=6),
        "odds_a": [1.50, 1.60, 1.02, 1.70, 1.55, 1.65],
        "odds_b": [2.60, 2.40, 12.0, 2.20, 2.50, 2.30],
        "close_a": [1.45, 1.62, 1.02, 1.60, 1.50, 1.70],
        "close_b": [2.70, 2.35, 12.5, 2.40, 2.60, 2.25],
        "outcome": [0, 0, 0, 0, 0, 0],
        "pred": [0, 0, 0, 0, 0, 0],
    })


def _prediction_backtest(games, **kwargs):
    return PredictionBacktest(
        games, ["odds_a", "odds_b"], "outcome", "date", "pred",
        strategy=FixedStake(100), **kwargs,
    )


class TestClosingOdds:
    """Closing odds are copied into both result frames."""

    def test_stored_per_outcome(self, games):
        backtest = _prediction_backtest(games, closing_odds_columns=["close_a", "close_b"])
        backtest.run()
        results = backtest.detailed_results
        assert results["bt_closing_odd_0"].tolist() == games["close_a"].tolist()
        assert results["bt_closing_odd_1"].tolist() == games["close_b"].tolist()
        assert "bt_closing_odd_0" in backtest.bookie_results.columns

    def test_clv_metric_available(self, games):
        backtest = _prediction_backtest(games, closing_odds_columns=["close_a", "close_b"])
        backtest.run()
        assert np.isfinite(backtest.calculate_metrics()["CLV [%]"])

    def test_absent_without_parameter(self, games):
        backtest = _prediction_backtest(games)
        backtest.run()
        assert not any(c.startswith("bt_closing_odd_") for c in backtest.detailed_results)
        assert np.isnan(backtest.calculate_metrics()["CLV [%]"])


class TestOddsHaircut:
    """The haircut lowers bet odds before staking and settlement."""

    def test_odds_and_profit_lowered(self, games):
        backtest = _prediction_backtest(games, odds_haircut=0.1)
        backtest.run()
        results = backtest.detailed_results
        expected = [max(1.0, o * 0.9) for o in games["odds_a"]]
        np.testing.assert_allclose(results["bt_odds"], expected)
        np.testing.assert_allclose(results["bt_profit"], [100 * (o - 1) for o in expected])

    def test_floor_at_one(self, games):
        backtest = _prediction_backtest(games, odds_haircut=0.1)
        backtest.run()
        assert backtest.detailed_results.loc[2, "bt_odds"] == 1.0

    def test_model_features_untouched(self, games):
        seen = []

        class Recorder:
            classes_ = np.array([0, 1])

            def fit(self, X, y):
                seen.append(X["odds_a"].tolist())
                return self

            def predict(self, X):
                return np.zeros(len(X), dtype=int)

            def predict_proba(self, X):
                return np.tile([0.7, 0.3], (len(X), 1))

        from sklearn.model_selection import TimeSeriesSplit

        backtest = ModelBacktest(
            games, ["odds_a", "odds_b"], "outcome", "date", Recorder(),
            strategy=FixedStake(100), cv_schema=TimeSeriesSplit(n_splits=2),
            feature_columns=["odds_a", "odds_b"], odds_haircut=0.1,
        )
        backtest.run()
        assert seen[0] == games["odds_a"].tolist()[: len(seen[0])]


class TestStressTest:
    """stress_test re-runs with each haircut and leaves the original intact."""

    def test_table_shape_and_monotonic_profit(self, games):
        backtest = _prediction_backtest(games)
        backtest.run()
        original = backtest.detailed_results.copy()
        table = backtest.stress_test(haircuts=[0.0, 0.02, 0.05])
        assert table.index.tolist() == [0.0, 0.02, 0.05]
        assert table.index.name == "odds_haircut"
        assert table["Yield [%]"].is_monotonic_decreasing
        pd.testing.assert_frame_equal(backtest.detailed_results, original)
        assert backtest.odds_haircut == 0.0

    def test_zero_haircut_matches_run(self, games):
        backtest = _prediction_backtest(games)
        backtest.run()
        table = backtest.stress_test(haircuts=[0.0], metrics=["Bankroll Final [$]"])
        assert table.loc[0.0, "Bankroll Final [$]"] == pytest.approx(
            backtest.detailed_results["bt_ending_bankroll"].iloc[-1]
        )

    def test_invalid_haircut_raises(self, games):
        with pytest.raises(ValueError, match="haircut"):
            _prediction_backtest(games).stress_test(haircuts=[1.2])


class TestValidation:
    """Invalid parameters are rejected at construction."""

    @pytest.mark.parametrize("kwargs, match", [
        ({"closing_odds_columns": ["close_a"]}, "one column per odds column"),
        ({"closing_odds_columns": ["close_a", "nope"]}, "not found"),
        ({"odds_haircut": 1.0}, "odds_haircut"),
        ({"odds_haircut": -0.1}, "odds_haircut"),
        ({"margin_method": "magic"}, "margin_method"),
    ])
    def test_invalid_parameters(self, games, kwargs, match):
        with pytest.raises(ValueError, match=match):
            _prediction_backtest(games, **kwargs)
