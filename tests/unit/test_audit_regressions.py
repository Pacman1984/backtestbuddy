"""
Regression tests for bugs found in the package audit.

Each test pins one fixed behavior: drawdown from the opening bankroll,
stable same-date ordering, Kelly requiring probabilities, stake
validation, explicit model features, and an empty odds histogram.
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from backtestbuddy.backtest.sport_backtest import ModelBacktest, PredictionBacktest
from backtestbuddy.metrics.sport_metrics import (
    _compound_period_returns,
    calculate_drawdowns,
    calculate_max_drawdown,
)
from backtestbuddy.plots.sport_plots import _underwater_pct, plot_odds_histogram
from backtestbuddy.strategies.sport_strategies import (
    FixedStake,
    KellyCriterion,
    ValueBet,
)


def _games(n=6, same_day=False):
    """Two-way market with alternating outcomes."""
    dates = (
        [pd.Timestamp("2023-01-01")] * n
        if same_day
        else pd.date_range("2023-01-01", periods=n)
    )
    return pd.DataFrame({
        "date": dates,
        "game_id": np.arange(n),
        "odds_0": [2.0] * n,
        "odds_1": [1.9] * n,
        "outcome": [i % 2 for i in range(n)],
        "pred": [0] * n,
        "p_0": [0.6] * n,
        "p_1": [0.4] * n,
    })


class TestDrawdownFromOpeningBankroll:
    """Losses on the first bets count against the opening bankroll."""

    @pytest.fixture
    def losing_start(self):
        return pd.DataFrame({
            "bt_starting_bankroll": [1000.0, 900.0],
            "bt_ending_bankroll": [900.0, 800.0],
        })

    def test_max_drawdown_includes_opening_bankroll(self, losing_start):
        assert calculate_max_drawdown(losing_start) == pytest.approx(-0.20)

    def test_drawdowns_includes_opening_bankroll(self, losing_start):
        magnitude, duration = calculate_drawdowns(losing_start)
        assert magnitude == pytest.approx(0.20)
        assert duration == 3

    def test_underwater_seeded_with_opening_bankroll(self):
        result = _underwater_pct(np.array([90.0, 80.0]), initial=100.0)
        np.testing.assert_allclose(result, [-10.0, -20.0])


class TestCompoundPeriodReturns:
    """Vectorized compounding matches prod(1 + r) - 1."""

    def test_total_loss_is_minus_one(self):
        data = pd.DataFrame({
            "bt_date_column": pd.to_datetime(["2023-01-01", "2023-01-01"]),
            "bt_profit": [-100.0, 0.0],
            "bt_starting_bankroll": [100.0, 0.0001],
        })
        returns = _compound_period_returns(data)
        assert returns.iloc[0] == pytest.approx(-1.0)

    def test_empty_days_are_skipped(self):
        data = pd.DataFrame({
            "bt_date_column": pd.to_datetime(["2023-01-01", "2023-01-05"]),
            "bt_profit": [10.0, -5.0],
            "bt_starting_bankroll": [100.0, 110.0],
        })
        returns = _compound_period_returns(data)
        assert len(returns) == 2
        np.testing.assert_allclose(returns.to_numpy(), [0.10, -5.0 / 110.0])


class TestSameDateOrder:
    """Same-date rows keep their input order after sorting."""

    def test_input_order_preserved(self):
        data = _games(n=200, same_day=True)
        backtest = PredictionBacktest(
            data, ["odds_0", "odds_1"], "outcome", "date", "pred",
            strategy=FixedStake(10),
        )
        assert backtest.data["game_id"].tolist() == list(range(200))


class TestKellyRequiresProbabilities:
    """Kelly without probability columns fails loudly instead of never betting."""

    def test_prediction_backtest_without_probs_raises(self):
        with pytest.raises(ValueError, match="requires model probabilities"):
            PredictionBacktest(
                _games(), ["odds_0", "odds_1"], "outcome", "date", "pred",
                strategy=KellyCriterion(),
            )

    def test_prediction_backtest_with_probs_bets(self):
        backtest = PredictionBacktest(
            _games(), ["odds_0", "odds_1"], "outcome", "date", "pred",
            strategy=KellyCriterion(), model_prob_columns=["p_0", "p_1"],
        )
        backtest.run()
        assert (backtest.detailed_results["bt_stake"] > 0).all()


class TestStrategyValidation:
    """Invalid sizing parameters are rejected at construction."""

    @pytest.mark.parametrize("factory", [
        lambda: FixedStake(-0.5),
        lambda: ValueBet(stake=-1),
        lambda: KellyCriterion(downscaling=-0.5),
        lambda: KellyCriterion(max_bet=1.5),
    ])
    def test_invalid_parameters_raise(self, factory):
        with pytest.raises(ValueError):
            factory()


class _RecordingModel:
    """Constant-probability model that remembers its training columns."""

    classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.columns = list(X.columns)
        return self

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.6, 0.4], (len(X), 1))


class TestModelFeatureColumns:
    """ModelBacktest trains only on the requested features."""

    def test_explicit_feature_columns(self):
        model = _RecordingModel()
        backtest = ModelBacktest(
            _games(n=30), ["odds_0", "odds_1"], "outcome", "date", model,
            feature_columns=["odds_0", "odds_1"],
        )
        backtest.run()
        assert model.columns == ["odds_0", "odds_1"]

    def test_missing_feature_column_raises(self):
        with pytest.raises(ValueError, match="not found"):
            ModelBacktest(
                _games(), ["odds_0", "odds_1"], "outcome", "date",
                _RecordingModel(), feature_columns=["nope"],
            )


class TestOddsHistogramNoBets:
    """The odds histogram renders a placeholder when nothing was staked."""

    def test_no_placed_bets_returns_figure(self):
        backtest = PredictionBacktest(
            _games(), ["odds_0", "odds_1"], "outcome", "date", "pred",
            strategy=FixedStake(0),
        )
        backtest.run()
        fig = plot_odds_histogram(backtest)
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 0
