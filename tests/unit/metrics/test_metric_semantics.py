"""
Unit tests for metric definitions that changed meaning.

Covers duration-weighted yearly ROI, Sharpe/Sortino on placed-bet periods
only, the corrected observed periods per year, calibration over all games,
and removal of the macro per-bet ROI from the summary table.
"""
import numpy as np
import pandas as pd
import pytest

from backtestbuddy.metrics.sport_metrics import (
    _all_outcome_probability_frame,
    _observed_periods_per_year,
    calculate_all_metrics,
    calculate_avg_roi_per_year_micro,
    calculate_brier_score,
    calculate_ece,
    calculate_log_loss,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
)


def _rows(dates, start, end, stake=None, bet_on=None):
    n = len(dates)
    return pd.DataFrame({
        "bt_date_column": pd.to_datetime(dates),
        "bt_starting_bankroll": start,
        "bt_ending_bankroll": end,
        "bt_profit": np.subtract(end, start),
        "bt_stake": stake if stake is not None else [10.0] * n,
        "bt_bet_on": bet_on if bet_on is not None else [0] * n,
    })


class TestYearlyRoiMicroWeighting:
    """Partial calendar years count by the days they cover."""

    def test_partial_years_are_weighted_by_days(self):
        data = _rows(
            ["2022-12-18", "2022-12-31", "2023-01-01", "2023-12-31", "2024-01-01"],
            start=[1000, 1000, 1000, 1000, 800],
            end=[1000, 950, 1000, 800, 820],
        )
        # 2022: 14 days, -5%; 2023: 365 days, -20%; 2024: 1 day, +2.5%
        expected = (14 * -5.0 + 365 * -20.0 + 1 * 2.5) / 380
        assert calculate_avg_roi_per_year_micro(data) == pytest.approx(expected)

    def test_full_years_keep_equal_weights(self):
        data = _rows(
            ["2021-01-01", "2021-12-31", "2022-01-01", "2022-12-31"],
            start=[1000, 1000, 1500, 1500],
            end=[1000, 1500, 1500, 1800],
        )
        assert calculate_avg_roi_per_year_micro(data) == pytest.approx(35.0)


class TestRatiosIgnoreSkippedOpportunities:
    """Adding rows without a placed bet does not change Sharpe or Sortino."""

    @pytest.fixture
    def bets(self):
        return _rows(
            ["2023-01-01", "2023-01-03", "2023-01-05", "2023-01-08"],
            start=[1000, 1100, 1050, 1150],
            end=[1100, 1050, 1150, 1080],
        )

    @pytest.fixture
    def bets_with_skips(self, bets):
        skipped = _rows(
            ["2023-01-02", "2023-01-04", "2023-01-06", "2023-01-07"],
            start=[1100, 1050, 1150, 1150],
            end=[1100, 1050, 1150, 1150],
            stake=[0.0] * 4,
            bet_on=[-1] * 4,
        )
        return (
            pd.concat([bets, skipped])
            .sort_values("bt_date_column", kind="mergesort")
            .reset_index(drop=True)
        )

    def test_sharpe_unchanged(self, bets, bets_with_skips):
        assert calculate_sharpe_ratio(bets_with_skips) == pytest.approx(
            calculate_sharpe_ratio(bets)
        )

    def test_sortino_unchanged(self, bets, bets_with_skips):
        assert calculate_sortino_ratio(bets_with_skips) == pytest.approx(
            calculate_sortino_ratio(bets)
        )


class TestObservedPeriodsPerYear:
    """Consecutive daily betting annualizes to a calendar year."""

    def test_ten_consecutive_days(self):
        dates = pd.date_range("2023-01-01", periods=10)
        data = _rows(dates, start=[1000] * 10, end=[1010] * 10)
        assert _observed_periods_per_year(data) == pytest.approx(365.25)

    def test_single_day(self):
        data = _rows(["2023-01-01"], start=[1000], end=[1010])
        assert _observed_periods_per_year(data) == pytest.approx(365.25)


class TestCalibrationScopeAll:
    """Calibration over every outcome of every game, bet or not."""

    @pytest.fixture
    def games(self):
        return pd.DataFrame({
            "bt_stake": [10.0, 0.0],
            "bt_bet_on": [0, -1],
            "bt_odds": [2.0, np.nan],
            "bt_win": [True, None],
            "bt_profit": [10.0, 0.0],
            "bt_actual_outcome": [0, 1],
            "bt_model_prob_0": [0.6, 0.7],
            "bt_model_prob_1": [0.4, 0.3],
        })

    def test_pairs_include_unbet_games(self, games):
        frame = _all_outcome_probability_frame(games)
        pairs = sorted(zip(frame["p"], frame["win"]))
        assert pairs == [(0.3, 1.0), (0.4, 0.0), (0.6, 1.0), (0.7, 0.0)]

    def test_brier_all_games(self, games):
        expected = np.mean([(0.6 - 1) ** 2, 0.4**2, 0.7**2, (0.3 - 1) ** 2])
        assert calculate_brier_score(games, scope="all") == pytest.approx(expected)

    def test_placed_scope_uses_bet_only(self, games):
        assert calculate_brier_score(games) == pytest.approx((0.6 - 1) ** 2)

    def test_log_loss_and_ece_accept_scope(self, games):
        assert np.isfinite(calculate_log_loss(games, scope="all"))
        assert 0.0 <= calculate_ece(games, scope="all") <= 1.0

    def test_no_probabilities_is_nan(self, games):
        data = games.drop(columns=["bt_model_prob_0", "bt_model_prob_1"])
        assert np.isnan(calculate_brier_score(data, scope="all"))

    def test_unknown_scope_raises(self, games):
        with pytest.raises(ValueError, match="scope"):
            calculate_brier_score(games, scope="everything")


class TestSummaryTableKeys:
    """Summary keys name the calibration scope; macro per-bet ROI is gone."""

    def test_keys(self):
        data = _rows(
            ["2023-01-01", "2023-01-02"],
            start=[1000, 1010],
            end=[1010, 1000],
        )
        data["bt_win"] = [True, False]
        data["bt_odds"] = [2.0, 2.0]
        metrics = calculate_all_metrics(data)
        assert "Avg. ROI per Bet [%] (macro)" not in metrics
        for name in ("Brier Score", "Log Loss", "ECE"):
            assert f"{name} (placed) [-]" in metrics
            assert f"{name} (all games) [-]" in metrics
            assert f"{name} [-]" not in metrics
