"""
Unit tests for closing line value and the margin-free market baseline.

Uses hand-computed bets: raw and no-vig CLV, the share of bets that beat
the closing line, NaN without closing odds, and market Brier / log loss.
"""
import numpy as np
import pandas as pd
import pytest

from backtestbuddy.metrics.sport_metrics import (
    calculate_all_metrics,
    calculate_beat_closing_line,
    calculate_clv,
    calculate_market_brier_score,
    calculate_market_log_loss,
)


@pytest.fixture
def bets():
    """Two placed bets and one skipped game with closing odds."""
    return pd.DataFrame({
        "bt_date_column": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
        "bt_stake": [10.0, 10.0, 0.0],
        "bt_bet_on": [0, 1, -1],
        "bt_odds": [2.20, 1.80, np.nan],
        "bt_win": [True, False, None],
        "bt_profit": [12.0, -10.0, 0.0],
        "bt_starting_bankroll": [1000.0, 1012.0, 1002.0],
        "bt_ending_bankroll": [1012.0, 1002.0, 1002.0],
        "bt_actual_outcome": [0, 0, 1],
        "bt_odd_0": [2.20, 2.10, 1.90],
        "bt_odd_1": [1.70, 1.80, 1.90],
        "bt_closing_odd_0": [2.00, 2.20, 1.90],
        "bt_closing_odd_1": [1.90, 1.70, 1.90],
    })


class TestClosingLineValue:
    """CLV compares the odds taken with the closing odds of the same outcome."""

    def test_raw_clv(self, bets):
        # bet 1: 2.20 / 2.00 - 1 = +10%; bet 2: 1.80 / 1.70 - 1 = +5.88%
        expected = (0.10 + (1.80 / 1.70 - 1)) / 2 * 100
        assert calculate_clv(bets) == pytest.approx(expected)

    def test_no_vig_clv(self, bets):
        fair_1 = (1 / 2.00) / (1 / 2.00 + 1 / 1.90)
        fair_2 = (1 / 1.70) / (1 / 2.20 + 1 / 1.70)
        expected = ((2.20 * fair_1 - 1) + (1.80 * fair_2 - 1)) / 2 * 100
        assert calculate_clv(bets, no_vig=True) == pytest.approx(expected)

    def test_no_vig_clv_method_changes_result(self, bets):
        assert calculate_clv(bets, no_vig=True, method="power") != pytest.approx(
            calculate_clv(bets, no_vig=True)
        )

    def test_beat_closing_line(self, bets):
        assert calculate_beat_closing_line(bets) == pytest.approx(100.0)
        bets.loc[0, "bt_closing_odd_0"] = 2.20  # equal odds do not count
        assert calculate_beat_closing_line(bets) == pytest.approx(50.0)

    def test_without_closing_odds_is_nan(self, bets):
        data = bets.drop(columns=["bt_closing_odd_0", "bt_closing_odd_1"])
        assert np.isnan(calculate_clv(data))
        assert np.isnan(calculate_beat_closing_line(data))

    def test_no_placed_bets_is_zero(self, bets):
        data = bets.assign(bt_stake=0.0, bt_bet_on=-1)
        assert calculate_clv(data) == 0.0


class TestMarketBaseline:
    """Margin-free market probabilities scored on every game."""

    def test_market_brier_even_market(self):
        data = pd.DataFrame({
            "bt_actual_outcome": [0],
            "bt_odd_0": [1.9],
            "bt_odd_1": [1.9],
        })
        assert calculate_market_brier_score(data) == pytest.approx(0.25)
        assert calculate_market_log_loss(data) == pytest.approx(-np.log(0.5))

    def test_market_brier_uses_all_games(self, bets):
        pairs = []
        for _, row in bets.iterrows():
            implied = np.array([1 / row["bt_odd_0"], 1 / row["bt_odd_1"]])
            fair = implied / implied.sum()
            for k in range(2):
                pairs.append((fair[k], 1.0 if row["bt_actual_outcome"] == k else 0.0))
        expected = np.mean([(p - y) ** 2 for p, y in pairs])
        assert calculate_market_brier_score(bets) == pytest.approx(expected)

    def test_without_outcomes_is_nan(self, bets):
        assert np.isnan(calculate_market_brier_score(bets.drop(columns="bt_actual_outcome")))


class TestSummaryKeys:
    """calculate_all_metrics reports the new keys and honors margin_method."""

    def test_keys_present(self, bets):
        metrics = calculate_all_metrics(bets)
        for key in (
            "CLV [%]",
            "CLV (no-vig) [%]",
            "Beat Closing Line [%]",
            "Brier Score (market, all games) [-]",
            "Log Loss (market, all games) [-]",
        ):
            assert key in metrics

    def test_margin_method_is_passed_through(self, bets):
        default = calculate_all_metrics(bets)
        power = calculate_all_metrics(bets, margin_method="power")
        assert power["CLV (no-vig) [%]"] == pytest.approx(
            calculate_clv(bets, no_vig=True, method="power")
        )
        assert power["CLV [%]"] == pytest.approx(default["CLV [%]"])
