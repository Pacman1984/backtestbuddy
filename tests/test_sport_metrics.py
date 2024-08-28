import pytest
import pandas as pd
import numpy as np
from src.metrics.sport_metrics import *

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'bt_date_column': pd.date_range(start='2023-01-01', periods=10),
        'bt_starting_bankroll': [1000] * 10,
        'bt_ending_bankroll': [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1500],
        'bt_profit': [0, 100, -50, 150, -50, 150, -50, 150, -50, 150],
        'bt_win': [False, True, False, True, False, True, False, True, False, True],
        'bt_odds': [1.5, 2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3, 1.6, 2.4],
        'bt_stake': [100] * 10
    })

class TestCalculateROI:
    def test_calculate_roi(self, sample_data):
        assert calculate_roi(sample_data) == pytest.approx(0.5)

    def test_calculate_roi_no_change(self):
        data = pd.DataFrame({'bt_starting_bankroll': [1000], 'bt_ending_bankroll': [1000]})
        assert calculate_roi(data) == 0

class TestCalculateSharpeRatio:
    def test_calculate_sharpe_ratio(self, sample_data):
        assert calculate_sharpe_ratio(sample_data) > 0

class TestCalculateMaxDrawdown:
    def test_calculate_max_drawdown(self, sample_data):
        assert calculate_max_drawdown(sample_data) == pytest.approx(-0.0454545, rel=1e-5)

class TestCalculateWinRate:
    def test_calculate_win_rate(self, sample_data):
        assert calculate_win_rate(sample_data) == 0.5

    def test_calculate_win_rate_no_bets(self):
        data = pd.DataFrame({'bt_win': []})
        assert calculate_win_rate(data) == 0

class TestCalculateAverageOdds:
    def test_calculate_average_odds(self, sample_data):
        assert calculate_average_odds(sample_data) == pytest.approx(1.95)

    def test_calculate_average_odds_empty(self):
        data = pd.DataFrame({'bt_odds': []})
        assert np.isnan(calculate_average_odds(data))

class TestCalculateTotalProfit:
    def test_calculate_total_profit(self, sample_data):
        assert calculate_total_profit(sample_data) == 500

class TestCalculateAverageStake:
    def test_calculate_average_stake(self, sample_data):
        assert calculate_average_stake(sample_data) == 100

class TestCalculateSortinoRatio:
    def test_calculate_sortino_ratio(self, sample_data):
        assert calculate_sortino_ratio(sample_data) > 0

class TestCalculateCalmarRatio:
    def test_calculate_calmar_ratio(self, sample_data):
        assert calculate_calmar_ratio(sample_data) > 0

class TestCalculateDrawdowns:
    def test_calculate_drawdowns(self, sample_data):
        avg_dd, avg_dur, max_dd, max_dur, median_dur = calculate_drawdowns(sample_data)
        assert avg_dd == pytest.approx(0.04035, rel=1e-3)
        assert avg_dur == 1
        assert max_dd == pytest.approx(0.0454545, rel=1e-5)
        assert max_dur == 1
        assert median_dur == 1

    def test_calculate_drawdowns_no_drawdown(self):
        data = pd.DataFrame({'bt_ending_bankroll': [1000, 1100, 1200, 1300]})
        avg_dd, avg_dur, max_dd, max_dur, median_dur = calculate_drawdowns(data)
        assert avg_dd == 0
        assert avg_dur == 0
        assert max_dd == 0
        assert max_dur == 0
        assert median_dur == 0

class TestCalculateBestWorstBets:
    def test_calculate_best_worst_bets(self, sample_data):
        best, worst = calculate_best_worst_bets(sample_data)
        assert best == 150
        assert worst == -50

class TestCalculateHighestOdds:
    def test_calculate_highest_odds(self, sample_data):
        highest_win, highest_lose = calculate_highest_odds(sample_data)
        assert highest_win == 2.4
        assert highest_lose == 1.9

class TestCalculateAllMetrics:
    def test_calculate_all_metrics(self, sample_data):
        metrics = calculate_all_metrics(sample_data)
        assert isinstance(metrics, dict)
        assert len(metrics) > 0
        assert metrics['ROI [%]'] == pytest.approx(50.0)
        assert metrics['Total Profit [$]'] == 500
        assert metrics['Win Rate [%]'] == 50.0
        assert metrics['Total Bets'] == 10