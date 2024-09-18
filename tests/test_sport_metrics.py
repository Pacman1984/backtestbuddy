import pytest
import pandas as pd
import numpy as np
from backtestbuddy.metrics.sport_metrics import *

@pytest.fixture
def sample_data():
    return pd.DataFrame({
        'bt_date_column': pd.date_range(start='2023-01-01', periods=11),
        'bt_starting_bankroll': [1000] * 11,
        'bt_ending_bankroll': [1000, 1100, 1050, 1200, 1150, 1300, 1250, 1400, 1350, 1500, 1500],
        'bt_profit': [0, 100, -50, 150, -50, 150, -50, 150, -50, 150, 0],
        'bt_win': [False, True, False, True, False, True, False, True, False, True, None],
        'bt_odds': [1.5, 2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3, 1.6, 2.4, None],
        'bt_stake': [100] * 10 + [0],
        'bt_bet_on': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, -1]  # Added this line
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

    def test_calculate_win_rate_empty_dataframe(self):
        data = pd.DataFrame({
            'bt_stake': [],
            'bt_bet_on': [],
            'bt_win': []
        })
        assert calculate_win_rate(data) == 0

    def test_calculate_win_rate_no_bets(self):
        data = pd.DataFrame({
            'bt_stake': [0, 0, 0],
            'bt_bet_on': [-1, -1, -1],
            'bt_win': [None, None, None]
        })
        assert calculate_win_rate(data) == 0

    def test_calculate_win_rate_all_wins(self):
        data = pd.DataFrame({
            'bt_stake': [100, 100, 100],
            'bt_bet_on': [0, 1, 0],
            'bt_win': [True, True, True]
        })
        assert calculate_win_rate(data) == 1.0

    def test_calculate_win_rate_mixed(self):
        data = pd.DataFrame({
            'bt_stake': [100, 0, 100, 100, 0],
            'bt_bet_on': [0, -1, 1, 0, -1],
            'bt_win': [True, None, False, True, None]
        })
        assert calculate_win_rate(data) == 2/3

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

    def test_calculate_average_stake_with_zero_stakes(self):
        data = pd.DataFrame({
            'bt_stake': [100, 100, 0, 100, 0]
        })
        assert calculate_average_stake(data) == 100

    def test_calculate_average_stake_all_zero(self):
        data = pd.DataFrame({
            'bt_stake': [0, 0, 0]
        })
        assert calculate_average_stake(data) == 0

class TestCalculateSortinoRatio:
    def test_calculate_sortino_ratio(self, sample_data):
        assert calculate_sortino_ratio(sample_data) > 0

class TestCalculateCalmarRatio:
    def test_calculate_calmar_ratio(self, sample_data):
        assert calculate_calmar_ratio(sample_data) > 0

class TestCalculateDrawdowns:
    def test_calculate_drawdowns(self, sample_data):
        max_dd, max_dur = calculate_drawdowns(sample_data)
        assert max_dd == pytest.approx(0.0454545, rel=1e-5)
        assert max_dur == 2

    def test_calculate_drawdowns_no_drawdown(self):
        data = pd.DataFrame({'bt_ending_bankroll': [1000, 1100, 1200, 1300]})
        max_dd, max_dur = calculate_drawdowns(data)
        assert max_dd == 0.0
        assert max_dur == 0

    def test_calculate_drawdowns_empty_data(self):
        data = pd.DataFrame({'bt_ending_bankroll': []})
        max_dd, max_dur = calculate_drawdowns(data)
        assert max_dd == 0.0
        assert max_dur == 0

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