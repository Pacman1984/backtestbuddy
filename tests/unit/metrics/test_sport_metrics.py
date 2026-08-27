"""
Unit tests for sport_metrics module.

Tests all metric calculation functions with deterministic data.
"""
import pytest
import pandas as pd
import numpy as np
from backtestbuddy.metrics.sport_metrics import *
from backtestbuddy.metrics.sport_metrics import (
    _observed_periods_per_year,
    _reliability_points,
)

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
    def test_calculate_sharpe_ratio(self):
        """Test Sharpe at P=252 equals annualized mean / sample std."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-02', '2023-01-03']
            ),
            'bt_profit': [100, -50, 200],
            'bt_starting_bankroll': [1000, 1000, 1000],
        })
        # r = [0.10, -0.05, 0.20]
        # mean = 0.083333..., sample std (ddof=1) = 0.12583057...
        # Sharpe = mean*252 / (std*sqrt(252))
        expected = 10.513149660756934
        assert calculate_sharpe_ratio(
            data, output_period=252
        ) == pytest.approx(expected)

    def test_sharpe_default_is_calendar_year(self):
        """Test default P=365.25 scales vs 252 by sqrt(365.25/252)."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-02', '2023-01-03']
            ),
            'bt_profit': [100, -50, 200],
            'bt_starting_bankroll': [1000, 1000, 1000],
        })
        trading = calculate_sharpe_ratio(data, output_period=252)
        calendar = calculate_sharpe_ratio(data)
        assert calendar == pytest.approx(
            trading * np.sqrt(365.25 / 252)
        )

    def test_sharpe_does_not_mutate_input(self):
        """Test Sharpe copies the frame instead of writing bt_date_column."""
        data = pd.DataFrame({
            'bt_date_column': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'bt_profit': [100, -50, 200],
            'bt_starting_bankroll': [1000, 1000, 1000],
        })
        original = data['bt_date_column'].tolist()
        calculate_sharpe_ratio(data)
        assert data['bt_date_column'].tolist() == original

    def test_sharpe_compounds_same_day_returns(self):
        """Test two bets on one day use prod(1+r)-1, not a simple sum."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-01', '2023-01-02']
            ),
            'bt_profit': [100, -50, 200],
            'bt_starting_bankroll': [1000, 1000, 1000],
        })
        # Day 1: (1.10)*(0.95)-1 = 0.045; day 2: 0.20
        # Sharpe of [0.045, 0.20]
        expected = 17.742697930831255
        wrong_sum = 18.708286933869704
        result = calculate_sharpe_ratio(data, output_period=252)
        assert result == pytest.approx(expected)
        assert result != pytest.approx(wrong_sum, rel=1e-4)

    def test_sharpe_zero_with_fewer_than_two_periods(self):
        """Test Sharpe is 0 when there is only one return period."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01']),
            'bt_profit': [100],
            'bt_starting_bankroll': [1000],
        })
        assert calculate_sharpe_ratio(data) == 0.0

    def test_observed_periods_per_year_uses_date_span(self):
        """Test obs/year P is n_unique_days / (span_days / 365.25)."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-02', '2023-01-03']
            ),
            'bt_profit': [100, -50, 200],
            'bt_starting_bankroll': [1000, 1000, 1000],
        })
        years = 2 / 365.25
        expected_p = 3 / years
        assert _observed_periods_per_year(data) == pytest.approx(expected_p)

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
        """Test win rate ignores no-bet rows among mixed results."""
        data = pd.DataFrame({
            'bt_stake': [100, 0, 100, 100, 0],
            'bt_bet_on': [0, -1, 1, 0, -1],
            'bt_win': [True, None, False, True, None]
        })
        assert calculate_win_rate(data) == 2/3

    def test_calculate_win_rate_zero_stake_not_a_bet(self):
        """Test stake 0 with bet_on set is not counted as a placed bet."""
        data = pd.DataFrame({
            'bt_stake': [0, 100],
            'bt_bet_on': [0, 1],
            'bt_win': [None, True],
        })
        assert calculate_win_rate(data) == 1.0

    def test_calculate_win_rate_skipped_outcome_not_a_bet(self):
        """Test a positive stake with bet_on -1 is not a placed bet."""
        data = pd.DataFrame({
            'bt_stake': [100, 100],
            'bt_bet_on': [-1, 1],
            'bt_win': [None, True],
        })
        assert calculate_win_rate(data) == 1.0

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
    def test_calculate_sortino_ratio(self):
        """Test Sortino uses downside deviation of all periods including zeros."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-02', '2023-01-03']
            ),
            'bt_starting_bankroll': [1000, 1000, 1000],
            'bt_profit': [100, -50, 200],
        })
        # r = [0.10, -0.05, 0.20]; shortfalls = [0, -0.05, 0]
        expected = 45.8257569495584
        assert calculate_sortino_ratio(
            data, output_period=252
        ) == pytest.approx(expected)

    def test_sortino_default_is_calendar_year(self):
        """Test default Sortino P=365.25 scales vs 252 by sqrt(365.25/252)."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-02', '2023-01-03']
            ),
            'bt_starting_bankroll': [1000, 1000, 1000],
            'bt_profit': [100, -50, 200],
        })
        trading = calculate_sortino_ratio(data, output_period=252)
        calendar = calculate_sortino_ratio(data)
        assert calendar == pytest.approx(
            trading * np.sqrt(365.25 / 252)
        )

    def test_sortino_ratio_no_negative_returns(self):
        """Test Sortino is inf when there is no downside and mean excess > 0."""
        data = pd.DataFrame({
            'bt_date_column': pd.date_range(start='2023-01-01', periods=5),
            'bt_starting_bankroll': [1000] * 5,
            'bt_profit': [100, 200, 50, 150, 100],
            'bt_stake': [100] * 5,
            'bt_bet_on': [1] * 5,
        })
        assert calculate_sortino_ratio(data) == float('inf')

    def test_sortino_zero_when_flat_returns(self):
        """Test Sortino is 0 when there is no downside and mean excess is 0."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-02', '2023-01-03']
            ),
            'bt_starting_bankroll': [1000, 1000, 1000],
            'bt_profit': [0, 0, 0],
        })
        assert calculate_sortino_ratio(data) == 0.0


class TestCalculateCalmarRatio:
    def test_calculate_calmar_ratio(self):
        """Test Calmar = geometric annual return / |return-curve max DD|."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2024-01-01']),
            'bt_profit': [100, -55],
            'bt_starting_bankroll': [1000, 1100],
            'bt_ending_bankroll': [1100, 1045],
        })
        years = 365 / 365.25
        r_annual = 1.045 ** (1 / years) - 1
        expected = r_annual / 0.05
        assert calculate_calmar_ratio(data) == pytest.approx(expected)

    def test_calmar_inf_when_no_drawdown_and_positive_return(self):
        """Test Calmar is inf when the return curve never draws down."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2024-01-01']),
            'bt_profit': [100, 100],
            'bt_starting_bankroll': [1000, 1100],
            'bt_ending_bankroll': [1100, 1200],
        })
        assert calculate_calmar_ratio(data) == float('inf')

    def test_calmar_compounds_same_day_returns(self):
        """Test same-day bets compound; sum would understate the drawdown."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(
                ['2023-01-01', '2023-01-01', '2024-01-01']
            ),
            'bt_profit': [100, -200, 300],
            'bt_starting_bankroll': [1000, 1000, 1000],
            'bt_ending_bankroll': [1100, 900, 1200],
        })
        # Compounded day-1 return: 1.10*0.80-1 = -0.12 (sum would be -0.10)
        # cumprod: 0.88, 0.88*1.30 = 1.144; max DD = 0.12
        years = 365 / 365.25
        r_annual = 1.144 ** (1 / years) - 1
        expected = r_annual / 0.12
        assert calculate_calmar_ratio(data) == pytest.approx(expected)

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

    def test_calculate_drawdowns_equal_peaks_uses_last_peak(self):
        # Equal peaks at indices 1 and 2; trough at index 4
        # Expect start at last peak (index 2), duration = 4 - 2 + 1 = 3
        data = pd.DataFrame({'bt_ending_bankroll': [100, 120, 120, 110, 90]})
        max_dd, max_dur = calculate_drawdowns(data)
        assert max_dd == pytest.approx(0.25, rel=1e-6)
        assert max_dur == 3

    def test_max_drawdown_matches_drawdowns_magnitude(self):
        """Test signed max drawdown is the negative of drawdowns magnitude."""
        data = pd.DataFrame({
            'bt_ending_bankroll': [1000, 850, 1100],
        })
        magnitude, duration = calculate_drawdowns(data)
        assert calculate_max_drawdown(data) == pytest.approx(-magnitude)
        assert magnitude == pytest.approx(0.15)
        assert duration == 2

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
        assert 'Risk-Adjusted Annual ROI [-]' in metrics
        assert 'CAGR [%]' in metrics
        assert metrics['Max Drawdown [%]'] == pytest.approx(4.54545, rel=1e-5)
        trading = metrics['Sharpe Ratio (252) [-]']
        calendar = metrics['Sharpe Ratio (365.25) [-]']
        assert calendar == pytest.approx(trading * np.sqrt(365.25 / 252))
        assert metrics['Sortino Ratio (365.25) [-]'] == pytest.approx(
            metrics['Sortino Ratio (252) [-]'] * np.sqrt(365.25 / 252)
        )
        for key in (
            'Sharpe Ratio (365.25) [-]',
            'Sharpe Ratio (252) [-]',
            'Sharpe Ratio (obs/year) [-]',
            'Sortino Ratio (365.25) [-]',
            'Sortino Ratio (252) [-]',
            'Sortino Ratio (obs/year) [-]',
        ):
            assert key in metrics

class TestCalculateAverageROIPerBet:
    def test_consistent_profits_micro(self):
        """Test case with consistent profits for micro-averaging"""
        data = pd.DataFrame({
            'bt_stake': [100, 100, 100],
            'bt_profit': [20, 20, 20],
            'bt_bet_on': [1, 1, 1]
        })
        assert calculate_avg_roi_per_bet_micro(data) == 20.0  # (20/100) * 100 = 20%

    def test_mixed_profits_losses_micro(self):
        """Test case with mixed profits and losses for micro-averaging"""
        data = pd.DataFrame({
            'bt_stake': [100, 100, 100],
            'bt_profit': [50, -50, 0],
            'bt_bet_on': [1, 1, 1]
        })
        assert calculate_avg_roi_per_bet_micro(data) == 0.0  # Average of (50%, -50%, 0%) = 0%

    def test_empty_dataframe_micro(self):
        """Test case with empty DataFrame for micro-averaging"""
        data = pd.DataFrame({
            'bt_stake': [],
            'bt_profit': [],
            'bt_bet_on': []
        })
        assert calculate_avg_roi_per_bet_micro(data) == 0.0

    def test_no_bets_placed_micro(self):
        """Test case with no bets placed for micro-averaging"""
        data = pd.DataFrame({
            'bt_stake': [0, 0, 0],
            'bt_profit': [0, 0, 0],
            'bt_bet_on': [-1, -1, -1]
        })
        assert calculate_avg_roi_per_bet_micro(data) == 0.0

    def test_zero_stake_row_not_included_in_micro_roi(self):
        """Test a zero-stake row is not treated as a bet in micro ROI."""
        data = pd.DataFrame({
            'bt_stake': [0, 100],
            'bt_profit': [0, 20],
            'bt_bet_on': [0, 1],
        })
        assert calculate_avg_roi_per_bet_micro(data) == 20.0

    def test_consistent_profits_macro(self):
        """Test case with consistent profits for macro-averaging"""
        data = pd.DataFrame({
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 1100, 1200],
            'bt_stake': [100, 100, 100],
            'bt_profit': [20, 20, 20],
            'bt_bet_on': [1, 1, 1]
        })
        # Total ROI = (1200 - 1000) / 1000 = 0.2 = 20%
        # Number of bets = 3
        # Macro ROI per bet = 20% / 3 = 6.67%
        assert calculate_avg_roi_per_bet_macro(data) == pytest.approx(6.67, rel=1e-2)

    def test_mixed_profits_losses_macro(self):
        """Test case with mixed profits and losses for macro-averaging"""
        data = pd.DataFrame({
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 1050, 1100],
            'bt_stake': [100, 100, 100],
            'bt_profit': [50, -50, 100],
            'bt_bet_on': [1, 1, 1]
        })
        # Total ROI = (1100 - 1000) / 1000 = 0.1 = 10%
        # Number of bets = 3
        # Macro ROI per bet = 10% / 3 = 3.33%
        assert calculate_avg_roi_per_bet_macro(data) == pytest.approx(3.33, rel=1e-2)

    def test_empty_dataframe_macro(self):
        """Test case with empty DataFrame for macro-averaging"""
        data = pd.DataFrame({
            'bt_starting_bankroll': [],
            'bt_ending_bankroll': [],
            'bt_stake': [],
            'bt_profit': [],
            'bt_bet_on': []
        })
        assert calculate_avg_roi_per_bet_macro(data) == 0.0

    def test_no_bets_placed_macro(self):
        """Test case with no bets placed for macro-averaging"""
        data = pd.DataFrame({
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000] * 3,
            'bt_stake': [0, 0, 0],
            'bt_profit': [0, 0, 0],
            'bt_bet_on': [-1, -1, -1]
        })
        assert calculate_avg_roi_per_bet_macro(data) == 0.0

class TestCalculateAverageROIPerYear:
    def test_single_year_profits_micro(self):
        """Test case with single year consistent profits for micro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-06-01', '2023-12-31']),
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 1200, 1500],
            'bt_profit': [0, 200, 300],
            'bt_bet_on': [1, 1, 1]
        })
        # Single year ROI = (1500 - 1000) / 1000 = 0.5 = 50%
        # Only one year, so micro average = 50%
        assert calculate_avg_roi_per_year_micro(data) == pytest.approx(50.0, rel=5e-2)

    def test_multiple_years_mixed_micro(self):
        """Test case with multiple years and different performance for micro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01', '2021-12-31',  # Year 1: 50% ROI
                '2022-01-01', '2022-12-31',  # Year 2: 20% ROI
                '2023-01-01', '2023-12-31'   # Year 3: 30% ROI
            ]),
            'bt_starting_bankroll': [1000, 1000, 1200, 1200, 1440, 1440],
            'bt_ending_bankroll': [1000, 1500, 1200, 1440, 1440, 1872],
            'bt_bet_on': [1] * 6
        })
        # Year 1 ROI = (1500 - 1000) / 1000 = 50%
        # Year 2 ROI = (1440 - 1200) / 1200 = 20%
        # Year 3 ROI = (1872 - 1440) / 1440 = 30%
        # Micro average = (50% + 20% + 30%) / 3 = 33.33%
        assert calculate_avg_roi_per_year_micro(data) == pytest.approx(33.33, rel=5e-2)

    def test_single_year_profits_macro(self):
        """Test case with single year consistent profits for macro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-06-01', '2023-12-31']),
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 1200, 1500],
            'bt_profit': [0, 200, 300],
            'bt_bet_on': [1, 1, 1]
        })
        # Total ROI = (1500 - 1000) / 1000 = 0.5 = 50%
        # Time period = 1 year
        # Macro average = 50% / 1 = 50%
        assert calculate_avg_roi_per_year_macro(data) == pytest.approx(50.0, rel=5e-2)

    def test_multiple_years_mixed_macro(self):
        """Test case with multiple years and different performance for macro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01',  # Start
                '2023-12-31'   # End (3 years)
            ]),
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 2500],
            'bt_bet_on': [1, 1]
        })
        # Total ROI = (2500 - 1000) / 1000 = 1.5 = 150%
        # Time period = 3 years
        # Macro average = 150% / 3 = 50%
        assert calculate_avg_roi_per_year_macro(data) == pytest.approx(50.0, rel=5e-2)

    def test_empty_dataframe_micro(self):
        """Test case with empty DataFrame for micro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.Series(dtype='datetime64[ns]'),
            'bt_starting_bankroll': [],
            'bt_ending_bankroll': [],
            'bt_bet_on': []
        })
        assert calculate_avg_roi_per_year_micro(data) == 0.0

    def test_empty_dataframe_macro(self):
        """Test case with empty DataFrame for macro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.Series(dtype='datetime64[ns]'),
            'bt_starting_bankroll': [],
            'bt_ending_bankroll': [],
            'bt_bet_on': []
        })
        assert calculate_avg_roi_per_year_macro(data) == 0.0

    def test_same_day_micro(self):
        """Test case with same day (zero years) for micro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-01-01']),
            'bt_starting_bankroll': [1000, 1000],
            'bt_ending_bankroll': [1000, 1100],
            'bt_bet_on': [1, 1]
        })
        # Single day in single year, ROI = 10%
        assert calculate_avg_roi_per_year_micro(data) == pytest.approx(10.0, rel=5e-2)

    def test_same_day_macro(self):
        """Test case with same day (zero years) for macro-averaging"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-01-01']),
            'bt_starting_bankroll': [1000, 1000],
            'bt_ending_bankroll': [1000, 1100],
            'bt_bet_on': [1, 1]
        })
        # Zero years duration should return 0
        assert calculate_avg_roi_per_year_macro(data) == 0.0

class TestCalculateRiskAdjustedAnnualROI:
    def test_normal_case_multi_year(self):
        """Test case with multiple years, positive returns and no drawdown"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01',  # Start
                '2023-12-31'   # End (3 years)
            ]),
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 1150],  # 15% total ROI over 3 years ≈ 5% annual
            'bt_bet_on': [1] * 2
        })
        # Total ROI = (1150 - 1000) / 1000 = 0.15 = 15%
        # Time period = 3 years
        # Annual ROI ≈ 5% = 0.05 decimal
        # Max drawdown = 0
        # Risk-adjusted = inf (no drawdown)
        result = calculate_risk_adjusted_annual_roi(data)
        assert result == float('inf')

    def test_complete_loss_multi_year(self):
        """Test case with complete loss over multiple years"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01',  # Start
                '2023-12-31'   # End (3 years)
            ]),
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 800],  # -20% total ROI over 3 years ≈ -6.67% annual
            'bt_bet_on': [1] * 2
        })
        # Total ROI = (800 - 1000) / 1000 = -0.2 = -20%
        # Time period = 3 years
        # Annual ROI ≈ -6.67% = -0.0667 decimal
        # Max drawdown = -0.2 (20%)
        # Risk-adjusted = -0.0667 / 0.2 = -0.3335 (unitless ratio)
        result = calculate_risk_adjusted_annual_roi(data)
        assert result < 0
        assert result == pytest.approx(-0.3335, rel=5e-2)

    def test_no_drawdown_multi_year(self):
        """Test case with no drawdown over multiple years"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01',  # Start
                '2023-12-31'   # End (3 years)
            ]),
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 1300],  # 30% total ROI over 3 years ≈ 10% annual
            'bt_bet_on': [1] * 2
        })
        # Total ROI = (1300 - 1000) / 1000 = 0.3 = 30%
        # Time period = 3 years
        # Annual ROI ≈ 10% = 0.10 decimal
        # Max drawdown = 0
        # Risk-adjusted = inf (no drawdown)
        assert calculate_risk_adjusted_annual_roi(data) == float('inf')

    def test_zero_drawdown_and_zero_roi_returns_zero(self):
        """Test risk-adjusted ROI is 0 when there is no drawdown and no return."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2021-01-01', '2021-12-31']),
            'bt_starting_bankroll': [1000, 1000],
            'bt_ending_bankroll': [1000, 1000],
            'bt_bet_on': [1, 1],
        })
        assert calculate_risk_adjusted_annual_roi(data) == 0.0

    def test_negative_roi_with_drawdown(self):
        """Test case with negative ROI and drawdown to ensure consistent sign"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01',  # Start
                '2021-06-30',  # Mid-year drawdown
                '2021-12-31'   # End (1 year)
            ]),
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 800, 900],  # -10% total ROI, with -20% max drawdown
            'bt_bet_on': [1] * 3
        })
        # Total ROI = (900 - 1000) / 1000 = -0.1 = -10%
        # Time period = 1 year
        # Annual ROI = -10% = -0.10 decimal
        # Max drawdown = -0.2 (20%)
        # Risk-adjusted = -0.10 / 0.2 = -0.50 (unitless ratio)
        result = calculate_risk_adjusted_annual_roi(data)
        assert result < 0  # Should be negative since ROI is negative
        assert result == pytest.approx(-0.50, rel=5e-2)

class TestCalculateCAGR:
    def test_normal_case_multi_year(self):
        """Test CAGR calculation over multiple years with consistent growth"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']),
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 1200, 1440]  # 20% growth each year
        })
        # CAGR = (1440/1000)^(1/2) - 1 = 0.20 = 20%
        assert calculate_cagr(data) == pytest.approx(20.0, rel=1e-2)

    def test_single_year_growth(self):
        """Test CAGR calculation for a single year period"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-12-31']),
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 1500]  # 50% growth
        })
        # For single year, CAGR equals simple return
        assert calculate_cagr(data) == pytest.approx(50.0, rel=1e-2)

    def test_partial_year(self):
        """Test CAGR calculation for a period less than a year"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-07-01']),  # 6 months
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 1200]  # 20% growth in 6 months
        })
        # CAGR = (1200/1000)^(1/0.5) - 1 = 0.44 = 44%
        assert calculate_cagr(data) == pytest.approx(44.4, rel=1e-2)

    def test_negative_growth(self):
        """Test CAGR calculation with negative growth"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2020-01-01', '2021-01-01', '2022-01-01']),
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 800, 640]  # -20% each year
        })
        # CAGR = (640/1000)^(1/2) - 1 = -0.20 = -20%
        assert calculate_cagr(data) == pytest.approx(-20.0, rel=1e-2)

    def test_no_change(self):
        """Test CAGR calculation when there's no change in value"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2020-01-01', '2021-01-01']),
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 1000]
        })
        assert calculate_cagr(data) == 0.0

    def test_empty_dataframe(self):
        """Test CAGR calculation with empty DataFrame"""
        data = pd.DataFrame({
            'bt_date_column': pd.Series([], dtype='datetime64[ns]'),
            'bt_starting_bankroll': pd.Series([], dtype='float64'),
            'bt_ending_bankroll': pd.Series([], dtype='float64')
        })
        assert calculate_cagr(data) == 0.0

    def test_same_day(self):
        """Test CAGR calculation when start and end dates are the same"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-01-01']),
            'bt_starting_bankroll': [1000] * 2,
            'bt_ending_bankroll': [1000, 1200]
        })
        assert calculate_cagr(data) == 0.0

    def test_lost_all(self):
        """Test CAGR calculation when all bets are lost and bankroll goes to near zero"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-06-01', '2023-12-31']),
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 100, 10]  # Lost 99% of bankroll in one year
        })
        # CAGR = (10/1000)^(1/1) - 1 = -0.99 = -99%
        assert calculate_cagr(data) == pytest.approx(-99.0, rel=1e-2)

    def test_complete_loss(self):
        """Test CAGR calculation when bankroll goes to exactly zero"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2023-06-01', '2023-12-31']),
            'bt_starting_bankroll': [1000] * 3,
            'bt_ending_bankroll': [1000, 500, 0]  # Complete loss to zero
        })
        # CAGR = (0/1000)^(1/1) - 1 = -100%
        assert calculate_cagr(data) == -100.0

    def test_zero_initial_value(self):
        """Test CAGR calculation with zero initial value"""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2024-01-01']),
            'bt_starting_bankroll': [0, 0],
            'bt_ending_bankroll': [0, 1000]
        })
        assert calculate_cagr(data) == 0.0

    def test_negative_final_value(self):
        """Test CAGR is 0 when final bankroll is negative."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2023-01-01', '2024-01-01']),
            'bt_starting_bankroll': [1000, 1000],
            'bt_ending_bankroll': [1000, -10],
        })
        assert calculate_cagr(data) == 0.0


class TestCalculateYield:
    def test_unequal_stakes_differ_from_micro_roi(self):
        """Test yield is profit/staked, not the mean of per-bet ROI."""
        data = pd.DataFrame({
            'bt_stake': [100, 300],
            'bt_profit': [20, -30],
            'bt_bet_on': [0, 1],
        })
        assert calculate_yield(data) == pytest.approx(-2.5)
        assert calculate_avg_roi_per_bet_micro(data) == pytest.approx(5.0)

    def test_yield_ignores_no_bet_rows(self):
        """Test skipped rows are excluded even if they have nonzero profit."""
        data = pd.DataFrame({
            'bt_stake': [0, 100],
            'bt_profit': [999, 25],
            'bt_bet_on': [-1, 0],
        })
        assert calculate_yield(data) == pytest.approx(25.0)

    def test_yield_empty_is_zero(self):
        """Test yield is 0 when no bets were placed."""
        data = pd.DataFrame({
            'bt_stake': [0],
            'bt_profit': [0],
            'bt_bet_on': [-1],
        })
        assert calculate_yield(data) == 0.0


class TestExpectedValueMetrics:
    def test_expected_profit_and_yield(self):
        """Test EV is stake * (p * odds - 1) on the selected outcome."""
        data = pd.DataFrame({
            'bt_stake': [50],
            'bt_odds': [2.0],
            'bt_profit': [50],
            'bt_win': [True],
            'bt_bet_on': [0],
            'bt_model_prob_0': [0.6],
            'bt_model_prob_1': [0.4],
        })
        assert calculate_expected_profit(data) == pytest.approx(10.0)
        assert calculate_expected_yield(data) == pytest.approx(20.0)
        assert calculate_realized_vs_expected_profit(data) == pytest.approx(
            40.0
        )

    def test_expected_profit_nan_without_model_probs(self):
        """Test EV metrics are nan when model probabilities are missing."""
        data = pd.DataFrame({
            'bt_stake': [100],
            'bt_odds': [2.0],
            'bt_profit': [100],
            'bt_win': [True],
            'bt_bet_on': [0],
        })
        assert np.isnan(calculate_expected_profit(data))
        assert np.isnan(calculate_expected_yield(data))
        assert np.isnan(calculate_realized_vs_expected_profit(data))

    def test_expected_uses_selected_outcome_prob(self):
        """Test EV uses bt_model_prob of bt_bet_on, not the other outcome."""
        data = pd.DataFrame({
            'bt_stake': [100],
            'bt_odds': [3.0],
            'bt_profit': [-100],
            'bt_win': [False],
            'bt_bet_on': [1],
            'bt_model_prob_0': [0.9],
            'bt_model_prob_1': [0.4],
        })
        # stake * (0.4 * 3 - 1) = 20
        assert calculate_expected_profit(data) == pytest.approx(20.0)


class TestImpliedProbAndOverround:
    def test_average_implied_prob(self):
        """Test implied probability is mean of 1/odds on placed bets."""
        data = pd.DataFrame({
            'bt_stake': [100, 100],
            'bt_odds': [2.0, 4.0],
            'bt_bet_on': [0, 1],
        })
        assert calculate_average_implied_prob(data) == pytest.approx(0.375)

    def test_average_overround(self):
        """Test overround is mean of sum(1/odds_k) - 1 as percent."""
        data = pd.DataFrame({
            'bt_stake': [100],
            'bt_odds': [2.0],
            'bt_bet_on': [0],
            'bt_odd_0': [2.0],
            'bt_odd_1': [1.8],
        })
        expected = (0.5 + 1.0 / 1.8 - 1.0) * 100
        assert calculate_average_overround(data) == pytest.approx(5.5555555556)

    def test_overround_nan_without_odd_columns(self):
        """Test overround is nan when bt_odd_* columns are missing."""
        data = pd.DataFrame({
            'bt_stake': [100],
            'bt_odds': [2.0],
            'bt_bet_on': [0],
        })
        assert np.isnan(calculate_average_overround(data))


class TestScoringRules:
    def test_brier_score_win_and_loss(self):
        """Test Brier is mean of (p - y)^2 on the selected outcome."""
        data = pd.DataFrame({
            'bt_stake': [100, 100],
            'bt_odds': [2.0, 2.0],
            'bt_profit': [100, -100],
            'bt_win': [True, False],
            'bt_bet_on': [0, 0],
            'bt_model_prob_0': [0.6, 0.6],
        })
        assert calculate_brier_score(data) == pytest.approx(0.26)

    def test_log_loss_win_and_loss(self):
        """Test log loss uses -log(p) on a win and -log(1-p) on a loss."""
        data = pd.DataFrame({
            'bt_stake': [100, 100],
            'bt_odds': [2.0, 2.0],
            'bt_profit': [100, -100],
            'bt_win': [True, False],
            'bt_bet_on': [0, 0],
            'bt_model_prob_0': [0.6, 0.6],
        })
        expected = (-np.log(0.6) + -np.log(0.4)) / 2
        assert calculate_log_loss(data) == pytest.approx(expected)

    def test_ece_single_bin(self):
        """Test ECE is |accuracy - confidence| when all p share a bin."""
        data = pd.DataFrame({
            'bt_stake': [100, 100],
            'bt_odds': [2.0, 2.0],
            'bt_profit': [100, -100],
            'bt_win': [True, False],
            'bt_bet_on': [0, 0],
            'bt_model_prob_0': [0.6, 0.6],
        })
        assert calculate_ece(data, n_bins=10) == pytest.approx(0.1)

    def test_scoring_nan_without_model_probs(self):
        """Test Brier, log loss, and ECE are nan without model probs."""
        data = pd.DataFrame({
            'bt_stake': [100],
            'bt_odds': [2.0],
            'bt_profit': [100],
            'bt_win': [True],
            'bt_bet_on': [0],
        })
        assert np.isnan(calculate_brier_score(data))
        assert np.isnan(calculate_log_loss(data))
        assert np.isnan(calculate_ece(data))


class TestReliabilityPoints:
    """Unit tests for equal-width calibration bins."""

    def test_equal_width_bins_merge_distinct_p_in_same_interval(self):
        """0.21 and 0.29 share (0.2, 0.3] when n_bins=10."""
        data = pd.DataFrame({
            'bt_stake': [100, 100, 100, 100],
            'bt_odds': [2.0, 2.0, 2.0, 2.0],
            'bt_profit': [-100, 100, -100, 100],
            'bt_win': [False, True, False, True],
            'bt_bet_on': [0, 0, 0, 0],
            'bt_model_prob_0': [0.21, 0.29, 0.21, 0.29],
        })
        points = _reliability_points(data, n_bins=10)
        assert len(points) == 1
        assert points["mean_p"].iloc[0] == pytest.approx(0.25)
        assert points["win_rate"].iloc[0] == pytest.approx(0.5)
        assert int(points["count"].iloc[0]) == 4

    def test_uses_selected_outcome_probability(self):
        """Decoy p on the other outcome is ignored."""
        data = pd.DataFrame({
            'bt_stake': [100, 100],
            'bt_odds': [2.0, 2.0],
            'bt_profit': [-100, -100],
            'bt_win': [False, False],
            'bt_bet_on': [1, 1],
            'bt_model_prob_0': [0.9, 0.9],
            'bt_model_prob_1': [0.2, 0.2],
        })
        points = _reliability_points(data, n_bins=10)
        assert len(points) == 1
        assert points["mean_p"].iloc[0] == pytest.approx(0.2)
        assert points["win_rate"].iloc[0] == pytest.approx(0.0)

    def test_empty_without_model_probs(self):
        """No probability columns yields an empty reliability table."""
        data = pd.DataFrame({
            'bt_stake': [100],
            'bt_odds': [2.0],
            'bt_profit': [100],
            'bt_win': [True],
            'bt_bet_on': [0],
        })
        points = _reliability_points(data)
        assert points.empty

