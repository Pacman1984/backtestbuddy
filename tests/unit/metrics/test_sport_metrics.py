"""
Unit tests for sport_metrics module.

Tests all metric calculation functions with deterministic data.
"""
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
    def test_calculate_sortino_ratio(self):
        """Test Sortino ratio with varying negative returns"""
        data = pd.DataFrame({
            'bt_date_column': pd.date_range(start='2023-01-01', periods=11),
            'bt_starting_bankroll': [1000] * 11,
            'bt_profit': [0, 100, -30, 150, -60, 150, -40, 150, -70, 150, 0],  # Varying negative returns
            'bt_stake': [100] * 10 + [0],
            'bt_bet_on': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, -1]
        })
        result = calculate_sortino_ratio(data)
        assert result > 0, f"Expected positive Sortino ratio, got {result}"
    
    def test_sortino_ratio_constant_negative_returns(self):
        """Test Sortino ratio when all negative returns are the same"""
        data = pd.DataFrame({
            'bt_date_column': pd.date_range(start='2023-01-01', periods=11),
            'bt_starting_bankroll': [1000] * 11,
            'bt_profit': [0, 100, -50, 150, -50, 150, -50, 150, -50, 150, 0],  # All negative returns same
            'bt_stake': [100] * 10 + [0],
            'bt_bet_on': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, -1]
        })
        result = calculate_sortino_ratio(data)
        # Should return a positive value - even with constant negative returns, 
        # downside deviation is calculable using the correct method
        assert result > 0, f"Expected positive Sortino ratio, got {result}"
    
    def test_sortino_ratio_no_negative_returns(self):
        """Test Sortino ratio when there are no negative returns (zero downside deviation)"""
        data = pd.DataFrame({
            'bt_date_column': pd.date_range(start='2023-01-01', periods=5),
            'bt_starting_bankroll': [1000] * 5,
            'bt_profit': [100, 200, 50, 150, 100],  # All positive
            'bt_stake': [100] * 5,
            'bt_bet_on': [1] * 5
        })
        result = calculate_sortino_ratio(data)
        # Should return inf when there are no negative returns and positive mean return
        assert result == float('inf'), f"Expected inf for no negative returns, got {result}"

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

    def test_calculate_drawdowns_equal_peaks_uses_last_peak(self):
        # Equal peaks at indices 1 and 2; trough at index 4
        # Expect start at last peak (index 2), duration = 4 - 2 + 1 = 3
        data = pd.DataFrame({'bt_ending_bankroll': [100, 120, 120, 110, 90]})
        max_dd, max_dur = calculate_drawdowns(data)
        assert max_dd == pytest.approx(0.25, rel=1e-6)
        assert max_dur == 3

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

