"""
Unit tests for risk-adjusted ROI calculation with units verification.

This test module verifies that the risk-adjusted annual ROI calculation
handles units correctly, converting percentages to decimals as needed
and returning a unitless ratio.
"""
import pytest
import pandas as pd
from backtestbuddy.metrics.sport_metrics import (
    calculate_risk_adjusted_annual_roi,
    calculate_avg_roi_per_year_macro,
    calculate_max_drawdown
)


class TestRiskAdjustedUnits:
    """Test to verify the units mismatch issue is resolved."""
    
    def test_units_correct_unitless_ratio(self):
        """
        Verify that Risk-Adjusted Annual ROI returns a correct unitless ratio.
        
        Expected behavior (FIXED):
        - avg_yearly_roi converted to decimal (10.0% -> 0.10)
        - max_drawdown is a decimal (-0.15 for 15% drawdown)
        - Division gives unitless ratio (0.10 / 0.15 = 0.667)
        """
        # Create test data with known values
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01',
                '2021-06-15',  # Mid-year drawdown point
                '2021-12-31'
            ]),
            'bt_starting_bankroll': [1000, 1000, 1000],
            'bt_ending_bankroll': [1000, 850, 1100],  # 10% gain, but 15% drawdown
            'bt_bet_on': [1, 1, 1]
        })
        
        # Check individual components
        avg_yearly_roi_pct = calculate_avg_roi_per_year_macro(data)
        max_drawdown = calculate_max_drawdown(data)
        
        print(f"\navg_yearly_roi (as %): {avg_yearly_roi_pct}")
        print(f"avg_yearly_roi (as decimal): {avg_yearly_roi_pct / 100}")
        print(f"max_drawdown: {max_drawdown}")
        print(f"abs(max_drawdown): {abs(max_drawdown)}")
        
        # avg_yearly_roi should be percentage (10.0)
        # Total ROI = (1100 - 1000) / 1000 = 0.1 = 10%
        # Duration = 1 year
        # Annual ROI = 10% / 1 year = 10%
        assert avg_yearly_roi_pct == pytest.approx(10.0, abs=0.1)
        
        # max_drawdown should be decimal (-0.15)
        # Peak at index 0 = 1000
        # Valley at index 1 = 850
        # Drawdown = (850 - 1000) / 1000 = -0.15
        assert max_drawdown == pytest.approx(-0.15, abs=0.01)
        
        # Fixed implementation returns unitless ratio
        result = calculate_risk_adjusted_annual_roi(data)
        print(f"Result (unitless ratio): {result}")
        
        # Expected unitless ratio:
        # Convert avg_yearly_roi to decimal: 10.0 / 100 = 0.10
        # Divide by abs(max_drawdown): 0.10 / 0.15 = 0.667
        expected_result = (avg_yearly_roi_pct / 100) / abs(max_drawdown)
        print(f"Expected unitless ratio: {expected_result}")
        assert result == pytest.approx(expected_result, abs=0.01)
        assert result == pytest.approx(0.667, abs=0.01)
    
    def test_edge_case_zero_drawdown(self):
        """Test edge case when MDD is zero - should return inf."""
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime(['2021-01-01', '2021-12-31']),
            'bt_starting_bankroll': [1000, 1000],
            'bt_ending_bankroll': [1000, 1200],  # Only gains, no drawdown
            'bt_bet_on': [1, 1]
        })
        
        result = calculate_risk_adjusted_annual_roi(data)
        
        # When MDD == 0, should return float('inf') per documented convention
        print(f"\nZero drawdown case - result: {result}")
        assert result == float('inf')
    
    def test_sign_consistency(self):
        """Test that signs are handled correctly."""
        # Negative ROI with positive drawdown magnitude
        data = pd.DataFrame({
            'bt_date_column': pd.to_datetime([
                '2021-01-01',
                '2021-12-31'
            ]),
            'bt_starting_bankroll': [1000, 1000],
            'bt_ending_bankroll': [1000, 800],  # -20% loss
            'bt_bet_on': [1, 1]
        })
        
        avg_yearly_roi_pct = calculate_avg_roi_per_year_macro(data)
        max_drawdown = calculate_max_drawdown(data)
        result = calculate_risk_adjusted_annual_roi(data)
        
        print(f"\nNegative ROI case:")
        print(f"avg_yearly_roi (as %): {avg_yearly_roi_pct}")
        print(f"avg_yearly_roi (as decimal): {avg_yearly_roi_pct / 100}")
        print(f"max_drawdown: {max_drawdown}")
        print(f"result (unitless ratio): {result}")
        
        # avg_yearly_roi should be negative percentage
        assert avg_yearly_roi_pct < 0
        # max_drawdown should be negative
        assert max_drawdown < 0
        # result should be negative (negative decimal / positive magnitude = negative)
        assert result < 0
        
        # Verify correct calculation
        # -20% as decimal = -0.20
        # abs(max_drawdown) = 0.20
        # result = -0.20 / 0.20 = -1.0
        expected = (avg_yearly_roi_pct / 100) / abs(max_drawdown)
        assert result == pytest.approx(expected, abs=0.01)
        assert result == pytest.approx(-1.0, abs=0.01)

