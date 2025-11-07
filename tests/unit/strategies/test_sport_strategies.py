"""
Unit tests for sport_strategies module.

Tests betting strategy implementations including FixedStake and KellyCriterion.
"""
import pytest
from backtestbuddy.strategies.sport_strategies import (
    BaseStrategy,
    FixedStake,
    KellyCriterion,
    get_default_strategy
)


class TestFixedStake:
    """Unit tests for FixedStake strategy."""
    
    def test_initialization_with_absolute_stake(self):
        """Test initialization with absolute stake value."""
        strategy = FixedStake(stake=100)
        assert strategy.stake == 100
        assert strategy.initial_bankroll is None
    
    def test_initialization_with_percentage_stake(self):
        """Test initialization with percentage stake value."""
        strategy = FixedStake(stake=0.1)
        assert strategy.stake == 0.1
        assert strategy.initial_bankroll is None
    
    def test_calculate_stake_absolute_value(self):
        """Test stake calculation with absolute value."""
        strategy = FixedStake(stake=100)
        odds = [2.0, 1.8]
        bankroll = 1000
        
        stake = strategy.calculate_stake(odds, bankroll)
        assert stake == 100
    
    def test_calculate_stake_percentage_value(self):
        """Test stake calculation with percentage value."""
        strategy = FixedStake(stake=0.1)  # 10%
        odds = [2.0, 1.8]
        bankroll = 1000
        
        stake = strategy.calculate_stake(odds, bankroll)
        assert stake == 100  # 10% of 1000
    
    def test_calculate_stake_respects_bankroll_limit(self):
        """Test that stake never exceeds available bankroll."""
        strategy = FixedStake(stake=500)
        odds = [2.0, 1.8]
        bankroll = 200  # Less than stake
        
        stake = strategy.calculate_stake(odds, bankroll)
        assert stake == 200  # Should be capped at bankroll
    
    def test_select_bet_with_prediction(self):
        """Test bet selection when prediction is provided."""
        strategy = FixedStake(stake=100)
        odds = [2.0, 1.8, 2.5]
        prediction = 1
        
        bet_on = strategy.select_bet(odds, prediction=prediction)
        assert bet_on == 1
    
    def test_select_bet_with_model_probs(self):
        """Test bet selection when model probabilities are provided."""
        strategy = FixedStake(stake=100)
        odds = [2.0, 1.8, 2.5]
        model_probs = [0.3, 0.6, 0.1]  # Highest prob is index 1
        
        bet_on = strategy.select_bet(odds, model_probs=model_probs)
        assert bet_on == 1
    
    def test_select_bet_defaults_to_lowest_odds(self):
        """Test that bet selection defaults to lowest odds when no prediction/probs."""
        strategy = FixedStake(stake=100)
        odds = [2.0, 1.5, 2.5]  # Lowest is index 1
        
        bet_on = strategy.select_bet(odds)
        assert bet_on == 1
    
    def test_get_bet_details(self):
        """Test that get_bet_details returns correct tuple."""
        strategy = FixedStake(stake=100)
        odds = [2.0, 1.8]
        bankroll = 1000
        prediction = 0
        
        stake, bet_on, additional_info = strategy.get_bet_details(odds, bankroll, prediction=prediction)
        assert stake == 100
        assert bet_on == 0
        assert isinstance(additional_info, dict)
    
    def test_str_representation_absolute(self):
        """Test string representation for absolute stake."""
        strategy = FixedStake(stake=100)
        str_repr = str(strategy)
        assert "Fixed Stake Strategy" in str_repr
        assert "$100.00" in str_repr
    
    def test_str_representation_percentage(self):
        """Test string representation for percentage stake."""
        strategy = FixedStake(stake=0.1)
        str_repr = str(strategy)
        assert "Fixed Stake Strategy" in str_repr
        assert "10.00%" in str_repr


class TestKellyCriterion:
    """Unit tests for KellyCriterion strategy."""
    
    def test_initialization_default_parameters(self):
        """Test initialization with default parameters."""
        strategy = KellyCriterion()
        assert strategy.downscaling == 0.5
        assert strategy.max_bet == 0.1
        assert strategy.min_kelly == 0
        assert strategy.min_prob == 0
    
    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        strategy = KellyCriterion(
            downscaling=0.75,
            max_bet=0.2,
            min_kelly=0.01,
            min_prob=0.55
        )
        assert strategy.downscaling == 0.75
        assert strategy.max_bet == 0.2
        assert strategy.min_kelly == 0.01
        assert strategy.min_prob == 0.55
    
    def test_calculate_kelly_fraction_positive_edge(self):
        """Test Kelly fraction calculation with positive edge."""
        strategy = KellyCriterion()
        odds = 2.0
        prob = 0.6  # Expected value: 0.6 * 2.0 - 1 = 0.2 (positive)
        
        kelly_fraction = strategy.calculate_kelly_fraction(odds, prob)
        assert kelly_fraction > 0
        assert kelly_fraction == pytest.approx(0.2)  # (0.6 * 1.0 - 0.4) / 1.0
    
    def test_calculate_kelly_fraction_negative_edge(self):
        """Test Kelly fraction calculation with negative edge returns zero."""
        strategy = KellyCriterion()
        odds = 2.0
        prob = 0.4  # Expected value: 0.4 * 2.0 - 1 = -0.2 (negative)
        
        kelly_fraction = strategy.calculate_kelly_fraction(odds, prob)
        assert kelly_fraction == 0  # Negative Kelly should return 0
    
    def test_calculate_kelly_fraction_no_edge(self):
        """Test Kelly fraction calculation with no edge."""
        strategy = KellyCriterion()
        odds = 2.0
        prob = 0.5  # Expected value: 0.5 * 2.0 - 1 = 0 (no edge)
        
        kelly_fraction = strategy.calculate_kelly_fraction(odds, prob)
        assert kelly_fraction == 0
    
    def test_calculate_stake_returns_zero_without_model_probs(self):
        """Test that stake is zero when model probabilities are not provided."""
        strategy = KellyCriterion()
        odds = [2.0, 1.8]
        bankroll = 1000
        
        stake = strategy.calculate_stake(odds, bankroll, model_probs=None)
        assert stake == 0
    
    def test_calculate_stake_with_positive_kelly(self):
        """Test stake calculation with positive Kelly fraction."""
        strategy = KellyCriterion(downscaling=1.0)
        odds = [2.0, 1.8]
        model_probs = [0.6, 0.4]  # Positive edge on first outcome
        bankroll = 1000
        
        stake = strategy.calculate_stake(odds, bankroll, model_probs=model_probs)
        assert stake > 0
        assert stake <= bankroll * strategy.max_bet  # Should respect max_bet
    
    def test_calculate_stake_respects_max_bet(self):
        """Test that stake respects max_bet parameter."""
        strategy = KellyCriterion(downscaling=1.0, max_bet=0.1)
        odds = [3.0, 1.5]
        model_probs = [0.8, 0.2]  # Very high edge
        bankroll = 1000
        
        stake = strategy.calculate_stake(odds, bankroll, model_probs=model_probs)
        assert stake <= bankroll * 0.1  # Should not exceed 10% of bankroll
    
    def test_calculate_stake_applies_downscaling(self):
        """Test that downscaling factor is applied correctly."""
        strategy_full = KellyCriterion(downscaling=1.0, max_bet=1.0)
        strategy_half = KellyCriterion(downscaling=0.5, max_bet=1.0)
        odds = [2.0, 1.8]
        model_probs = [0.6, 0.4]
        bankroll = 1000
        
        stake_full = strategy_full.calculate_stake(odds, bankroll, model_probs=model_probs)
        stake_half = strategy_half.calculate_stake(odds, bankroll, model_probs=model_probs)
        
        assert stake_half == pytest.approx(stake_full * 0.5)
    
    def test_calculate_stake_respects_min_kelly(self):
        """Test that stake is zero when Kelly fraction below min_kelly."""
        strategy = KellyCriterion(min_kelly=0.05)
        odds = [2.0, 1.8]
        model_probs = [0.52, 0.48]  # Small positive edge
        bankroll = 1000
        
        # Calculate expected Kelly fraction
        kelly = strategy.calculate_kelly_fraction(odds[0], model_probs[0])
        
        if kelly < 0.05:
            stake = strategy.calculate_stake(odds, bankroll, model_probs=model_probs)
            assert stake == 0
    
    def test_calculate_stake_respects_min_prob(self):
        """Test that stake is zero when probability below min_prob."""
        strategy = KellyCriterion(min_prob=0.6)
        odds = [2.0, 1.8]
        model_probs = [0.55, 0.45]  # Below min_prob threshold
        bankroll = 1000
        
        stake = strategy.calculate_stake(odds, bankroll, model_probs=model_probs)
        assert stake == 0
    
    def test_select_bet_returns_highest_kelly(self):
        """Test that select_bet returns outcome with highest Kelly fraction."""
        strategy = KellyCriterion()
        odds = [2.0, 2.5, 1.8]
        model_probs = [0.55, 0.65, 0.45]  # Middle has highest Kelly
        
        bet_on = strategy.select_bet(odds, model_probs=model_probs)
        
        # Calculate Kelly fractions to verify
        kelly_fractions = [strategy.calculate_kelly_fraction(o, p) for o, p in zip(odds, model_probs)]
        expected_bet = kelly_fractions.index(max(kelly_fractions))
        
        assert bet_on == expected_bet
    
    def test_select_bet_returns_minus_one_without_probs(self):
        """Test that select_bet returns -1 when no model probabilities provided."""
        strategy = KellyCriterion()
        odds = [2.0, 1.8]
        
        bet_on = strategy.select_bet(odds, model_probs=None)
        assert bet_on == -1
    
    def test_select_bet_returns_minus_one_below_min_kelly(self):
        """Test that select_bet returns -1 when all Kelly fractions below min_kelly."""
        strategy = KellyCriterion(min_kelly=0.1)
        odds = [2.0, 1.8]
        model_probs = [0.51, 0.49]  # Very small edge
        
        bet_on = strategy.select_bet(odds, model_probs=model_probs)
        # This might return -1 depending on the exact Kelly calculation
        # Verify that the logic is consistent
        kelly_fractions = [strategy.calculate_kelly_fraction(o, p) for o, p in zip(odds, model_probs)]
        if max(kelly_fractions) <= 0.1:
            assert bet_on == -1
    
    def test_get_bet_details_includes_kelly_fractions(self):
        """Test that get_bet_details returns Kelly fractions in additional_info."""
        strategy = KellyCriterion()
        odds = [2.0, 1.8]
        model_probs = [0.6, 0.4]
        bankroll = 1000
        
        stake, bet_on, additional_info = strategy.get_bet_details(
            odds, bankroll, model_probs=model_probs
        )
        
        assert 'kelly_fraction_0' in additional_info
        assert 'kelly_fraction_1' in additional_info
        assert isinstance(additional_info['kelly_fraction_0'], (int, float))
    
    def test_str_representation(self):
        """Test string representation includes all parameters."""
        strategy = KellyCriterion(
            downscaling=0.5,
            max_bet=0.1,
            min_kelly=0.01,
            min_prob=0.55
        )
        str_repr = str(strategy)
        assert "Kelly Criterion Strategy" in str_repr
        assert "downscaling=0.5" in str_repr
        assert "max_bet=0.1" in str_repr
        assert "min_kelly=0.01" in str_repr
        assert "min_prob=0.55" in str_repr


class TestGetDefaultStrategy:
    """Unit tests for get_default_strategy function."""
    
    def test_returns_fixed_stake(self):
        """Test that default strategy is FixedStake."""
        strategy = get_default_strategy()
        assert isinstance(strategy, FixedStake)
    
    def test_default_stake_is_one_percent(self):
        """Test that default stake is 1% of bankroll."""
        strategy = get_default_strategy()
        assert strategy.stake == 0.01

