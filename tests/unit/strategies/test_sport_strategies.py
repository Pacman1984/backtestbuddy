"""
Unit tests for sport_strategies module.

Tests betting strategy implementations including FixedStake, KellyCriterion,
ValueBet, UnitStake, and OddsFilter.
"""
import pytest
from backtestbuddy.strategies.sport_strategies import (
    BaseStrategy,
    FixedStake,
    KellyCriterion,
    OddsFilter,
    UnitStake,
    ValueBet,
    get_default_strategy
)


class TestFixedStake:
    """Unit tests for FixedStake strategy."""
    
    def test_initialization_with_absolute_stake(self):
        """Test initialization with absolute stake value."""
        strategy = FixedStake(stake=100)
        assert strategy.stake == 100
    
    def test_initialization_with_percentage_stake(self):
        """Test initialization with percentage stake value."""
        strategy = FixedStake(stake=0.1)
        assert strategy.stake == 0.1
    
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

    def test_get_bet_details_uses_select_bet_model_probs(self):
        """Test get_bet_details bets on the highest model probability."""
        strategy = FixedStake(stake=100)
        stake, bet_on, _ = strategy.get_bet_details(
            odds=[2.0, 1.8, 2.5],
            current_bankroll=1000,
            model_probs=[0.3, 0.6, 0.1],
        )
        assert stake == 100
        assert bet_on == 1
    
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

    def test_calculate_kelly_fraction_odds_at_or_below_one(self):
        """Test Kelly fraction is 0 when decimal odds are not greater than 1."""
        strategy = KellyCriterion()
        assert strategy.calculate_kelly_fraction(1.0, 0.9) == 0.0
        assert strategy.calculate_kelly_fraction(0.5, 0.9) == 0.0
    
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


class TestValueBet:
    """Unit tests for ValueBet strategy."""

    def test_skips_when_ev_equals_min_ev(self):
        """EV exactly equal to min_ev is not a bet."""
        strategy = ValueBet(min_ev=0.0, stake=25)
        stake, bet_on, _ = strategy.get_bet_details(
            odds=[2.0, 3.0],
            bankroll=1000,
            model_probs=[0.5, 0.2],
        )
        assert stake == 0.0
        assert bet_on == -1

    def test_skips_when_ev_at_or_below_min_ev(self):
        """Skip when the best EV is not strictly above min_ev."""
        strategy = ValueBet(min_ev=0.10, stake=25)
        stake, bet_on, _ = strategy.get_bet_details(
            odds=[2.0, 2.0],
            bankroll=1000,
            model_probs=[0.54, 0.45],
        )
        assert stake == 0.0
        assert bet_on == -1

    def test_places_bet_when_ev_exceeds_min_ev(self):
        """Place a bet when the best EV is above min_ev."""
        strategy = ValueBet(min_ev=0.15, stake=25)
        stake, bet_on, _ = strategy.get_bet_details(
            odds=[2.0, 1.8],
            bankroll=1000,
            model_probs=[0.6, 0.4],
        )
        assert stake == 25
        assert bet_on == 0

    def test_selects_higher_ev_not_higher_probability(self):
        """Pick the higher EV even when that outcome has lower probability."""
        strategy = ValueBet(min_ev=0.0, stake=25)
        bet_on = strategy.select_bet(
            odds=[1.5, 3.5],
            model_probs=[0.80, 0.40],
        )
        assert bet_on == 1

    def test_ignores_prediction_when_selecting_by_ev(self):
        """A class prediction does not override the highest-EV outcome."""
        strategy = ValueBet(min_ev=0.0, stake=25)
        bet_on = strategy.select_bet(
            odds=[1.5, 3.5],
            model_probs=[0.80, 0.40],
            prediction=0,
        )
        assert bet_on == 1

    def test_skips_without_model_probs(self):
        """Stake 0 and skip when model probabilities are missing."""
        strategy = ValueBet(stake=25)
        assert strategy.calculate_stake([2.0, 1.8], 1000) == 0.0
        assert strategy.select_bet([2.0, 1.8]) == -1

    def test_skips_when_prob_length_mismatches_odds(self):
        """Skip when model_probs is not aligned with odds."""
        strategy = ValueBet(stake=25)
        assert strategy.select_bet([2.0, 1.8], model_probs=[0.6]) == -1

    def test_fraction_stake_uses_current_bankroll(self):
        """stake < 1 is a fraction of current bankroll."""
        strategy = ValueBet(min_ev=0.0, stake=0.2)
        stake = strategy.calculate_stake(
            [2.0, 1.8], 500, model_probs=[0.6, 0.4]
        )
        assert stake == 100

    def test_caps_absolute_stake_at_bankroll(self):
        """Absolute stake cannot exceed current bankroll."""
        strategy = ValueBet(min_ev=0.0, stake=200)
        stake = strategy.calculate_stake(
            [2.0, 1.8], 50, model_probs=[0.6, 0.4]
        )
        assert stake == 50

    def test_get_bet_details_includes_per_outcome_ev(self):
        """Extra info reports EV for every outcome."""
        strategy = ValueBet(stake=25)
        _, _, info = strategy.get_bet_details(
            [2.0, 1.8], 1000, model_probs=[0.6, 0.4]
        )
        assert info["ev_0"] == pytest.approx(0.2)
        assert info["ev_1"] == pytest.approx(-0.28)


class TestUnitStake:
    """Unit tests for UnitStake plans."""

    def test_rejects_unknown_plan(self):
        """Unknown plan names raise ValueError."""
        with pytest.raises(ValueError, match="plan"):
            UnitStake(unit=10, plan="martingale")

    def test_rejects_non_positive_unit(self):
        """unit must be positive."""
        with pytest.raises(ValueError, match="unit"):
            UnitStake(unit=0)

    def test_loss_win_impact_differ_at_odds_three(self):
        """At odds 3.0 the three plans size different stakes."""
        odds = [3.0, 1.5]
        bankroll = 1000
        unit = 90
        prediction = 0
        loss = UnitStake(unit=unit, plan="loss")
        win = UnitStake(unit=unit, plan="win")
        impact = UnitStake(unit=unit, plan="impact")
        s_loss, on_loss, _ = loss.get_bet_details(
            odds, bankroll, prediction=prediction
        )
        s_win, on_win, _ = win.get_bet_details(
            odds, bankroll, prediction=prediction
        )
        s_impact, on_impact, _ = impact.get_bet_details(
            odds, bankroll, prediction=prediction
        )
        assert on_loss == on_win == on_impact == 0
        assert s_loss == 90
        assert s_win == 45
        assert s_impact == 30
        assert len({s_loss, s_win, s_impact}) == 3

    def test_win_plan_skips_odds_at_one(self):
        """Unit-win cannot size a bet when decimal odds are 1."""
        strategy = UnitStake(unit=90, plan="win")
        stake, bet_on, _ = strategy.get_bet_details(
            [1.0, 2.0], 1000, prediction=0
        )
        assert stake == 0.0
        assert bet_on == -1

    def test_win_caps_at_bankroll(self):
        """Unit-win is capped when unit/(odds-1) exceeds bankroll."""
        strategy = UnitStake(unit=90, plan="win")
        stake = strategy.calculate_stake(
            [1.1, 3.0], 20, prediction=0
        )
        assert stake == 20

    def test_win_plan_skips_odds_below_one(self):
        """Unit-win skips when decimal odds are below 1."""
        strategy = UnitStake(unit=90, plan="win")
        stake, bet_on, _ = strategy.get_bet_details(
            [0.5, 2.0], 1000, prediction=0
        )
        assert stake == 0.0
        assert bet_on == -1

    def test_loss_caps_at_bankroll(self):
        """Unit-loss is capped at current bankroll."""
        strategy = UnitStake(unit=90, plan="loss")
        stake = strategy.calculate_stake(
            [3.0, 1.5], 20, prediction=0
        )
        assert stake == 20

    def test_selects_highest_probability_for_sizing(self):
        """Model probabilities override prediction for unit-win sizing."""
        strategy = UnitStake(unit=90, plan="win")
        stake, bet_on, _ = strategy.get_bet_details(
            [3.0, 1.5],
            1000,
            model_probs=[0.3, 0.7],
            prediction=0,
        )
        assert bet_on == 1
        assert stake == 180

    def test_bookie_path_sizes_the_favorite(self):
        """Without probs or prediction, unit-impact sizes the lowest odds."""
        strategy = UnitStake(unit=90, plan="impact")
        stake = strategy.calculate_stake([3.0, 1.5], 1000)
        assert stake == 60
        assert strategy.select_bet([3.0, 1.5]) == 1

    def test_get_bet_details_includes_plan_metadata(self):
        """Extra info reports the plan name and unit."""
        strategy = UnitStake(unit=90, plan="impact")
        _, _, info = strategy.get_bet_details(
            [3.0, 1.5], 1000, prediction=0
        )
        assert info["plan"] == "impact"
        assert info["unit"] == 90


class TestOddsFilter:
    """Unit tests for OddsFilter wrapping."""

    def test_skips_out_of_band_favorite_and_picks_in_band(self):
        """Without probs, bet the in-band favorite, not the overall favorite."""
        strategy = OddsFilter(min_odds=1.5, stake=25)
        stake, bet_on, _ = strategy.get_bet_details([1.2, 2.5], 1000)
        assert bet_on == 1
        assert stake == 25

    def test_skips_when_nothing_in_band(self):
        """Skip the market when every outcome is outside the band."""
        strategy = OddsFilter(min_odds=3.0, stake=25)
        stake, bet_on, _ = strategy.get_bet_details([1.2, 2.5], 1000)
        assert stake == 0.0
        assert bet_on == -1

    def test_max_odds_excludes_predicted_longshot(self):
        """A prediction on odds above max_odds is dropped."""
        odds = [1.5, 5.0]
        assert FixedStake(25).select_bet(odds, prediction=1) == 1
        filtered = OddsFilter(min_odds=1.01, max_odds=2.0, stake=25)
        assert filtered.select_bet(odds, prediction=1) == 0

    def test_drops_out_of_band_prediction(self):
        """An out-of-band prediction is ignored; the in-band favorite is used."""
        strategy = OddsFilter(min_odds=2.0, stake=25)
        bet_on = strategy.select_bet([1.2, 3.0], prediction=0)
        assert bet_on == 1

    def test_inclusive_bounds(self):
        """Outcomes exactly on min_odds or max_odds stay eligible."""
        strategy = OddsFilter(min_odds=1.5, max_odds=2.0, stake=25)
        assert strategy.select_bet([1.5, 3.0]) == 0
        assert strategy.select_bet([1.2, 2.0]) == 1

    def test_maps_prediction_into_the_filtered_list(self):
        """A predicted in-band longshot is kept, not replaced by the favorite."""
        strategy = OddsFilter(min_odds=2.0, stake=25)
        bet_on = strategy.select_bet([1.2, 3.0, 4.0], prediction=2)
        assert bet_on == 2

    def test_wraps_valuebet_and_skips_when_only_in_band_has_no_ev(self):
        """Filtering out the only +EV outcome skips the market."""
        inner = ValueBet(min_ev=0.0, stake=25)
        strategy = OddsFilter(min_odds=1.5, inner=inner)
        stake, bet_on, _ = strategy.get_bet_details(
            [1.3, 2.2], 1000, model_probs=[0.85, 0.40]
        )
        assert stake == 0.0
        assert bet_on == -1
        assert inner.select_bet([1.3, 2.2], [0.85, 0.40]) == 0

    def test_wraps_unit_win_and_sizes_on_in_band_odds(self):
        """Unit-win stake uses the in-band odds, not the excluded favorite."""
        inner = UnitStake(unit=90, plan="win")
        strategy = OddsFilter(min_odds=2.0, inner=inner)
        stake, bet_on, _ = strategy.get_bet_details([1.4, 3.0], 1000)
        assert bet_on == 1
        assert stake == 45

    def test_requires_probabilities_follows_inner(self):
        """OddsFilter inherits whether the inner strategy needs probs."""
        wrapped_fixed = OddsFilter(inner=FixedStake(25))
        wrapped_value = OddsFilter(inner=ValueBet())
        assert wrapped_fixed.requires_probabilities is False
        assert wrapped_value.requires_probabilities is True


