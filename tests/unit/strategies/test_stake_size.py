"""
Unit tests for explicit stake sizes and the legacy stake= argument.

amount= and fraction= are unambiguous; the old stake= still works with
its below-1-is-a-fraction rule but warns. Mixing arguments is an error.
"""
import pytest

from backtestbuddy.strategies.sport_strategies import (
    FixedStake,
    OddsFilter,
    StakeSize,
    ValueBet,
)


class TestStakeSize:
    """Resolving amounts and fractions against a bankroll."""

    def test_fraction(self):
        assert StakeSize(fraction=0.02).resolve(1000) == pytest.approx(20)

    def test_amount_capped_at_bankroll(self):
        assert StakeSize(amount=50).resolve(30) == 30

    def test_amount_below_one_is_an_amount(self):
        assert StakeSize(amount=0.5).resolve(1000) == 0.5

    @pytest.mark.parametrize("kwargs", [{}, {"amount": 1, "fraction": 0.1}])
    def test_exactly_one_required(self, kwargs):
        with pytest.raises(ValueError, match="exactly one"):
            StakeSize(**kwargs)

    @pytest.mark.parametrize("kwargs", [{"amount": -1}, {"fraction": 1.5}, {"fraction": -0.1}])
    def test_out_of_range(self, kwargs):
        with pytest.raises(ValueError):
            StakeSize(**kwargs)


class TestFixedStakeSizes:
    """FixedStake with explicit and legacy sizes."""

    def test_amount(self):
        assert FixedStake(amount=10).calculate_stake([2.0, 2.0], 1000) == 10

    def test_fraction(self):
        assert FixedStake(fraction=0.02).calculate_stake([2.0, 2.0], 1000) == pytest.approx(20)

    def test_fraction_of_one_is_whole_bankroll(self):
        assert FixedStake(fraction=1.0).calculate_stake([2.0, 2.0], 800) == 800

    @pytest.mark.parametrize("legacy, expected", [(0.99, 990.0), (1.0, 1.0), (10, 10.0)])
    def test_legacy_rule_kept_and_warns(self, legacy, expected):
        with pytest.warns(DeprecationWarning, match="ambiguous"):
            strategy = FixedStake(legacy)
        assert strategy.calculate_stake([2.0, 2.0], 1000) == pytest.approx(expected)
        assert strategy.stake == legacy

    def test_explicit_form_does_not_warn(self, recwarn):
        FixedStake(amount=10)
        FixedStake(fraction=0.1)
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_mixing_raises(self):
        with pytest.raises(ValueError, match="only one"):
            FixedStake(10, amount=10)

    def test_missing_size_raises(self):
        with pytest.raises(ValueError, match="needs amount"):
            FixedStake()

    def test_str(self):
        assert "2.00% of current bankroll" in str(FixedStake(fraction=0.02))
        assert "$10.00" in str(FixedStake(amount=10))


class TestValueBetAndOddsFilterSizes:
    """ValueBet and OddsFilter accept the same sizes."""

    def test_value_bet_default_is_one_unit(self, recwarn):
        strategy = ValueBet(min_ev=0.0)
        assert strategy.calculate_stake([2.5, 1.6], 1000, model_probs=[0.5, 0.5]) == 1
        assert not [w for w in recwarn if issubclass(w.category, DeprecationWarning)]

    def test_value_bet_fraction(self):
        strategy = ValueBet(min_ev=0.0, fraction=0.05)
        assert strategy.calculate_stake([2.5, 1.6], 1000, model_probs=[0.5, 0.5]) == pytest.approx(50)

    def test_value_bet_legacy_warns(self):
        with pytest.warns(DeprecationWarning):
            ValueBet(0.0, 0.05)

    def test_odds_filter_default_inner_size(self):
        strategy = OddsFilter(min_odds=1.5, fraction=0.1)
        assert strategy.inner.size.fraction == 0.1

    def test_odds_filter_legacy_warns(self):
        with pytest.warns(DeprecationWarning):
            OddsFilter(min_odds=1.5, stake=10)
