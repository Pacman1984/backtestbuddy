"""
Unit tests for bookmaker margin removal.

Checks hand-computed values, that every method returns probabilities that
sum to 1, that power and Shin shift probability toward favourites, and
the handling of invalid odds.
"""
import numpy as np
import pytest

from backtestbuddy.odds.margin import MARGIN_METHODS, remove_margin


class TestKnownValues:
    """Hand-computed fair probabilities."""

    def test_multiplicative(self):
        # implied 1/1.5 = 0.6667, 1/2.5 = 0.4, sum 1.0667
        np.testing.assert_allclose(
            remove_margin([1.5, 2.5]), [0.625, 0.375]
        )

    def test_additive(self):
        # overround 0.0667 split equally: 0.6667 - 0.0333, 0.4 - 0.0333
        np.testing.assert_allclose(
            remove_margin([1.5, 2.5], method="additive"), [0.63333333, 0.36666667]
        )

    def test_power_solves_exponent(self):
        odds = np.array([1.2, 4.5])
        fair = remove_margin(odds, method="power")
        # fair_i = (1/odds_i) ** k for one k shared by both outcomes
        k = np.log(fair) / np.log(1 / odds)
        assert k[0] == pytest.approx(k[1])

    def test_shin_equals_additive_for_two_outcomes(self):
        """Shin and additive coincide on two-way markets."""
        odds = [[1.2, 4.5], [1.05, 12.0], [1.9, 1.9]]
        np.testing.assert_allclose(
            remove_margin(odds, method="shin"),
            remove_margin(odds, method="additive"),
            atol=1e-9,
        )


class TestProperties:
    """Behavior shared by all methods."""

    @pytest.mark.parametrize("method", MARGIN_METHODS)
    def test_rows_sum_to_one(self, method):
        odds = [[1.9, 1.9], [1.2, 4.5], [2.5, 3.2, 2.9], [1.01, 30.0]]
        for row in odds:
            assert remove_margin(row, method=method).sum() == pytest.approx(1.0)

    @pytest.mark.parametrize("method", MARGIN_METHODS)
    def test_symmetric_market_is_even(self, method):
        np.testing.assert_allclose(remove_margin([1.9, 1.9], method=method), [0.5, 0.5])

    @pytest.mark.parametrize("method", ["power", "shin"])
    def test_longshot_methods_favour_the_favourite(self, method):
        multiplicative = remove_margin([1.2, 4.5])
        other = remove_margin([1.2, 4.5], method=method)
        assert other[0] > multiplicative[0]
        assert other[1] < multiplicative[1]

    def test_three_way_shin_differs_from_additive(self):
        odds = [1.5, 4.0, 7.0]
        assert not np.allclose(
            remove_margin(odds, method="shin"), remove_margin(odds, method="additive")
        )

    def test_2d_input_keeps_shape(self):
        result = remove_margin(np.array([[1.9, 1.9], [1.2, 4.5]]))
        assert result.shape == (2, 2)


class TestInvalidInput:
    """Invalid markets yield NaN rows or errors."""

    @pytest.mark.parametrize("row", [[1.0, 2.0], [np.nan, 2.0], [0.5, 3.0]])
    def test_invalid_odds_row_is_nan(self, row):
        result = remove_margin([row, [1.9, 1.9]])
        assert np.isnan(result[0]).all()
        np.testing.assert_allclose(result[1], [0.5, 0.5])

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="method"):
            remove_margin([1.9, 1.9], method="magic")

    def test_single_outcome_raises(self):
        with pytest.raises(ValueError, match="two outcomes"):
            remove_margin([1.5])

    def test_book_without_margin_uses_multiplicative(self):
        # implied sum 0.7125 < 1: no insider share to solve for
        np.testing.assert_allclose(
            remove_margin([2.5, 3.2], method="shin"), remove_margin([2.5, 3.2])
        )
