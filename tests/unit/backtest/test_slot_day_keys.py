"""
Tests for slot_column and day_column of PredictionBacktest.

slot_column replaces the date value as the settlement slot of
settle_simultaneous; day_column replaces the calendar day of every
BetHistory day total and query (daily limits, pause after losses).
Both default to None, which must keep the 0.3.1 results exactly.
"""
import numpy as np
import pandas as pd
import pytest

import backtestbuddy as btb
from backtestbuddy.strategies.history import BetHistory


def _random_data():
    """300 matches over 40 days at three kick-off times; the model has an edge."""
    rng = np.random.default_rng(7)
    n = 300
    days = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 40, n), unit="D")
    hours = pd.to_timedelta(rng.choice([12, 15, 18], n), unit="h")
    p = rng.uniform(0.25, 0.75, n)
    market = np.clip(p + rng.normal(0, 0.08, n), 0.1, 0.9)
    return pd.DataFrame({
        "date": days + hours, "a": np.round(1.0 / (market * 1.03), 2), "b": np.round(1.0 / ((1 - market) * 1.03), 2),
        "y": (rng.uniform(size=n) > p).astype(int),
        "pa": p, "pb": 1 - p, "pred": (p < 0.5).astype(int),
    })


def _kelly():
    return btb.Strategy(btb.HighestKelly(), btb.SimultaneousKelly(fraction=0.5, max_bet=0.2, max_total=0.3))


STRATEGIES = {
    "fixed": lambda: btb.FixedStake(fraction=0.05),
    "kelly": lambda: btb.KellyCriterion(),
    "simkelly": _kelly,
    "daily": lambda: btb.RiskLimits(btb.FixedStake(fraction=0.05), max_daily_fraction=0.12),
    "pause": lambda: btb.RiskLimits(btb.FixedStake(fraction=0.05), pause_after_losses=2),
    "stop": lambda: btb.RiskLimits(btb.KellyCriterion(), stop_at_drawdown=0.2),
    "all": lambda: btb.RiskLimits(_kelly(), max_daily_fraction=0.15, pause_after_losses=2,
                                  stop_at_drawdown=0.35, min_stake=5),
}

# (final bankroll, total stake, bets placed, sum of stake * row number), computed with
# backtestbuddy 0.3.1 (commit 0cb9dd9) before slot_column and day_column existed
EXPECTED_031 = {
    ('fixed', False): (1034.300093, 12221.574708, 300, 1765013.67),
    ('fixed', True): (980.885502, 12118.989373, 300, 1733851.5118),
    ('kelly', False): (4695.921588, 47629.393672, 254, 9981029.581),
    ('kelly', True): (5237.417913, 50921.176715, 254, 10794293.7645),
    ('simkelly', False): (5732.895513, 61611.87884, 254, 13354110.6023),
    ('simkelly', True): (7186.260672, 70740.714309, 254, 15663147.575),
    ('daily', False): (1122.554989, 4067.258571, 120, 579082.9605),
    ('daily', True): (1105.96543, 4037.80808, 120, 572822.3695),
    ('pause', False): (983.174543, 8394.00359, 219, 1170369.0556),
    ('pause', True): (1116.970797, 11274.833481, 267, 1607477.7444),
    ('stop', False): (753.945225, 246.054775, 5, 788.7201),
    ('stop', True): (744.592618, 255.407382, 5, 820.8346),
    ('all', False): (1051.840396, 2417.63518, 49, 159621.9785),
    ('all', True): (1292.217252, 3192.123006, 57, 215884.7826),
}


def _backtest(data, strategy, settle, **kwargs):
    bt = btb.PredictionBacktest(data, ["a", "b"], "y", "date", "pred", strategy=strategy,
                                model_prob_columns=["pa", "pb"], settle_simultaneous=settle, **kwargs)
    bt.run()
    return bt


def _fingerprint(results):
    stakes = results["bt_stake"].to_numpy(float)
    return (float(results["bt_ending_bankroll"].iloc[-1]), float(stakes.sum()), int((stakes > 0).sum()),
            float((stakes * np.arange(1, len(stakes) + 1)).sum()))


class TestDefaultsUnchanged:
    @pytest.mark.parametrize("name,settle", list(EXPECTED_031))
    def test_matches_031(self, name, settle):
        results = _backtest(_random_data(), STRATEGIES[name](), settle).detailed_results
        final, total, n_bets, weighted = _fingerprint(results)
        expected = EXPECTED_031[(name, settle)]
        assert final == pytest.approx(expected[0], abs=1e-5)
        assert total == pytest.approx(expected[1], abs=1e-5)
        assert n_bets == expected[2]
        assert weighted == pytest.approx(expected[3], abs=1e-3)

    @pytest.mark.parametrize("name,settle", list(EXPECTED_031))
    def test_explicit_none_equals_omitted(self, name, settle):
        data = _random_data()
        omitted = _backtest(data, STRATEGIES[name](), settle)
        explicit = _backtest(data, STRATEGIES[name](), settle, slot_column=None, day_column=None)
        pd.testing.assert_frame_equal(omitted.detailed_results, explicit.detailed_results)
        pd.testing.assert_frame_equal(omitted.bookie_results, explicit.bookie_results)

    @pytest.mark.parametrize("name,settle", list(EXPECTED_031))
    def test_keys_equal_to_defaults_give_same_results(self, name, settle):
        """slot = date value and day = calendar day reproduce the default rules."""
        data = _random_data()
        data["slot"] = data["date"]
        data["day"] = data["date"].dt.normalize()
        keys = {"day_column": "day", **({"slot_column": "slot"} if settle else {})}
        default = _backtest(data, STRATEGIES[name](), settle)
        keyed = _backtest(data, STRATEGIES[name](), settle, **keys)
        pd.testing.assert_frame_equal(default.detailed_results, keyed.detailed_results)
        pd.testing.assert_frame_equal(default.bookie_results, keyed.bookie_results)


@pytest.fixture
def two_copies():
    """One day drawn twice (copies 1 and 2): 4 matches each at the same timestamp."""
    one = pd.DataFrame({
        "date": pd.Timestamp("2024-03-01 12:00"),
        "a": 2.0, "b": 2.0, "y": [1, 1, 0, 0], "pred": 0, "pa": 0.6, "pb": 0.4,
    })
    data = pd.concat([one.assign(copy=1), one.assign(copy=2)], ignore_index=True)
    data["day"] = data["copy"].map(lambda c: f"2024-03-01/{c}")
    return data


class TestDayColumn:
    @pytest.mark.parametrize("settle", [False, True])
    def test_shared_day_shares_the_daily_limit(self, two_copies, settle):
        two_copies["y"] = [0, 0, 1, 1] * 2
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), max_daily_fraction=0.02, min_stake=1)
        results = _backtest(two_copies, strategy, settle).detailed_results
        # 2% of 1000 = 20 for the whole calendar day: copy 1 uses it up
        assert results["bt_stake"].tolist() == pytest.approx([10, 10, 0, 0, 0, 0, 0, 0])

    @pytest.mark.parametrize("settle", [False, True])
    def test_own_day_key_gives_own_daily_limit(self, two_copies, settle):
        # Copy 1 wins its first two bets, so copy 2's limit is 2% of 1020 (sequential) or
        # of 1000 (one slot); min_stake drops the 0.40 left over in sequential mode
        two_copies["y"] = [0, 0, 1, 1] * 2
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), max_daily_fraction=0.02, min_stake=1)
        results = _backtest(two_copies, strategy, settle, day_column="day").detailed_results
        assert results["bt_stake"].tolist() == pytest.approx([10, 10, 0, 0, 10, 10, 0, 0])

    def test_pause_after_losses_per_day_key(self, two_copies):
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), pause_after_losses=2)
        shared = _backtest(two_copies, strategy, False).detailed_results
        own = _backtest(two_copies, strategy, False, day_column="day").detailed_results
        # Two losses of copy 1 pause the calendar day, copy 2 included
        assert (shared["bt_stake"] > 0).tolist() == [True, True] + [False] * 6
        # With its own day key copy 2 starts without a losing streak
        assert (own["bt_stake"] > 0).tolist() == [True, True, False, False, True, True, False, False]

    def test_streak_counts_only_bets_of_the_key_when_interleaved(self, two_copies):
        """Rows alternate between the copies; each copy's streak skips the other's bets."""
        data = two_copies.iloc[[0, 4, 1, 5, 2, 6, 3, 7]].reset_index(drop=True)
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), pause_after_losses=2)
        results = _backtest(data, strategy, False, day_column="day").detailed_results
        assert (results["bt_stake"] > 0).tolist() == [True, True, True, True, False, False, False, False]

    def test_dates_and_metrics_stay_real(self, two_copies):
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), max_daily_fraction=0.02)
        bt = _backtest(two_copies, strategy, True, day_column="day")
        assert (bt.detailed_results["bt_date_column"] == pd.Timestamp("2024-03-01 12:00")).all()
        metrics = bt.calculate_metrics()
        assert bt.evaluate(["Total Bets", "Yield [%]"]) == {
            "Total Bets": metrics["Total Bets"], "Yield [%]": metrics["Yield [%]"]}

    def test_history_returns_to_calendar_days_after_run(self, two_copies):
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), max_daily_fraction=0.02)
        history = _backtest(two_copies, strategy, False, day_column="day").history
        assert history.staked_on("2024-03-01") == 0.0  # nothing is stored under the calendar day
        history.set_day_key("2024-03-01/1")
        assert history.staked_on("2024-03-01") == pytest.approx(20.0)


class TestSlotColumn:
    def test_shared_slot_settles_together(self, two_copies):
        two_copies["slot"] = "2024-03-01 12:00"
        bt = _backtest(two_copies, btb.FixedStake(fraction=0.1), True, slot_column="slot", day_column="day")
        assert bt.detailed_results["bt_sizing_bankroll"].tolist() == pytest.approx([1000] * 8)

    def test_own_slots_settle_one_after_another(self, two_copies):
        two_copies["slot"] = two_copies["copy"]
        results = _backtest(two_copies, btb.FixedStake(fraction=0.1), True, slot_column="slot",
                            day_column="day").detailed_results
        # Copy 1 loses 2 x 100 and wins 2 x 100: copy 2 is sized from the bankroll after copy 1
        assert results["bt_sizing_bankroll"].tolist() == pytest.approx([1000] * 8)
        two_copies.loc[two_copies["copy"] == 1, "y"] = 1  # copy 1 now loses all 4 bets
        results = _backtest(two_copies, btb.FixedStake(fraction=0.1), True, slot_column="slot",
                            day_column="day").detailed_results
        assert results["bt_sizing_bankroll"].tolist() == pytest.approx([1000] * 4 + [600] * 4)

    def test_own_slots_see_each_others_results(self, two_copies):
        """Separate slots are what lets a copy react to its twin's losses."""
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), pause_after_losses=2)
        two_copies["y"] = [0, 0, 1, 1] * 2  # each copy ends with two losses
        two_copies["slot"] = "same"
        shared = _backtest(two_copies, strategy, True, slot_column="slot").detailed_results
        two_copies["slot"] = two_copies["copy"]
        separate = _backtest(two_copies, strategy, True, slot_column="slot").detailed_results
        assert (shared["bt_stake"] > 0).all()
        assert (separate["bt_stake"] > 0).tolist() == [True] * 4 + [False] * 4

    def test_slot_may_span_several_date_values(self):
        data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01 12:00", "2024-01-01 18:00", "2024-01-02 12:00"]),
            "a": 2.0, "b": 2.0, "y": [1, 1, 0], "pred": 0, "pa": 0.6, "pb": 0.4, "slot": ["d1", "d1", "d2"],
        })
        results = _backtest(data, btb.FixedStake(fraction=0.1), True, slot_column="slot").detailed_results
        # Both d1 bets are sized from 1000 although their date values differ
        assert results["bt_sizing_bankroll"].tolist() == pytest.approx([1000, 1000, 800])


class TestValidation:
    @pytest.mark.parametrize("param", ["slot_column", "day_column"])
    def test_missing_column(self, two_copies, param):
        with pytest.raises(ValueError, match=f"{param} 'nope' not found"):
            _backtest(two_copies, btb.FixedStake(amount=10), True, **{param: "nope"})

    @pytest.mark.parametrize("param", ["slot_column", "day_column"])
    def test_missing_values(self, two_copies, param):
        two_copies["key"] = two_copies["day"]
        two_copies.loc[3, "key"] = None
        with pytest.raises(ValueError, match="has missing values in 1 rows"):
            _backtest(two_copies, btb.FixedStake(amount=10), True, **{param: "key"})

    @pytest.mark.parametrize("param", ["slot_column", "day_column"])
    def test_unhashable_values(self, two_copies, param):
        two_copies["key"] = [[1]] * len(two_copies)
        with pytest.raises(ValueError, match="must hold hashable values"):
            _backtest(two_copies, btb.FixedStake(amount=10), True, **{param: "key"})

    def test_slot_needs_simultaneous_mode(self, two_copies):
        with pytest.raises(ValueError, match="slot_column requires settle_simultaneous=True"):
            _backtest(two_copies, btb.FixedStake(amount=10), False, slot_column="copy")

    def test_slot_rows_must_be_adjacent_in_date_order(self, two_copies):
        data = two_copies.iloc[[0, 4, 1, 5, 2, 6, 3, 7]].reset_index(drop=True)
        with pytest.raises(ValueError, match="rows of slot 1 are not adjacent"):
            _backtest(data, btb.FixedStake(amount=10), True, slot_column="copy")

    def test_slot_reused_on_a_later_date_is_rejected(self):
        data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
            "a": 2.0, "b": 2.0, "y": 0, "pred": 0, "pa": 0.6, "pb": 0.4, "slot": ["x", "y", "x"],
        })
        with pytest.raises(ValueError, match="rows of slot 'x' are not adjacent"):
            _backtest(data, btb.FixedStake(amount=10), True, slot_column="slot")

    def test_day_keys_need_not_be_adjacent(self, two_copies):
        data = two_copies.iloc[[0, 4, 1, 5, 2, 6, 3, 7]].reset_index(drop=True)
        _backtest(data, btb.FixedStake(amount=10), True, day_column="day")


class TestBetHistoryDayKey:
    def test_key_replaces_calendar_day(self):
        history = BetHistory(1000)
        history.set_day_key("copy 1")
        history.record("2024-01-01", stake=10, profit=-10, bankroll_before=1000, bankroll_after=990)
        history.set_day_key("copy 2")
        assert history.staked_on("2024-01-01") == 0.0
        assert history.bankroll_at_start_of("2024-01-01") == 990
        history.add_open_stake("2024-01-01", 5)
        assert history.staked_on("2024-01-01") == 5.0
        history.set_day_key("copy 1")
        assert history.staked_on("2024-01-01") == 10.0
        assert history.profit_on("2024-01-01") == -10.0
        assert history.losing_streak_on("2024-01-01") == 1
        history.set_day_key(None)
        assert history.staked_on("2024-01-01") == 0.0

    def test_unhashable_key(self):
        with pytest.raises(TypeError):
            BetHistory(1000).set_day_key(["copy 1"])
