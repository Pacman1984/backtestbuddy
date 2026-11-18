"""
Tests for slot_column and day_column of PredictionBacktest.

slot_column replaces the date value as the settlement slot of
settle_simultaneous. day_column adds a key to the calendar day: every
BetHistory day total and query (daily limits, pause after losses) then
uses the pair (calendar day, key). Both default to None, which must keep
the 0.3.1 results exactly.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtestbuddy as btb
from backtestbuddy.strategies.history import BetHistory

DATA_DIR = Path(__file__).parent / "data"
REFERENCE_INPUT = DATA_DIR / "reference_input.csv.gz"
REFERENCE_FILE = DATA_DIR / "reference_0_3_1.csv.gz"
REFERENCE_COLUMNS = ["bt_stake", "bt_bet_on", "bt_win", "bt_profit", "bt_starting_bankroll",
                     "bt_ending_bankroll", "bt_risk_limit", "bt_sizing_bankroll"]


def _random_data():
    """120 matches over 15 days at three kick-off times; the model has an edge."""
    rng = np.random.default_rng(7)
    n = 120
    days = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 15, n), unit="D")
    hours = pd.to_timedelta(rng.choice([12, 15, 18], n), unit="h")
    p = rng.uniform(0.25, 0.75, n)
    market = np.clip(p + rng.normal(0, 0.08, n), 0.1, 0.9)
    return pd.DataFrame({
        "date": days + hours, "a": np.round(1.0 / (market * 1.03), 2), "b": np.round(1.0 / ((1 - market) * 1.03), 2),
        "y": (rng.uniform(size=n) > p).astype(int),
        "pa": p, "pb": 1 - p, "pred": (p < 0.5).astype(int),
    })


def reference_input():
    """The stored output of ``_random_data``.

    NumPy versions may differ in the last bit of generated floats, so the exact
    comparison with 0.3.1 reads the inputs it was computed from.
    """
    return pd.read_csv(REFERENCE_INPUT, parse_dates=["date"], float_precision="round_trip")


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
SCENARIOS = [(name, settle) for name in STRATEGIES for settle in (False, True)]


def _backtest(data, strategy, settle, **kwargs):
    bt = btb.PredictionBacktest(data, ["a", "b"], "y", "date", "pred", strategy=strategy,
                                model_prob_columns=["pa", "pb"], settle_simultaneous=settle, **kwargs)
    bt.run()
    return bt


def reference_frame(bt, name, settle):
    """Result columns compared exactly against 0.3.1 (also used to write the reference file)."""
    results = bt.detailed_results
    frame = pd.DataFrame({"scenario": f"{name}/{settle}", "row": range(len(results))})
    for column in REFERENCE_COLUMNS:
        frame[column] = results[column] if column in results else np.nan
    frame["bookie_stake"] = bt.bookie_results["bt_stake"].to_numpy()
    frame["bookie_ending_bankroll"] = bt.bookie_results["bt_ending_bankroll"].to_numpy()
    return frame


def _as_text(series):
    return series.map(lambda value: "" if value is None or value != value else str(value)).tolist()


class TestDefaultsUnchanged:
    @pytest.fixture(scope="class")
    def reference(self):
        # Written by backtestbuddy 0.3.1 (commit 0cb9dd9) from REFERENCE_INPUT with
        # float_format="%.17g", so every float round-trips exactly
        return pd.read_csv(REFERENCE_FILE, keep_default_na=False, na_values=[""], float_precision="round_trip")

    @pytest.mark.parametrize("name,settle", SCENARIOS)
    def test_identical_to_031(self, reference, name, settle):
        expected = reference[reference["scenario"] == f"{name}/{settle}"].reset_index(drop=True)
        actual = reference_frame(_backtest(reference_input(), STRATEGIES[name](), settle), name, settle)
        assert len(actual) == len(expected) == 120
        for column in ["bt_stake", "bt_bet_on", "bt_profit", "bt_starting_bankroll", "bt_ending_bankroll",
                       "bt_sizing_bankroll", "bookie_stake", "bookie_ending_bankroll"]:
            np.testing.assert_array_equal(actual[column].to_numpy(float), expected[column].to_numpy(float),
                                          err_msg=column)
        for column in ["bt_win", "bt_risk_limit"]:
            assert _as_text(actual[column]) == _as_text(expected[column].astype(object)), column

    @pytest.mark.parametrize("name,settle", SCENARIOS)
    def test_explicit_none_equals_omitted(self, name, settle):
        data = _random_data()
        omitted = _backtest(data, STRATEGIES[name](), settle)
        explicit = _backtest(data, STRATEGIES[name](), settle, slot_column=None, day_column=None)
        pd.testing.assert_frame_equal(omitted.detailed_results, explicit.detailed_results)
        pd.testing.assert_frame_equal(omitted.bookie_results, explicit.bookie_results)

    @pytest.mark.parametrize("name", list(STRATEGIES))
    def test_slot_equal_to_date_value_gives_same_results(self, name):
        data = _random_data()
        data["slot"] = data["date"]
        default = _backtest(data, STRATEGIES[name](), True)
        keyed = _backtest(data, STRATEGIES[name](), True, slot_column="slot")
        pd.testing.assert_frame_equal(default.detailed_results, keyed.detailed_results)

    @pytest.mark.parametrize("name,settle", SCENARIOS)
    def test_one_shared_day_key_gives_same_results(self, name, settle):
        """A single key for all rows splits nothing, so limits act as on calendar days."""
        data = _random_data()
        data["day"] = "all"
        default = _backtest(data, STRATEGIES[name](), settle)
        keyed = _backtest(data, STRATEGIES[name](), settle, day_column="day")
        pd.testing.assert_frame_equal(default.detailed_results.drop(columns="day"),
                                      keyed.detailed_results.drop(columns="day"))


@pytest.fixture
def two_copies():
    """One day drawn twice (copies 1 and 2): 4 matches each at the same timestamp."""
    one = pd.DataFrame({
        "date": pd.Timestamp("2024-03-01 12:00"),
        "a": 2.0, "b": 2.0, "y": [1, 1, 0, 0], "pred": 0, "pa": 0.6, "pb": 0.4,
    })
    return pd.concat([one.assign(copy=1), one.assign(copy=2)], ignore_index=True)


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
        results = _backtest(two_copies, strategy, settle, day_column="copy").detailed_results
        assert results["bt_stake"].tolist() == pytest.approx([10, 10, 0, 0, 10, 10, 0, 0])

    @pytest.mark.parametrize("order", [[0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]])
    def test_day_start_in_shared_slot_is_slot_opening_bankroll(self, two_copies, order):
        """Acceptance test: two copies of one day in one shared slot, each with its own day key.

        Both day starts are 1000, the bankroll before the shared slot, not the bankroll after
        the other key's bets in the same slot; the same in both row orders.
        """
        two_copies["y"] = [0, 0, 0, 0, 1, 1, 1, 1]  # copy 1 wins 4 x 100, copy 2 loses 4 x 100
        data = two_copies.iloc[order].reset_index(drop=True)
        data["slot"] = "shared"
        strategy = btb.RiskLimits(btb.FixedStake(fraction=0.1), max_daily_fraction=0.5)
        bt = _backtest(data, strategy, True, slot_column="slot", day_column="copy")
        assert bt.detailed_results["bt_sizing_bankroll"].tolist() == pytest.approx([1000] * 8)
        day = pd.Timestamp("2024-03-01 12:00")
        assert bt.history.bankroll_at_start_of(day, key=1) == pytest.approx(1000)
        assert bt.history.bankroll_at_start_of(day, key=2) == pytest.approx(1000)

    def test_daily_cap_after_shared_slot_does_not_depend_on_row_order(self, two_copies):
        """A later slot of the same day caps each key at 50% of 1000 in both orders."""
        late = two_copies.assign(date=pd.Timestamp("2024-03-01 18:00"), y=0)
        caps = []
        for order in ([0, 1, 2, 3, 4, 5, 6, 7], [4, 5, 6, 7, 0, 1, 2, 3]):
            early = two_copies.assign(y=[0, 0, 0, 0, 1, 1, 1, 1]).iloc[order]
            data = pd.concat([early, late], ignore_index=True)
            strategy = btb.RiskLimits(btb.FixedStake(fraction=0.1), max_daily_fraction=0.5)
            results = _backtest(data, strategy, True, day_column="copy").detailed_results
            staked = results.groupby([results["bt_date_column"], results["copy"]])["bt_stake"].sum()
            caps.append(staked.to_dict())
        assert caps[0] == pytest.approx(caps[1])
        # Each copy: 400 in the first slot, then at most 500 - 400 = 100 in the second
        assert sum(stake for (date, _), stake in caps[0].items() if date.hour == 18) == pytest.approx(200)

    def test_pause_after_losses_per_day_key(self, two_copies):
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), pause_after_losses=2)
        shared = _backtest(two_copies, strategy, False).detailed_results
        own = _backtest(two_copies, strategy, False, day_column="copy").detailed_results
        # Two losses of copy 1 pause the calendar day, copy 2 included
        assert (shared["bt_stake"] > 0).tolist() == [True, True] + [False] * 6
        # With its own day key copy 2 starts without a losing streak
        assert (own["bt_stake"] > 0).tolist() == [True, True, False, False, True, True, False, False]

    def test_streak_counts_only_bets_of_the_key_when_interleaved(self, two_copies):
        """Rows alternate between the copies; each copy's streak skips the other's bets."""
        data = two_copies.iloc[[0, 4, 1, 5, 2, 6, 3, 7]].reset_index(drop=True)
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), pause_after_losses=2)
        results = _backtest(data, strategy, False, day_column="copy").detailed_results
        assert (results["bt_stake"] > 0).tolist() == [True, True, True, True, False, False, False, False]

    def test_stop_at_drawdown_sees_the_whole_history(self, two_copies):
        """Drawdown is global: losses of copy 1 stop copy 2 even with separate day keys."""
        two_copies["y"] = 1  # every bet loses
        strategy = btb.RiskLimits(btb.FixedStake(fraction=0.1), stop_at_drawdown=0.25)
        results = _backtest(two_copies, strategy, False, day_column="copy").detailed_results
        # 1000 -> 900 -> 810 -> 729 (27% down): no further bet, for either copy
        assert (results["bt_stake"] > 0).tolist() == [True, True, True] + [False] * 5
        assert (results["bt_risk_limit"].iloc[3:] == "stop_at_drawdown").all()

    def test_global_losing_streak_unchanged(self, two_copies):
        """Strategies reading history.losing_streak (e.g. SmallerAfterLosses) still see all bets."""
        bt = _backtest(two_copies.assign(y=1), btb.FixedStake(amount=10), False, day_column="copy")
        assert bt.history.losing_streak == 8

    def test_query_for_another_day_of_the_same_key(self):
        """During a run, a query for yesterday returns yesterday of the current row's key."""
        seen = []

        class Recorder(btb.FixedStake):
            def get_bet_details(self, odds, bankroll, model_probs=None, prediction=None, **kwargs):
                if kwargs["date"].day == 2:
                    seen.append(kwargs["history"].profit_on(kwargs["date"] - pd.Timedelta(days=1)))
                return super().get_bet_details(odds, bankroll, model_probs, prediction, **kwargs)

        data = pd.DataFrame({
            "date": pd.to_datetime(["2024-03-01", "2024-03-01", "2024-03-02", "2024-03-02"]),
            "a": 2.0, "b": 2.0, "y": [0, 1, 0, 0], "pred": 0, "pa": 0.6, "pb": 0.4, "copy": [1, 2, 1, 2],
        })
        _backtest(data, Recorder(amount=10), False, day_column="copy")
        assert seen == [10.0, -10.0]  # copy 1 won yesterday, copy 2 lost

    def test_dates_and_metrics_stay_real(self, two_copies):
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), max_daily_fraction=0.02)
        bt = _backtest(two_copies, strategy, True, day_column="copy")
        assert (bt.detailed_results["bt_date_column"] == pd.Timestamp("2024-03-01 12:00")).all()
        metrics = bt.calculate_metrics()
        assert bt.evaluate(["Total Bets", "Yield [%]"]) == {
            "Total Bets": metrics["Total Bets"], "Yield [%]": metrics["Yield [%]"]}

    def test_queries_after_the_run(self, two_copies):
        """After the run, key= selects a copy; without key only unkeyed bets count (none here)."""
        two_copies["y"] = [1, 1, 0, 0, 0, 0, 0, 0]
        history = _backtest(two_copies, btb.FixedStake(amount=10), False, day_column="copy").history
        day = "2024-03-01"
        assert history.staked_on(day, key=1) == pytest.approx(40.0)
        assert history.profit_on(day, key=1) == pytest.approx(0.0)
        assert history.profit_on(day, key=2) == pytest.approx(40.0)
        assert history.bankroll_at_start_of(day, key=2) == pytest.approx(1000.0)
        assert history.losing_streak_on(day, key=1) == 0
        assert history.staked_on(day) == 0.0
        assert history.profit_on(day) == 0.0
        assert history.losing_streak_on(day) == 0
        assert history.bankroll_at_start_of(day) == history.bankroll  # no unkeyed bet that day


class TestSlotColumn:
    def test_shared_slot_settles_together(self, two_copies):
        two_copies["slot"] = "2024-03-01 12:00"
        bt = _backtest(two_copies, btb.FixedStake(fraction=0.1), True, slot_column="slot", day_column="copy")
        assert bt.detailed_results["bt_sizing_bankroll"].tolist() == pytest.approx([1000] * 8)

    def test_own_slots_settle_one_after_another(self, two_copies):
        two_copies.loc[two_copies["copy"] == 1, "y"] = 1  # copy 1 loses all 4 bets
        results = _backtest(two_copies, btb.FixedStake(fraction=0.1), True, slot_column="copy",
                            day_column="copy").detailed_results
        assert results["bt_sizing_bankroll"].tolist() == pytest.approx([1000] * 4 + [600] * 4)

    def test_own_slots_see_each_others_results(self, two_copies):
        """Separate slots are what lets a copy react to its twin's losses."""
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), pause_after_losses=2)
        two_copies["y"] = [0, 0, 1, 1] * 2  # each copy ends with two losses
        two_copies["slot"] = "same"
        shared = _backtest(two_copies, strategy, True, slot_column="slot").detailed_results
        separate = _backtest(two_copies, strategy, True, slot_column="copy").detailed_results
        assert (shared["bt_stake"] > 0).all()
        assert (separate["bt_stake"] > 0).tolist() == [True] * 4 + [False] * 4

    def test_slot_may_span_two_calendar_days(self):
        """Without day_column, a slot over two calendar days keeps one daily limit per day."""
        data = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01 22:00", "2024-01-02 01:00", "2024-01-02 18:00"]),
            "a": 2.0, "b": 2.0, "y": [1, 1, 0], "pred": 0, "pa": 0.6, "pb": 0.4,
            "slot": ["night", "night", "evening"],
        })
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), max_daily_fraction=0.01)
        bt = _backtest(data, strategy, True, slot_column="slot")
        results = bt.detailed_results
        # Both night bets are sized from 1000 and each uses its own day's limit of 10
        assert results["bt_sizing_bankroll"].tolist() == pytest.approx([1000, 1000, 980])
        assert results["bt_stake"].tolist() == pytest.approx([10, 10, 0])
        assert results["bt_risk_limit"].iloc[2] == "max_daily_fraction"
        # Jan 2 started inside the night slot, so its start is the slot's opening bankroll
        assert bt.history.bankroll_at_start_of("2024-01-02") == pytest.approx(1000)


class TestValidation:
    @pytest.mark.parametrize("param", ["slot_column", "day_column"])
    def test_missing_column(self, two_copies, param):
        with pytest.raises(ValueError, match=f"{param} 'nope' not found"):
            _backtest(two_copies, btb.FixedStake(amount=10), True, **{param: "nope"})

    @pytest.mark.parametrize("param", ["slot_column", "day_column"])
    @pytest.mark.parametrize("missing", [None, np.nan])
    def test_missing_values(self, two_copies, param, missing):
        two_copies["key"] = two_copies["copy"].astype(object)
        two_copies.loc[3, "key"] = missing
        with pytest.raises(ValueError, match="has missing values in 1 rows"):
            _backtest(two_copies, btb.FixedStake(amount=10), True, **{param: "key"})

    @pytest.mark.parametrize("param", ["slot_column", "day_column"])
    def test_unhashable_values(self, two_copies, param):
        two_copies["key"] = [[1]] * len(two_copies)
        with pytest.raises(ValueError, match="must hold hashable values"):
            _backtest(two_copies, btb.FixedStake(amount=10), True, **{param: "key"})

    @pytest.mark.parametrize("param", ["slot_column", "day_column"])
    def test_equal_keys_of_different_types(self, two_copies, param):
        two_copies["key"] = pd.Series([1, 1, 1, 1, True, True, True, True], dtype=object)
        with pytest.raises(ValueError, match="compare equal but differ in type"):
            _backtest(two_copies, btb.FixedStake(amount=10), True, **{param: "key"})

    def test_mixed_types_that_differ_are_fine(self, two_copies):
        two_copies["key"] = pd.Series(["a"] * 4 + [2] * 4, dtype=object)
        strategy = btb.RiskLimits(btb.FixedStake(amount=10), max_daily_fraction=0.02)
        results = _backtest(two_copies, strategy, True, slot_column="key", day_column="key").detailed_results
        # Key "a" loses its two bets; key 2 then starts its day at 980 (limit 19.60)
        assert results["bt_stake"].tolist() == pytest.approx([10, 10, 0, 0, 10, 9.6, 0, 0])

    def test_day_column_must_not_be_the_date_column(self, two_copies):
        with pytest.raises(ValueError, match="day_column must not be the date column"):
            _backtest(two_copies, btb.FixedStake(amount=10), False, day_column="date")

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
        _backtest(data, btb.FixedStake(amount=10), True, day_column="copy")


class TestBetHistoryKeys:
    def test_key_adds_to_calendar_day(self):
        history = BetHistory(1000)
        history.record("2024-01-01", stake=10, profit=-10, bankroll_before=1000, bankroll_after=990, key="c1")
        history.record("2024-01-02", stake=20, profit=20, bankroll_before=990, bankroll_after=1010, key="c1")
        history.add_open_stake("2024-01-02", 5, key="c2")
        assert history.staked_on("2024-01-01", key="c1") == 10.0
        assert history.staked_on("2024-01-02", key="c1") == 20.0
        assert history.staked_on("2024-01-02", key="c2") == 5.0
        assert history.profit_on("2024-01-01", key="c1") == -10.0
        assert history.bankroll_at_start_of("2024-01-02", key="c1") == 990.0
        assert history.bankroll_at_start_of("2024-01-02", key="c2") == 1010.0  # no bet yet: current
        assert history.losing_streak_on("2024-01-01", key="c1") == 1
        assert history.losing_streak_on("2024-01-02", key="c1") == 0
        assert history.staked_on("2024-01-01") == 0.0  # unkeyed bucket is empty

    def test_day_start_bankroll_argument(self):
        history = BetHistory(1000)
        history.record("2024-01-01", stake=10, profit=10, bankroll_before=1400, bankroll_after=1410,
                       day_start_bankroll=1000)
        assert history.bankroll_at_start_of("2024-01-01") == 1000.0

    @pytest.mark.parametrize("key,error", [(["c1"], TypeError), (float("nan"), ValueError)])
    def test_invalid_key(self, key, error):
        with pytest.raises(error):
            BetHistory(1000).record("2024-01-01", stake=10, profit=10, bankroll_before=1000,
                                    bankroll_after=1010, key=key)
        with pytest.raises(error):
            BetHistory(1000).add_open_stake("2024-01-01", 5, key=key)

    def test_pickle_from_before_day_keys(self):
        """A ledger pickled by 0.3.1 lacks the new attributes and must still load and work."""
        history = BetHistory(1000)
        history.record("2024-01-01", stake=10, profit=-10, bankroll_before=1000, bankroll_after=990)
        state = dict(history.__dict__)
        for name in ("_bet_keys", "_streak_by_day", "_row_key"):
            del state[name]
        old = BetHistory.__new__(BetHistory)
        old.__setstate__(state)
        restored = pickle.loads(pickle.dumps(old))
        assert restored.losing_streak_on("2024-01-01") == 1
        restored.record("2024-01-01", stake=10, profit=-10, bankroll_before=990, bankroll_after=980, key="c1")
        assert restored.losing_streak_on("2024-01-01", key="c1") == 1
        assert restored.losing_streak_on("2024-01-01") == 0
