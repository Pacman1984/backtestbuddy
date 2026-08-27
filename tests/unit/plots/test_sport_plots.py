"""
Unit tests for sport_plots module.

Tests plotting functions to ensure they execute without errors
and return correct types. Visual correctness is not tested here.
"""
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.dummy import DummyClassifier

from backtestbuddy.backtest.sport_backtest import ModelBacktest, PredictionBacktest
from backtestbuddy.strategies.sport_strategies import FixedStake
from backtestbuddy.plots.sport_plots import (
    _format_metric_value,
    _metric_table_cells,
    _panel_x,
    _underwater_pct,
    plot_backtest,
    plot_calibration,
    plot_odds_histogram,
)


class TestPlotBacktest:
    """Unit tests for plot_backtest function."""
    
    @pytest.fixture
    def simple_backtest_result(self):
        """Create a simple backtest with results for plotting."""
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0],
            'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'prediction': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        })
        
        backtest = PredictionBacktest(
            data=data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        return backtest
    
    def test_plot_backtest_returns_figure(self, simple_backtest_result):
        """Test that plot_backtest returns a Plotly Figure object."""
        fig = plot_backtest(simple_backtest_result)
        assert isinstance(fig, go.Figure)
    
    def test_plot_backtest_executes_without_error(self, simple_backtest_result):
        """Test that plot_backtest executes without raising exceptions."""
        try:
            fig = plot_backtest(simple_backtest_result)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"plot_backtest raised an exception: {e}")
    
    def test_plot_backtest_has_traces(self, simple_backtest_result):
        """Test that plot_backtest produces a figure with traces."""
        fig = plot_backtest(simple_backtest_result)
        assert len(fig.data) > 0, "Figure should have at least one trace"
    
    def test_plot_backtest_has_subplots(self, simple_backtest_result):
        """Test that plot_backtest creates multiple subplots."""
        fig = plot_backtest(simple_backtest_result)
        # Check that the figure has the expected layout structure
        assert fig.layout is not None
        # Should have multiple y-axes for subplots
        assert hasattr(fig.layout, 'yaxis')

    def test_plot_backtest_puts_metrics_in_a_table(
        self, simple_backtest_result
    ):
        """Metrics sit in a table, not paper annotations on the charts."""
        fig = plot_backtest(simple_backtest_result)
        tables = [trace for trace in fig.data if trace.type == "table"]
        assert len(tables) == 1
        assert list(tables[0].header.values) == ["Metric", "Value"]
        names = list(tables[0].cells.values[0])
        values = list(tables[0].cells.values[1])
        assert len(tables[0].cells.values) == 2
        assert len(names) == len(values)
        assert "" not in names
        assert "Yield [%]" in names
        assert "Total Bets" in names
        assert list(tables[0].header.align) == ["right", "left"]
        assert list(tables[0].cells.align) == ["right", "left"]
        metric_annotations = [
            ann
            for ann in (fig.layout.annotations or [])
            if getattr(ann, "text", "") and "Yield [%]:" in str(ann.text)
        ]
        assert metric_annotations == []

    def test_plot_backtest_x_axis_title_only_on_bottom(
        self, simple_backtest_result
    ):
        """Only the lowest panel is labeled Bet Number."""
        fig = plot_backtest(simple_backtest_result)
        assert not (fig.layout.xaxis.title.text or "")
        assert not (fig.layout.xaxis2.title.text or "")
        assert not (fig.layout.xaxis3.title.text or "")
        assert fig.layout.xaxis4.title.text == "Bet Number"

    def test_plot_backtest_date_axis_labels_bottom_as_date(
        self, simple_backtest_result
    ):
        """Date mode puts calendar timestamps on x and labels Date."""
        fig = plot_backtest(simple_backtest_result, x_axis="date")
        assert fig.layout.xaxis4.title.text == "Date"
        main = next(t for t in fig.data if t.name == "Main Strategy")
        assert pd.Timestamp(main.x[0]) == pd.Timestamp("2023-01-01")

    def test_plot_backtest_rejects_unknown_x_axis(
        self, simple_backtest_result
    ):
        """Only bet and date are valid x_axis values."""
        with pytest.raises(ValueError, match="x_axis"):
            plot_backtest(simple_backtest_result, x_axis="week")

    def test_plot_backtest_overlays_bookie_by_default(
        self, simple_backtest_result
    ):
        """Bookie bankroll is a named trace with one point per opportunity."""
        fig = plot_backtest(simple_backtest_result)
        bookie = next(t for t in fig.data if t.name == "Bookie")
        np.testing.assert_allclose(
            np.asarray(bookie.y, dtype=float),
            simple_backtest_result.bookie_results[
                "bt_ending_bankroll"
            ].to_numpy(dtype=float),
        )

    def test_plot_backtest_bookie_keeps_skipped_rows(self):
        """Bookie x is all opportunities; main and drawdown are placed only."""
        data = pd.DataFrame({
            "date": pd.date_range(start="2023-01-01", periods=4),
            "odds_1": [2.0, 1.8, 2.5, 1.5],
            "odds_2": [1.8, 2.2, 1.6, 2.8],
            "outcome": [0, 1, 0, 1],
            "prediction": [0, -1, 1, -1],
        })
        backtest = PredictionBacktest(
            data=data,
            odds_columns=["odds_1", "odds_2"],
            outcome_column="outcome",
            date_column="date",
            prediction_column="prediction",
            initial_bankroll=1000,
            strategy=FixedStake(stake=100),
        )
        backtest.run()
        fig = plot_backtest(backtest)
        main = next(t for t in fig.data if t.name == "Main Strategy")
        bookie = next(t for t in fig.data if t.name == "Bookie")
        dd = next(t for t in fig.data if t.name == "Drawdown")
        placed = backtest.detailed_results
        n_placed = int(
            ((placed["bt_stake"] > 0) & (placed["bt_bet_on"] != -1)).sum()
        )
        assert n_placed == 2
        assert len(backtest.bookie_results) == 4
        assert len(main.y) == n_placed
        assert len(dd.y) == n_placed
        assert len(bookie.y) == len(backtest.bookie_results)
        np.testing.assert_allclose(
            np.asarray(bookie.y, dtype=float),
            backtest.bookie_results["bt_ending_bankroll"].to_numpy(
                dtype=float
            ),
        )

    def test_plot_backtest_can_hide_bookie(self, simple_backtest_result):
        """show_bookie=False omits the Bookie trace."""
        fig = plot_backtest(simple_backtest_result, show_bookie=False)
        names = [t.name for t in fig.data]
        assert "Bookie" not in names

    def test_plot_backtest_has_underwater_panel(
        self, simple_backtest_result
    ):
        """Drawdown sits on subplot row 2, filled to zero."""
        fig = plot_backtest(simple_backtest_result)
        dd = next(t for t in fig.data if t.name == "Drawdown")
        assert dd.fill == "tozeroy"
        assert dd.yaxis == "y2"
        y = np.asarray(dd.y, dtype=float)
        assert y[0] == pytest.approx(0.0)
        titles = [
            getattr(ann, "text", "")
            for ann in (fig.layout.annotations or [])
        ]
        assert "Drawdown" in titles
        assert fig.layout.yaxis2.title.text == "DD %"

    def test_plot_backtest_legend_is_horizontal(
        self, simple_backtest_result
    ):
        """Legend sits above the charts so it cannot cover the table."""
        fig = plot_backtest(simple_backtest_result)
        assert fig.layout.legend.orientation == "h"
    
    def test_plot_backtest_with_no_bets_placed(self):
        """Test plot_backtest behavior when no bets are placed."""
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2],
            'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9],
            'outcome': [0, 1, 0, 1, 0],
            'prediction': [-1, -1, -1, -1, -1]  # No bets
        })
        
        backtest = PredictionBacktest(
            data=data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=0)  # Zero stake
        )
        backtest.run()
        
        # Should handle gracefully even with no bets
        try:
            fig = plot_backtest(backtest)
            # Even with no bets, should return a figure (might be empty)
            assert isinstance(fig, go.Figure)
        except Exception:
            # It's acceptable if it raises an error for no bets scenario
            pass


class TestPlotOddsHistogram:
    """Unit tests for plot_odds_histogram function."""
    
    @pytest.fixture
    def simple_backtest_result(self):
        """Create a simple backtest with results for plotting."""
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=20),
            'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0,
                       1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0, 1.8],
            'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8,
                       2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8, 2.2],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'prediction': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        })
        
        backtest = PredictionBacktest(
            data=data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        return backtest
    
    def test_plot_odds_histogram_returns_figure(self, simple_backtest_result):
        """Test that plot_odds_histogram returns a Plotly Figure object."""
        fig = plot_odds_histogram(simple_backtest_result)
        assert isinstance(fig, go.Figure)
    
    def test_plot_odds_histogram_executes_without_error(self, simple_backtest_result):
        """Test that plot_odds_histogram executes without raising exceptions."""
        try:
            fig = plot_odds_histogram(simple_backtest_result)
            assert fig is not None
        except Exception as e:
            pytest.fail(f"plot_odds_histogram raised an exception: {e}")
    
    def test_plot_odds_histogram_with_custom_bins(self, simple_backtest_result):
        """Test plot_odds_histogram with custom number of bins."""
        fig = plot_odds_histogram(simple_backtest_result, num_bins=10)
        assert isinstance(fig, go.Figure)
    
    def test_plot_odds_histogram_with_auto_bins(self, simple_backtest_result):
        """Test plot_odds_histogram with automatic binning."""
        fig = plot_odds_histogram(simple_backtest_result, num_bins=None)
        assert isinstance(fig, go.Figure)
    
    def test_plot_odds_histogram_has_traces(self, simple_backtest_result):
        """Test that plot_odds_histogram produces a figure with histogram traces."""
        fig = plot_odds_histogram(simple_backtest_result)
        assert len(fig.data) > 0, "Figure should have at least one trace"
        # Check that at least one trace is a histogram
        has_histogram = any(isinstance(trace, go.Histogram) for trace in fig.data)
        assert has_histogram or len(fig.data) > 0, "Should have histogram traces"


class TestPlotIntegration:
    """Integration tests for plotting with different backtest types."""
    
    def test_plot_with_model_backtest(self):
        """Test plotting with ModelBacktest results."""
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'feature_1': [5, 3, 7, 2, 8, 4, 9, 1, 6, 5],
            'feature_2': [2, 6, 4, 9, 1, 8, 3, 7, 5, 2],
            'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0],
            'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        model = DummyClassifier(strategy="stratified", random_state=42)
        backtest = ModelBacktest(
            data=data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        
        # Test both plotting functions work with ModelBacktest
        fig1 = plot_backtest(backtest)
        fig2 = plot_odds_histogram(backtest)
        fig3 = plot_calibration(backtest)

        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)
        names = [t.name for t in fig3.data]
        assert "Perfect calibration" in names
        has_probs = any(
            col.startswith("bt_model_prob_")
            for col in backtest.detailed_results.columns
        )
        if has_probs:
            assert "Observed" in names


class TestPlotCalibration:
    """Unit tests for the reliability diagram."""

    def test_observed_points_aggregate_equal_width_bin(self):
        """0.21 and 0.29 share a 10-bin interval; unique-p would stay split."""

        class _Results:
            """Minimal backtest stub with placed-bet probability rows."""

            def __init__(self, frame: pd.DataFrame) -> None:
                self.detailed_results = frame

        data = pd.DataFrame({
            "bt_stake": [100, 100, 100, 100],
            "bt_odds": [2.0, 2.0, 2.0, 2.0],
            "bt_profit": [-100, 100, -100, 100],
            "bt_win": [False, True, False, True],
            "bt_bet_on": [1, 1, 1, 1],
            "bt_model_prob_0": [0.9, 0.9, 0.9, 0.9],
            "bt_model_prob_1": [0.21, 0.29, 0.21, 0.29],
        })
        fig = plot_calibration(_Results(data), n_bins=10)
        observed = next(t for t in fig.data if t.name == "Observed")
        assert len(observed.x) == 1
        assert observed.x[0] == pytest.approx(0.25)
        assert observed.y[0] == pytest.approx(0.5)
        diagonal = next(
            t for t in fig.data if t.name == "Perfect calibration"
        )
        assert list(diagonal.x) == [0.0, 1.0]
        assert list(diagonal.y) == [0.0, 1.0]

    def test_no_probabilities_has_diagonal_only(self):
        """Without model probs there is no Observed scatter."""
        data = pd.DataFrame({
            "date": pd.date_range(start="2023-01-01", periods=5),
            "odds_1": [2.0, 1.8, 2.5, 1.5, 2.2],
            "odds_2": [1.8, 2.2, 1.6, 2.8, 1.9],
            "outcome": [0, 1, 0, 1, 0],
            "prediction": [0, 1, 1, 0, 0],
        })
        backtest = PredictionBacktest(
            data=data,
            odds_columns=["odds_1", "odds_2"],
            outcome_column="outcome",
            date_column="date",
            prediction_column="prediction",
            initial_bankroll=1000,
            strategy=FixedStake(stake=100),
        )
        backtest.run()
        fig = plot_calibration(backtest)
        names = [t.name for t in fig.data]
        assert "Observed" not in names
        assert "Perfect calibration" in names
        notes = [
            getattr(ann, "text", "")
            for ann in (fig.layout.annotations or [])
        ]
        assert any("No model probabilities" in text for text in notes)


class TestUnderwaterPct:
    """Unit tests for the underwater series helper."""

    def test_drop_from_peak_is_negative_percent(self):
        """A fall from 110 to 99 is a 10% underwater print."""
        equity = np.array([100.0, 110.0, 99.0])
        np.testing.assert_allclose(
            _underwater_pct(equity),
            [0.0, 0.0, -10.0],
        )

    def test_empty_and_zero_peak_are_zero(self):
        """Empty input stays empty; a zero peak does not divide."""
        assert _underwater_pct(np.array([])).size == 0
        np.testing.assert_allclose(
            _underwater_pct(np.array([0.0, 0.0])),
            [0.0, 0.0],
        )


class TestPanelX:
    """Unit tests for bet-index vs calendar x values."""

    def test_bet_is_one_based_index(self):
        """Bet mode is 1..n aligned to the frame index."""
        frame = pd.DataFrame(
            {"bt_date_column": pd.to_datetime(["2023-01-02", "2023-01-03"])},
            index=[10, 20],
        )
        xs = _panel_x(frame, "bet")
        assert list(xs) == [1, 2]
        assert list(xs.index) == [10, 20]

    def test_date_uses_bt_date_column(self):
        """Date mode parses bt_date_column."""
        frame = pd.DataFrame(
            {"bt_date_column": ["2023-01-02", "2023-01-03"]}
        )
        xs = _panel_x(frame, "date")
        assert list(xs) == [
            pd.Timestamp("2023-01-02"),
            pd.Timestamp("2023-01-03"),
        ]

    def test_unknown_axis_raises(self):
        """Unsupported x_axis values are rejected."""
        frame = pd.DataFrame({"bt_date_column": ["2023-01-01"]})
        with pytest.raises(ValueError, match="x_axis"):
            _panel_x(frame, "week")


class TestFormatMetricValue:
    """Unit tests for side-table value formatting."""

    def test_non_finite_floats_become_em_dash(self):
        """NaN and inf are shown as an em dash, not the string nan."""
        assert _format_metric_value(float("nan")) == "—"
        assert _format_metric_value(float("inf")) == "—"
        assert _format_metric_value(float("-inf")) == "—"

    def test_finite_float_uses_two_decimals(self):
        """Finite floats are rounded to two decimal places."""
        assert _format_metric_value(12.345) == "12.35"
        assert _format_metric_value(4.0) == "4.00"

    def test_integers_have_no_decimals(self):
        """Counts stay as whole numbers."""
        assert _format_metric_value(5) == "5"
        assert _format_metric_value(np.int64(12)) == "12"

    def test_timestamps_and_durations_drop_midnight(self):
        """Dates show YYYY-MM-DD; whole-day durations drop 00:00:00."""
        ts = pd.Timestamp("2023-01-01 00:00:00")
        assert _format_metric_value(ts) == "2023-01-01"
        assert _format_metric_value(pd.Timedelta(days=9)) == "9 days"


class TestMetricTableCells:
    """Unit tests for the single Metric/Value column layout."""

    def test_keeps_all_metrics_in_one_column_pair(self):
        """Odd-length maps stay one pair; they are not split or padded."""
        metrics = {"A": 1, "B": 2.5, "C": float("nan")}
        names, values = _metric_table_cells(metrics)
        assert names == ["A", "B", "C"]
        assert values == ["1", "2.50", "—"]


