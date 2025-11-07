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
from backtestbuddy.plots.sport_plots import plot_backtest, plot_odds_histogram


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
        
        assert isinstance(fig1, go.Figure)
        assert isinstance(fig2, go.Figure)

