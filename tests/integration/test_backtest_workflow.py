"""
Integration tests for complete backtest workflows.

Tests end-to-end scenarios with multiple components working together.
These tests verify that the full backtest pipeline works correctly
from data input through strategy execution to results generation.
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from backtestbuddy.backtest.sport_backtest import ModelBacktest, PredictionBacktest
from backtestbuddy.strategies.sport_strategies import FixedStake, KellyCriterion


class TestBacktestWorkflow:
    """Integration tests for complete backtest workflows."""
    
    @pytest.fixture
    def deterministic_data(self):
        """Create deterministic test data for reproducible integration tests."""
        return pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=20),
            'feature_1': [5, 3, 7, 2, 8, 4, 9, 1, 6, 5, 3, 7, 2, 8, 4, 9, 1, 6, 5, 3],
            'feature_2': [2, 6, 4, 9, 1, 8, 3, 7, 5, 2, 6, 4, 9, 1, 8, 3, 7, 5, 2, 6],
            'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0, 
                       1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0, 1.8],
            'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8, 
                       2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8, 2.2],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model with fixed random seed."""
        return DummyClassifier(strategy="stratified", random_state=42)

    @pytest.fixture
    def kelly_strategy(self):
        """Create a Kelly Criterion strategy."""
        return KellyCriterion(downscaling=1.0)

    @pytest.fixture
    def fractional_kelly_strategy(self):
        """Create a fractional Kelly Criterion strategy."""
        return KellyCriterion(downscaling=0.5)

    def test_complete_backtest_workflow_with_fixed_stake(self, deterministic_data, dummy_model):
        """Test complete workflow from initialization to metrics calculation with FixedStake."""
        # Initialize backtest
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        
        # Run backtest
        backtest.run()
        
        # Verify results exist
        assert backtest.detailed_results is not None
        assert backtest.bookie_results is not None
        
        # Get results
        detailed = backtest.get_detailed_results()
        bookie = backtest.get_bookie_results()
        
        assert len(detailed) > 0
        assert len(bookie) > 0
        assert len(detailed) == len(bookie)
        
        # Calculate metrics
        metrics = backtest.calculate_metrics()
        assert isinstance(metrics, dict)
        assert 'ROI [%]' in metrics
        assert 'Win Rate [%]' in metrics
        
        # Verify bankroll progression is continuous
        for i in range(1, len(detailed)):
            assert detailed['bt_starting_bankroll'].iloc[i] == detailed['bt_ending_bankroll'].iloc[i-1]

    def test_strategy_comparison_workflow(self, deterministic_data, dummy_model, kelly_strategy, fractional_kelly_strategy):
        """Test comparing multiple strategies in a complete workflow."""
        strategies = [
            FixedStake(stake=100),
            kelly_strategy,
            fractional_kelly_strategy
        ]
        results = []

        for strategy in strategies:
            backtest = ModelBacktest(
                data=deterministic_data,
                odds_columns=['odds_1', 'odds_2'],
                outcome_column='outcome',
                date_column='date',
                model=dummy_model,
                initial_bankroll=1000,
                strategy=strategy
            )
            backtest.run()
            final_bankroll = backtest.detailed_results['bt_ending_bankroll'].iloc[-1]
            metrics = backtest.calculate_metrics()
            results.append({
                'strategy': strategy.__class__.__name__,
                'final_bankroll': final_bankroll,
                'roi': metrics['ROI [%]'],
                'win_rate': metrics['Win Rate [%]']
            })

        # Verify we got results for all strategies
        assert len(results) == 3
        assert all(isinstance(r['final_bankroll'], (int, float)) for r in results)
        assert all(isinstance(r['roi'], (int, float)) for r in results)

    def test_prediction_backtest_complete_workflow(self):
        """Test complete workflow with PredictionBacktest from data to metrics."""
        # Create data with predictions
        data = pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=10),
            'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2, 1.9, 2.3, 1.7, 2.1, 2.0],
            'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.4, 2.0, 1.8],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
            'prediction': [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
        })
        
        # Initialize and run backtest
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
        
        # Get results
        detailed = backtest.get_detailed_results()
        bookie = backtest.get_bookie_results()
        
        # Verify results structure
        assert len(detailed) == 10
        assert len(bookie) == 10
        
        # Calculate and verify metrics
        metrics = backtest.calculate_metrics()
        assert metrics['Total Bets'] == 10
        assert metrics['Total Opportunities'] == 10
        assert metrics['Bet Frequency [%]'] == 100.0
        
        # Verify win rate calculation is reasonable (should be between 0 and 100)
        assert 0 <= metrics['Win Rate [%]'] <= 100

    def test_bankroll_evolution_workflow(self, deterministic_data, dummy_model):
        """Test that bankroll evolves correctly through the entire backtest."""
        initial_bankroll = 1000
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=initial_bankroll,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        
        results = backtest.detailed_results
        
        # First bet starts with initial bankroll
        assert results['bt_starting_bankroll'].iloc[0] == initial_bankroll
        
        # Verify each bet's ending bankroll = starting + profit
        for i in range(len(results)):
            expected_ending = results['bt_starting_bankroll'].iloc[i] + results['bt_profit'].iloc[i]
            assert results['bt_ending_bankroll'].iloc[i] == pytest.approx(expected_ending)
        
        # Verify bankroll continuity
        for i in range(1, len(results)):
            assert results['bt_starting_bankroll'].iloc[i] == results['bt_ending_bankroll'].iloc[i-1]
        
        # Verify final bankroll matches last ending bankroll
        metrics = backtest.calculate_metrics()
        assert metrics['Bankroll Final [$]'] == results['bt_ending_bankroll'].iloc[-1]

    def test_metrics_consistency_across_workflow(self, deterministic_data, dummy_model):
        """Test that metrics are consistent when calculated at different points."""
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        
        # Calculate metrics twice
        metrics1 = backtest.calculate_metrics()
        metrics2 = backtest.calculate_metrics()
        
        # Metrics should be identical
        assert metrics1 == metrics2
        
        # Verify key metrics are reasonable
        assert isinstance(metrics1['ROI [%]'], (int, float))
        assert isinstance(metrics1['Total Bets'], int)
        assert metrics1['Total Bets'] > 0
        assert 0 <= metrics1['Bet Frequency [%]'] <= 100

    def test_model_training_and_prediction_workflow(self, deterministic_data):
        """Test that model is trained and used for predictions correctly in workflow."""
        class TrackingDummyClassifier(DummyClassifier):
            """DummyClassifier that tracks fit and predict calls."""
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.fit_called = 0
                self.predict_called = 0
                self.predict_proba_called = 0
            
            def fit(self, X, y):
                self.fit_called += 1
                return super().fit(X, y)
            
            def predict(self, X):
                self.predict_called += 1
                return super().predict(X)
            
            def predict_proba(self, X):
                self.predict_proba_called += 1
                return super().predict_proba(X)
        
        model = TrackingDummyClassifier(strategy="stratified", random_state=42)
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        
        # Verify model was used
        assert model.fit_called > 0, "Model fit should be called during backtest"
        assert model.predict_called > 0, "Model predict should be called during backtest"
        assert model.predict_proba_called > 0, "Model predict_proba should be called during backtest"

