"""
Unit tests for sport_backtest module.

Tests individual backtest class behaviors with controlled test data.
These are unit tests focusing on specific methods and edge cases.
For end-to-end workflow tests, see tests/integration/test_backtest_workflow.py
"""
import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from backtestbuddy.backtest.sport_backtest import BaseBacktest, ModelBacktest, PredictionBacktest
from backtestbuddy.strategies.sport_strategies import FixedStake, KellyCriterion


class TestModelBacktest:
    """Unit tests for ModelBacktest class."""
    
    @pytest.fixture
    def deterministic_data(self):
        """Create deterministic test data with known values."""
        return pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=12),
            'feature_1': [5, 3, 7, 2, 8, 4, 6, 9, 1, 5, 7, 3],
            'feature_2': [2, 6, 4, 9, 1, 8, 3, 7, 5, 2, 4, 6],
            'odds_1': [2.0, 1.8, 2.5, 1.5, 2.2, 1.9, 2.1, 1.7, 2.3, 2.0, 1.6, 2.4],
            'odds_2': [1.8, 2.2, 1.6, 2.8, 1.9, 2.1, 1.7, 2.3, 1.8, 2.0, 2.5, 1.6],
            'outcome': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        })

    @pytest.fixture
    def dummy_model(self):
        """Create a dummy model for testing."""
        return DummyClassifier(strategy="stratified", random_state=42)

    def test_initialization(self, deterministic_data, dummy_model):
        """Test that ModelBacktest initializes correctly with required parameters."""
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model
        )
        assert backtest.data.equals(deterministic_data)
        assert backtest.model == dummy_model
        assert backtest.odds_columns == ['odds_1', 'odds_2']
        assert backtest.outcome_column == 'outcome'
        assert backtest.date_column == 'date'
        assert backtest.initial_bankroll == 1000.0  # default value

    def test_run_method_produces_results(self, deterministic_data, dummy_model):
        """Test that run() method produces non-null results."""
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
        assert backtest.detailed_results is not None
        assert backtest.bookie_results is not None
        assert len(backtest.detailed_results) > 0
        assert len(backtest.bookie_results) > 0

    def test_fixed_stake_strategy_uses_constant_stake(self, deterministic_data, dummy_model):
        """Test that FixedStake strategy uses constant stake amount."""
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
        assert backtest.detailed_results is not None
        assert 'bt_stake' in backtest.detailed_results.columns
        # All non-zero stakes should be 100
        non_zero_stakes = backtest.detailed_results[backtest.detailed_results['bt_stake'] > 0]['bt_stake']
        assert (non_zero_stakes == 100).all()

    def test_fixed_stake_percentage_strategy(self, deterministic_data, dummy_model):
        """Test that percentage-based FixedStake calculates stakes correctly."""
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=0.5)  # 50% stake
        )
        backtest.run()
        
        results = backtest.detailed_results
        
        # First bet should be 50% of 1000 = 500
        assert results['bt_stake'].iloc[0] == 500
        
        # Subsequent bets should be 50% of the current bankroll
        for i in range(1, len(results)):
            expected_stake = results['bt_starting_bankroll'].iloc[i] * 0.5
            assert results['bt_stake'].iloc[i] == pytest.approx(expected_stake)
            
        # All stakes should be less than or equal to the current bankroll
        assert (results['bt_stake'] <= results['bt_starting_bankroll']).all()

    def test_kelly_criterion_strategy_respects_bankroll(self, deterministic_data, dummy_model):
        """Test that Kelly Criterion never suggests betting more than bankroll."""
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=KellyCriterion(downscaling=1.0)
        )
        backtest.run()
        stakes = backtest.detailed_results['bt_stake']
        bankrolls = backtest.detailed_results['bt_starting_bankroll']
        assert (stakes >= 0).all()  # Kelly should never suggest negative stakes
        assert (stakes <= bankrolls).all()  # Kelly should never suggest betting more than the bankroll

    def test_fractional_kelly_vs_full_kelly(self, deterministic_data, dummy_model):
        """Test that fractional Kelly produces smaller or equal stakes than full Kelly."""
        full_kelly_backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=KellyCriterion(downscaling=1.0)
        )
        fractional_kelly_backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=KellyCriterion(downscaling=0.5)
        )
        full_kelly_backtest.run()
        fractional_kelly_backtest.run()
        
        full_kelly_stakes = full_kelly_backtest.detailed_results['bt_stake']
        fractional_kelly_stakes = fractional_kelly_backtest.detailed_results['bt_stake']
        
        # Fractional Kelly stakes should be less than or equal to full Kelly
        assert (fractional_kelly_stakes <= full_kelly_stakes * 1.01).all()  # Allow small floating-point tolerance

    def test_incorrect_predict_proba_output_raises_error(self, deterministic_data):
        """Test that incorrect model output shape raises descriptive error."""
        class IncorrectProbabilityClassifier:
            def fit(self, X, y):
                pass

            def predict(self, X):
                return np.zeros(len(X))

            def predict_proba(self, X):
                # Return incorrect number of probabilities
                return np.random.random((len(X), 3))  # 3 probabilities instead of 2

        incorrect_model = IncorrectProbabilityClassifier()
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=incorrect_model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )

        with pytest.raises(ValueError) as excinfo:
            backtest.run()

        assert "The model's predict_proba output shape (3) doesn't match the number of odds columns (2)" in str(excinfo.value)
        assert "Example of correct output:" in str(excinfo.value)
        assert "Example of incorrect output:" in str(excinfo.value)

    def test_model_probabilities_are_stored(self, deterministic_data):
        """Test that model probabilities are correctly stored in results."""
        class ProbabilityDummyClassifier(DummyClassifier):
            def predict_proba(self, X):
                # Return fixed probabilities for deterministic testing
                return np.array([[0.6, 0.4]] * len(X))

        prob_model = ProbabilityDummyClassifier(strategy="stratified")
        backtest = ModelBacktest(
            data=deterministic_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=prob_model,
            initial_bankroll=1000,
            strategy=KellyCriterion()
        )
        backtest.run()
        assert backtest.detailed_results is not None
        assert 'bt_model_prob_0' in backtest.detailed_results.columns
        assert 'bt_model_prob_1' in backtest.detailed_results.columns


class TestPredictionBacktest:
    """Unit tests for PredictionBacktest class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create deterministic sample data for testing."""
        return pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'odds_1': [1.5, 2.0, 1.8, 2.2, 1.9],
            'odds_2': [2.5, 1.8, 2.2, 1.7, 2.1],
            'outcome': [0, 1, 0, 1, 0],
            'prediction': [0, 1, 1, 0, 0]
        })

    def test_initialization(self, sample_data):
        """Test that PredictionBacktest initializes correctly."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction'
        )
        assert backtest.data.equals(sample_data)
        assert backtest.prediction_column == 'prediction'

    def test_run_method_produces_results(self, sample_data):
        """Test that run() method produces results with correct length."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        assert backtest.detailed_results is not None
        assert backtest.bookie_results is not None
        assert len(backtest.detailed_results) == 5
        assert len(backtest.bookie_results) == 5

    def test_detailed_results_content(self, sample_data):
        """Test that detailed results contain expected columns."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        results = backtest.detailed_results
        expected_columns = ['bt_index', 'bt_predicted_outcome', 'bt_actual_outcome', 
                            'bt_starting_bankroll', 'bt_ending_bankroll', 'bt_stake', 
                            'bt_win', 'bt_profit', 'bt_roi']
        for col in expected_columns:
            assert col in results.columns

    def test_bookie_results_content(self, sample_data):
        """Test that bookie results contain expected columns."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        results = backtest.bookie_results
        expected_columns = ['bt_index', 'bt_predicted_outcome', 'bt_actual_outcome', 
                            'bt_starting_bankroll', 'bt_ending_bankroll', 'bt_stake', 
                            'bt_win', 'bt_profit', 'bt_roi']
        for col in expected_columns:
            assert col in results.columns

    def test_get_detailed_results(self, sample_data):
        """Test that get_detailed_results() returns DataFrame after run."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        results = backtest.get_detailed_results()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5

    def test_get_bookie_results(self, sample_data):
        """Test that get_bookie_results() returns DataFrame after run."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        results = backtest.get_bookie_results()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5

    def test_calculate_metrics(self, sample_data):
        """Test that calculate_metrics() returns expected metric keys."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        metrics = backtest.calculate_metrics()
        assert isinstance(metrics, dict)
        expected_metrics = [
            'Backtest Start Date', 'Backtest End Date', 'Backtest Duration',
            'ROI [%]', 'Total Profit [$]', 'Bankroll Final [$]', 'Bankroll Peak [$]', 'Bankroll Valley [$]',
            'Sharpe Ratio [-]', 'Sortino Ratio [-]', 'Calmar Ratio [-]',
            'Max Drawdown [%]', 'Max. Drawdown Duration [bets]',
            'Win Rate [%]', 'Average Odds [-]', 'Highest Winning Odds [-]', 'Highest Losing Odds [-]',
            'Average Stake [$]', 'Best Bet [$]', 'Worst Bet [$]',
            'Total Bets', 'Total Opportunities', 'Bet Frequency [%]'
        ]
        for metric in expected_metrics:
            assert metric in metrics, f"Expected metric '{metric}' not found in calculated metrics"

    def test_plot_method_runs_without_error(self, sample_data):
        """Test that plot() method runs without error after backtest."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )
        backtest.run()
        # This test just checks if the plot method runs without error
        backtest.plot()

    def test_missing_prediction_column_raises_error(self, sample_data):
        """Test that missing prediction column raises ValueError."""
        with pytest.raises(ValueError):
            PredictionBacktest(
                data=sample_data,
                odds_columns=['odds_1', 'odds_2'],
                outcome_column='outcome',
                date_column='date',
                prediction_column='non_existent_column'
            )

    def test_get_results_before_run_raises_error(self, sample_data):
        """Test that calling get_results before run() raises ValueError."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction'
        )
        with pytest.raises(ValueError):
            backtest.get_detailed_results()
        with pytest.raises(ValueError):
            backtest.get_bookie_results()

    def test_calculate_metrics_before_run_raises_error(self, sample_data):
        """Test that calling calculate_metrics before run() raises ValueError."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction'
        )
        with pytest.raises(ValueError):
            backtest.calculate_metrics()

    def test_plot_before_run_raises_error(self, sample_data):
        """Test that calling plot before run() raises ValueError."""
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction'
        )
        with pytest.raises(ValueError):
            backtest.plot()

