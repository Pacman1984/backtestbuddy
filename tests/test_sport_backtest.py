import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from backtestbuddy.backtest.sport_backtest import BaseBacktest, ModelBacktest, PredictionBacktest
from backtestbuddy.strategies.sport_strategies import FixedStake, KellyCriterion


class TestModelBacktest:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=20),
            'feature_1': np.random.randint(1, 10, 20),
            'feature_2': np.random.randint(1, 10, 20),
            'odds_1': np.random.uniform(1.5, 3.0, 20),
            'odds_2': np.random.uniform(1.5, 3.0, 20),
            'outcome': np.random.randint(0, 2, 20)
        })

    @pytest.fixture
    def dummy_model(self):
        return DummyClassifier(strategy="stratified", random_state=42)

    @pytest.fixture
    def kelly_strategy(self):
        return KellyCriterion(downscaling=1.0)

    @pytest.fixture
    def fractional_kelly_strategy(self):
        return KellyCriterion(downscaling=0.5)

    @pytest.fixture
    def backtest(self, sample_data, dummy_model):
        return ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )

    def test_initialization(self, sample_data, dummy_model):
        backtest = ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model
        )
        assert backtest.data.equals(sample_data)
        assert backtest.model == dummy_model

    def test_run_method(self, backtest):
        backtest.run()
        assert backtest.detailed_results is not None
        assert backtest.bookie_results is not None
        assert len(backtest.detailed_results) > 0
        assert len(backtest.bookie_results) > 0

    def test_fixed_stake_strategy(self, sample_data, dummy_model):
        backtest = ModelBacktest(
            data=sample_data,
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
        assert (backtest.detailed_results['bt_stake'] == 100).all()

    def test_fixed_stake_percentage_strategy(self, sample_data, dummy_model):
        # Test with percentage stake (0.5 = 50%)
        backtest = ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=FixedStake(stake=0.5)  # 50% stake
        )
        backtest.run()
        
        # Check that stakes are correctly calculated as percentage of bankroll
        results = backtest.detailed_results
        
        # First bet should be 50% of 1000 = 500
        assert results['bt_stake'].iloc[0] == 500
        
        # Subsequent bets should be 50% of the current bankroll
        for i in range(1, len(results)):
            expected_stake = results['bt_starting_bankroll'].iloc[i] * 0.5
            assert results['bt_stake'].iloc[i] == pytest.approx(expected_stake)
            
        # All stakes should be less than or equal to the current bankroll
        assert (results['bt_stake'] <= results['bt_starting_bankroll']).all()

    def test_kelly_criterion_strategy(self, sample_data, dummy_model, kelly_strategy):
        backtest = ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=kelly_strategy
        )
        backtest.run()
        assert backtest.detailed_results is not None
        assert 'bt_stake' in backtest.detailed_results.columns
        assert not (backtest.detailed_results['bt_stake'] == 0).all()
        assert (backtest.detailed_results['bt_stake'] <= backtest.detailed_results['bt_starting_bankroll']).all()

    def test_fractional_kelly_strategy(self, sample_data, dummy_model, fractional_kelly_strategy):
        backtest = ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=fractional_kelly_strategy
        )
        backtest.run()
        assert backtest.detailed_results is not None
        assert 'bt_stake' in backtest.detailed_results.columns
        assert not (backtest.detailed_results['bt_stake'] == 0).all()
        assert (backtest.detailed_results['bt_stake'] <= 0.5 * backtest.detailed_results['bt_starting_bankroll']).all()

    def test_strategy_comparison(self, sample_data, dummy_model, kelly_strategy, fractional_kelly_strategy):
        strategies = [
            FixedStake(stake=100),
            kelly_strategy,
            fractional_kelly_strategy
        ]
        results = []

        for strategy in strategies:
            backtest = ModelBacktest(
                data=sample_data,
                odds_columns=['odds_1', 'odds_2'],
                outcome_column='outcome',
                date_column='date',
                model=dummy_model,
                initial_bankroll=1000,
                strategy=strategy
            )
            backtest.run()
            final_bankroll = backtest.detailed_results['bt_ending_bankroll'].iloc[-1]
            results.append((strategy.__class__.__name__, final_bankroll))

        assert len(results) == 3
        assert all(isinstance(r[1], (int, float)) for r in results)

    def test_model_probabilities(self, sample_data):
        class ProbabilityDummyClassifier(DummyClassifier):
            def predict_proba(self, X):
                return np.random.random((len(X), 2))

        prob_model = ProbabilityDummyClassifier(strategy="stratified")
        backtest = ModelBacktest(
            data=sample_data,
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

    def test_kelly_criterion_bet_sizing(self, sample_data, dummy_model, kelly_strategy):
        backtest = ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=kelly_strategy
        )
        backtest.run()
        stakes = backtest.detailed_results['bt_stake']
        bankrolls = backtest.detailed_results['bt_starting_bankroll']
        assert (stakes >= 0).all()  # Kelly should never suggest negative stakes
        assert (stakes <= bankrolls).all()  # Kelly should never suggest betting more than the bankroll

    def test_fractional_kelly_vs_full_kelly(self, sample_data, dummy_model, kelly_strategy, fractional_kelly_strategy):
        full_kelly_backtest = ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=kelly_strategy
        )
        fractional_kelly_backtest = ModelBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            model=dummy_model,
            initial_bankroll=1000,
            strategy=fractional_kelly_strategy
        )
        full_kelly_backtest.run()
        fractional_kelly_backtest.run()
        
        full_kelly_stakes = full_kelly_backtest.detailed_results['bt_stake']
        fractional_kelly_stakes = fractional_kelly_backtest.detailed_results['bt_stake']
        
        # Print out the stakes for debugging
        print("Full Kelly stakes:", full_kelly_stakes)
        print("Fractional Kelly stakes:", fractional_kelly_stakes)

        # Check if stakes are close enough, allowing for small floating-point differences
        # np.testing.assert_array_less(fractional_kelly_stakes, full_kelly_stakes * 1.01)

        # If you want to keep the original assertion, you can uncomment the line below
        assert (fractional_kelly_stakes <= full_kelly_stakes).all()

    def test_incorrect_predict_proba_output(self, sample_data):
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
            data=sample_data,
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


class TestPredictionBacktest:
    @pytest.fixture
    def sample_data(self):
        return pd.DataFrame({
            'date': pd.date_range(start='2023-01-01', periods=5),
            'odds_1': [1.5, 2.0, 1.8, 2.2, 1.9],
            'odds_2': [2.5, 1.8, 2.2, 1.7, 2.1],
            'outcome': [0, 1, 0, 1, 0],
            'prediction': [0, 1, 1, 0, 0]
        })

    @pytest.fixture
    def backtest(self, sample_data):
        return PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction',
            initial_bankroll=1000,
            strategy=FixedStake(stake=100)
        )

    def test_initialization(self, sample_data):
        backtest = PredictionBacktest(
            data=sample_data,
            odds_columns=['odds_1', 'odds_2'],
            outcome_column='outcome',
            date_column='date',
            prediction_column='prediction'
        )
        assert backtest.data.equals(sample_data)
        assert backtest.prediction_column == 'prediction'

    def test_run_method(self, backtest):
        backtest.run()
        assert backtest.detailed_results is not None
        assert backtest.bookie_results is not None
        assert len(backtest.detailed_results) == 5
        assert len(backtest.bookie_results) == 5

    def test_detailed_results_content(self, backtest):
        backtest.run()
        results = backtest.detailed_results
        expected_columns = ['bt_index', 'bt_predicted_outcome', 'bt_actual_outcome', 
                            'bt_starting_bankroll', 'bt_ending_bankroll', 'bt_stake', 
                            'bt_win', 'bt_profit', 'bt_roi']
        for col in expected_columns:
            assert col in results.columns

    def test_bookie_results_content(self, backtest):
        backtest.run()
        results = backtest.bookie_results
        expected_columns = ['bt_index', 'bt_predicted_outcome', 'bt_actual_outcome', 
                            'bt_starting_bankroll', 'bt_ending_bankroll', 'bt_stake', 
                            'bt_win', 'bt_profit', 'bt_roi']
        for col in expected_columns:
            assert col in results.columns

    def test_get_detailed_results(self, backtest):
        backtest.run()
        results = backtest.get_detailed_results()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5

    def test_get_bookie_results(self, backtest):
        backtest.run()
        results = backtest.get_bookie_results()
        assert isinstance(results, pd.DataFrame)
        assert len(results) == 5

    def test_calculate_metrics(self, backtest):
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

    def test_plot_method(self, backtest):
        backtest.run()
        # This test just checks if the plot method runs without error
        backtest.plot()

    def test_missing_prediction_column(self, sample_data):
        with pytest.raises(ValueError):
            PredictionBacktest(
                data=sample_data,
                odds_columns=['odds_1', 'odds_2'],
                outcome_column='outcome',
                date_column='date',
                prediction_column='non_existent_column'
            )

    def test_get_results_before_run(self, backtest):
        with pytest.raises(ValueError):
            backtest.get_detailed_results()
        with pytest.raises(ValueError):
            backtest.get_bookie_results()

    def test_calculate_metrics_before_run(self, backtest):
        with pytest.raises(ValueError):
            backtest.calculate_metrics()

    def test_plot_before_run(self, backtest):
        with pytest.raises(ValueError):
            backtest.plot()