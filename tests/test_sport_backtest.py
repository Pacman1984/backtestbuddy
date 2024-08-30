import pytest
import pandas as pd
import numpy as np
from sklearn.dummy import DummyClassifier
from src.backtest.sport_backtest import BaseBacktest, ModelBacktest, PredictionBacktest
from src.strategies.sport_strategies import FixedStake



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
        return DummyClassifier(strategy="most_frequent")

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
            'ROI [%]', 'Total Profit [$]', 'Bankroll Final [$]', 'Bankroll Peak [$]', 'Bankroll Valley [$]',
            'Sharpe Ratio [-]', 'Sortino Ratio [-]', 'Calmar Ratio [-]',
            'Max Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration [bets]',
            'Avg. Drawdown Duration [bets]', 'Median Drawdown Duration [bets]',
            'Win Rate [%]', 'Average Odds [-]', 'Highest Winning Odds [-]', 'Highest Losing Odds [-]',
            'Average Stake [$]', 'Best Bet [$]', 'Worst Bet [$]',
            'Total Bets'
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