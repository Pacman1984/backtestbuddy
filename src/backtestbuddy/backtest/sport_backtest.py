from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit

from backtestbuddy.metrics.sport_metrics import calculate_all_metrics
from backtestbuddy.plots.sport_plots import plot_backtest, plot_odds_histogram
from backtestbuddy.strategies.sport_strategies import (
    BaseStrategy,
    FixedStake,
    get_default_strategy,
)


class BaseBacktest(ABC):
    """
    Abstract base class for backtesting strategies.

    This class provides a framework for implementing different backtesting
    approaches. It should be subclassed to create specific backtesting strategies.

    Attributes:
        data (pd.DataFrame): The dataset to be used for backtesting.
        odds_columns (List[str]): The names of the columns containing odds for each outcome.
        outcome_column (str): The name of the column containing the actual outcomes.
        date_column (str): The name of the column containing the date information.
        initial_bankroll (float): The initial bankroll for the simulation.
        model (Optional[Any]): The model to be used for predictions, if applicable.
        strategy (BaseStrategy): The betting strategy to be used.
        cv_schema (Any): The cross-validation schema to be used.
        detailed_results (Optional[pd.DataFrame]): Detailed results of the backtest.
    """

    def __init__(self, 
                 data: pd.DataFrame, 
                 odds_columns: List[str],
                 outcome_column: str,
                 date_column: str,
                 initial_bankroll: float = 1000.0,
                 model: Optional[Any] = None, 
                 strategy: Optional[BaseStrategy] = None, 
                 cv_schema: Optional[Any] = None):
        """
        Initialize the BaseBacktest.

        Args:
            data (pd.DataFrame): The dataset to be used for backtesting.
            odds_columns (List[str]): The names of the columns containing odds for each outcome.
            outcome_column (str): The name of the column containing the actual outcomes.
            date_column (str): The name of the column containing the date information.
            initial_bankroll (float): The initial bankroll for the simulation. Defaults to 1000.0.
            model (Optional[Any], optional): The model to be used for predictions. Defaults to None.
            strategy (Optional[BaseStrategy], optional): The betting strategy to be used. 
                Defaults to a default strategy if None is provided.
            cv_schema (Optional[Any], optional): The cross-validation schema to be used. 
                Defaults to TimeSeriesSplit with 5 splits if None is provided.
        """
        self.data = data.sort_values(date_column).reset_index(drop=True)
        self.odds_columns = odds_columns
        self.outcome_column = outcome_column
        self.date_column = date_column
        self.initial_bankroll = initial_bankroll
        self.model = model
        self.strategy = strategy if strategy is not None else get_default_strategy()
        self.cv_schema = cv_schema if cv_schema is not None else TimeSeriesSplit(n_splits=5)
        self.detailed_results: Optional[pd.DataFrame] = None
        self.bookie_results: Optional[pd.DataFrame] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.model_prob_columns: Optional[List[str]] = None

    @abstractmethod
    def run(self) -> None:
        """
        Run the backtest.

        This method should be implemented by subclasses to perform the actual
        backtesting logic.

        Raises:
            NotImplementedError: If the method is not implemented by a subclass.
        """
        pass


    def get_detailed_results(self) -> pd.DataFrame:
        """
        Get the detailed results of the backtest.

        Returns:
            pd.DataFrame: A DataFrame containing detailed results for each bet.

        Raises:
            ValueError: If the backtest has not been run yet.
        """
        if self.detailed_results is None:
            raise ValueError("Backtest has not been run yet. Call run() first.")
        return self.detailed_results

    def get_bookie_results(self) -> pd.DataFrame:
        """
        Get the results of the bookie strategy simulation.

        Returns:
            pd.DataFrame: A DataFrame containing the results of the bookie strategy simulation.

        Raises:
            ValueError: If the backtest has not been run yet.
        """
        if self.bookie_results is None:
            raise ValueError("Backtest has not been run yet. Call run() first.")
        return self.bookie_results

    def _simulate_bet(self, fold: int, index: int, prediction: int, bet_on: int, actual_outcome: int, odds: List[float], current_bankroll: float, stake: float, model_probs: Optional[List[float]] = None, **additional_info: Any) -> Dict[str, Any]:
        """
        Process a bet by placing it, simulating its outcome, and generating the result.

        Args:
            fold (int): The current fold number.
            index (int): The index of the current data point.
            prediction (int): The predicted outcome index.
            bet_on (int): The index of the outcome to bet on (may differ from prediction).
            actual_outcome (int): The actual outcome index (0-based).
            odds (List[float]): The odds for each possible outcome, aligned with prediction/outcome indices.
            current_bankroll (float): The current bankroll before placing the bet.
            stake (float): The stake amount for the bet.
            model_probs (Optional[List[float]]): The model's predicted probabilities for each outcome.
            **additional_info: Additional information to be included in the result.

        Returns:
            Dict[str, Any]: A dictionary containing all the result information, including:
                'bt_index': Index of the current data point.
                'bt_fold': Current fold number.
                'bt_predicted_outcome': Predicted outcome index.
                'bt_bet_on': Index of the outcome bet on (may differ from prediction).
                'bt_actual_outcome': Actual outcome index.
                'bt_starting_bankroll': Bankroll before the bet.
                'bt_ending_bankroll': Bankroll after the bet.
                'bt_stake': Amount staked on the bet.
                'bt_potential_return': Potential return if the bet wins.
                'bt_win': Boolean indicating if the bet won (None if no bet was placed).
                'bt_profit': Profit (positive) or loss (negative) from the bet.
                'bt_roi': Return on Investment as a percentage.
                'bt_odds': The odds for the outcome bet on (None if no bet was placed).
                'bt_odd_{i}': The odds for each possible outcome.
                'bt_model_prob_{i}': The model probability for each possible outcome (if provided).
                'bt_date_column': The date of the event from the specified date column.
                Additional keys from **additional_info will also be included.

        Example:
            For a 2-way betting event (e.g., tennis match):
            odds = [2.0, 1.8]  # [player1_win_odds, player2_win_odds]
            prediction = 0  # Predicting player 1 to win
            bet_on = 0  # Betting on player 1 to win
            actual_outcome = 1  # Player 2 actually won

            For a 3-way betting event (e.g., football match):
            odds = [2.5, 3.0, 2.8]  # [home_win_odds, draw_odds, away_win_odds]
            prediction = 1  # Predicting a draw
            bet_on = 0  # Betting on home team to win
            actual_outcome = 0  # Home team actually won
        """
        # Handle the case where stake is 0 or bet_on is -1 (no bet)
        if stake == 0 or bet_on == -1:
            outcome = {
                "bt_win": None,
                "bt_profit": 0,
                "bt_roi": 0
            }
            potential_return = 0
        else:
            potential_return = stake * odds[bet_on]
            if actual_outcome == bet_on:
                outcome = {
                    "bt_win": True,
                    "bt_profit": potential_return - stake,
                    "bt_roi": (potential_return - stake) / stake * 100
                }
            else:
                outcome = {
                    "bt_win": False,
                    "bt_profit": -stake,
                    "bt_roi": -100
                }

        # Update bankroll
        ending_bankroll = current_bankroll + outcome['bt_profit']

        # Generate the complete result
        result = {
            'bt_index': index,
            'bt_fold': fold,
            'bt_predicted_outcome': prediction,
            'bt_bet_on': bet_on,
            'bt_actual_outcome': actual_outcome,
            'bt_starting_bankroll': current_bankroll,
            'bt_ending_bankroll': ending_bankroll,
            'bt_odds': odds[bet_on] if bet_on != -1 else None,
            **outcome,
            'bt_stake': stake,
            'bt_potential_return': potential_return,
            'bt_date_column': self.data.iloc[index][self.date_column],
        }

        # Add individual odds for each outcome
        for i, odd in enumerate(odds):
            result[f'bt_odd_{i}'] = odd

        if model_probs:
            for i, prob in enumerate(model_probs):
                result[f'bt_model_prob_{i}'] = prob

        # Add any additional information as bt_* columns
        for key, value in additional_info.items():
            result[f'bt_{key}'] = value

        return result


    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics based on the backtest results.
        """
        if self.detailed_results is None:
            raise ValueError("Backtest results are not available. Make sure to run the backtest first.")
        
        self.metrics = calculate_all_metrics(self.detailed_results)
        return self.metrics

    def plot(self):
        """
        Generate and display a plot of the backtest results.

        This method creates a plot showing the bankroll over time, 
        win/loss markers, and ROI for each bet.

        Raises:
            ValueError: If the backtest has not been run yet.
        """
        if self.detailed_results is None:
            raise ValueError("Backtest has not been run yet. Call run() first.")
        
        fig = plot_backtest(self)
        fig.show()

    def plot_odds_distribution(self, num_bins: Optional[int] = None) -> go.Figure:
        """
        Generate a histogram plot of the odds distribution for the main strategy.
        
        Args:
            num_bins (Optional[int]): The number of bins to use for the histogram. If None, an automatic binning strategy is used.
        
        Returns:
            go.Figure: A Plotly figure object containing the odds histogram.
        """
        return plot_odds_histogram(self, num_bins)

    def _simulate_bookie_bet(self, fold: int, index: int, odds: List[float], actual_outcome: int, current_bankroll: float) -> Dict[str, Any]:
        """
        Simulate a bet using the bookie strategy (betting on the outcome with the lowest odds).

        Args:
            fold (int): The current fold number.
            index (int): The index of the current data point.
            odds (List[float]): The odds for each possible outcome.
            actual_outcome (int): The actual outcome index (0-based).
            current_bankroll (float): The current bankroll before placing the bet.

        Returns:
            Dict[str, Any]: A dictionary containing the result information for the bookie bet.
        """
        # For bookie strategy, we still bet on the lowest odds
        bet_on = odds.index(min(odds))
        stake = self.strategy.calculate_stake(odds, current_bankroll)
        potential_return = stake * odds[bet_on]

        if stake == 0:
            win = None
            profit = 0
            roi = 0
        else:
            if actual_outcome == bet_on:
                win = True
                profit = potential_return - stake
                roi = (potential_return - stake) / stake * 100
            else:
                win = False
                profit = -stake
                roi = -100

        ending_bankroll = current_bankroll + profit

        result = {
            'bt_index': index,
            'bt_fold': fold,
            'bt_predicted_outcome': bet_on,  # For bookie strategy, prediction is same as bet_on
            'bt_bet_on': bet_on,
            'bt_actual_outcome': actual_outcome,
            'bt_starting_bankroll': current_bankroll,
            'bt_ending_bankroll': ending_bankroll,
            'bt_stake': stake,
            'bt_potential_return': potential_return if stake != 0 else 0,
            'bt_win': win,
            'bt_profit': profit,
            'bt_roi': roi,
            'bt_odds': odds[bet_on],  # Add the odds for the bet placed
            'bt_date_column': self.data.iloc[index][self.date_column],
        }

        # Add individual odds for each outcome
        for i, odd in enumerate(odds):
            result[f'bt_odd_{i}'] = odd

        return result


class ModelBacktest(BaseBacktest):
    """
    A backtester class for strategies that use a predictive model.

    This class implements the backtesting logic for strategies where a model
    is used to make predictions before applying the betting strategy.

    Attributes:
        Inherits all attributes from BaseBacktest.
        model (Any): The predictive model to be used in the backtest.
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 odds_columns: List[str],
                 outcome_column: str,
                 date_column: str,
                 model: Any,
                 initial_bankroll: float = 1000.0,
                 strategy: Optional[BaseStrategy] = None, 
                 cv_schema: Optional[Any] = None):
        """
        Initialize the ModelBacktester.

        Args:
            data (pd.DataFrame): The dataset to be used for backtesting.
            odds_columns (List[str]): The names of the columns containing odds for each outcome.
            outcome_column (str): The name of the column containing the actual outcomes.
            date_column (str): The name of the column containing the date information.
            model (Any): The predictive model to be used.
            initial_bankroll (float): The initial bankroll for the simulation. Defaults to 1000.0.
            strategy (Optional[BaseStrategy], optional): The betting strategy to be used.
            cv_schema (Optional[Any], optional): The cross-validation schema to be used.
        """
        super().__init__(data, odds_columns, outcome_column, date_column, initial_bankroll, model, strategy, cv_schema)

    def run(self) -> None:
        """
        Run the model-based backtest.

        This method implements the backtesting logic for a model-based strategy.
        It uses the model to make predictions, applies the betting strategy,
        and populates self.detailed_results with the outcomes.
        """
        all_results = []
        bookie_results = []
        current_bankroll = self.initial_bankroll
        bookie_bankroll = self.initial_bankroll

        # Define feature columns, excluding date and outcome columns
        feature_columns = [col for col in self.data.columns if col not in [self.date_column, self.outcome_column]]

        for fold, (train_index, test_index) in enumerate(self.cv_schema.split(self.data)):
            X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
            y_train, y_test = X_train[self.outcome_column], X_test[self.outcome_column]

            # Train the model using only feature columns
            self.model.fit(X_train[feature_columns], y_train)

            # Make predictions using only feature columns
            predictions = self.model.predict(X_test[feature_columns])
            probabilities = self.model.predict_proba(X_test[feature_columns])

            for i, (prediction, probs) in enumerate(zip(predictions, probabilities)):
                odds = X_test.iloc[i][self.odds_columns].tolist()
                actual_outcome = y_test.iloc[i]

                
                # Ensure probs match the number of odds
                if len(probs) != len(odds):
                    raise ValueError(f"Number of probabilities ({len(probs)}) doesn't match number of odds ({len(odds)})")
                
                # Convert probs to a list if it's not already
                probs_list = probs.tolist() if isinstance(probs, np.ndarray) else list(probs)
                
                bet_details = self.strategy.get_bet_details(odds, current_bankroll, model_probs=probs_list, prediction=prediction)
                stake, bet_on, additional_info = bet_details

                result = self._simulate_bet(fold, test_index[i], prediction, bet_on, actual_outcome, odds, current_bankroll, stake, model_probs=probs_list, **additional_info)
                current_bankroll = result['bt_ending_bankroll']
                
                # Add all original features to the result
                result.update(X_test.iloc[i].to_dict())
                all_results.append(result)

                # Simulate bookie bet
                bookie_result = self._simulate_bookie_bet(fold, test_index[i], odds, actual_outcome, bookie_bankroll)
                bookie_bankroll = bookie_result['bt_ending_bankroll']
                
                # Add all original features to the bookie result
                bookie_result.update(X_test.iloc[i].to_dict())
                bookie_results.append(bookie_result)

        self.detailed_results = pd.DataFrame(all_results)
        self.bookie_results = pd.DataFrame(bookie_results)
       


class PredictionBacktest(BaseBacktest):
    """
    A backtester class for strategies that use pre-computed predictions.

    This class implements the backtesting logic for strategies where predictions
    are already available in the dataset, and only the betting strategy needs to be applied.
    Unlike the ModelBacktest, this class doesn't use cross-validation (cv_schema) because:
    1. The predictions are pre-computed, so there's no need to train a model on different folds.
    2. We assume the predictions were generated using appropriate methods to avoid look-ahead bias.
    3. The entire dataset can be used sequentially, simulating a real-world betting scenario.

    Attributes:
        Inherits all attributes from BaseBacktest.
        prediction_column (str): The name of the column in the dataset that contains the predictions.
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 odds_columns: List[str],
                 outcome_column: str,
                 date_column: str,
                 prediction_column: str,
                 initial_bankroll: float = 1000.0,
                 strategy: Optional[BaseStrategy] = None,
                 model_prob_columns: Optional[List[str]] = None):
        """
        Initialize the PredictionBacktester.

        This initializer doesn't include a cv_schema parameter because the PredictionBacktest
        doesn't require cross-validation. The predictions are assumed to be pre-computed correctly,
        taking into account any necessary time-based splits or other methodologies to prevent data leakage.

        Args:
            data (pd.DataFrame): The dataset to be used for backtesting. Should include pre-computed predictions.
            odds_columns (List[str]): The names of the columns containing odds for each outcome.
            outcome_column (str): The name of the column containing the actual outcomes.
            date_column (str): The name of the column containing the date information.
            prediction_column (str): The name of the column in the dataset that contains the predictions.
            initial_bankroll (float): The initial bankroll for the simulation. Defaults to 1000.0.
            strategy (Optional[BaseStrategy], optional): The betting strategy to be used.
            model_prob_columns (Optional[List[str]], optional): The names of the columns in the dataset that contain the model probabilities.

        Raises:
            ValueError: If the specified prediction_column is not found in the dataset.
            ValueError: If the strategy requires model probabilities but model_prob_columns is not provided.
        """
        super().__init__(data, odds_columns, outcome_column, date_column, initial_bankroll, None, strategy, None)
        
        if prediction_column not in data.columns:
            raise ValueError(f"Prediction column '{prediction_column}' not found in the dataset.")
        
        self.prediction_column = prediction_column
        self.model_prob_columns = model_prob_columns

        # Check if the strategy requires model probabilities
        if hasattr(self.strategy, 'requires_probabilities') and self.strategy.requires_probabilities:
            if not model_prob_columns:
                raise ValueError("The selected strategy requires model probabilities, but model_prob_columns was not provided.")
            for col in model_prob_columns:
                if col not in data.columns:
                    raise ValueError(f"Model probability column '{col}' not found in the dataset.")

    def run(self) -> None:
        """
        Run the prediction-based backtest and bookie simulation.

        This method implements the backtesting logic for a strategy based on pre-computed predictions
        and also simulates a bookie strategy. It populates self.detailed_results with the outcomes
        of the prediction-based strategy and self.bookie_results with the outcomes of the bookie strategy.

        Unlike the ModelBacktest, this method doesn't use cross-validation splits. Instead, it processes
        the entire dataset sequentially, which is appropriate when:
        1. Predictions are pre-computed and assumed to be generated without look-ahead bias.
        2. We want to simulate a continuous betting scenario, where each bet is placed based on
           information available up to that point in time.
        3. The dataset is already arranged in chronological order, representing the actual sequence
           of betting opportunities.

        This approach allows for a more realistic simulation of a betting strategy's performance
        over time, as it would be applied in a real-world scenario.
        """
        all_results = []
        bookie_results = []
        current_bankroll = self.initial_bankroll
        bookie_bankroll = self.initial_bankroll

        for i, row in self.data.iterrows():
            prediction = row[self.prediction_column]
            actual_outcome = row[self.outcome_column]
            odds = row[self.odds_columns].tolist()
            model_probs = row[self.model_prob_columns].tolist() if self.model_prob_columns else None

            bet_details = self.strategy.get_bet_details(odds, current_bankroll, model_probs=model_probs, prediction=prediction)
            stake, bet_on, additional_info = bet_details

            result = self._simulate_bet(0, i, prediction, bet_on, actual_outcome, odds, current_bankroll, stake, model_probs=model_probs, **additional_info)
            current_bankroll = result['bt_ending_bankroll']
            
            result.update(row.to_dict())
            all_results.append(result)

            bookie_result = self._simulate_bookie_bet(0, i, odds, actual_outcome, bookie_bankroll)
            bookie_bankroll = bookie_result['bt_ending_bankroll']
            
            bookie_result.update(row.to_dict())
            bookie_results.append(bookie_result)

        self.detailed_results = pd.DataFrame(all_results)
        self.bookie_results = pd.DataFrame(bookie_results)