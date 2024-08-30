from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit

from src.metrics.sport_metrics import calculate_all_metrics
from src.plots.sport_plots import plot_backtest, plot_odds_histogram
from src.strategies.sport_strategies import (BaseStrategy, FixedStake,
                                             get_default_strategy)


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

    def _simulate_bet(self, fold: int, index: int, prediction: int, actual_outcome: int, odds: List[float], current_bankroll: float, **kwargs: Any) -> Dict[str, Any]:
        """
        Process a bet by placing it, simulating its outcome, and generating the result.

        Args:
            fold (int): The current fold number.
            index (int): The index of the current data point.
            prediction (int): The predicted outcome index (0-based).
            actual_outcome (int): The actual outcome index (0-based).
            odds (List[float]): The odds for each possible outcome, aligned with prediction/outcome indices.
            current_bankroll (float): The current bankroll before placing the bet.
            **kwargs: Additional data that might be used in custom betting strategies.

        Returns:
            Dict[str, Any]: A dictionary containing all the result information, including:
                'bt_index': Index of the current data point.
                'bt_fold': Current fold number.
                'bt_predicted_outcome': Predicted outcome index.
                'bt_actual_outcome': Actual outcome index.
                'bt_starting_bankroll': Bankroll before the bet.
                'bt_ending_bankroll': Bankroll after the bet.
                'bt_stake': Amount staked on the bet.
                'bt_potential_return': Potential return if the bet wins.
                'bt_win': Boolean indicating if the bet won.
                'bt_profit': Profit (positive) or loss (negative) from the bet.
                'bt_roi': Return on Investment as a percentage.
                'bt_odds': The odds for the predicted outcome.

        Example:
            For a 2-way betting event (e.g., tennis match):
            odds = [2.0, 1.8]  # [player1_win_odds, player2_win_odds]
            prediction = 0  # Predicting player 1 to win
            actual_outcome = 1  # Player 2 actually won

            For a 3-way betting event (e.g., football match):
            odds = [2.5, 3.0, 2.8]  # [home_win_odds, draw_odds, away_win_odds]
            prediction = 1  # Predicting a draw
            actual_outcome = 0  # Home team actually won
        """
        # Place the bet
        stake = self.strategy.calculate_stake(odds[prediction], current_bankroll, **kwargs)
        potential_return = stake * odds[prediction]
        bet = {"bt_stake": stake, "bt_potential_return": potential_return}

        # Simulate the outcome of the bet
        if actual_outcome == prediction:
            outcome = {
                "bt_win": True,
                "bt_profit": bet["bt_potential_return"] - bet["bt_stake"],
                "bt_roi": (bet["bt_potential_return"] - bet["bt_stake"]) / bet["bt_stake"] * 100
            }
        else:
            outcome = {
                "bt_win": False,
                "bt_profit": -bet["bt_stake"],
                "bt_roi": -100
            }

        # Update bankroll
        ending_bankroll = current_bankroll + outcome['bt_profit']

        # Generate the complete result
        result = {
            'bt_index': index,
            'bt_fold': fold,
            'bt_predicted_outcome': prediction,
            'bt_actual_outcome': actual_outcome,
            'bt_starting_bankroll': current_bankroll,
            'bt_ending_bankroll': ending_bankroll,
            'bt_odds': odds[prediction],
            'bt_date_column': self.data.iloc[index][self.date_column],  # Add this line
            **bet,
            **outcome
        }

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
        prediction = odds.index(min(odds))  # Bet on the outcome with the lowest odds
        stake = self.strategy.calculate_stake(odds[prediction], current_bankroll)
        potential_return = stake * odds[prediction]

        if actual_outcome == prediction:
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
            'bt_predicted_outcome': prediction,
            'bt_actual_outcome': actual_outcome,
            'bt_starting_bankroll': current_bankroll,
            'bt_ending_bankroll': ending_bankroll,
            'bt_stake': stake,
            'bt_potential_return': potential_return,
            'bt_win': win,
            'bt_profit': profit,
            'bt_roi': roi,
            'bt_odds': odds[prediction],
            'bt_date_column': self.data.iloc[index][self.date_column],
        }

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

            # Simulate bets
            for i, prediction in enumerate(predictions):
                odds = X_test.iloc[i][self.odds_columns].tolist()
                actual_outcome = y_test.iloc[i]

                # Simulate bet based on prediction
                result = self._simulate_bet(fold, test_index[i], prediction, actual_outcome, odds, current_bankroll)
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
                 strategy: Optional[BaseStrategy] = None):
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

        Raises:
            ValueError: If the specified prediction_column is not found in the dataset.
        """
        super().__init__(data, odds_columns, outcome_column, date_column, initial_bankroll, None, strategy, None)
        
        if prediction_column not in data.columns:
            raise ValueError(f"Prediction column '{prediction_column}' not found in the dataset.")
        
        self.prediction_column = prediction_column

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

            # Simulate bet based on prediction
            result = self._simulate_bet(0, i, prediction, actual_outcome, odds, current_bankroll)
            current_bankroll = result['bt_ending_bankroll']
            
            # Add all original features to the result
            result.update(row.to_dict())
            all_results.append(result)

            # Simulate bookie bet
            bookie_result = self._simulate_bookie_bet(0, i, odds, actual_outcome, bookie_bankroll)
            bookie_bankroll = bookie_result['bt_ending_bankroll']
            
            # Add all original features to the bookie result
            bookie_result.update(row.to_dict())
            bookie_results.append(bookie_result)

        self.detailed_results = pd.DataFrame(all_results)
        self.bookie_results = pd.DataFrame(bookie_results)