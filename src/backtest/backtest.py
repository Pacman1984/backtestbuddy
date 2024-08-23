from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.strategies.base import BaseStrategy, FixedStake, get_default_strategy


class BaseBacktest(ABC):
    """
    Abstract base class for backtesting strategies.

    This class provides a framework for implementing different backtesting
    approaches. It should be subclassed to create specific backtesting strategies.

    Attributes:
        data (pd.DataFrame): The dataset to be used for backtesting.
        odds_columns (List[str]): The names of the columns containing odds for each outcome.
        outcome_column (str): The name of the column containing the actual outcomes.
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
            initial_bankroll (float): The initial bankroll for the simulation. Defaults to 1000.0.
            model (Optional[Any], optional): The model to be used for predictions. Defaults to None.
            strategy (Optional[BaseStrategy], optional): The betting strategy to be used. 
                Defaults to a default strategy if None is provided.
            cv_schema (Optional[Any], optional): The cross-validation schema to be used. 
                Defaults to TimeSeriesSplit with 5 splits if None is provided.
        """
        self.data = data
        self.odds_columns = odds_columns
        self.outcome_column = outcome_column
        self.initial_bankroll = initial_bankroll
        self.model = model
        self.strategy = strategy if strategy is not None else get_default_strategy()
        self.cv_schema = cv_schema if cv_schema is not None else TimeSeriesSplit(n_splits=5)
        self.detailed_results: Optional[pd.DataFrame] = None

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

        
    def _simulate_bet(self, fold: int, index: int, prediction: int, actual_outcome: int, odds: List[float], current_bankroll: float, **kwargs: Any) -> Dict[str, Any]:
        """
        Process a bet by placing it, simulating its outcome, and generating the result.

        Args:
            fold (int): The current fold number.
            index (int): The index of the current data point.
            prediction (int): The predicted outcome.
            actual_outcome (int): The actual outcome.
            odds (List[float]): The odds for each outcome.
            current_bankroll (float): The current bankroll before placing the bet.
            **kwargs: Additional data that might be used in custom betting strategies.


        Returns:
            Dict[str, Any]: A dictionary containing all the result information, including bet details and outcomes.
                keys: 'bt_index', 'bt_fold', 'bt_predicted_outcome', 'bt_actual_outcome', 'bt_starting_bankroll', 
                'bt_ending_bankroll', 'bt_stake', 'bt_potential_return', 'bt_win', 'bt_profit', 'bt_roi'
        """
        # Place the bet
        stake = self.strategy.calculate_stake(odds[prediction], current_bankroll, **kwargs) # Calculate stake based on the strategy
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
            **bet,
            **outcome
        }

        return result

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics based on the backtest results.

        This method computes relevant performance metrics after the backtest has been run.
        It assumes that self.results has been populated by the run() method.

        Returns:
            Dict[str, Any]: A dictionary containing the calculated metrics.

        Raises:
            ValueError: If calculate_metrics is called before run() or if results are empty.
        """
        if self.results is None:
            raise ValueError("Backtest results are not available. Make sure to run the backtest first.")

        metrics = {}

        # Calculate ROI
        initial_bankroll = self.initial_bankroll
        final_bankroll = self.results['final_bankroll']
        metrics['ROI'] = ((final_bankroll - initial_bankroll) / initial_bankroll) * 100

        # Calculate win rate
        total_bets = self.results['total_bets']
        winning_bets = self.results['winning_bets']
        metrics['win_rate'] = (winning_bets / total_bets) * 100 if total_bets else 0

        # Calculate average odds
        metrics['average_odds'] = self.results['total_odds'] / total_bets if total_bets else 0

        # Calculate profit
        metrics['total_profit'] = final_bankroll - initial_bankroll

        # Calculate average stake
        metrics['average_stake'] = self.results['total_stake'] / total_bets if total_bets else 0

        return metrics


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
            model (Any): The predictive model to be used.
            initial_bankroll (float): The initial bankroll for the simulation. Defaults to 1000.0.
            strategy (Optional[BaseStrategy], optional): The betting strategy to be used.
            cv_schema (Optional[Any], optional): The cross-validation schema to be used.
        """
        super().__init__(data, odds_columns, outcome_column, initial_bankroll, model, strategy, cv_schema)

    def run(self) -> None:
        """
        Run the model-based backtest.

        This method implements the backtesting logic for a model-based strategy.
        It uses the model to make predictions, applies the betting strategy,
        and populates self.detailed_results with the outcomes.
        """
        all_results = []
        current_bankroll = self.initial_bankroll

        for fold, (train_index, test_index) in enumerate(self.cv_schema.split(self.data)):
            X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
            y_train, y_test = X_train[self.outcome_column], X_test[self.outcome_column]

            # Train the model (including odds columns as features)
            self.model.fit(X_train.drop([self.outcome_column], axis=1), y_train)

            # Make predictions (including odds columns as features)
            predictions = self.model.predict(X_test.drop([self.outcome_column], axis=1))

            # Simulate bets
            for i, prediction in enumerate(predictions):
                odds = X_test.iloc[i][self.odds_columns].tolist()
                actual_outcome = y_test.iloc[i]

                result = self._simulate_bet(fold, i, prediction, actual_outcome, odds, current_bankroll)
                current_bankroll = result['bt_ending_bankroll']
                
                # Add all original features to the result
                result.update(X_test.iloc[i].to_dict())

                all_results.append(result)

        self.detailed_results = pd.DataFrame(all_results)


class PredictionBacktest(BaseBacktest):
    """
    A backtester class for strategies that use pre-computed predictions.

    This class implements the backtesting logic for strategies where predictions
    are already available in the dataset, and only the betting strategy needs to be applied.

    Attributes:
        Inherits all attributes from BaseBacktest.
        prediction_column (str): The name of the column in the dataset that contains the predictions.
    """

    def __init__(self, 
                 data: pd.DataFrame,
                 odds_columns: List[str],
                 outcome_column: str,
                 prediction_column: str,
                 initial_bankroll: float = 1000.0,
                 strategy: Optional[BaseStrategy] = None, 
                 cv_schema: Optional[Any] = None):
        """
        Initialize the PredictionBacktester.

        Args:
            data (pd.DataFrame): The dataset to be used for backtesting. Should include pre-computed predictions.
            odds_columns (List[str]): The names of the columns containing odds for each outcome.
            outcome_column (str): The name of the column containing the actual outcomes.
            prediction_column (str): The name of the column in the dataset that contains the predictions.
            initial_bankroll (float): The initial bankroll for the simulation. Defaults to 1000.0.
            strategy (Optional[BaseStrategy], optional): The betting strategy to be used.
            cv_schema (Optional[Any], optional): The cross-validation schema to be used. 
                Defaults to None, which will use TimeSeriesSplit with 5 splits.
        """
        super().__init__(data, odds_columns, outcome_column, initial_bankroll, None, strategy, cv_schema)
        self.prediction_column = prediction_column

    def run(self) -> None:
        """
        Run the prediction-based backtest.

        This method implements the backtesting logic for a strategy based on pre-computed predictions.
        It applies the betting strategy to the predictions in the specified column of the dataset
        and populates self.detailed_results with the outcomes.
        """
        if self.prediction_column not in self.data.columns:
            raise ValueError(f"Prediction column '{self.prediction_column}' not found in the dataset.")

        all_results = []
        current_bankroll = self.initial_bankroll

        for fold, (_, test_index) in enumerate(self.cv_schema.split(self.data)):
            X_test = self.data.iloc[test_index]

            for i, row in X_test.iterrows():
                prediction = row[self.prediction_column]
                actual_outcome = row[self.outcome_column]
                odds = row[self.odds_columns].tolist()

                result = self._simulate_bet(fold, i, prediction, actual_outcome, odds, current_bankroll)
                current_bankroll = result['bt_ending_bankroll'] # Update current bankroll for next bet in the loop
                
                # Add all original features to the result
                result.update(row.to_dict())

                all_results.append(result)

        self.detailed_results = pd.DataFrame(all_results)