import copy
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import TimeSeriesSplit

from backtestbuddy.metrics.sport_metrics import calculate_all_metrics
from backtestbuddy.odds.margin import MARGIN_METHODS
from backtestbuddy.plots.sport_plots import (
    plot_backtest,
    plot_calibration,
    plot_odds_histogram,
)
from backtestbuddy.strategies.history import BetHistory
from backtestbuddy.strategies.sport_strategies import (
    BaseStrategy,
    FixedStake,
    get_default_strategy,
)

STRESS_TEST_METRICS = (
    "ROI [%]",
    "Yield [%]",
    "CAGR [%]",
    "Max Drawdown [%]",
    "Sharpe Ratio (365.25) [-]",
    "Total Bets",
    "Bankroll Final [$]",
)


class _BetRow(NamedTuple):
    """Inputs for one match, prepared before the betting loop."""

    fold: int
    index: int
    position: int
    prediction: Any
    probs: Optional[List[float]]
    odds: List[float]
    outcome: Any


class _ResultColumns:
    """Collect result rows column by column and build the DataFrame once.

    Equivalent to ``pd.DataFrame(list_of_row_dicts)``: columns appear in the
    order they are first seen, and rows that lack a column get NaN. Building
    from column lists avoids the per-row dict merging that dominated runtime.
    """

    def __init__(self) -> None:
        self.columns: Dict[str, List[Any]] = {}
        self.first_row: Dict[str, int] = {}
        self.n_rows = 0

    def add(self, row: Dict[str, Any]) -> None:
        """Append one result row."""
        for key, value in row.items():
            column = self.columns.get(key)
            if column is None:
                column = self.columns[key] = [np.nan] * self.n_rows
                self.first_row[key] = self.n_rows
            column.append(value)
        self.n_rows += 1
        for column in self.columns.values():
            if len(column) < self.n_rows:
                column.append(np.nan)

    def to_frame(self, original: pd.DataFrame) -> pd.DataFrame:
        """Build the results and attach the original data columns.

        Column order matches the former per-row ``dict.update`` plus
        ``pd.DataFrame(list_of_dicts)``: result columns of the first row,
        then the data columns, then result columns first seen in a later
        row (e.g. strategy info missing on early rows). A data column with
        the same name as a result column replaces it in place.

        Args:
            original: Data rows in processing order, one per result row.

        Returns:
            Results DataFrame with a default RangeIndex.
        """
        original = original.reset_index(drop=True)
        early = [c for c in self.columns if self.first_row[c] == 0]
        late = [c for c in self.columns if self.first_row[c] > 0]
        frame = pd.DataFrame({c: self.columns[c] for c in early})
        for column in original.columns:
            frame[column] = original[column]
        for column in late:
            if column not in original.columns:
                frame[column] = self.columns[column]
        return frame


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
        closing_odds_columns (Optional[List[str]]): Reference odds per outcome for CLV.
        odds_haircut (float): Fraction by which bet odds are lowered before staking and settlement.
        margin_method (str): Method used to remove the bookmaker margin in metrics.
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
                 cv_schema: Optional[Any] = None,
                 closing_odds_columns: Optional[List[str]] = None,
                 odds_haircut: float = 0.0,
                 margin_method: str = "multiplicative",
                 settle_simultaneous: bool = False):
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
            closing_odds_columns (Optional[List[str]], optional): Reference odds for each
                outcome, in the same order as ``odds_columns``. Usually closing odds, but any
                later or sharper line works. Enables the CLV metrics. Defaults to None.
            odds_haircut (float): Lower every bet odd by this fraction before the strategy
                sees it and before settlement, e.g. ``0.02`` for 2% worse prices. Odds are
                floored at 1.0. Model features and closing odds are not changed. Defaults to 0.
            margin_method (str): How metrics remove the bookmaker margin: ``multiplicative``,
                ``additive``, ``power``, or ``shin``. Defaults to ``multiplicative``.
            settle_simultaneous (bool): Treat rows with the same date value as one slot of
                simultaneous matches. All bets of a slot are sized from the bankroll at the
                start of the slot, their total stake is capped at that bankroll, and they settle
                together, so no bet can use the result of another match in the same slot.
                Defaults to False (rows settle one after another).

        Raises:
            ValueError: If closing odds columns are missing or do not match ``odds_columns``
                in length, ``odds_haircut`` is outside ``[0, 1)``, or ``margin_method`` is
                unknown.
        """
        if closing_odds_columns is not None:
            if len(closing_odds_columns) != len(odds_columns):
                raise ValueError("closing_odds_columns must have one column per odds column")
            missing = [col for col in closing_odds_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Closing odds columns not found in the dataset: {missing}")
        if not 0 <= odds_haircut < 1:
            raise ValueError("odds_haircut must be in [0, 1)")
        if margin_method not in MARGIN_METHODS:
            raise ValueError(f"margin_method must be one of {MARGIN_METHODS}, got {margin_method!r}")
        # Stable sortkeeps the input order of same-date rows reproducible.
        self.data = data.sort_values(date_column, kind="mergesort").reset_index(drop=True)
        self.odds_columns = odds_columns
        self.outcome_column = outcome_column
        self.date_column = date_column
        self.initial_bankroll = initial_bankroll
        self.model = model
        self.strategy = strategy if strategy is not None else get_default_strategy()
        self.cv_schema = cv_schema if cv_schema is not None else TimeSeriesSplit(n_splits=5)
        self.detailed_results: Optional[pd.DataFrame] = None
        self.bookie_results: Optional[pd.DataFrame] = None
        self.history: Optional[BetHistory] = None
        self.metrics: Optional[Dict[str, Any]] = None
        self.model_prob_columns: Optional[List[str]] = None
        self.closing_odds_columns = closing_odds_columns
        self.odds_haircut = odds_haircut
        self.margin_method = margin_method
        self.settle_simultaneous = settle_simultaneous

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


    def _bet_odds(self, odds: List[float]) -> List[float]:
        """Apply ``odds_haircut`` to market odds, floored at 1.0.

        Args:
            odds: Decimal odds for each outcome.

        Returns:
            Odds the bettor actually gets. Unchanged when the haircut is 0.

        Example:
            haircut 0.02, odds ``[2.0, 1.02]`` → ``[1.96, 1.0]``.
        """
        if not self.odds_haircut:
            return odds
        lowered = [odd * (1 - self.odds_haircut) for odd in odds]
        return [1.0 if odd < 1.0 else odd for odd in lowered]

    @staticmethod
    def _record_bet(history: BetHistory, result: Dict[str, Any]) -> None:
        """Add a settled result row to the history if a bet was placed."""
        if result['bt_stake'] > 0 and result['bt_bet_on'] != -1:
            history.record(
                result['bt_date_column'],
                stake=result['bt_stake'],
                profit=result['bt_profit'],
                bankroll_before=result['bt_starting_bankroll'],
                bankroll_after=result['bt_ending_bankroll'],
                won=bool(result['bt_win']),
            )

    def _column_values(self, columns: Optional[List[str]]) -> Optional[List[List[Any]]]:
        """Python values of each column, converted once before the loop.

        ``Series.tolist`` returns the same Python types the old per-row
        ``to_dict("records")`` produced, so strategies see identical inputs.
        """
        if columns is None:
            return None
        return [self.data[col].tolist() for col in columns]

    @staticmethod
    def _row_values(columns: Optional[List[List[Any]]], position: int) -> Optional[List[Any]]:
        """Values of one row across pre-converted columns, or None."""
        if columns is None:
            return None
        return [values[position] for values in columns]

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

    def _simulate_bet(self, fold: int, index: int, prediction: int, bet_on: int, actual_outcome: int, odds: List[float], current_bankroll: float, stake: float, model_probs: Optional[List[float]] = None, closing_odds: Optional[List[float]] = None, **additional_info: Any) -> Dict[str, Any]:
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
            closing_odds (Optional[List[float]]): Reference odds for each outcome, stored as
                ``bt_closing_odd_{i}``.
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
                'bt_closing_odd_{i}': The reference odds for each outcome (if provided).
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
            'bt_date_column': self._dates[index],
        }

        # Add individual odds for each outcome
        for i, odd in enumerate(odds):
            result[f'bt_odd_{i}'] = odd

        if model_probs:
            for i, prob in enumerate(model_probs):
                result[f'bt_model_prob_{i}'] = prob

        if closing_odds is not None:
            for i, odd in enumerate(closing_odds):
                result[f'bt_closing_odd_{i}'] = odd

        # Add any additional information as bt_* columns
        for key, value in additional_info.items():
            result[f'bt_{key}'] = value

        return result


    def _slots(self, rows: List[_BetRow]) -> Iterator[List[_BetRow]]:
        """Group rows into settlement slots.

        Sequential mode yields one row per slot. Simultaneous mode groups
        consecutive rows with the same date value (rows are date-sorted).
        """
        if not self.settle_simultaneous:
            for row in rows:
                yield [row]
            return
        slot: List[_BetRow] = []
        for row in rows:
            if slot and self._dates[row.position] != self._dates[slot[0].position]:
                yield slot
                slot = []
            slot.append(row)
        if slot:
            yield slot

    def _simulate(self, rows: List[_BetRow]) -> None:
        """Bet on every prepared row and store results, bookie results, and history.

        Args:
            rows: Matches in betting order with their predictions and odds.
        """
        all_results = _ResultColumns()
        bookie_results = _ResultColumns()
        # Fresh ledger per run, so repeated runs (stress_test) never share state
        history = BetHistory(self.initial_bankroll)
        closing_columns = self._column_values(self.closing_odds_columns)
        current_bankroll = self.initial_bankroll
        bookie_bankroll = self.initial_bankroll
        adjust = getattr(self.strategy, "adjust_simultaneous", None)

        for slot in self._slots(rows):
            simultaneous = self.settle_simultaneous
            slot_bankroll = current_bankroll
            slot_bookie_bankroll = bookie_bankroll

            # Decide every bet of the slot before any of them settles
            decisions = []
            for row in slot:
                stake, bet_on, additional_info = self.strategy.get_bet_details(
                    row.odds, slot_bankroll, model_probs=row.probs, prediction=row.prediction,
                    history=history, date=self._dates[row.position])
                decisions.append([stake, bet_on, additional_info])
            if simultaneous and adjust is not None:
                adjusted = adjust([d[0] if d[1] != -1 else 0.0 for d in decisions], slot_bankroll)
                for decision, stake in zip(decisions, adjusted):
                    decision[0] = stake

            available = slot_bankroll
            bookie_available = slot_bookie_bankroll
            settled = []
            for row, (stake, bet_on, additional_info) in zip(slot, decisions):
                if simultaneous:
                    # Simultaneous bets cannot stake more than the slot's bankroll in total
                    if bet_on != -1 and stake > 0:
                        stake = min(stake, max(available, 0.0))
                        available -= stake
                    if stake <= 0:
                        stake, bet_on = 0.0, -1
                closing = self._row_values(closing_columns, row.position)
                result = self._simulate_bet(row.fold, row.index, row.prediction, bet_on, row.outcome, row.odds,
                                            current_bankroll, stake, model_probs=row.probs, closing_odds=closing,
                                            **additional_info)
                if simultaneous:
                    result['bt_sizing_bankroll'] = slot_bankroll
                current_bankroll = result['bt_ending_bankroll']
                if simultaneous:
                    settled.append(result)
                else:
                    self._record_bet(history, result)
                all_results.add(result)

                bookie_result = self._simulate_bookie_bet(
                    row.fold, row.index, row.odds, row.outcome, bookie_bankroll, closing_odds=closing,
                    sizing_bankroll=slot_bookie_bankroll if simultaneous else None,
                    max_stake=max(bookie_available, 0.0) if simultaneous else None)
                bookie_available -= bookie_result['bt_stake']
                bookie_bankroll = bookie_result['bt_ending_bankroll']
                bookie_results.add(bookie_result)

            # Results of a simultaneous slot become visible to strategies only afterwards
            for result in settled:
                self._record_bet(history, result)

        # Original columns are attached once instead of copied into every row
        original = self.data.iloc[[row.position for row in rows]]
        self.detailed_results = all_results.to_frame(original)
        self.bookie_results = bookie_results.to_frame(original)
        self.history = history

    def calculate_metrics(self) -> Dict[str, Any]:
        """
        Calculate performance metrics based on the backtest results.
        """
        if self.detailed_results is None:
            raise ValueError("Backtest results are not available. Make sure to run the backtest first.")
        
        self.metrics = calculate_all_metrics(self.detailed_results, margin_method=self.margin_method)
        return self.metrics

    def stress_test(self,
                    haircuts: Sequence[float] = (0.0, 0.01, 0.02, 0.03),
                    metrics: Sequence[str] = STRESS_TEST_METRICS) -> pd.DataFrame:
        """
        Re-run the backtest with progressively worse odds.

        Each run lowers every bet odd by one haircut fraction (see ``odds_haircut``). The
        strategy decides and bets settle at the lowered odds; model features, model
        probabilities, and closing odds stay as they are. This shows how much of the edge
        survives when you cannot get the odds in the data, e.g. because you bet after the
        opening line has moved.

        Args:
            haircuts (Sequence[float]): Odds reductions to test, each in ``[0, 1)``.
            metrics (Sequence[str]): Keys of ``calculate_all_metrics`` to report.

        Returns:
            pd.DataFrame: One row per haircut (index ``odds_haircut``), one column per metric.
            This backtest's own results are left unchanged.

        Example:
            >>> backtest.stress_test(haircuts=[0, 0.02])["Yield [%]"]
        """
        rows = {}
        for haircut in haircuts:
            if not 0 <= haircut < 1:
                raise ValueError("Each haircut must be in [0, 1)")
            # Shallow copy: data is shared read-only; run() replaces the result frames
            trial = copy.copy(self)
            trial.odds_haircut = haircut
            trial.run()
            summary = calculate_all_metrics(trial.detailed_results, margin_method=self.margin_method)
            rows[haircut] = {name: summary[name] for name in metrics}
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index.name = "odds_haircut"
        return table

    def plot(self, x_axis: str = "bet", show_bookie: bool = True):
        """
        Generate and display a plot of the backtest results.

        This method creates a plot showing the bankroll over time,
        underwater drawdown, win/loss markers, and ROI for each bet.

        Args:
            x_axis: ``"bet"`` (placed-bet index) or ``"date"``.
            show_bookie: Overlay ``bookie_results`` on the bankroll panel.

        Raises:
            ValueError: If the backtest has not been run yet.
        """
        if self.detailed_results is None:
            raise ValueError("Backtest has not been run yet. Call run() first.")

        fig = plot_backtest(self, x_axis=x_axis, show_bookie=show_bookie)
        fig.show()

    def plot_calibration(self, n_bins: Optional[int] = None, scope: str = "placed") -> go.Figure:
        """Plot a reliability diagram for model probabilities.

        Args:
            n_bins: Equal-width bins on ``[0, 1]``. Defaults to ECE bins.
            scope: ``"placed"`` (selected outcome of placed bets) or ``"all"``
                (every outcome of every game).

        Returns:
            Plotly figure. Empty of observed points when no probabilities
            were stored.

        Raises:
            ValueError: If the backtest has not been run yet.
        """
        if self.detailed_results is None:
            raise ValueError("Backtest has not been run yet. Call run() first.")
        return plot_calibration(self, n_bins=n_bins, scope=scope)

    def plot_odds_distribution(self, num_bins: Optional[int] = None) -> go.Figure:
        """
        Generate a histogram plot of the odds distribution for the main strategy.
        
        Args:
            num_bins (Optional[int]): The number of bins to use for the histogram. If None, an automatic binning strategy is used.
        
        Returns:
            go.Figure: A Plotly figure object containing the odds histogram.
        """
        return plot_odds_histogram(self, num_bins)

    def _simulate_bookie_bet(self, fold: int, index: int, odds: List[float], actual_outcome: int, current_bankroll: float, closing_odds: Optional[List[float]] = None, sizing_bankroll: Optional[float] = None, max_stake: Optional[float] = None) -> Dict[str, Any]:
        """
        Simulate a bet using the bookie strategy (lowest-odds favorite).

        Stake sizing reuses ``self.strategy.calculate_stake`` without model
        probabilities. KellyCriterion and ValueBet therefore always stake 0
        here; the bookie path is only a valid benchmark with strategies that
        do not require model probabilities (e.g. FixedStake, UnitStake).

        Args:
            fold: Current fold number.
            index: Index of the current data point.
            odds: Decimal odds for each possible outcome.
            actual_outcome: Actual outcome index (0-based).
            current_bankroll: Bankroll before placing the bet.
            closing_odds: Reference odds for each outcome, if available.
            sizing_bankroll: Bankroll the stake is sized from; defaults to
                ``current_bankroll`` (differs only for simultaneous slots).
            max_stake: Upper bound on the stake (unstaked funds of a slot).

        Returns:
            Result row for the bookie bet.
        """
        # For bookie strategy, we still bet on the lowest odds
        bet_on = odds.index(min(odds))
        stake = self.strategy.calculate_stake(odds, current_bankroll if sizing_bankroll is None else sizing_bankroll)
        if max_stake is not None:
            stake = min(stake, max_stake)
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
            'bt_date_column': self._dates[index],
        }

        # Add individual odds for each outcome
        for i, odd in enumerate(odds):
            result[f'bt_odd_{i}'] = odd

        if closing_odds is not None:
            for i, odd in enumerate(closing_odds):
                result[f'bt_closing_odd_{i}'] = odd

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
                 cv_schema: Optional[Any] = None,
                 feature_columns: Optional[List[str]] = None,
                 closing_odds_columns: Optional[List[str]] = None,
                 odds_haircut: float = 0.0,
                 margin_method: str = "multiplicative",
                 settle_simultaneous: bool = False):
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
            feature_columns (Optional[List[str]], optional): Columns passed to the model.
                Defaults to every column except the date and outcome columns. Set this
                explicitly to keep IDs, prediction or post-event columns out of the model.
                Closing odds are post-event data for a bettor; exclude them here.
            closing_odds_columns (Optional[List[str]], optional): Reference odds for CLV.
            odds_haircut (float): Fraction by which bet odds are lowered. Defaults to 0.
            margin_method (str): Margin removal method for metrics.
            settle_simultaneous (bool): Settle same-date rows as one slot. Defaults to False.

        Raises:
            ValueError: If a feature column is missing from the dataset.
        """
        super().__init__(data, odds_columns, outcome_column, date_column, initial_bankroll, model, strategy, cv_schema,
                         closing_odds_columns, odds_haircut, margin_method, settle_simultaneous)
        if feature_columns is not None:
            missing = [col for col in feature_columns if col not in data.columns]
            if missing:
                raise ValueError(f"Feature columns not found in the dataset: {missing}")
        self.feature_columns = feature_columns

    def run(self) -> None:
        """
        Run the model-based backtest.

        This method implements the backtesting logic for a model-based strategy.
        It uses the model to make predictions, applies the betting strategy,
        and populates self.detailed_results with the outcomes.
        """
        self._dates = self.data[self.date_column].tolist()
        rows: List[_BetRow] = []

        # Define feature columns, excluding date and outcome columns unless set explicitly
        feature_columns = self.feature_columns or [col for col in self.data.columns if col not in [self.date_column, self.outcome_column]]

        # Train and predict every fold first; betting never feeds back into the model
        for fold, (train_index, test_index) in enumerate(self.cv_schema.split(self.data)):
            X_train, X_test = self.data.iloc[train_index], self.data.iloc[test_index]
            y_train, y_test = X_train[self.outcome_column], X_test[self.outcome_column]

            # Train the model using only feature columns
            self.model.fit(X_train[feature_columns], y_train)

            # Make predictions using only feature columns
            predictions = self.model.predict(X_test[feature_columns])
            probabilities = self.model.predict_proba(X_test[feature_columns])

            # Check if probabilities match the number of odds columns
            if probabilities.shape[1] != len(self.odds_columns):
                raise ValueError(
                    f"The model's predict_proba output shape ({probabilities.shape[1]}) "
                    f"doesn't match the number of odds columns ({len(self.odds_columns)}). "
                    f"Expected shape: (n_samples, {len(self.odds_columns)})"
                    "\nExample of correct output:"
                    f"\n[[0.3, 0.7], [0.6, 0.4], ...] for {len(self.odds_columns)} outcomes"
                    "\nExample of incorrect output:"
                    "\n[[1.0], [1.0], ...] or [[0.3, 0.5, 0.2], [0.1, 0.7, 0.2], ...]"
                )

            # Convert once per fold; per-row .iloc lookups dominate runtime otherwise
            odds_rows = [self._bet_odds(row) for row in X_test[self.odds_columns].to_numpy(dtype=float).tolist()]
            outcomes = y_test.tolist()

            for i, (prediction, probs) in enumerate(zip(predictions, probabilities)):
                odds = odds_rows[i]
                # Ensure probs match the number of odds
                if len(probs) != len(odds):
                    raise ValueError(f"Number of probabilities ({len(probs)}) doesn't match number of odds ({len(odds)})")
                # Convert probs to a list if it's not already
                probs_list = probs.tolist() if isinstance(probs, np.ndarray) else list(probs)
                rows.append(_BetRow(fold, test_index[i], test_index[i], prediction, probs_list, odds, outcomes[i]))

        self._simulate(rows)
       


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
                 model_prob_columns: Optional[List[str]] = None,
                 closing_odds_columns: Optional[List[str]] = None,
                 odds_haircut: float = 0.0,
                 margin_method: str = "multiplicative",
                 settle_simultaneous: bool = False):
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
            closing_odds_columns (Optional[List[str]], optional): Reference odds for CLV.
            odds_haircut (float): Fraction by which bet odds are lowered. Defaults to 0.
            margin_method (str): Margin removal method for metrics.
            settle_simultaneous (bool): Settle same-date rows as one slot. Defaults to False.

        Raises:
            ValueError: If the specified prediction_column is not found in the dataset.
            ValueError: If the strategy requires model probabilities but model_prob_columns is not provided.
        """
        super().__init__(data, odds_columns, outcome_column, date_column, initial_bankroll, None, strategy, None,
                         closing_odds_columns, odds_haircut, margin_method, settle_simultaneous)
        
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
        # Convert every needed column to Python values once; the loop then only indexes lists
        self._dates = self.data[self.date_column].tolist()
        predictions = self.data[self.prediction_column].tolist()
        outcomes = self.data[self.outcome_column].tolist()
        odds_columns = self._column_values(self.odds_columns)
        prob_columns = self._column_values(self.model_prob_columns) if self.model_prob_columns else None

        rows = [
            _BetRow(0, i, i, predictions[i], self._row_values(prob_columns, i),
                    self._bet_odds(self._row_values(odds_columns, i)), outcomes[i])
            for i in range(len(self.data))
        ]
        self._simulate(rows)