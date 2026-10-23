__version__ = "0.2.0"

from .backtest.sport_backtest import BaseBacktest, ModelBacktest, PredictionBacktest
from .strategies.base import BaseStrategy
from .strategies.history import BetHistory, SettledBet
from .strategies.sport_strategies import (
    EdgeProportionalStake,
    FixedStake,
    KellyCriterion,
    OddsFilter,
    RiskLimits,
    UnitStake,
    ValueBet,
)
from .strategies.toolkit import (
    BetContext,
    EdgeProportional,
    Favourite,
    HighestEV,
    HighestKelly,
    HighestProbability,
    KellySize,
    ModelPick,
    Prediction,
    Selector,
    SimultaneousKelly,
    Sizer,
    StakeSize,
    Strategy,
    UnitPlan,
)
from .metrics.sport_metrics import METRIC_NAMES, calculate_all_metrics, calculate_metrics
from .odds.margin import remove_margin
from .tuning import split_by_date
from .plots.sport_plots import (
    plot_backtest,
    plot_calibration,
    plot_odds_histogram,
)

__all__ = [
    # Backtests
    "BaseBacktest",
    "ModelBacktest",
    "PredictionBacktest",
    # Ready-made strategies and wrappers
    "BaseStrategy",
    "FixedStake",
    "KellyCriterion",
    "ValueBet",
    "UnitStake",
    "EdgeProportionalStake",
    "OddsFilter",
    "RiskLimits",
    # Strategy toolkit
    "Strategy",
    "Selector",
    "Sizer",
    "BetContext",
    "ModelPick",
    "HighestProbability",
    "Prediction",
    "Favourite",
    "HighestEV",
    "HighestKelly",
    "StakeSize",
    "KellySize",
    "SimultaneousKelly",
    "UnitPlan",
    "EdgeProportional",
    # Memory
    "BetHistory",
    "SettledBet",
    # Metrics, odds, tuning, plots
    "calculate_all_metrics",
    "calculate_metrics",
    "METRIC_NAMES",
    "split_by_date",
    "remove_margin",
    "plot_backtest",
    "plot_calibration",
    "plot_odds_histogram",
]
