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
from .metrics.sport_metrics import calculate_all_metrics
from .odds.margin import remove_margin
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
    # Metrics, odds, plots
    "calculate_all_metrics",
    "remove_margin",
    "plot_backtest",
    "plot_calibration",
    "plot_odds_histogram",
]
