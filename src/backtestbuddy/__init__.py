__version__ = "0.2.0"

from .backtest.sport_backtest import BaseBacktest, ModelBacktest, PredictionBacktest
from .strategies.history import BetHistory, SettledBet
from .strategies.sport_strategies import (
    BaseStrategy,
    FixedStake,
    KellyCriterion,
    OddsFilter,
    RiskLimits,
    StakeSize,
    UnitStake,
    ValueBet,
)
from .metrics.sport_metrics import calculate_all_metrics
from .odds.margin import remove_margin
from .plots.sport_plots import (
    plot_backtest,
    plot_calibration,
    plot_odds_histogram,
)

__all__ = [
    "BaseBacktest",
    "ModelBacktest",
    "PredictionBacktest",
    "BaseStrategy",
    "FixedStake",
    "KellyCriterion",
    "ValueBet",
    "UnitStake",
    "OddsFilter",
    "RiskLimits",
    "StakeSize",
    "BetHistory",
    "SettledBet",
    "calculate_all_metrics",
    "remove_margin",
    "plot_backtest",
    "plot_calibration",
    "plot_odds_histogram",
]
