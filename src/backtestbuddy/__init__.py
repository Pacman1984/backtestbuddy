__version__ = "0.1.13"

from .backtest.sport_backtest import BaseBacktest, ModelBacktest, PredictionBacktest
from .strategies.sport_strategies import (
    BaseStrategy,
    FixedStake,
    KellyCriterion,
    OddsFilter,
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
    "calculate_all_metrics",
    "remove_margin",
"plot_backtest",
    "plot_calibration",
    "plot_odds_histogram",
]