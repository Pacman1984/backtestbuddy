"""Helpers for hyperparameter searches over strategies and stake sizes.

Optimizing many settings on one history mostly finds settings that were
lucky there. Keep a holdout period the search never sees, tune on the
rest, and check the winner once on the holdout. ``split_by_date`` makes
that split; ``backtest.evaluate()`` gives the fast objective value.
"""

from typing import Any, Optional, Tuple

import pandas as pd


def split_by_date(
    data: pd.DataFrame,
    date_column: str,
    holdout_from: Any = None,
    holdout_fraction: Optional[float] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split matches into a tuning period and a later holdout period.

    Pass exactly one of ``holdout_from`` and ``holdout_fraction``. Matches on
    the cutoff date go to the holdout, so no date is split across both parts.

    Args:
        data: One row per match.
        date_column: Column with the match date.
        holdout_from: First date of the holdout (anything ``pd.Timestamp``
            accepts).
        holdout_fraction: Share of the distinct dates, counted from the end,
            that form the holdout, in ``(0, 1)``.

    Returns:
        ``(tuning, holdout)``, both in date order with their original index.

    Raises:
        ValueError: If not exactly one split argument is given, the fraction
            is out of range, or either part would be empty.

    Example:
        >>> tuning, holdout = split_by_date(data, "date", holdout_from="2024-01-01")
        >>> tuning, holdout = split_by_date(data, "date", holdout_fraction=0.2)
    """
    if (holdout_from is None) == (holdout_fraction is None):
        raise ValueError("Pass exactly one of holdout_from= or holdout_fraction=")
    dates = pd.to_datetime(data[date_column])
    if holdout_fraction is not None:
        if not 0 < holdout_fraction < 1:
            raise ValueError("holdout_fraction must be in (0, 1)")
        distinct = dates.drop_duplicates().sort_values()
        n_holdout = max(1, int(round(len(distinct) * holdout_fraction)))
        cutoff = distinct.iloc[len(distinct) - n_holdout]
    else:
        cutoff = pd.Timestamp(holdout_from)
    ordered = data.assign(_date=dates).sort_values("_date", kind="mergesort")
    is_holdout = ordered["_date"] >= cutoff
    tuning = ordered.loc[~is_holdout].drop(columns="_date")
    holdout = ordered.loc[is_holdout].drop(columns="_date")
    if tuning.empty or holdout.empty:
        raise ValueError(f"Split at {cutoff.date()} leaves an empty part")
    return tuning, holdout
