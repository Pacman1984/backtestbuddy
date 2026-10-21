"""Sports-betting performance metrics computed from backtest result rows.

All money metrics use ``bt_starting_bankroll`` / ``bt_ending_bankroll`` /
``bt_profit``. Risk ratios (Sharpe, Sortino, Calmar) use per-bet simple
returns ``profit / starting_bankroll``, compounded inside each return period
without inserting empty calendar days.

A bet is counted only when ``bt_stake > 0`` and ``bt_bet_on != -1``.

Yield is ``sum(profit) / sum(stake)``. Expected P&L, Brier, log-loss,
and ECE use the model probability of the selected outcome. Implied
probability is ``1 / odds``; overround is ``sum_k 1/odds_k - 1``.

Sharpe and Sortino annualize with ``output_period`` periods per year.
The sports default is ``365.25`` (calendar). ``252`` is the equity
trading-year convention used through 0.1.13. Observed periods/year uses
the sample density of periods with at least one placed bet.

Calibration metrics (Brier, log-loss, ECE) have two scopes: ``"placed"``
scores the selected outcome of placed bets only; ``"all"`` scores every
outcome of every game that has model probabilities. The market baseline
scores the margin-free probabilities of ``bt_odd_*`` the same way.

Closing line value (CLV) compares the odds taken with reference odds in
``bt_closing_odd_*``; those columns are optional.
"""

from typing import Any, Dict, List, Literal, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

from backtestbuddy.odds.margin import MarginMethod, remove_margin

TRADING_DAYS_PER_YEAR= 252.0
CALENDAR_DAYS_PER_YEAR = 365.25
LOG_LOSS_CLIP = 1e-15
ECE_BINS = 10

CalibrationScope = Literal["placed", "all"]


class DrawdownWindow(NamedTuple):
    """Peak-to-trough window on an equity curve.

    Attributes:
        magnitude: Positive drawdown fraction (0.15 = 15%).
        start: Index of the last peak at or before the trough.
        end: Index of the maximum-drawdown trough.
        duration: Inclusive length ``end - start + 1``.
    """

    magnitude: float
    start: int
    end: int
    duration: int


def _filter_placed_bets(detailed_results: pd.DataFrame) -> pd.DataFrame:
    """Return rows where a bet was actually placed.

    Matches ``BaseBacktest._simulate_bet``: no bet when stake is 0 or
    ``bet_on == -1``.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Subset of ``detailed_results`` with ``bt_stake > 0`` and
        ``bt_bet_on != -1``.
    """
    if len(detailed_results) == 0:
        return detailed_results
    return detailed_results[
        (detailed_results["bt_stake"] > 0)
        & (detailed_results["bt_bet_on"] != -1)
    ]


def _numbered_columns(detailed_results: pd.DataFrame, prefix: str) -> List[str]:
    """Return ``{prefix}{i}`` columns sorted by outcome index.

    Args:
        detailed_results: Backtest result rows.
        prefix: Column prefix such as ``bt_odd_``.

    Returns:
        Matching column names, e.g. ``bt_odd_0``, ``bt_odd_1``.
    """
    cols = [
        c
        for c in detailed_results.columns
        if c.startswith(prefix) and c[len(prefix) :].isdigit()
    ]
    return sorted(cols, key=lambda name: int(name[len(prefix) :]))


def _outcome_odd_columns(detailed_results: pd.DataFrame) -> List[str]:
    """Return ``bt_odd_{i}`` columns sorted by outcome index.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Column names such as ``bt_odd_0``, ``bt_odd_1``.
    """
    return _numbered_columns(detailed_results, "bt_odd_")


def _selected_model_probability(bet_placed: pd.DataFrame) -> pd.Series:
    """Model probability of the outcome that was bet on.

    Args:
        bet_placed: Placed-bet rows.

    Returns:
        Series aligned to ``bet_placed``; NaN when ``bt_model_prob_{k}``
        is missing.
    """
    if bet_placed.empty:
        return pd.Series(dtype=float)
    bet_on = bet_placed["bt_bet_on"].to_numpy()
    values = np.full(len(bet_placed), np.nan, dtype=float)
    for outcome in pd.unique(bet_on):
        if pd.isna(outcome) or int(outcome) < 0:
            continue
        outcome_i = int(outcome)
        col = f"bt_model_prob_{outcome_i}"
        if col not in bet_placed.columns:
            continue
        mask = bet_on == outcome
        values[mask] = pd.to_numeric(
            bet_placed.loc[mask, col], errors="coerce"
        ).to_numpy()
    return pd.Series(values, index=bet_placed.index)


def _placed_probability_frame(detailed_results: pd.DataFrame) -> pd.DataFrame:
    """Placed bets that have a selected-outcome model probability.

    Returns:
        Frame with ``stake``, ``odds``, ``profit``, ``win``, ``p``. Empty
        if no placed bets have model probabilities.
    """
    bets = _filter_placed_bets(detailed_results)
    empty = pd.DataFrame(columns=["stake", "odds", "profit", "win", "p"])
    if bets.empty:
        return empty
    frame = pd.DataFrame(
        {
            "stake": bets["bt_stake"].astype(float),
            "odds": bets["bt_odds"].astype(float),
            "profit": bets["bt_profit"].astype(float),
            "win": (bets["bt_win"] == True).astype(float),
            "p": _selected_model_probability(bets),
        }
    )
    return frame.dropna(subset=["p", "odds", "stake"])


def _all_outcome_probability_frame(detailed_results: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, outcome) with the model probability and result.

    Every row with a known ``bt_actual_outcome`` contributes one pair per
    ``bt_model_prob_{k}`` column, whether or not a bet was placed. This
    scores the model on all games instead of only the bets it selected.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Frame with ``p`` and ``win`` (1.0 if outcome ``k`` happened). Empty
        if there are no model probability columns.

    Example:
        One game, probs ``[0.6, 0.4]``, outcome 0 → rows ``(0.6, 1)``,
        ``(0.4, 0)``.
    """
    empty = pd.DataFrame(columns=["p", "win"])
    prefix = "bt_model_prob_"
    prob_cols = [
        c
        for c in detailed_results.columns
        if c.startswith(prefix) and c[len(prefix) :].isdigit()
    ]
    if (
        len(detailed_results) == 0
        or not prob_cols
        or "bt_actual_outcome" not in detailed_results.columns
    ):
        return empty

    actual = pd.to_numeric(
        detailed_results["bt_actual_outcome"], errors="coerce"
    ).to_numpy(dtype=float)
    known = ~np.isnan(actual)
    parts = []
    for col in sorted(prob_cols, key=lambda c: int(c[len(prefix) :])):
        outcome = int(col[len(prefix) :])
        probs = pd.to_numeric(detailed_results[col], errors="coerce")
        parts.append(
            pd.DataFrame(
                {
                    "p": probs.to_numpy(dtype=float)[known],
                    "win": (actual[known] == outcome).astype(float),
                }
            )
        )
    return pd.concat(parts, ignore_index=True).dropna(subset=["p"])


def _calibration_frame(
    detailed_results: pd.DataFrame, scope: CalibrationScope
) -> pd.DataFrame:
    """Probability/result pairs for a calibration scope.

    Args:
        detailed_results: Backtest result rows.
        scope: ``"placed"`` (selected outcome of placed bets) or ``"all"``
            (every outcome of every game).

    Returns:
        Frame with at least ``p`` and ``win``.

    Raises:
        ValueError: If ``scope`` is unknown.
    """
    if scope == "placed":
        return _placed_probability_frame(detailed_results)
    if scope == "all":
        return _all_outcome_probability_frame(detailed_results)
    raise ValueError("scope must be 'placed' or 'all'.")


def _compound_period_returns(
    detailed_results: pd.DataFrame,
    return_period: int = 1,
) -> pd.Series:
    """Compound per-bet simple returns into period returns.

    Per-bet return is ``bt_profit / bt_starting_bankroll``. Same-period bets
    compound as ``prod(1 + r) - 1``. Only placed bets enter (when the stake
    columns are present), so periods with skipped opportunities only are
    dropped like empty calendar days. The result therefore does not depend
    on whether skipped opportunities are in ``detailed_results``.

    Args:
        detailed_results: Backtest result rows with dates and P&L.
        return_period: Length of each bucket in calendar days.

    Returns:
        Period returns indexed by period start. Empty input yields an empty
        Series.

    Example:
        Two same-day returns of 0.10 and -0.05 compound to
        ``1.10 * 0.95 - 1 = 0.045``, not ``0.10 - 0.05 = 0.05``.
    """
    if {"bt_stake", "bt_bet_on"}.issubset(detailed_results.columns):
        detailed_results = _filter_placed_bets(detailed_results)
    if len(detailed_results) == 0:
        return pd.Series(dtype=float)

    dates = pd.to_datetime(detailed_results["bt_date_column"])
    per_bet = (
        detailed_results["bt_profit"] / detailed_results["bt_starting_bankroll"]
    ).astype(float)
    per_bet.index = pd.DatetimeIndex(dates)

    # prod(1 + r) - 1 == expm1(sum(log1p(r))); a vectorized sum avoids a
    # Python call per bucket. A -100% bet gives -inf; older pandas (1.3)
    # sums -inf + 0 to NaN, so wiped-out periods are set to -1 explicitly.
    freq = f"{return_period}D"
    with np.errstate(divide="ignore"):
        log_growth = np.log1p(per_bet)
    grouped = log_growth.groupby(pd.Grouper(freq=freq))
    sizes = grouped.size()
    period_returns = np.expm1(grouped.sum())
    wiped_out = (per_bet <= -1.0).groupby(pd.Grouper(freq=freq)).any()
    period_returns[wiped_out] = -1.0
    return period_returns[sizes > 0].astype(float)


def _observed_periods_per_year(
    detailed_results: pd.DataFrame,
    return_period: int = 1,
) -> float:
    """Observed non-empty return periods per calendar year.

    ``P = n_periods / years`` with ``years = (span_days + return_period) /
    365.25``. Adding one period to the span counts the last bucket, so
    ``n`` consecutive daily periods give ``P = 365.25``. This matches the
    density of the Sharpe/Sortino series (empty periods are not filled).
    Prefer this over 252 or 365.25 when betting days are sparse.

    Args:
        detailed_results: Backtest result rows with dates and P&L.
        return_period: Same bucket length as Sharpe/Sortino.

    Returns:
        Periods per year. ``365.25`` if there are no periods.

    Example:
        50 non-empty daily buckets over 365 days → about ``50.0``; 10
        consecutive betting days → ``365.25``.
    """
    returns = _compound_period_returns(detailed_results, return_period)
    if len(returns) == 0 or len(detailed_results) == 0:
        return CALENDAR_DAYS_PER_YEAR

    dates = pd.to_datetime(detailed_results["bt_date_column"])
    span_days = (dates.max() - dates.min()).days + return_period
    years = span_days / CALENDAR_DAYS_PER_YEAR
    return float(len(returns) / years)


def _opening_bankroll(detailed_results: pd.DataFrame) -> Optional[float]:
    """Bankroll before the first row, or None if it is not available.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        ``bt_starting_bankroll`` of the first row.
    """
    if (
        len(detailed_results) == 0
        or "bt_starting_bankroll" not in detailed_results.columns
    ):
        return None
    return float(detailed_results["bt_starting_bankroll"].iloc[0])


def _equity_drawdown_stats(
    equity_curve: np.ndarray,
    initial: Optional[float] = None,
) -> DrawdownWindow:
    """Compute max drawdown on a bankroll / equity series.

    Duration starts at the last peak at or before the trough
    (``end - start + 1``).

    Args:
        equity_curve: Sequential bankroll values.
        initial: Opening bankroll. When given it is prepended as index 0,
            so losses on the first bets count as drawdown. Indices in the
            result are then shifted by one relative to ``equity_curve``.

    Returns:
        DrawdownWindow with positive magnitude. All zeros if there is no
        drawdown or the series is empty.

    Example:
        ``[900, 800]`` with ``initial=1000`` → magnitude 0.2.
    """
    if len(equity_curve) == 0:
        return DrawdownWindow(0.0, 0, 0, 0)

    equity = np.asarray(equity_curve, dtype=float)
    if initial is not None:
        equity = np.concatenate(([float(initial)], equity))
    cummax = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = (cummax - equity) / cummax
    drawdown = np.nan_to_num(drawdown, nan=0.0, posinf=0.0, neginf=0.0)
    magnitude = float(np.max(drawdown))

    if magnitude == 0:
        return DrawdownWindow(0.0, 0, 0, 0)

    end = int(np.argmax(drawdown))
    peak_mask = equity[: end + 1] == cummax[: end + 1]
    start = int(np.where(peak_mask)[0][-1])
    duration = end - start + 1
    return DrawdownWindow(magnitude, start, end, duration)


def calculate_roi(detailed_results: pd.DataFrame) -> float:
    """Calculate total return on investment as a decimal.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        ``(final_bankroll - initial_bankroll) / initial_bankroll``.
        ``0.0`` for an empty DataFrame.

    Example:
        Start 1000, end 1500 → ``0.5``.
    """
    if len(detailed_results) == 0:
        return 0.0
    initial_bankroll = detailed_results["bt_starting_bankroll"].iloc[0]
    final_bankroll = detailed_results["bt_ending_bankroll"].iloc[-1]
    if initial_bankroll == 0:
        return 0.0
    return (final_bankroll - initial_bankroll) / initial_bankroll


def calculate_sharpe_ratio(
    detailed_results: pd.DataFrame,
    return_period: int = 1,
    output_period: float = CALENDAR_DAYS_PER_YEAR,
) -> float:
    """Calculate the annualized Sharpe ratio (risk-free rate = 0).

    Mean and sample standard deviation (``ddof=1``) of compounded period
    returns are scaled by ``output_period`` and ``sqrt(output_period)``.
    ``Sharpe(P) = (mean / std) * sqrt(P)``.

    Default ``output_period=365.25`` is a calendar year (sports). Pass
    ``252`` for the equity trading-year scale used through 0.1.13, or
    ``_observed_periods_per_year(...)`` for sample density.

    Args:
        detailed_results: Backtest result rows.
        return_period: Return bucket length in calendar days.
        output_period: Periods per year for annualization.

    Returns:
        Sharpe ratio, or ``0.0`` if fewer than two periods, volatility is
        zero, or there is no data.

    Example:
        Same returns at ``P=365.25`` vs ``P=252`` differ by
        ``sqrt(365.25 / 252)``.
    """
    returns = _compound_period_returns(detailed_results, return_period)
    if len(returns) < 2:
        return 0.0

    annualized_mean_return = returns.mean() * output_period
    annualized_std_return = returns.std(ddof=1) * np.sqrt(output_period)

    if annualized_std_return == 0 or pd.isna(annualized_std_return):
        return 0.0

    return float(annualized_mean_return / annualized_std_return)


def calculate_max_drawdown(detailed_results: pd.DataFrame) -> float:
    """Calculate maximum bankroll drawdown as a signed decimal.

    Args:
        detailed_results: Backtest result rows with ``bt_ending_bankroll``.

    The opening bankroll (``bt_starting_bankroll`` of the first row) is
    the first peak, so early losses count.

    Returns:
        Negative fraction (e.g. ``-0.15`` for a 15% peak-to-trough drop),
        or ``0.0`` if there is no drawdown.

    Example:
        Equity 1000 → 850 → 1100 → ``-0.15``.
    """
    if "bt_ending_bankroll" not in detailed_results.columns:
        return 0.0
    magnitude, _, _, _ = _equity_drawdown_stats(
        detailed_results["bt_ending_bankroll"].to_numpy(),
        initial=_opening_bankroll(detailed_results),
    )
    return -magnitude


def calculate_win_rate(detailed_results: pd.DataFrame) -> float:
    """Calculate win rate among placed bets.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Winning bets / placed bets, or ``0.0`` if no bets were placed.
    """
    bet_placed = _filter_placed_bets(detailed_results)
    total_bets = len(bet_placed)
    if total_bets == 0:
        return 0.0
    winning_bets = bet_placed["bt_win"].sum()
    return winning_bets / total_bets


def calculate_average_odds(detailed_results: pd.DataFrame) -> float:
    """Calculate the mean decimal odds of the given rows.

    Args:
        detailed_results: Rows whose ``bt_odds`` should be averaged.

    Returns:
        Mean of ``bt_odds`` (NaN if empty).
    """
    return detailed_results["bt_odds"].mean()


def calculate_total_profit(detailed_results: pd.DataFrame) -> float:
    """Calculate the sum of per-row profit.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Sum of ``bt_profit``.
    """
    return detailed_results["bt_profit"].sum()


def calculate_average_stake(detailed_results: pd.DataFrame) -> float:
    """Calculate the mean stake among non-zero stakes.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Mean of ``bt_stake`` where stake > 0, or ``0.0`` if none.
    """
    non_zero_stakes = detailed_results["bt_stake"][
        detailed_results["bt_stake"] > 0
    ]
    return non_zero_stakes.mean() if len(non_zero_stakes) > 0 else 0.0


def calculate_sortino_ratio(
    detailed_results: pd.DataFrame,
    return_period: int = 1,
    output_period: float = CALENDAR_DAYS_PER_YEAR,
    target_return: float = 0.0,
) -> float:
    """Calculate the annualized Sortino ratio.

    Downside deviation is ``sqrt(mean(min(0, r - tau)^2))`` over all
    periods (zeros for upside periods), then scaled by
    ``sqrt(output_period)``. This uses ``N`` in the mean, not sample
    ``N-1``. Same ``output_period`` convention as Sharpe: default 365.25
    (calendar), pass 252 for the 0.1.13 trading-year scale.

    Args:
        detailed_results: Backtest result rows.
        return_period: Return bucket length in calendar days.
        output_period: Periods per year for annualization.
        target_return: Minimum acceptable return per period (default 0).

    Returns:
        Sortino ratio. ``inf`` if downside deviation is 0 and mean excess
        return is positive; ``0.0`` if there is no downside and mean excess
        is not positive.

    Example:
        All-positive profits with target 0 → ``inf``.
    """
    returns = _compound_period_returns(detailed_results, return_period)
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - target_return
    shortfalls = np.minimum(0, excess_returns)
    downside_deviation_period = np.sqrt(np.mean(shortfalls**2))
    downside_deviation_annual = downside_deviation_period * np.sqrt(
        output_period
    )
    annualized_mean_excess_return = excess_returns.mean() * output_period

    if downside_deviation_annual == 0:
        if annualized_mean_excess_return > 0:
            return float("inf")
        return 0.0

    return float(annualized_mean_excess_return / downside_deviation_annual)


def calculate_calmar_ratio(
    detailed_results: pd.DataFrame,
    return_period: int = 1,
    output_period: int = 365,
    years: Optional[float] = None,
) -> float:
    """Calculate the Calmar ratio using geometric annual return.

    Maximum drawdown here is on the wealth index of compounded period
    returns, starting at 1.0 so an opening loss is a drawdown. This is
    not the bankroll equity curve used by ``calculate_drawdowns``.

    Args:
        detailed_results: Backtest result rows.
        return_period: Return bucket length in calendar days.
        output_period: Fallback periods-per-year if the date span is 0.
        years: Optional holding period in years. If None, inferred from
            dates.

    Returns:
        Calmar ratio. ``inf`` if drawdown is 0 and geometric annual return
        is positive; ``0.0`` if there is no data, no drawdown with
        non-positive return, or a zero holding period.
    """
    if len(detailed_results) == 0:
        return 0.0

    returns = _compound_period_returns(detailed_results, return_period)
    if len(returns) == 0:
        return 0.0

    cumulative_returns = (1 + returns).cumprod()
    wealth = pd.concat(
        [pd.Series([1.0], dtype=float), cumulative_returns],
        ignore_index=True,
    )
    peak = wealth.cummax()
    drawdown = (wealth - peak) / peak
    max_drawdown = float(drawdown.min())

    r_total = float(cumulative_returns.iloc[-1] - 1)

    if years is None:
        dates = pd.to_datetime(detailed_results["bt_date_column"])
        k_years = (dates.max() - dates.min()).days / 365.25
        if k_years == 0:
            k_years = len(returns) / output_period
            if k_years == 0:
                return 0.0
    else:
        k_years = years

    r_annual = (1 + r_total) ** (1 / k_years) - 1

    if max_drawdown == 0:
        if r_annual > 0:
            return float("inf")
        return 0.0

    return float(r_annual / abs(max_drawdown))


def calculate_drawdowns(detailed_results: pd.DataFrame) -> Tuple[float, int]:
    """Calculate max bankroll drawdown magnitude and duration in rows.

    Args:
        detailed_results: Backtest result rows with ``bt_ending_bankroll``.

    Returns:
        Tuple of (positive drawdown fraction, duration in rows). Duration
        is 0 when there is no drawdown.

    Example:
        Equity 100, 120, 120, 110, 90 → magnitude 0.25, duration 3
        (last peak at the second 120 through the trough).
    """
    if len(detailed_results) == 0:
        return 0.0, 0
    window = _equity_drawdown_stats(
        detailed_results["bt_ending_bankroll"].to_numpy(),
        initial=_opening_bankroll(detailed_results),
    )
    return window.magnitude, window.duration


def calculate_best_worst_bets(
    detailed_results: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate best and worst per-row profit.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        ``(max profit, min profit)``.
    """
    best_bet = detailed_results["bt_profit"].max()
    worst_bet = detailed_results["bt_profit"].min()
    return best_bet, worst_bet


def calculate_highest_odds(
    detailed_results: pd.DataFrame,
) -> Tuple[float, float]:
    """Calculate highest winning odds and highest losing odds.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        ``(highest winning odds, highest losing odds)``. ``0`` if a side
        has no rows.
    """
    winning_bets = detailed_results[detailed_results["bt_win"] == True]
    losing_bets = detailed_results[detailed_results["bt_win"] == False]

    highest_winning_odds = (
        winning_bets["bt_odds"].max() if not winning_bets.empty else 0
    )
    highest_losing_odds = (
        losing_bets["bt_odds"].max() if not losing_bets.empty else 0
    )

    return highest_winning_odds, highest_losing_odds


def calculate_avg_roi_per_bet_micro(detailed_results: pd.DataFrame) -> float:
    """Calculate the mean of per-bet ROI percentages.

    Each placed bet contributes ``(profit / stake) * 100``. Equal weight
    per bet.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Mean per-bet ROI in percent, or ``0.0`` if no bets were placed.

    Example:
        Stakes 100 with profits 20, -50, 30 → mean of 20%, -50%, 30% = 0%.
    """
    bet_placed = _filter_placed_bets(detailed_results)
    if bet_placed.empty:
        return 0.0
    stakes = bet_placed["bt_stake"].replace(0, np.nan)
    roi_per_bet = bet_placed["bt_profit"] / stakes * 100
    return float(roi_per_bet.mean()) if roi_per_bet.notna().any() else 0.0


def calculate_avg_roi_per_bet_macro(detailed_results: pd.DataFrame) -> float:
    """Calculate total ROI divided by the number of placed bets.

    Not part of ``calculate_all_metrics``: bankroll ROI compounds across
    bets, so dividing it by the bet count has no per-bet meaning when
    stakes scale with bankroll. Use ``calculate_yield`` per bet instead.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        ``(total ROI decimal / n_bets) * 100``, or ``0.0`` if no bets.

    Example:
        20% total ROI over 10 bets → 2% per bet.
    """
    bet_placed = _filter_placed_bets(detailed_results)
    if bet_placed.empty:
        return 0.0

    total_roi = calculate_roi(detailed_results)
    num_bets = len(bet_placed)
    return (total_roi / num_bets) * 100 if num_bets > 0 else 0.0


def calculate_avg_roi_per_year_micro(detailed_results: pd.DataFrame) -> float:
    """Calculate the duration-weighted mean of calendar-year simple ROIs.

    Each year's ROI is ``(year_end - year_start) / year_start * 100``.
    Each year is weighted by the days it covers (first to last row in
    that year, inclusive), so a two-week partial year does not count as
    much as a full one. This is not CAGR.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Weighted mean of yearly ROIs in percent, or ``0.0`` if empty.

    Example:
        A 14-day year at -3.7% and a 365-day year at -25.5% →
        ``(14 * -3.7 + 365 * -25.5) / 379 ≈ -24.7``.
    """
    if len(detailed_results) == 0:
        return 0.0

    dates = pd.to_datetime(detailed_results["bt_date_column"])
    detailed_results = detailed_results.copy()
    detailed_results["year"] = dates.dt.year
    detailed_results["date"] = dates

    yearly_rois = []
    weights = []
    for _, group in detailed_results.groupby("year"):
        initial_bankroll = group["bt_starting_bankroll"].iloc[0]
        final_bankroll = group["bt_ending_bankroll"].iloc[-1]
        if initial_bankroll == 0:
            continue
        yearly_roi = (
            (final_bankroll - initial_bankroll) / initial_bankroll
        ) * 100
        yearly_rois.append(yearly_roi)
        weights.append((group["date"].max() - group["date"].min()).days + 1)

    if not yearly_rois:
        return 0.0
    return float(np.average(yearly_rois, weights=weights))


def calculate_avg_roi_per_year_macro(detailed_results: pd.DataFrame) -> float:
    """Calculate total ROI divided by calendar years (linear, not CAGR).

    Args:
        detailed_results: Backtest result rows.

    Returns:
        ``(total ROI decimal / years) * 100``, or ``0.0`` if empty or
        duration is 0.
    """
    if len(detailed_results) == 0:
        return 0.0

    total_roi = calculate_roi(detailed_results)
    dates = pd.to_datetime(detailed_results["bt_date_column"])
    years = (dates.max() - dates.min()).days / 365.25
    if years == 0:
        return 0.0
    return (total_roi / years) * 100


def calculate_risk_adjusted_annual_roi(
    detailed_results: pd.DataFrame,
) -> float:
    """Calculate annual macro ROI (decimal) per unit of max drawdown.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Unitless ratio. ``inf`` if max drawdown is 0 and yearly ROI is
        positive; ``0.0`` if max drawdown is 0 and yearly ROI is not
        positive.

    Example:
        10% annual ROI with 15% max drawdown → ``0.10 / 0.15 ≈ 0.667``.
    """
    avg_yearly_roi_pct = calculate_avg_roi_per_year_macro(detailed_results)
    avg_yearly_roi = avg_yearly_roi_pct / 100.0
    max_drawdown = calculate_max_drawdown(detailed_results)

    if max_drawdown == 0:
        if avg_yearly_roi > 0:
            return float("inf")
        return 0.0

    return avg_yearly_roi / abs(max_drawdown)


def calculate_cagr(detailed_results: pd.DataFrame) -> float:
    """Calculate compound annual growth rate of the bankroll.

    ``CAGR = (end / start) ** (1 / years) - 1``, returned as percent.
    Years use ``(max_date - min_date).days / 365.25``.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        CAGR in percent. ``0.0`` if empty, zero duration, zero start, or
        negative final bankroll. Complete loss to 0 is ``-100.0``.
    """
    if len(detailed_results) == 0:
        return 0.0

    initial_value = detailed_results["bt_starting_bankroll"].iloc[0]
    final_value = detailed_results["bt_ending_bankroll"].iloc[-1]

    dates = pd.to_datetime(detailed_results["bt_date_column"])
    years = (dates.max() - dates.min()).days / 365.25

    if years == 0 or initial_value == 0:
        return 0.0
    if final_value < 0:
        return 0.0
    if final_value == 0:
        return -100.0

    cagr = (pow(final_value / initial_value, 1 / years) - 1) * 100
    return cagr


def calculate_yield(detailed_results: pd.DataFrame) -> float:
    """Calculate yield: profit over total amount staked, as percent.

    Distinct from bankroll ROI, which is relative to opening bankroll.
    Distinct from micro ROI-per-bet, which equal-weights each bet.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        ``sum(profit) / sum(stake) * 100`` on placed bets, or ``0.0``
        if none were placed or total stake is 0.

    Example:
        Stakes 100 and 300, profits 20 and -30 → ``-10 / 400 * 100 = -2.5``.
    """
    bet_placed = _filter_placed_bets(detailed_results)
    if bet_placed.empty:
        return 0.0
    total_stake = float(bet_placed["bt_stake"].sum())
    if total_stake == 0:
        return 0.0
    return float(bet_placed["bt_profit"].sum() / total_stake * 100)


def calculate_expected_profit(detailed_results: pd.DataFrame) -> float:
    """Calculate expected profit from model p and decimal odds.

    Per placed bet: ``stake * (p * odds - 1)``, using the model
    probability of the selected outcome.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.

    Returns:
        Sum of expected profits, ``0.0`` if no placed bets, ``nan`` if
        no selected-outcome model probabilities.

    Example:
        Stake 100, p=0.6, odds=2.0 → ``100 * (1.2 - 1) = 20``.
    """
    frame = _placed_probability_frame(detailed_results)
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    if frame.empty:
        return float("nan")
    expected = frame["stake"] * (frame["p"] * frame["odds"] - 1.0)
    return float(expected.sum())


def calculate_expected_yield(detailed_results: pd.DataFrame) -> float:
    """Calculate expected yield: expected profit over staked amount.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.

    Returns:
        Expected profit / stake on rows with model probabilities, as
        percent. ``0.0`` if no placed bets, ``nan`` if no probabilities.

    Example:
        Expected profit 20 on stake 100 → ``20.0``.
    """
    frame = _placed_probability_frame(detailed_results)
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    if frame.empty:
        return float("nan")
    total_stake = float(frame["stake"].sum())
    if total_stake == 0:
        return 0.0
    expected = float(
        (frame["stake"] * (frame["p"] * frame["odds"] - 1.0)).sum()
    )
    return expected / total_stake * 100


def calculate_realized_vs_expected_profit(
    detailed_results: pd.DataFrame,
) -> float:
    """Calculate realized profit minus expected profit on the same bets.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.

    Returns:
        ``sum(profit) - expected profit`` on rows with model
        probabilities. ``0.0`` if no placed bets, ``nan`` if no
        probabilities.

    Example:
        Expected 20, realized 100 → ``80``.
    """
    frame = _placed_probability_frame(detailed_results)
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    if frame.empty:
        return float("nan")
    expected = float(
        (frame["stake"] * (frame["p"] * frame["odds"] - 1.0)).sum()
    )
    return float(frame["profit"].sum() - expected)


def calculate_average_implied_prob(detailed_results: pd.DataFrame) -> float:
    """Calculate mean raw implied probability of the selected odds.

    ``implied = 1 / decimal odds``. Vig is not stripped; see overround.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Mean of ``1 / bt_odds`` on placed bets, or ``nan`` if none.

    Example:
        Odds 2.0 and 4.0 → mean of 0.5 and 0.25 = ``0.375``.
    """
    bet_placed = _filter_placed_bets(detailed_results)
    if bet_placed.empty:
        return float("nan")
    odds = pd.to_numeric(bet_placed["bt_odds"], errors="coerce")
    implied = 1.0 / odds.where(odds > 0)
    if implied.notna().sum() == 0:
        return float("nan")
    return float(implied.mean())


def calculate_average_overround(detailed_results: pd.DataFrame) -> float:
    """Calculate mean book overround across all outcome odds on a row.

    Overround is ``sum_k 1/odds_k - 1``, reported as percent. Uses
    ``bt_odd_*`` columns.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Mean overround in percent, or ``nan`` if no outcome-odd columns
        or no placed bets.

    Example:
        Odds 2.0 and 1.8 → ``(0.5 + 1/1.8 - 1) * 100 ≈ 5.556``.
    """
    bet_placed = _filter_placed_bets(detailed_results)
    if bet_placed.empty:
        return float("nan")
    odd_cols = _outcome_odd_columns(bet_placed)
    if not odd_cols:
        return float("nan")
    implied_sum = None
    for col in odd_cols:
        odds = pd.to_numeric(bet_placed[col], errors="coerce")
        inv = 1.0 / odds.where(odds > 0)
        implied_sum = inv if implied_sum is None else implied_sum + inv
    overround = implied_sum - 1.0
    if overround.notna().sum() == 0:
        return float("nan")
    return float(overround.mean() * 100)


def _brier(frame: pd.DataFrame) -> float:
    """Mean squared error of ``p`` vs ``win`` in a probability frame."""
    return float(((frame["p"] - frame["win"]) ** 2).mean())


def _log_loss(frame: pd.DataFrame) -> float:
    """Binary log loss of ``p`` vs ``win`` with ``p`` clipped away from 0/1."""
    p = frame["p"].clip(LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    y = frame["win"]
    return float((-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean())


def calculate_brier_score(
    detailed_results: pd.DataFrame,
    scope: CalibrationScope = "placed",
) -> float:
    """Calculate the Brier score of model probabilities.

    ``mean((p - y)^2)`` with ``y=1`` if the outcome happened, else ``0``.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.
        scope: ``"placed"`` scores the selected outcome of placed bets;
            ``"all"`` scores every outcome of every game.

    Returns:
        Mean squared error of p vs outcome. ``nan`` if no probabilities;
        ``0.0`` if no placed bets in ``"placed"`` scope.

    Example:
        p=0.6 win → ``(0.6-1)^2 = 0.16``.
    """
    if scope == "placed" and _filter_placed_bets(detailed_results).empty:
        return 0.0
    frame = _calibration_frame(detailed_results, scope)
    if frame.empty:
        return float("nan")
    return _brier(frame)


def calculate_log_loss(
    detailed_results: pd.DataFrame,
    scope: CalibrationScope = "placed",
) -> float:
    """Calculate binary log loss of model probabilities.

    ``-mean(y log p + (1-y) log(1-p))``. ``p`` is clipped to
    ``[1e-15, 1-1e-15]``.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.
        scope: ``"placed"`` or ``"all"``, as in ``calculate_brier_score``.

    Returns:
        Mean log loss. ``nan`` if no probabilities; ``0.0`` if no
        placed bets in ``"placed"`` scope.

    Example:
        p=0.6 win → ``-log(0.6)``.
    """
    if scope == "placed" and _filter_placed_bets(detailed_results).empty:
        return 0.0
    frame = _calibration_frame(detailed_results, scope)
    if frame.empty:
        return float("nan")
    return _log_loss(frame)


def _reliability_points(
    detailed_results: pd.DataFrame,
    n_bins: int = ECE_BINS,
    scope: CalibrationScope = "placed",
) -> pd.DataFrame:
    """Equal-width reliability points for a calibration diagram.

    Bins are ``np.linspace(0, 1, n_bins + 1)``. Empty bins are omitted.
    Each row is the mean predicted probability, the observed win rate,
    and the number of placed bets in that bin.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.
        n_bins: Number of equal-width probability bins.
        scope: ``"placed"`` or ``"all"``, as in ``calculate_brier_score``.

    Returns:
        Frame with ``mean_p``, ``win_rate``, and ``count``. Empty if
        there are no probabilities in the scope.

    Example:
        All ``p=0.6`` and half win → one row ``(0.6, 0.5, N)``.
    """
    frame = _calibration_frame(detailed_results, scope)
    empty = pd.DataFrame(columns=["mean_p", "win_rate", "count"])
    if frame.empty:
        return empty
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    frame = frame.copy()
    frame["bin"] = pd.cut(frame["p"], bins=bins, include_lowest=True)
    rows = []
    for _, group in frame.groupby("bin", observed=True):
        if group.empty:
            continue
        rows.append(
            {
                "mean_p": float(group["p"].mean()),
                "win_rate": float(group["win"].mean()),
                "count": int(len(group)),
            }
        )
    if not rows:
        return empty
    return pd.DataFrame(rows)


def calculate_ece(
    detailed_results: pd.DataFrame,
    n_bins: int = ECE_BINS,
    scope: CalibrationScope = "placed",
) -> float:
    """Calculate expected calibration error of model probabilities.

    Equal-width bins on ``[0, 1]``. Empty bins are skipped.
    ``ECE = sum_m (n_m / N) * |acc_m - conf_m|``.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.
        n_bins: Number of equal-width probability bins.
        scope: ``"placed"`` or ``"all"``, as in ``calculate_brier_score``.

    Returns:
        ECE in ``[0, 1]``. ``nan`` if no probabilities; ``0.0`` if no
        placed bets in ``"placed"`` scope.

    Example:
        All p=0.6 and half win → one occupied bin, ECE = ``|0.5-0.6|=0.1``.
    """
    if scope == "placed" and _filter_placed_bets(detailed_results).empty:
        return 0.0
    points = _reliability_points(detailed_results, n_bins=n_bins, scope=scope)
    if points.empty:
        return float("nan")
    n = int(points["count"].sum())
    gap = (points["win_rate"] - points["mean_p"]).abs()
    return float((points["count"] / n * gap).sum())


def _market_probability_frame(
    detailed_results: pd.DataFrame,
    method: MarginMethod = "multiplicative",
) -> pd.DataFrame:
    """One row per (game, outcome) with the margin-free market probability.

    Fair probabilities come from ``bt_odd_*`` (the odds bet into) with
    ``remove_margin``. Every game with a known outcome counts, bet or not.

    Args:
        detailed_results: Backtest result rows.
        method: Margin removal method.

    Returns:
        Frame with ``p`` and ``win``. Empty without odds or outcomes.
    """
    empty = pd.DataFrame(columns=["p", "win"])
    odd_cols = _outcome_odd_columns(detailed_results)
    if (
        len(detailed_results) == 0
        or len(odd_cols) < 2
        or "bt_actual_outcome" not in detailed_results.columns
    ):
        return empty
    actual = pd.to_numeric(
        detailed_results["bt_actual_outcome"], errors="coerce"
    ).to_numpy(dtype=float)
    known = ~np.isnan(actual)
    odds = detailed_results[odd_cols].apply(pd.to_numeric, errors="coerce")
    fair = remove_margin(odds.to_numpy(dtype=float)[known], method)
    outcomes = np.arange(len(odd_cols))
    frame = pd.DataFrame(
        {
            "p": fair.ravel(),
            "win": (actual[known][:, None] == outcomes).astype(float).ravel(),
        }
    )
    return frame.dropna(subset=["p"])


def calculate_market_brier_score(
    detailed_results: pd.DataFrame,
    method: MarginMethod = "multiplicative",
) -> float:
    """Calculate the Brier score of margin-free market probabilities.

    Same pairs as ``calculate_brier_score(scope="all")`` but with the
    bookmaker's fair probabilities instead of the model's. A model whose
    all-games Brier score is lower than this beats the market line.

    Args:
        detailed_results: Backtest result rows with ``bt_odd_*``.
        method: Margin removal method (see ``remove_margin``).

    Returns:
        Brier score, or ``nan`` without odds or outcomes.

    Example:
        Odds ``[1.9, 1.9]``, outcome 0 → ``mean(0.5^2, 0.5^2) = 0.25``.
    """
    frame = _market_probability_frame(detailed_results, method)
    if frame.empty:
        return float("nan")
    return _brier(frame)


def calculate_market_log_loss(
    detailed_results: pd.DataFrame,
    method: MarginMethod = "multiplicative",
) -> float:
    """Calculate the log loss of margin-free market probabilities.

    Args:
        detailed_results: Backtest result rows with ``bt_odd_*``.
        method: Margin removal method (see ``remove_margin``).

    Returns:
        Log loss, or ``nan`` without odds or outcomes.
    """
    frame = _market_probability_frame(detailed_results, method)
    if frame.empty:
        return float("nan")
    return _log_loss(frame)


def _closing_line_frame(
    detailed_results: pd.DataFrame,
    method: MarginMethod = "multiplicative",
) -> pd.DataFrame:
    """Odds taken vs closing odds of the selected outcome on placed bets.

    Args:
        detailed_results: Backtest result rows with ``bt_closing_odd_*``.
        method: Margin removal method for the fair closing probability.

    Returns:
        Frame with ``odds`` (taken), ``close`` (raw closing odds of the
        selected outcome), and ``fair_p`` (margin-free closing
        probability). Empty without closing columns or placed bets.
    """
    empty = pd.DataFrame(columns=["odds", "close", "fair_p"])
    bets = _filter_placed_bets(detailed_results)
    close_cols = _numbered_columns(bets, "bt_closing_odd_")
    if bets.empty or len(close_cols) < 2:
        return empty
    closing = (
        bets[close_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    )
    fair = remove_margin(closing, method)
    selected = bets["bt_bet_on"].to_numpy().astype(int)
    in_range = selected < len(close_cols)
    rows = np.arange(len(bets))[in_range]
    cols = selected[in_range]
    frame = pd.DataFrame(
        {
            "odds": pd.to_numeric(bets["bt_odds"], errors="coerce").to_numpy()[in_range],
            "close": closing[rows, cols],
            "fair_p": fair[rows, cols],
        }
    )
    return frame.dropna(subset=["odds", "close"])


def _has_closing_odds(detailed_results: pd.DataFrame) -> bool:
    return len(_numbered_columns(detailed_results, "bt_closing_odd_")) >= 2


def calculate_clv(
    detailed_results: pd.DataFrame,
    no_vig: bool = False,
    method: MarginMethod = "multiplicative",
) -> float:
    """Calculate mean closing line value of placed bets, in percent.

    Raw CLV per bet is ``odds_taken / closing_odds - 1`` on the selected
    outcome. No-vig CLV is ``odds_taken * fair_closing_p - 1``: the
    expected return per unit if the margin-free closing line is the true
    probability. Positive CLV means you got better prices than the
    reference line. The reference is whatever ``closing_odds_columns``
    held in the backtest; it does not have to be the literal close.

    Args:
        detailed_results: Backtest result rows with ``bt_closing_odd_*``.
        no_vig: Use the margin-free closing probability.
        method: Margin removal method when ``no_vig`` is True.

    Returns:
        Mean CLV in percent. ``nan`` without closing odds; ``0.0`` if no
        bets were placed.

    Example:
        Took 2.20, closed 2.00 at ``[2.00, 1.90]``: raw ``+10%``;
        multiplicative fair p ``0.4872`` → no-vig ``2.2 * 0.4872 - 1 ≈ +7.2%``.
    """
    if not _has_closing_odds(detailed_results):
        return float("nan")
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    frame = _closing_line_frame(detailed_results, method)
    if no_vig:
        frame = frame.dropna(subset=["fair_p"])
        values = frame["odds"] * frame["fair_p"] - 1.0
    else:
        values = frame["odds"] / frame["close"] - 1.0
    if values.empty:
        return float("nan")
    return float(values.mean() * 100)


def calculate_beat_closing_line(detailed_results: pd.DataFrame) -> float:
    """Calculate the share of placed bets taken above the closing odds.

    Args:
        detailed_results: Backtest result rows with ``bt_closing_odd_*``.

    Returns:
        Percent of placed bets with ``odds_taken > closing_odds`` on the
        selected outcome. ``nan`` without closing odds; ``0.0`` if no
        bets were placed.
    """
    if not _has_closing_odds(detailed_results):
        return float("nan")
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    frame = _closing_line_frame(detailed_results)
    if frame.empty:
        return float("nan")
    return float((frame["odds"] > frame["close"]).mean() * 100)


def calculate_all_metrics(
    detailed_results: pd.DataFrame,
    margin_method: MarginMethod = "multiplicative",
) -> Dict[str, Any]:
    """Calculate the full metrics dictionary for a backtest.

    Drawdown, win rate, odds, and stake stats use placed bets only.
    ROI, CAGR, Sharpe, Sortino, Calmar, and risk-adjusted ROI use the
    full result set so the holding period stays the full backtest.

    Sharpe and Sortino are reported three times with the scale in the
    key: calendar ``365.25``, trading-year ``252``, and observed
    non-empty periods per year. There is no unnamed key.

    Sharpe and Sortino use only periods with placed bets.

    Yield is profit over total staked. Expected profit / yield use the
    model probability of the selected outcome. Brier / log-loss / ECE are
    reported twice: ``(placed)`` on the selected outcome of placed bets,
    ``(all games)`` on every outcome of every game. Both need
    ``bt_model_prob_*``. The ``(market, all games)`` scores use the
    margin-free odds instead and need no model.

    CLV metrics need ``bt_closing_odd_*`` and are ``nan`` without them.

    Args:
        detailed_results: Full backtest result rows.
        margin_method: Margin removal for no-vig CLV and the market
            scores: ``multiplicative``, ``additive``, ``power``, ``shin``.

    Returns:
        Mapping of metric names to values.
    """
    bet_placed = _filter_placed_bets(detailed_results)

    start_date = detailed_results["bt_date_column"].min()
    end_date = detailed_results["bt_date_column"].max()
    duration = end_date - start_date

    bankroll_final = detailed_results["bt_ending_bankroll"].iloc[-1]
    bankroll_peak = detailed_results["bt_ending_bankroll"].max()
    bankroll_valley = detailed_results["bt_ending_bankroll"].min()

    max_drawdown, max_drawdown_duration = calculate_drawdowns(bet_placed)
    best_bet, worst_bet = calculate_best_worst_bets(bet_placed)
    highest_winning_odds, highest_losing_odds = calculate_highest_odds(
        bet_placed
    )

    n_opportunities = len(detailed_results)
    n_bets = len(bet_placed)
    bet_frequency = (
        (n_bets / n_opportunities) * 100 if n_opportunities else 0.0
    )
    obs_periods_per_year = _observed_periods_per_year(detailed_results)

    return {
        "Backtest Start Date": start_date,
        "Backtest End Date": end_date,
        "Backtest Duration": duration,
        "ROI [%]": calculate_roi(detailed_results) * 100,
        "Yield [%]": calculate_yield(detailed_results),
        "Expected Yield [%]": calculate_expected_yield(detailed_results),
        "Avg. ROI per Bet [%] (micro)": calculate_avg_roi_per_bet_micro(
            detailed_results
        ),
        "Avg. ROI per Year [%] (micro)": calculate_avg_roi_per_year_micro(
            detailed_results
        ),
        "Avg. ROI per Year [%] (macro)": calculate_avg_roi_per_year_macro(
            detailed_results
        ),
        "CAGR [%]": calculate_cagr(detailed_results),
        "Risk-Adjusted Annual ROI [-]": calculate_risk_adjusted_annual_roi(
            detailed_results
        ),
        "Total Profit [$]": calculate_total_profit(detailed_results),
        "Expected Profit [$]": calculate_expected_profit(detailed_results),
        "Realized vs Expected Profit [$]": (
            calculate_realized_vs_expected_profit(detailed_results)
        ),
        "Bankroll Final [$]": bankroll_final,
        "Bankroll Peak [$]": bankroll_peak,
        "Bankroll Valley [$]": bankroll_valley,
        "Sharpe Ratio (365.25) [-]": calculate_sharpe_ratio(detailed_results),
        "Sharpe Ratio (252) [-]": calculate_sharpe_ratio(
            detailed_results, output_period=TRADING_DAYS_PER_YEAR
        ),
        "Sharpe Ratio (obs/year) [-]": calculate_sharpe_ratio(
            detailed_results,
            output_period=obs_periods_per_year,
        ),
        "Sortino Ratio (365.25) [-]": calculate_sortino_ratio(detailed_results),
        "Sortino Ratio (252) [-]": calculate_sortino_ratio(
            detailed_results, output_period=TRADING_DAYS_PER_YEAR
        ),
        "Sortino Ratio (obs/year) [-]": calculate_sortino_ratio(
            detailed_results,
            output_period=obs_periods_per_year,
        ),
        "Calmar Ratio [-]": calculate_calmar_ratio(detailed_results),
        "Max Drawdown [%]": max_drawdown * 100,
        "Max. Drawdown Duration [bets]": max_drawdown_duration,
        "Win Rate [%]": calculate_win_rate(bet_placed) * 100,
        "Average Odds [-]": calculate_average_odds(bet_placed),
        "Average Implied Prob [-]": calculate_average_implied_prob(
            detailed_results
        ),
        "Average Overround [%]": calculate_average_overround(
            detailed_results
        ),
        "Brier Score (placed) [-]": calculate_brier_score(detailed_results),
        "Brier Score (all games) [-]": calculate_brier_score(
            detailed_results, scope="all"
        ),
        "Log Loss (placed) [-]": calculate_log_loss(detailed_results),
        "Log Loss (all games) [-]": calculate_log_loss(
            detailed_results, scope="all"
        ),
        "ECE (placed) [-]": calculate_ece(detailed_results),
        "ECE (all games) [-]": calculate_ece(detailed_results, scope="all"),
        "Brier Score (market, all games) [-]": calculate_market_brier_score(
            detailed_results, method=margin_method
        ),
        "Log Loss (market, all games) [-]": calculate_market_log_loss(
            detailed_results, method=margin_method
        ),
        "CLV [%]": calculate_clv(detailed_results),
        "CLV (no-vig) [%]": calculate_clv(
            detailed_results, no_vig=True, method=margin_method
        ),
        "Beat Closing Line [%]": calculate_beat_closing_line(detailed_results),
        "Highest Winning Odds [-]": highest_winning_odds,
        "Highest Losing Odds [-]": highest_losing_odds,
        "Average Stake [$]": calculate_average_stake(bet_placed),
        "Best Bet [$]": best_bet,
        "Worst Bet [$]": worst_bet,
        "Total Bets": n_bets,
        "Total Opportunities": n_opportunities,
        "Bet Frequency [%]": bet_frequency,
    }
