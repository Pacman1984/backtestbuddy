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
the sample density of non-empty return buckets.
"""

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252.0
CALENDAR_DAYS_PER_YEAR = 365.25
LOG_LOSS_CLIP = 1e-15
ECE_BINS = 10


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


def _outcome_odd_columns(detailed_results: pd.DataFrame) -> List[str]:
    """Return ``bt_odd_{i}`` columns sorted by outcome index.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Column names such as ``bt_odd_0``, ``bt_odd_1``.
    """
    cols = [
        c
        for c in detailed_results.columns
        if c.startswith("bt_odd_") and c[len("bt_odd_") :].isdigit()
    ]

    def _index(name: str) -> int:
        return int(name[len("bt_odd_") :])

    return sorted(cols, key=_index)


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


def _compound_period_returns(
    detailed_results: pd.DataFrame,
    return_period: int = 1,
) -> pd.Series:
    """Compound per-bet simple returns into period returns.

    Per-bet return is ``bt_profit / bt_starting_bankroll``. Same-period bets
    compound as ``prod(1 + r) - 1``. Empty calendar days are not filled with
    zeros.

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
    if len(detailed_results) == 0:
        return pd.Series(dtype=float)

    df = detailed_results.copy()
    dates = pd.to_datetime(df["bt_date_column"])
    per_bet = (df["bt_profit"] / df["bt_starting_bankroll"]).astype(float)
    per_bet.index = pd.DatetimeIndex(dates)

    grouped = per_bet.groupby(pd.Grouper(freq=f"{return_period}D"))

    def _period_return(group: pd.Series) -> float:
        if group.empty:
            return np.nan
        return float((1.0 + group).prod() - 1.0)

    period_returns = grouped.apply(_period_return)
    return period_returns.dropna()


def _observed_periods_per_year(
    detailed_results: pd.DataFrame,
    return_period: int = 1,
) -> float:
    """Observed non-empty return periods per calendar year.

    ``P = n_periods / years`` with ``years = (max_date - min_date).days /
    365.25``. This matches the density of the Sharpe/Sortino series (empty
    calendar days are not filled). Prefer this over 252 or 365.25 when
    betting days are sparse.

    Args:
        detailed_results: Backtest result rows with dates and P&L.
        return_period: Same bucket length as Sharpe/Sortino.

    Returns:
        Periods per year. ``365.25`` if there are no periods or the date
        span is zero.

    Example:
        50 non-empty daily buckets over 365 days → about ``50.0``.
    """
    returns = _compound_period_returns(detailed_results, return_period)
    if len(returns) == 0 or len(detailed_results) == 0:
        return CALENDAR_DAYS_PER_YEAR

    dates = pd.to_datetime(detailed_results["bt_date_column"])
    years = (dates.max() - dates.min()).days / CALENDAR_DAYS_PER_YEAR
    if years == 0:
        return CALENDAR_DAYS_PER_YEAR
    return float(len(returns) / years)


def _equity_drawdown_stats(equity_curve: np.ndarray) -> DrawdownWindow:
    """Compute max drawdown on a bankroll / equity series.

    Duration starts at the last peak at or before the trough
    (``end - start + 1``).

    Args:
        equity_curve: Sequential bankroll values.

    Returns:
        DrawdownWindow with positive magnitude. All zeros if there is no
        drawdown or the series is empty.
    """
    if len(equity_curve) == 0:
        return DrawdownWindow(0.0, 0, 0, 0)

    equity = np.asarray(equity_curve, dtype=float)
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

    Returns:
        Negative fraction (e.g. ``-0.15`` for a 15% peak-to-trough drop),
        or ``0.0`` if there is no drawdown.

    Example:
        Equity 1000 → 850 → 1100 → ``-0.15``.
    """
    if "bt_ending_bankroll" not in detailed_results.columns:
        return 0.0
    magnitude, _, _, _ = _equity_drawdown_stats(
        detailed_results["bt_ending_bankroll"].to_numpy()
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
        detailed_results["bt_ending_bankroll"].to_numpy()
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
    """Calculate the arithmetic mean of calendar-year simple ROIs.

    Each year's ROI is ``(year_end - year_start) / year_start * 100``.
    This is not CAGR.

    Args:
        detailed_results: Backtest result rows.

    Returns:
        Mean of yearly ROIs in percent, or ``0.0`` if empty.
    """
    if len(detailed_results) == 0:
        return 0.0

    dates = pd.to_datetime(detailed_results["bt_date_column"])
    detailed_results = detailed_results.copy()
    detailed_results["year"] = dates.dt.year

    yearly_rois = []
    for _, group in detailed_results.groupby("year"):
        initial_bankroll = group["bt_starting_bankroll"].iloc[0]
        final_bankroll = group["bt_ending_bankroll"].iloc[-1]
        if initial_bankroll == 0:
            continue
        yearly_roi = (
            (final_bankroll - initial_bankroll) / initial_bankroll
        ) * 100
        yearly_rois.append(yearly_roi)

    return float(np.mean(yearly_rois)) if yearly_rois else 0.0


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


def calculate_brier_score(detailed_results: pd.DataFrame) -> float:
    """Calculate Brier score of the selected-outcome model probability.

    ``mean((p - y)^2)`` with ``y=1`` on a win and ``0`` on a loss.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.

    Returns:
        Mean squared error of p vs outcome. ``nan`` if no probabilities;
        ``0.0`` if no placed bets.

    Example:
        p=0.6 win → ``(0.6-1)^2 = 0.16``.
    """
    frame = _placed_probability_frame(detailed_results)
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    if frame.empty:
        return float("nan")
    return float(((frame["p"] - frame["win"]) ** 2).mean())


def calculate_log_loss(detailed_results: pd.DataFrame) -> float:
    """Calculate binary log loss of the selected-outcome probability.

    ``-mean(y log p + (1-y) log(1-p))``. ``p`` is clipped to
    ``[1e-15, 1-1e-15]``.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.

    Returns:
        Mean log loss. ``nan`` if no probabilities; ``0.0`` if no
        placed bets.

    Example:
        p=0.6 win → ``-log(0.6)``.
    """
    frame = _placed_probability_frame(detailed_results)
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    if frame.empty:
        return float("nan")
    p = frame["p"].clip(LOG_LOSS_CLIP, 1.0 - LOG_LOSS_CLIP)
    y = frame["win"]
    return float((-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))).mean())


def _reliability_points(
    detailed_results: pd.DataFrame,
    n_bins: int = ECE_BINS,
) -> pd.DataFrame:
    """Equal-width reliability points for a calibration diagram.

    Bins are ``np.linspace(0, 1, n_bins + 1)``. Empty bins are omitted.
    Each row is the mean predicted probability, the observed win rate,
    and the number of placed bets in that bin.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.
        n_bins: Number of equal-width probability bins.

    Returns:
        Frame with ``mean_p``, ``win_rate``, and ``count``. Empty if
        there are no selected-outcome probabilities.

    Example:
        All ``p=0.6`` and half win → one row ``(0.6, 0.5, N)``.
    """
    frame = _placed_probability_frame(detailed_results)
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
) -> float:
    """Calculate expected calibration error of selected-outcome p.

    Equal-width bins on ``[0, 1]``. Empty bins are skipped.
    ``ECE = sum_m (n_m / N) * |acc_m - conf_m|``.

    Args:
        detailed_results: Backtest result rows with ``bt_model_prob_*``.
        n_bins: Number of equal-width probability bins.

    Returns:
        ECE in ``[0, 1]``. ``nan`` if no probabilities; ``0.0`` if no
        placed bets.

    Example:
        All p=0.6 and half win → one occupied bin, ECE = ``|0.5-0.6|=0.1``.
    """
    if _filter_placed_bets(detailed_results).empty:
        return 0.0
    points = _reliability_points(detailed_results, n_bins=n_bins)
    if points.empty:
        return float("nan")
    n = int(points["count"].sum())
    gap = (points["win_rate"] - points["mean_p"]).abs()
    return float((points["count"] / n * gap).sum())


def calculate_all_metrics(detailed_results: pd.DataFrame) -> Dict[str, Any]:
    """Calculate the full metrics dictionary for a backtest.

    Drawdown, win rate, odds, and stake stats use placed bets only.
    ROI, CAGR, Sharpe, Sortino, Calmar, and risk-adjusted ROI use the
    full result set so the holding period stays the full backtest.

    Sharpe and Sortino are reported three times with the scale in the
    key: calendar ``365.25``, trading-year ``252``, and observed
    non-empty periods per year. There is no unnamed key.

    Yield is profit over total staked. Expected profit / yield and
    Brier / log-loss / ECE use the model probability of the selected
    outcome when ``bt_model_prob_*`` is present.

    Args:
        detailed_results: Full backtest result rows.

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
        "Avg. ROI per Bet [%] (macro)": calculate_avg_roi_per_bet_macro(
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
        "Brier Score [-]": calculate_brier_score(detailed_results),
        "Log Loss [-]": calculate_log_loss(detailed_results),
        "ECE [-]": calculate_ece(detailed_results),
        "Highest Winning Odds [-]": highest_winning_odds,
        "Highest Losing Odds [-]": highest_losing_odds,
        "Average Stake [$]": calculate_average_stake(bet_placed),
        "Best Bet [$]": best_bet,
        "Worst Bet [$]": worst_bet,
        "Total Bets": n_bets,
        "Total Opportunities": n_opportunities,
        "Bet Frequency [%]": bet_frequency,
    }
