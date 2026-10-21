"""
Cross-check every metric against an independent reference implementation.

The reference below is deliberately naive: plain Python loops over result
rows, with sklearn for Brier score and log loss. Each metric from
``calculate_all_metrics`` must match it on randomized backtests that mix
strategies, 2-way and 3-way markets, same-day bets, gaps, and multiple
calendar years. A formula change in the package without a matching change
here fails this test.
"""
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import brier_score_loss, log_loss

from backtestbuddy.backtest.sport_backtest import PredictionBacktest
from backtestbuddy.metrics.sport_metrics import calculate_all_metrics
from backtestbuddy.strategies.sport_strategies import (
    FixedStake,
    KellyCriterion,
    OddsFilter,
    UnitStake,
    ValueBet,
)

N_SCENARIOS = 20
NOT_COMPARED = {
    "Backtest Start Date",
    "Backtest End Date",
    "Backtest Duration",
    "Max. Drawdown Duration [bets]",
}


def _mean(values):
    return sum(values) / len(values)


def _placed(rows):
    return [r for r in rows if r["bt_stake"] > 0 and r["bt_bet_on"] != -1]


def _years(rows):
    dates = [pd.Timestamp(r["bt_date_column"]) for r in rows]
    return (max(dates) - min(dates)).days / 365.25


def _daily_returns(rows):
    """Compounded daily returns of placed bets; days without bets are absent."""
    buckets = {}
    for r in _placed(rows):
        day = pd.Timestamp(r["bt_date_column"]).normalize()
        buckets.setdefault(day, []).append(r["bt_profit"] / r["bt_starting_bankroll"])
    returns = []
    for day in sorted(buckets):
        growth = 1.0
        for x in buckets[day]:
            growth *= 1 + x
        returns.append(growth - 1)
    return returns


def _periods_per_year(rows):
    dates = [pd.Timestamp(r["bt_date_column"]) for r in rows]
    span_days = (max(dates) - min(dates)).days + 1
    return len(_daily_returns(rows)) / (span_days / 365.25)


def _sharpe(rows, periods):
    r = _daily_returns(rows)
    if len(r) < 2:
        return 0.0
    m = _mean(r)
    sd = math.sqrt(sum((v - m) ** 2 for v in r) / (len(r) - 1))
    return 0.0 if sd == 0 else m / sd * math.sqrt(periods)


def _sortino(rows, periods):
    r = _daily_returns(rows)
    downside = math.sqrt(_mean([min(0.0, v) ** 2 for v in r]))
    if downside == 0:
        return math.inf if _mean(r) > 0 else 0.0
    return _mean(r) / downside * math.sqrt(periods)


def _max_drawdown(rows):
    peak = rows[0]["bt_starting_bankroll"]
    worst = 0.0
    for r in rows:
        peak = max(peak, r["bt_ending_bankroll"])
        worst = max(worst, (peak - r["bt_ending_bankroll"]) / peak)
    return worst


def _calmar(rows):
    wealth, peak, worst = 1.0, 1.0, 0.0
    for v in _daily_returns(rows):
        wealth *= 1 + v
        peak = max(peak, wealth)
        worst = max(worst, (peak - wealth) / peak)
    annual = wealth ** (1 / _years(rows)) - 1
    if worst == 0:
        return math.inf if annual > 0 else 0.0
    return annual / worst


def _yearly_roi_weighted(rows):
    by_year = {}
    for r in rows:
        by_year.setdefault(pd.Timestamp(r["bt_date_column"]).year, []).append(r)
    total, weight_sum = 0.0, 0
    for group in by_year.values():
        start = group[0]["bt_starting_bankroll"]
        if not start:
            continue
        dates = [pd.Timestamp(r["bt_date_column"]) for r in group]
        weight = (max(dates) - min(dates)).days + 1
        total += weight * (group[-1]["bt_ending_bankroll"] / start - 1) * 100
        weight_sum += weight
    return total / weight_sum


def _ece(pairs):
    bins = {}
    for p, y in pairs:
        bins.setdefault(max(0, math.ceil(p * 10) - 1), []).append((p, y))
    return sum(
        len(b) / len(pairs) * abs(_mean([y for _, y in b]) - _mean([p for p, _ in b]))
        for b in bins.values()
    )


def _fair(odds):
    """Multiplicative margin removal for one market."""
    implied = [1 / o for o in odds]
    return [q / sum(implied) for q in implied]


def reference_metrics(rows, n_outcomes):
    placed = _placed(rows)
    market_pairs = [
        (p, 1.0 if r["bt_actual_outcome"] == k else 0.0)
        for r in rows
        for k, p in enumerate(_fair([r[f"bt_odd_{i}"] for i in range(n_outcomes)]))
    ]
    closing = [
        (r["bt_odds"], [r[f"bt_closing_odd_{i}"] for i in range(n_outcomes)], int(r["bt_bet_on"]))
        for r in placed
    ]
    stake = sum(r["bt_stake"] for r in placed)
    profit = sum(r["bt_profit"] for r in placed)
    selected = [
        (r[f"bt_model_prob_{int(r['bt_bet_on'])}"], 1.0 if r["bt_win"] else 0.0, r)
        for r in placed
    ]
    expected = sum(r["bt_stake"] * (p * r["bt_odds"] - 1) for p, _, r in selected)
    placed_pairs = [(p, y) for p, y, _ in selected]
    all_pairs = [
        (r[f"bt_model_prob_{k}"], 1.0 if r["bt_actual_outcome"] == k else 0.0)
        for r in rows
        for k in range(n_outcomes)
    ]
    roi = rows[-1]["bt_ending_bankroll"] / rows[0]["bt_starting_bankroll"] - 1
    years = _years(rows)
    periods = _periods_per_year(rows)

    def brier(pairs):
        return brier_score_loss([y for _, y in pairs], [p for p, _ in pairs])

    def logloss(pairs):
        return log_loss([y for _, y in pairs], [p for p, _ in pairs], labels=[0, 1])

    return {
        "ROI [%]": roi * 100,
        "Yield [%]": profit / stake * 100,
        "Expected Yield [%]": expected / stake * 100,
        "Avg. ROI per Bet [%] (micro)": _mean(
            [r["bt_profit"] / r["bt_stake"] * 100 for r in placed]
        ),
        "Avg. ROI per Year [%] (micro)": _yearly_roi_weighted(rows),
        "Avg. ROI per Year [%] (macro)": roi / years * 100,
        "CAGR [%]": ((1 + roi) ** (1 / years) - 1) * 100,
        "Risk-Adjusted Annual ROI [-]": (roi / years) / _max_drawdown(rows),
        "Total Profit [$]": sum(r["bt_profit"] for r in rows),
        "Expected Profit [$]": expected,
        "Realized vs Expected Profit [$]": profit - expected,
        "Bankroll Final [$]": rows[-1]["bt_ending_bankroll"],
        "Bankroll Peak [$]": max(r["bt_ending_bankroll"] for r in rows),
        "Bankroll Valley [$]": min(r["bt_ending_bankroll"] for r in rows),
        "Sharpe Ratio (365.25) [-]": _sharpe(rows, 365.25),
        "Sharpe Ratio (252) [-]": _sharpe(rows, 252),
        "Sharpe Ratio (obs/year) [-]": _sharpe(rows, periods),
        "Sortino Ratio (365.25) [-]": _sortino(rows, 365.25),
        "Sortino Ratio (252) [-]": _sortino(rows, 252),
        "Sortino Ratio (obs/year) [-]": _sortino(rows, periods),
        "Calmar Ratio [-]": _calmar(rows),
        "Max Drawdown [%]": _max_drawdown(placed) * 100,
        "Win Rate [%]": sum(1 for r in placed if r["bt_win"]) / len(placed) * 100,
        "Average Odds [-]": _mean([r["bt_odds"] for r in placed]),
        "Average Implied Prob [-]": _mean([1 / r["bt_odds"] for r in placed]),
        "Average Overround [%]": _mean([
            (sum(1 / r[f"bt_odd_{k}"] for k in range(n_outcomes)) - 1) * 100
            for r in placed
        ]),
        "Brier Score (placed) [-]": brier(placed_pairs),
        "Brier Score (all games) [-]": brier(all_pairs),
        "Log Loss (placed) [-]": logloss(placed_pairs),
        "Log Loss (all games) [-]": logloss(all_pairs),
        "ECE (placed) [-]": _ece(placed_pairs),
        "ECE (all games) [-]": _ece(all_pairs),
        "Brier Score (market, all games) [-]": brier(market_pairs),
        "Log Loss (market, all games) [-]": logloss(market_pairs),
        "CLV [%]": _mean([taken / close[k] - 1 for taken, close, k in closing]) * 100,
        "CLV (no-vig) [%]": _mean(
            [taken * _fair(close)[k] - 1 for taken, close, k in closing]
        ) * 100,
        "Beat Closing Line [%]": _mean(
            [1.0 if taken > close[k] else 0.0 for taken, close, k in closing]
        ) * 100,
        "Highest Winning Odds [-]": max(r["bt_odds"] for r in placed if r["bt_win"]),
        "Highest Losing Odds [-]": max(r["bt_odds"] for r in placed if not r["bt_win"]),
        "Average Stake [$]": stake / len(placed),
        "Best Bet [$]": max(r["bt_profit"] for r in placed),
        "Worst Bet [$]": min(r["bt_profit"] for r in placed),
        "Total Bets": len(placed),
        "Total Opportunities": len(rows),
        "Bet Frequency [%]": len(placed) / len(rows) * 100,
    }


def random_backtest(seed):
    """Run a randomized backtest; returns (results, number of outcomes)."""
    rng = np.random.default_rng(seed)
    n_outcomes = 2 if seed % 2 else 3
    n = int(rng.integers(150, 500))
    gaps = rng.choice([0, 0, 1, 2, 7, 30], size=n)
    dates = (
        pd.Timestamp("2019-03-01")
        + pd.to_timedelta(np.cumsum(gaps), unit="D")
        + pd.to_timedelta(rng.integers(0, 20, n), unit="h")
    )
    true_p = rng.dirichlet(np.ones(n_outcomes) * 2, size=n)
    margin = rng.uniform(1.02, 1.08, n)[:, None]
    odds = np.round(1 / (true_p * margin), 2).clip(1.01)
    model_p = np.clip(true_p + rng.normal(0, 0.06, true_p.shape), 0.01, None)
    model_p /= model_p.sum(axis=1, keepdims=True)
    data = pd.DataFrame({
        "date": dates,
        "outcome": [rng.choice(n_outcomes, p=p) for p in true_p],
        "pred": model_p.argmax(axis=1),
    })
    # Closing line drifts a few percent away from the odds taken
    closing = np.round(odds * np.exp(rng.normal(0, 0.04, odds.shape)), 2).clip(1.01)
    for k in range(n_outcomes):
        data[f"odds_{k}"] = odds[:, k]
        data[f"close_{k}"] = closing[:, k]
        data[f"p_{k}"] = model_p[:, k]
    strategy = [
        KellyCriterion(0.5, 0.1),
        ValueBet(0.02, 0.03),
        FixedStake(25),
        UnitStake(20, "win"),
        OddsFilter(1.6, 4.0, KellyCriterion(1.0, 0.05)),
    ][seed % 5]
    backtest = PredictionBacktest(
        data,
        [f"odds_{k}" for k in range(n_outcomes)],
        "outcome",
        "date",
        "pred",
        strategy=strategy,
        model_prob_columns=[f"p_{k}" for k in range(n_outcomes)],
        closing_odds_columns=[f"close_{k}" for k in range(n_outcomes)],
    )
    backtest.run()
    return backtest.detailed_results, n_outcomes


@pytest.mark.integration
@pytest.mark.metrics
@pytest.mark.parametrize("seed", range(N_SCENARIOS))
def test_metrics_match_reference(seed):
    """Every metric equals the naive reference on a random backtest."""
    results, n_outcomes = random_backtest(seed)
    if (results["bt_stake"] > 0).sum() < 2 or not results["bt_win"].any():
        pytest.skip("Scenario placed too few bets to compare all metrics")

    actual = calculate_all_metrics(results)
    expected = reference_metrics(results.to_dict("records"), n_outcomes)

    assert set(actual) - NOT_COMPARED == set(expected)
    for name, value in expected.items():
        assert float(actual[name]) == pytest.approx(value, rel=1e-9, abs=1e-9), name
