# Odds Module

The `odds` module converts bookmaker odds into fair probabilities. It does not depend on any data provider: pass decimal odds from any source.

## `remove_margin`

```python
import pandas as pd
from backtestbuddy import remove_margin

remove_margin([1.9, 1.9])                       # array([0.5, 0.5])
remove_margin([1.2, 4.5], method="shin")        # favourite gets more than multiplicative

# Many markets at once, e.g. the 1X2 columns of a DataFrame
df = pd.DataFrame({"odds_home": [2.10, 1.55], "odds_draw": [3.40, 4.00], "odds_away": [3.60, 6.50]})
fair = remove_margin(df[["odds_home", "odds_draw", "odds_away"]].to_numpy(), method="power")
df[["p_home", "p_draw", "p_away"]] = fair
```

- Input: decimal odds of one market, shape `(n_outcomes,)`, or of many markets, shape `(n_markets, n_outcomes)`. At least two outcomes per market.
- Output: fair probabilities with the same shape; each row sums to 1.
- Invalid rows (any odds `<= 1`, NaN, or infinite) return NaN for the whole row.

## Methods

Implied probabilities $q_i = 1 / odds_i$ sum to $S > 1$; the excess $S - 1$ is the overround (margin).

| Method | Formula | Notes |
|---|---|---|
| `multiplicative` (default) | $p_i = q_i / S$ | Spreads the margin in proportion to probability. Simple and common. |
| `additive` | $p_i = q_i - (S - 1) / n$ | Equal share of the margin per outcome. Long shots with a big margin can go negative. |
| `power` | $p_i = q_i^k$ with $k$ such that $\sum q_i^k = 1$ | Takes more margin from long shots. |
| `shin` | Shin (1993) with insider share $z$ | Takes more margin from long shots. Equal to `additive` for two-way markets. |

Power and Shin fit the favourite-longshot bias better than multiplicative: favourites get a higher fair probability, long shots a lower one. This matters most for markets with heavy favourites, e.g. tennis.

A book with $S \le 1$ (no margin) has no insider share; `shin` then falls back to `multiplicative`.

## Where the package uses it

Backtests take `margin_method=` (default `"multiplicative"`). It is used by:

- `CLV (no-vig) [%]`: odds taken vs the margin-free closing probability.
- `Brier Score (market, all games) [-]` and `Log Loss (market, all games) [-]`: the market baseline your model has to beat.

`calculate_all_metrics(results, margin_method=...)` accepts the same argument.
