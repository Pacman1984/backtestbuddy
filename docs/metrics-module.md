# Metrics Module

`backtest.calculate_metrics()` (or `calculate_all_metrics(detailed_results, margin_method="multiplicative")` from `backtestbuddy.metrics.sport_metrics`) returns one dictionary with every metric below. This page lists them first in a quick reference, then explains each one with its formula and the results columns it reads.

A **placed bet** is a row with `bt_stake > 0` and `bt_bet_on != -1`. Values that cannot be computed, e.g. CLV without closing odds, are `nan` and appear as "—" in the plot table.

## Quick reference

| Key | Meaning | Better when |
|---|---|---|
| `Backtest Start Date`, `End Date`, `Duration` | First date, last date, and the span between | |
| `ROI [%]` | Bankroll growth from start to end | higher |
| `Yield [%]` | Profit per unit staked | higher |
| `Expected Yield [%]` | Yield your model expected on the bets placed | higher |
| `Avg. ROI per Bet [%] (micro)` | Mean of each bet's profit / stake | higher |
| `Avg. ROI per Year [%] (micro)` | Mean calendar-year ROI, weighted by days covered | higher |
| `Avg. ROI per Year [%] (macro)` | Total ROI / years (linear) | higher |
| `CAGR [%]` | Compound annual growth rate of the bankroll | higher |
| `Risk-Adjusted Annual ROI [-]` | Annual ROI (macro) / max drawdown | higher |
| `Total Profit [$]` | Sum of all profits | higher |
| `Expected Profit [$]` | Profit your model expected on the bets placed | higher |
| `Realized vs Expected Profit [$]` | Actual minus expected profit (luck plus miscalibration) | near 0 |
| `Bankroll Final / Peak / Valley [$]` | Last, highest, and lowest bankroll | higher |
| `Sharpe Ratio (365.25 / 252 / obs/year) [-]` | Annualized return / volatility, three annualization scales | higher |
| `Sortino Ratio (365.25 / 252 / obs/year) [-]` | Like Sharpe, but only downside volatility counts | higher |
| `Calmar Ratio [-]` | Annual growth / max drawdown of the return curve | higher |
| `Max Drawdown [%]` | Worst fall from a bankroll peak | lower |
| `Max. Drawdown Duration [bets]` | Bets from that peak to the trough | lower |
| `Win Rate [%]` | Share of placed bets that won | context |
| `Average Odds [-]` | Mean odds of placed bets | context |
| `Average Implied Prob [-]` | Mean `1 / odds` of placed bets, margin included | context |
| `Average Overround [%]` | Mean bookmaker margin of the markets bet | lower |
| `Brier Score (placed / all games) [-]` | Squared error of model probabilities | lower |
| `Log Loss (placed / all games) [-]` | Log loss of model probabilities | lower |
| `ECE (placed / all games) [-]` | Calibration error of model probabilities | lower |
| `Brier Score / Log Loss (market, all games) [-]` | Same scores for the margin-free odds; the baseline to beat | (compare) |
| `CLV [%]` | Mean of odds taken / closing odds − 1 | higher |
| `CLV (no-vig) [%]` | Expected return against the margin-free closing line | higher |
| `Beat Closing Line [%]` | Share of bets taken above the closing odds | higher |
| `Highest Winning / Losing Odds [-]` | Largest odds that won / lost | context |
| `Average Stake [$]` | Mean stake of placed bets | context |
| `Best Bet / Worst Bet [$]` | Largest single profit / loss | context |
| `Total Bets`, `Total Opportunities`, `Bet Frequency [%]` | Bets placed, rows, and their ratio | context |

## Overall Sport Performance Metrics

### ROI (Return on Investment)

- Description: Measures the profitability of your strategy relative to the initial investment.
- Formula: $ROI = \frac{Final Bankroll - Initial Bankroll}{Initial Bankroll} \times 100\%$
- Calculation: `(Final Bankroll - Initial Bankroll) / Initial Bankroll * 100`
- Columns used: `bt_starting_bankroll` (first row), `bt_ending_bankroll` (last row)

### Total Profit

- Description: The total amount of money gained or lost during the backtest period.
- Formula: $Total Profit = \sum_{i=1}^{n} Profit_i$
- Calculation: Sum of all individual bet profits
- Column used: `bt_profit`

### Yield

- Description: Industry betting yield: total profit over total amount staked. Not bankroll ROI.
- Formula: $Yield = \frac{\sum profit}{\sum stake} \times 100\%$
- Calculation: Placed bets only (`bt_stake > 0` and `bt_bet_on != -1`)
- Columns used: `bt_profit`, `bt_stake`, `bt_bet_on`
- Note: With unequal stakes, Yield differs from Avg. ROI per Bet (micro). Yield is the per-bet return measure; the former `Avg. ROI per Bet [%] (macro)` (bankroll ROI / number of bets) was removed from `calculate_all_metrics` because bankroll ROI compounds across bets.

### Avg. ROI per Bet (micro)

- Description: Mean of each placed bet's own return, so every bet counts equally regardless of stake.
- Formula: $\overline{profit_i / stake_i} \times 100\%$
- Columns used: `bt_profit`, `bt_stake`, `bt_bet_on`
- Example: stakes 100, profits 20, −50, 30 → mean of 20%, −50%, 30% = 0%.

### Avg. ROI per Year (micro / macro)

- Micro: duration-weighted mean of calendar-year ROIs. Each year's ROI is `(year_end / year_start - 1) * 100`, weighted by the days that year covers (first to last row in the year, inclusive). A two-week partial year therefore does not count as much as a full year.
- Macro: total ROI divided by the backtest length in years (linear, not compounded; see CAGR for the geometric rate).
- Columns used: `bt_starting_bankroll`, `bt_ending_bankroll`, `bt_date_column`

### CAGR (Compound Annual Growth Rate)

- Description: The constant yearly growth rate that turns the starting bankroll into the final bankroll.
- Formula: $CAGR = \left(\frac{Final\ Bankroll}{Initial\ Bankroll}\right)^{1/years} - 1$, in percent, with $years = (\max date - \min date).days / 365.25$
- Edge cases: `0.0` for an empty frame or a zero-length span; `-100.0` if the bankroll ends at 0
- Columns used: `bt_starting_bankroll` (first row), `bt_ending_bankroll` (last row), `bt_date_column`
- Example: 1000 → 1210 over two years → 10%.

### Expected Yield / Expected Profit / Realized vs Expected Profit

- Description: Model-implied P&L vs what actually happened, using the probability of the **selected** outcome.
- Per-bet expected profit: $stake \times (p \cdot odds - 1)$
- Keys:
  - `Expected Profit [$]`: sum of per-bet expected profit
  - `Expected Yield [%]`: expected profit / stake on those rows, times 100
  - `Realized vs Expected Profit [$]`: `sum(profit) - expected profit` on the same rows
- Columns used: `bt_stake`, `bt_odds`, `bt_profit`, `bt_bet_on`, `bt_model_prob_{k}`
- Edge cases: `0.0` if no placed bets; `nan` if model probabilities are missing

### Bankroll Final

- Description: The final value of your bankroll at the end of the backtest period.
- Calculation: Last value in the `bt_ending_bankroll` column
- Column used: `bt_ending_bankroll`

### Bankroll Peak

- Description: The highest value your bankroll reached during the backtest period.
- Formula: $Bankroll Peak = \max(bt\_ending\_bankroll)$
- Calculation: Maximum value in the `bt_ending_bankroll` column
- Column used: `bt_ending_bankroll`

### Bankroll Valley

- Description: The lowest value your bankroll reached during the backtest period.
- Formula: $Bankroll Valley = \min(bt\_ending\_bankroll)$
- Calculation: Minimum value in the `bt_ending_bankroll` column
- Column used: `bt_ending_bankroll`

## Risk-Adjusted Performance Metrics

### Sharpe Ratio

- Description: Measures risk-adjusted return using period returns with a risk-free rate of 0.
- Formula: $Sharpe = \frac{\bar{r} \cdot P}{s \cdot \sqrt{P}} = \frac{\bar{r}}{s}\sqrt{P}$ where $\bar{r}$ and $s$ are the mean and sample standard deviation (`ddof=1`) of compounded period returns, and $P$ is `output_period`.
- Calculation:
  1. Per-bet return of **placed bets only**: `r = bt_profit / bt_starting_bankroll`
  2. Compound inside each `return_period`-day bucket: `prod(1 + r) - 1`. Periods without a placed bet are skipped, whether they are empty calendar days or contain only skipped opportunities, so the ratio does not depend on whether skipped rows are in the data.
  3. Annualize mean with `* output_period` and sample std with `* sqrt(output_period)`
  4. Sharpe = annualized mean / annualized std (0 if fewer than two periods or std is 0)
- Columns used: `bt_profit`, `bt_starting_bankroll`, `bt_date_column`
- Annualization (`P`). Same series, three reported keys:
  - `Sharpe Ratio (365.25) [-]`: `P = 365.25` (calendar year; sports default of `calculate_sharpe_ratio`)
  - `Sharpe Ratio (252) [-]`: `P = 252` (equity trading-year; 0.1.13 default)
  - `Sharpe Ratio (obs/year) [-]`: `P = n_periods / years` with `years = ((max_date - min_date).days + return_period) / 365.25` (sample density of periods with placed bets; `n` consecutive betting days give `P = 365.25`)
- Note: This is not excess-return Sharpe with a non-zero $r_f$. Calmar and CAGR already use calendar years. 252 overstates annualization when betting days are sparse; obs/year does not. Call `calculate_sharpe_ratio(..., output_period=P)` for any other `P`.

### Sortino Ratio

- Description: Sharpe-style ratio that only penalizes returns below a target $\tau$ (default 0).
- Formula: $Sortino = \frac{\text{annualized mean excess return}}{\text{annualized downside deviation}}$
- Downside deviation: $\sqrt{\text{mean}(\min(0, r - \tau)^2)}$ over **all** periods (zeros for upside periods), then $\times \sqrt{P}$. Uses $N$ in the mean, not sample $N-1$.
- Calculation:
  1. Same compounded period returns as Sharpe
  2. `excess = period_return - target_return`
  3. Downside deviation as above, annualized with `sqrt(output_period)`
  4. Mean excess annualized with `* output_period`
  5. Sortino = annualized mean excess / annualized downside deviation
- Edge cases: `inf` if downside deviation is 0 and mean excess > 0; `0.0` if downside is 0 and mean excess is not positive
- Columns used: `bt_profit`, `bt_starting_bankroll`, `bt_date_column`
- Annualization: same three `P` values and key names as Sharpe (`Sortino Ratio (365.25) [-]`, `Sortino Ratio (252) [-]`, `Sortino Ratio (obs/year) [-]`).

### Calmar Ratio

- Description: Geometric annual return divided by the magnitude of maximum drawdown on the **compounded return curve** (not the bankroll equity curve used by Max Drawdown).
- Formula: $Calmar = \frac{R_{annual}}{|\text{return-curve max DD}|}$
- Calculation:
  1. Same compounded period returns as Sharpe
  2. Wealth index starts at 1, then `cumulative = (1 + r).cumprod()` so the first period can be a drawdown
  3. Return-curve max DD: `min((cumulative - cummax) / cummax)`
  4. $R_{total} = cumulative[-1] - 1$
  5. $K_{years} = (\max date - \min date).days / 365.25$
  6. $R_{annual} = (1 + R_{total})^{1/K_{years}} - 1$
  7. Calmar = $R_{annual} / |max DD|$
- Edge cases: `inf` if drawdown is 0 and $R_{annual} > 0$; `0.0` if there is no data or no drawdown with non-positive return
- Columns used: `bt_profit`, `bt_starting_bankroll`, `bt_date_column`

### Risk-Adjusted Annual ROI

- Description: Measures the annual return per unit of maximum drawdown risk as a unitless ratio. Returns `inf` when there is no drawdown.
- Formula: $Risk-Adjusted\ Annual\ ROI = \frac{Average\ Yearly\ ROI\ (decimal)}{|Maximum\ Drawdown|}$
- Calculation:
  1. Calculate average yearly ROI as percentage using macro method:
     ```python
     avg_yearly_roi_pct = calculate_avg_roi_per_year_macro(detailed_results)
     ```
  2. Convert to decimal: `avg_yearly_roi = avg_yearly_roi_pct / 100.0`
  3. Maximum drawdown with `calculate_max_drawdown(detailed_results)`: the bankroll curve starting from the opening bankroll, as a negative decimal (see [Max Drawdown](#max-drawdown)).
  4. Risk-Adjusted Annual ROI:
     - If `max_drawdown == 0` and yearly ROI > 0: `float('inf')`
     - If `max_drawdown == 0` and yearly ROI is not positive: `0.0`
     - Otherwise: `avg_yearly_roi / abs(max_drawdown)`
- Output: Unitless ratio (e.g., 0.667 means 0.667 units of annual return per unit of drawdown)
- Interpretation:
  - Positive values: Strategy is profitable
  - Negative values: Strategy is unprofitable
  - Higher absolute values: Better risk-adjusted performance
  - `inf`: Perfect performance with no drawdown
- Columns used: `bt_starting_bankroll`, `bt_ending_bankroll`, `bt_date_column`

## Drawdown Analysis

### Max Drawdown

- Description: Largest peak-to-trough decline on the **bankroll** equity curve (`bt_ending_bankroll`).
- Formula: magnitude $= \max((peak - equity) / peak)$; reported as a **positive** percent in `calculate_all_metrics` (`Max Drawdown [%]`). The helper `calculate_max_drawdown` returns the signed decimal (`-magnitude`).
- Opening bankroll: `bt_starting_bankroll` of the first row is the first peak, so losses on the first bets count as drawdown.
- Peak rule: duration starts at the **last** peak at or before the trough (`end - start + 1` rows).
- Row filter: `calculate_all_metrics` uses placed bets only (`bt_stake > 0` and `bt_bet_on != -1`).
- Column used: `bt_ending_bankroll`

This is not the same as Calmar's return-curve drawdown.

### Max Drawdown Duration

- Description: Inclusive number of **placed bets** from the last peak before the max-drawdown trough to that trough.
- Calculation: `duration = end - start + 1` on the bet-placed equity curve
- Column used: `bt_ending_bankroll`

## Betting Performance Metrics

### Win Rate

- Description: The percentage of **placed** bets that won.
- Formula: $Win Rate = \frac{Winning Bets}{Placed Bets} \times 100\%$
- Placed bet: `bt_stake > 0` and `bt_bet_on != -1` (same rule as `_simulate_bet`)
- Column used: `bt_win`, `bt_stake`, `bt_bet_on`

### Average Odds

- Description: The average odds of all bets placed.
- Calculation: mean of `bt_odds` over placed bets
- Column used: `bt_odds`

### Highest Winning Odds

- Description: The highest odds of a winning bet.
- Calculation: `winning_bets['bt_odds'].max()`
- Columns used: `bt_odds`, `bt_win`

### Highest Losing Odds

- Description: The highest odds of a losing bet.
- Calculation: `losing_bets['bt_odds'].max()`
- Columns used: `bt_odds`, `bt_win`

### Average Stake

- Description: The average amount staked per placed bet (zero stakes excluded).
- Calculation: mean of `bt_stake` where `bt_stake > 0`
- Column used: `bt_stake`

### Best Bet

- Description: The highest profit achieved from a single bet.
- Calculation: maximum `bt_profit` over placed bets
- Column used: `bt_profit`

### Worst Bet

- Description: The largest loss incurred from a single bet.
- Calculation: minimum `bt_profit` over placed bets
- Column used: `bt_profit`

## Additional Information

### Total Bets

- Description: The total number of bets placed during the backtest period.
- Calculation: Count of rows where `bt_stake > 0` and `bt_bet_on != -1`
- Columns used: `bt_stake`, `bt_bet_on`

### Total Opportunities

- Description: The total number of betting opportunities during the backtest period.
- Calculation: Count of all rows in the detailed results dataframe
- All columns

### Bet Frequency

- Description: The percentage of opportunities where a bet was placed.
- Formula: $Bet Frequency = \frac{Total Bets}{Total Opportunities} \times 100\%$
- Calculation: `(Total Bets / Total Opportunities) * 100`
- Columns used: `bt_stake`, `bt_bet_on`

### Backtest Duration

- Description: The time period covered by the backtest.
- Formula: $Backtest Duration = End Date - Start Date$
- Calculation: `End Date - Start Date`
- Column used: `bt_date_column`

## Probability and market

### Average Implied Probability

- Description: Mean raw implied probability of the odds you took: `1 / bt_odds`. Vig is not stripped.
- Formula: $\overline{1 / odds}$
- Columns used: `bt_odds`, `bt_stake`, `bt_bet_on`
- Edge cases: `nan` if no placed bets

### Average Overround

- Description: Mean book margin on the full outcome set for each placed-bet row.
- Formula: $\overline{\sum_k 1/odds_k - 1} \times 100\%$
- Columns used: `bt_odd_0`, `bt_odd_1`, … (not the selected `bt_odds` column alone)
- Edge cases: `nan` if those columns or placed bets are missing

### Calibration scopes

Brier, Log Loss, and ECE are reported in two scopes. Each function takes `scope="placed"` (default) or `scope="all"`; `plot_calibration` takes the same argument.

- `(placed)`: the model probability of the **selected outcome of placed bets**. These are the games where the strategy disagreed with the market, so this measures calibration where money was at risk, not the model overall.
- `(all games)`: every outcome of every game with model probabilities, bet or not. Each `(game, outcome k)` pair is one binary prediction with `y = 1` if outcome `k` happened. Use this for the model's overall calibration.

### Brier Score

- Description: Mean squared error of model probability vs result, in the chosen scope.
- Formula: $\mathrm{Brier} = \mathrm{mean}((p - y)^2)$ with $y=1$ if the outcome happened, else $0$
- Columns used: `bt_model_prob_{k}`, `bt_bet_on`, `bt_win` (placed); `bt_model_prob_{k}`, `bt_actual_outcome` (all games)
- Edge cases: `0.0` if no placed bets (placed scope); `nan` if no model probabilities

### Log Loss

- Description: Binary log loss of model probabilities in the chosen scope. `p` clipped to `[1e-15, 1-1e-15]`.
- Formula: $-\mathrm{mean}(y \log p + (1-y)\log(1-p))$
- Columns used: same as Brier
- Edge cases: same as Brier

### ECE (Expected Calibration Error)

- Description: Weighted mean absolute gap between accuracy and confidence in equal-width probability bins on `[0, 1]` (default 10 bins). Empty bins skipped.
- Formula: $\mathrm{ECE} = \sum_m (n_m / N)\,|\mathrm{acc}_m - \mathrm{conf}_m|$
- Columns used: same as Brier
- Edge cases: same as Brier; `calculate_ece(..., n_bins=)` to change the bin count

### Market baseline (Brier / Log Loss, market, all games)

- Description: The same all-games scores, but with the bookmaker's margin-free probabilities from `bt_odd_*` instead of the model's. If `Brier Score (all games)` is below `Brier Score (market, all games)`, the model predicts outcomes better than the line it bets into.
- Calculation: `remove_margin(bt_odd_*, method=margin_method)` per game, then one binary pair per (game, outcome) as in the `(all games)` scope.
- Columns used: `bt_odd_{k}`, `bt_actual_outcome`. No model needed.
- Edge cases: `nan` without odds columns or outcomes.

## Closing line value

These metrics need `closing_odds_columns` in the backtest (stored as `bt_closing_odd_{k}`) and are `nan` without them. The reference line can be any later or sharper set of odds; it does not have to be the literal close.

### CLV

- Description: Mean of how much better your odds were than the reference odds of the same outcome, over placed bets.
- Formula: $\mathrm{CLV} = \overline{odds_{taken} / odds_{close} - 1} \times 100\%$
- Example: took 2.20, closed 2.00 → +10%.

### CLV (no-vig)

- Description: Expected return per unit if the margin-free closing line is the true probability.
- Formula: $\overline{odds_{taken} \cdot p^{fair}_{close} - 1} \times 100\%$, with $p^{fair}_{close}$ from `remove_margin(bt_closing_odd_*, margin_method)`.
- Note: Stricter than raw CLV because the closing margin is removed. Positive no-vig CLV over many bets is strong evidence of an edge.

### Beat Closing Line

- Description: Percent of placed bets whose odds were strictly higher than the closing odds of the selected outcome. Equal odds do not count.
- Note: Read together with CLV. A high share with positive CLV means consistently better prices; a low share with positive CLV means a few large outliers carry the result.

Edge cases for all three: `0.0` if no bets were placed.

