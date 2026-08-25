# Metrics Module

The Metrics Module in BacktestBuddy provides a comprehensive set of performance metrics to evaluate your betting strategies. This document describes each metric, how it's calculated, and which columns from the detailed results dataframe are used. Right now, the only module that implements these metrics is the `sport_metrics.py` file.

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
- Formula: $Sharpe = \frac{\bar{r} \cdot P}{s \cdot \sqrt{P}}$ where $\bar{r}$ and $s$ are the mean and sample standard deviation (`ddof=1`) of compounded period returns, and $P$ is `output_period` (default 252).
- Calculation:
  1. Per-bet return: `r = bt_profit / bt_starting_bankroll`
  2. Compound inside each `return_period`-day bucket: `prod(1 + r) - 1` (empty calendar days are not filled with zeros)
  3. Annualize mean with `* output_period` and sample std with `* sqrt(output_period)`
  4. Sharpe = annualized mean / annualized std (0 if fewer than two periods or std is 0)
- Columns used: `bt_profit`, `bt_starting_bankroll`, `bt_date_column`
- Note: This is not excess-return Sharpe with a non-zero $r_f$. Default 252 is a trading-year scale; Calmar and CAGR use calendar years (`days / 365.25`).

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
  3. Calculate maximum drawdown:
     ```python
     equity_curve = detailed_results['bt_ending_bankroll']
     peak = equity_curve.cummax()
     drawdown = (equity_curve - peak) / peak
     max_drawdown = drawdown.min()
     ```
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
- Calculation: `detailed_results['bt_odds'].mean()`
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
- Calculation: `detailed_results['bt_profit'].max()`
- Column used: `bt_profit`

### Worst Bet

- Description: The largest loss incurred from a single bet.
- Calculation: `detailed_results['bt_profit'].min()`
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

These metrics provide a comprehensive overview of your betting strategy's performance, allowing you to assess its profitability, risk, and consistency. Use them to compare different strategies and optimize your approach to sports betting.
