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

- Description: Measures the risk-adjusted return of the betting strategy using excess returns.
- Formula: $Sharpe Ratio = \frac{Annualized Mean Excess Return}{Annualized Standard Deviation of Excess Returns}$ where excess returns are $r - r_f$ per period ($r_f$ is the risk-free rate, defaulting to 0 for sports betting).
- Calculation:
  1. Calculate returns: `returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']`
  2. Resample returns based on the return period (e.g., daily, weekly) using `returns.resample(f'{return_period}D').sum()`
  3. Calculate excess returns per period: `excess_returns = returns - r_f` (where $r_f = 0$ by default)
  4. Calculate annualized mean excess return: `annualized_mean_excess_return = excess_returns.mean() * output_period`
  5. Calculate annualized standard deviation of excess returns: `annualized_std_excess_return = excess_returns.std() * np.sqrt(output_period)`
  6. Sharpe Ratio = `annualized_mean_excess_return / annualized_std_excess_return`
- Columns used: `bt_profit`, `bt_starting_bankroll`, `bt_date_column`
- Note: Returns are computed from `bt_profit / bt_starting_bankroll` and resampled according to the return period before calculating excess returns and annualization.

### Sortino Ratio

- Description: Similar to Sharpe Ratio, but only considers downside risk relative to a target return threshold.
- Formula: $Sortino Ratio = \frac{Annualized Mean Excess Return}{Annualized Downside Deviation}$ where downside deviation measures volatility below the target return $\tau$.
- Target Return ($\tau$): Minimum acceptable return (MAR) threshold per period. Default is 0.0 (break-even), representing the natural baseline for sports betting.
- Downside Deviation: The annualized standard deviation of returns below the target return, calculated as $\sqrt{\text{mean}(\min(0, r - \tau)^2)}$ per period, then annualized.
- Calculation:
  1. Calculate returns: `returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']`
  2. Resample returns based on the return period (e.g., daily, weekly) using `returns.resample(f'{return_period}D').sum()`
  3. Calculate excess returns: `excess_returns = returns - target_return` (where $target\_return = \tau$, default 0.0)
  4. Calculate shortfalls: `shortfalls = min(0, excess_returns)` (negative excess returns only, zeros for positive)
  5. Calculate downside deviation per period: `downside_deviation_period = sqrt(mean(shortfalls^2))`
  6. Annualize downside deviation: `downside_deviation_annual = downside_deviation_period * sqrt(output_period)`
  7. Calculate annualized mean excess return: `annualized_mean_excess_return = excess_returns.mean() * output_period`
  8. Sortino Ratio = `annualized_mean_excess_return / downside_deviation_annual`
- Columns used: `bt_profit`, `bt_starting_bankroll`, `bt_date_column`
- Note: Returns are computed from `bt_profit / bt_starting_bankroll` and resampled according to the return period. The downside deviation uses the corrected formula $\sqrt{\text{mean}(\min(0, r - \tau)^2)}$ rather than filtering negative returns.

### Calmar Ratio

- Description: Measures the risk-adjusted return relative to maximum drawdown using geometric annual return.
- Formula: $Calmar Ratio = \frac{Geometric Annual Return}{|Maximum Drawdown|}$ where geometric annual return is computed over the lookback period and maximum drawdown magnitude is used in the denominator.
- Calculation:
  1. Calculate returns: `returns = detailed_results['bt_profit'] / detailed_results['bt_starting_bankroll']`
  2. Resample returns based on the return period (e.g., daily, weekly) using `returns.resample(f'{return_period}D').sum()`
  3. Calculate cumulative returns: `cumulative_returns = (1 + returns).cumprod()`
  4. Calculate maximum drawdown:

     ```python
     peak = cumulative_returns.cummax()
     drawdown = (cumulative_returns - peak) / peak
     max_drawdown = drawdown.min()
     ```

  5. Calculate total cumulative return: `R_total = cumulative_returns.iloc[-1] - 1`
  6. Calculate years from date range: `K_years = (date_range.max() - date_range.min()).days / 365.25` (or use provided `years` parameter)
  7. Calculate geometric annual return: `R_annual = (1 + R_total) ** (1 / K_years) - 1`
  8. Calmar Ratio = `R_annual / abs(max_drawdown)`
- Columns used: `bt_profit`, `bt_starting_bankroll`, `bt_date_column`
- Note: Returns are computed from `bt_profit / bt_starting_bankroll` and resampled according to the return period. The geometric annual return uses compound annual growth rate (CAGR) over the actual time period, not arithmetic mean scaling. The maximum drawdown magnitude (absolute value) is used in the denominator.

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
     - If `max_drawdown == 0`: returns `float('inf')`
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

- Description: The largest peak-to-trough decline in the bankroll.
- Formula: $Max Drawdown = \min(\frac{Trough Value - Peak Value}{Peak Value})$
- Calculation:

  ```python
  equity_curve = detailed_results['bt_ending_bankroll']
  peak = equity_curve.cummax()
  drawdown = (equity_curve - peak) / peak
  max_drawdown = drawdown.min()
  ```

- Column used: `bt_ending_bankroll`

### Average Drawdown

- Description: The average of all drawdowns during the backtest period.
- Calculation: Mean of all drawdowns calculated in the `calculate_drawdowns` function
- Column used: `bt_ending_bankroll`

### Max Drawdown Duration

- Description: The longest period (in number of bets) that the bankroll was in a drawdown state.
- Calculation: Longest consecutive sequence of declining `bt_ending_bankroll` values
- Column used: `bt_ending_bankroll`

### Average Drawdown Duration

- Description: The average duration of all drawdowns during the backtest period.
- Calculation: Mean duration of all drawdown periods calculated in the `calculate_drawdowns` function
- Column used: `bt_ending_bankroll`

### Median Drawdown Duration

- Description: The median duration of all drawdowns during the backtest period.
- Calculation: Median duration of all drawdown periods calculated in the `calculate_drawdowns` function
- Column used: `bt_ending_bankroll`

## Betting Performance Metrics

### Win Rate

- Description: The percentage of bets that resulted in a profit.
- Formula: $Win Rate = \frac{Number of Winning Bets}{Total Number of Bets} \times 100\%$
- Calculation: `(detailed_results['bt_win'].sum() / len(detailed_results)) * 100`
- Column used: `bt_win`

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

- Description: The average amount staked per bet.
- Calculation: `detailed_results['bt_stake'].mean()`
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
- Calculation: Count of rows in the detailed results dataframe where a bet was placed
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
