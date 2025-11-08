# Plots Module

The Plots module provides visualization functionality for backtest results. Currently, the only module that implements these plotting functions is the `sport_plots.py` file.

## Sport Plots

The `plots` module contains functions for visualizing backtest results and analyzing betting patterns. All plotting functions use Plotly to create interactive visualizations.

### Functions

#### `plot_backtest`

Creates a comprehensive plot of the backtest results, including the main strategy performance, bookie strategy comparison, and Max Drawdown visualization.

**Signature:**

```python
def plot_backtest(backtest: Any) -> go.Figure
```

**Description:**

This function generates a multi-panel plot with three subplots:

1. **Bankroll Over Time**: Shows the bankroll progression for the main strategy with:
   - Bankroll line over time
   - Win markers (green triangles) indicating winning bets
   - Loss markers (red triangles) indicating losing bets
   - Max Drawdown highlighted with a shaded region
   - Interactive hover information showing game index, date, starting/ending bankroll, stake, and stake percentage

2. **ROI**: Displays the Return on Investment for each bet as scatter points
   - Shows individual bet performance
   - Includes a zero line reference

3. **Stake Percentage**: Shows the stake percentage (stake/bankroll) for each bet as a bar chart
   - Helps visualize bet sizing over time

**Parameters:**

- `backtest` (Any): An instance of a Backtest class (e.g., `ModelBacktest` or `PredictionBacktest`) containing the results. The backtest must have been run (i.e., `backtest.run()` must have been called) and must have a `detailed_results` attribute populated.

**Returns:**

- `go.Figure`: A Plotly figure object containing the backtest results plot. You can call `.show()` on this figure to display it, or use it in other Plotly operations.

**Features:**

- Only includes bets that were actually placed (filters out non-bets)
- Displays all calculated metrics as annotations on the right side of the plot
- Interactive hover tooltips with detailed bet information
- Max Drawdown visualization with annotation showing percentage and duration

**Example Usage:**

```python
from backtestbuddy.backtest.sport_backtest import PredictionBacktest
from backtestbuddy.strategies.sport_strategies import FixedStake

# ... setup backtest ...
backtest = PredictionBacktest(...)
backtest.run()

# Plot the results
backtest.plot()  # Uses plot_backtest internally and displays the figure

# Or use the function directly for more control
from backtestbuddy.plots.sport_plots import plot_backtest
fig = plot_backtest(backtest)
fig.show()  # Display the plot
fig.write_html("backtest_results.html")  # Save to HTML file
```

#### `plot_odds_histogram`

Creates a histogram plot of the odds distribution, showing the frequency of winning and losing bets at different odds ranges, along with break-even win rate indicators.

**Signature:**

```python
def plot_odds_histogram(backtest: Any, num_bins: Optional[int] = None) -> go.Figure
```

**Description:**

This function generates a histogram visualization that helps analyze betting patterns:

- **Stacked bars** showing winning bets (green) and losing bets (red) at different odds ranges
- **Break-even win rate lines** (blue dotted) indicating the minimum win rate needed to break even at each odds level
- **Average odds** vertical line (blue dashed) with annotation
- **Median odds** vertical line (purple dotted) with annotation

The histogram uses logarithmic binning to better represent the distribution of odds, which typically span a wide range.

**Parameters:**

- `backtest` (Any): An instance of a Backtest class containing the results. The backtest must have been run and must have a `detailed_results` attribute populated.
- `num_bins` (Optional[int]): The number of bins to use for the histogram. If `None`, an automatic binning strategy is used (square root rule: `sqrt(number_of_bets)`). Defaults to `None`.

**Returns:**

- `go.Figure`: A Plotly figure object containing the odds histogram. You can call `.show()` on this figure to display it, or use it in other Plotly operations.

**Features:**

- Logarithmic binning for better visualization of odds distribution
- Break-even win rate calculation: For odds `o`, the break-even win rate is `1/o`
- Only includes bets that were actually placed
- Interactive hover tooltips
- Visual indicators for average and median odds

**Example Usage:**

```python
from backtestbuddy.backtest.sport_backtest import PredictionBacktest
from backtestbuddy.strategies.sport_strategies import FixedStake

# ... setup backtest ...
backtest = PredictionBacktest(...)
backtest.run()

# Plot odds distribution
fig = backtest.plot_odds_distribution()  # Uses plot_odds_histogram internally
fig.show()

# Or use the function directly with custom binning
from backtestbuddy.plots.sport_plots import plot_odds_histogram
fig = plot_odds_histogram(backtest, num_bins=20)
fig.show()
fig.write_html("odds_distribution.html")  # Save to HTML file
```

## Using Plot Functions

### Direct Method Calls

Both plotting functions can be called directly from the backtest instance:

```python
# After running a backtest
backtest.run()

# Plot backtest results (displays automatically)
backtest.plot()

# Plot odds distribution (returns figure, doesn't auto-display)
fig = backtest.plot_odds_distribution()
fig.show()
```

### Importing Functions Directly

You can also import and use the plotting functions directly:

```python
from backtestbuddy.plots.sport_plots import plot_backtest, plot_odds_histogram

# After running backtest
fig1 = plot_backtest(backtest)
fig1.show()

fig2 = plot_odds_histogram(backtest, num_bins=15)
fig2.show()
```

### Customizing Plots

Since the functions return Plotly `Figure` objects, you can customize them further:

```python
fig = backtest.plot_odds_distribution()

# Customize the layout
fig.update_layout(
    title="My Custom Title",
    width=1200,
    height=600
)

# Update axis labels
fig.update_xaxes(title_text="Custom X Label")
fig.update_yaxes(title_text="Custom Y Label")

fig.show()
```

### Exporting Plots

Plotly figures can be exported to various formats:

```python
fig = backtest.plot()

# Save as HTML (interactive)
fig.write_html("backtest_results.html")

# Save as static image (requires kaleido)
fig.write_image("backtest_results.png")
fig.write_image("backtest_results.pdf")
```

## Integration with Backtest Classes

Both `ModelBacktest` and `PredictionBacktest` classes provide convenient methods that wrap these plotting functions:

- `backtest.plot()` - Calls `plot_backtest()` and displays the figure
- `backtest.plot_odds_distribution(num_bins)` - Calls `plot_odds_histogram()` and returns the figure

These methods handle the integration automatically, ensuring the backtest has been run before plotting.
