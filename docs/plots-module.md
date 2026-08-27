# Plots Module

The Plots module provides visualization functionality for backtest results. Currently, the only module that implements these plotting functions is the `sport_plots.py` file.

## Sport Plots

The `plots` module contains functions for visualizing backtest results and analyzing betting patterns. All plotting functions use Plotly to create interactive visualizations.

### Functions

#### `plot_backtest`

Creates a four-panel plot of **placed bets** for the main strategy: bankroll (max drawdown window from the last peak to the trough), underwater drawdown, per-bet ROI, and stake as a percent of starting bankroll. A right-hand table shows `calculate_all_metrics`.

**Signature:**

```python
def plot_backtest(
    backtest: Any,
    x_axis: Literal["bet", "date"] = "bet",
    show_bookie: bool = True,
) -> go.Figure
```

**Description:**

1. **Bankroll Over Time**: Main strategy bankroll, win/loss markers, max-drawdown vrect. Optional dotted **Bookie** overlay from `bookie_results`.
2. **Drawdown**: Full underwater path `(equity / running peak - 1) * 100`.
3. **ROI**: Per-bet return with a zero line.
4. **Stake Percentage**: Stake / starting bankroll.
5. **Metrics table**: All `calculate_all_metrics` values. Non-finite values show as "—".

**Parameters:**

- `backtest`: A run backtest (`detailed_results` populated).
- `x_axis`: `"bet"` (compact placed-bet index, default) or `"date"` (`bt_date_column`). Use `"date"` when the strategy skips bets that the bookie still takes, so the overlay lines up.
- `show_bookie`: Overlay `bookie_results` on the bankroll panel. KellyCriterion and ValueBet stake 0 on the bookie path, so that line is flat.

**Example Usage:**

```python
from backtestbuddy.plots.sport_plots import plot_backtest

backtest.run()
backtest.plot()  # displays plot_backtest
fig = plot_backtest(backtest, x_axis="date")
fig.show()
```

#### `plot_calibration`

Reliability diagram for the model probability of the **selected** outcome. Equal-width bins on `[0, 1]`, same as `calculate_ece`. Marker size scales with bets in the bin. The diagonal is perfect calibration. Without `bt_model_prob_*`, only the diagonal is drawn.

```python
def plot_calibration(
    backtest: Any,
    n_bins: Optional[int] = None,
) -> go.Figure
```

```python
fig = backtest.plot_calibration()
fig.show()
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

The plotting functions can be called from the backtest instance:

```python
backtest.run()
backtest.plot(x_axis="date")
fig = backtest.plot_odds_distribution()
fig.show()
fig = backtest.plot_calibration()
fig.show()
```

### Importing Functions Directly

You can also import and use the plotting functions directly:

```python
from backtestbuddy.plots.sport_plots import (
    plot_backtest,
    plot_calibration,
    plot_odds_histogram,
)

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

- `backtest.plot(x_axis="bet", show_bookie=True)` - Calls `plot_backtest()` and displays the figure
- `backtest.plot_odds_distribution(num_bins)` - Calls `plot_odds_histogram()` and returns the figure
- `backtest.plot_calibration(n_bins)` - Calls `plot_calibration()` and returns the figure

These methods handle the integration automatically, ensuring the backtest has been run before plotting.
