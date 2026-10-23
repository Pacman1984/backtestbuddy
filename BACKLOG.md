# Backlog

Planned work that is designed but not built yet. Each item says why it matters, how it should work, and what must not break. Pick an item, open a branch, and move the finished entry to the changelog.

Status: 🟢 ready to build · 🟡 needs a design decision · ⚪ idea

---

## ⚪ Optional compiled kernel (numba, Rust only if needed)

**Why.** Only relevant if `evaluate()` (0.13–0.25 s per trial on 10,000 matches) is still too slow for very large searches or Monte Carlo runs. A lean loop without per-match Python objects measured about 0.07 s with Kelly.

**Design.** Step 1: numba kernels for the preset strategies behind an optional extra (`pip install backtestbuddy[fast]`). Step 2, only if numba is not enough: a Rust core via PyO3/maturin, same API. Custom Python strategies always keep the Python path, since a per-bet call from Rust back into Python would cancel most of the gain.

**Cost to weigh.** Platform wheels for Windows, macOS, Linux (incl. ARM) × Python 3.9-3.13, a heavier CI, and two languages to maintain.

**Effort.** Medium (numba) to large (Rust).

---

## 🟡 Several bets per match (Dutching)

**Why.** Dutching backs several outcomes of one match so that any of them returns the same profit. Today the engine places at most one bet per row.

**Decision needed.** Results layout: one row per bet (simple, but `detailed_results` is no longer one row per match) or one row per match with list columns. One row per bet plus a `bt_match_id` column is the likely choice.

**Effort.** Large; touches settlement, results, metrics that assume one bet per row, and plots.

---

## 🟡 Exchange betting: lay bets and commission

**Why.** On exchanges (Betfair, Smarkets, …) you can bet *against* an outcome, and winnings pay commission. A large group of serious bettors works only this way.

**Design.**

- `commission=0.02` on backtests: winning profit times `(1 - commission)`. Small and independent; can ship first.
- Lay bets: stake as liability or backer's stake, settlement reversed. Needs a `bt_side` (`back` / `lay`) column and lay-aware metrics (yield on liability).

**Decision needed.** Whether lay strategies select via the same selectors with a `side` flag or via dedicated lay selectors.

**Effort.** Commission is small; lay bets are medium to large.

---

## ⚪ Progression systems (Martingale, Fibonacci, D'Alembert)

**Why.** Well known and often asked for. They have no positive expectation; the value is in showing *why* they fail, e.g. with the stress test and the drawdown metrics.

**Design.** Sizers reading `BetHistory.losing_streak` / `last_bet`. No engine change needed now that the ledger exists. Docs must state the risk clearly.

**Effort.** Small.

---

## ⚪ Void and push results

**Why.** Some markets refund the stake (tennis retirements at some bookmakers, Asian handicap pushes). Today a bet only wins or loses.

**Design.** Allow a "void" outcome value (e.g. `-1` in `outcome_column`): profit 0, stake returned, excluded from win rate but counted as placed.

**Effort.** Small to medium; win rate, calibration, and CLV must skip voids.

---

## ⚪ HTML report

**Why.** A single shareable HTML file with metric tiles, equity curve, drawdown, monthly heatmap, calibration, and per-segment breakdown, without running a server.

**Design.** `backtest.report("report.html")` rendering the existing Plotly figures plus new ones into one self-contained page.

**Effort.** Medium.
