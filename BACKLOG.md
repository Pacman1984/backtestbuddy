# Backlog

Planned work that is designed but not built yet. Each item says why it matters, how it should work, and what must not break. Pick an item, open a branch, and move the finished entry to the changelog.

Status: 🟢 ready to build · 🟡 needs a design decision · ⚪ idea

---

## 🟢 Strategy toolkit: separate selection from sizing

**Why.** Each strategy decides both *which outcome* to bet on and *how much*. Combinations need a new class each time: "value bets only, but sized with Kelly" is impossible today without writing one. `get_bet_details` is near-identical boilerplate in every class, and `select_bet` runs twice per match (inside `calculate_stake` and again in `get_bet_details`).

**Design.** Two small building blocks plus one class that combines them:

```python
class Selector(ABC):           # which outcome, or -1 to skip
    def select(self, ctx: BetContext) -> int: ...

class Sizer(ABC):              # how much on that outcome
    def size(self, ctx: BetContext, outcome: int) -> float: ...

class Strategy(BaseStrategy):  # implements calculate_stake / select_bet / get_bet_details once
    def __init__(self, select: Selector, size: Sizer): ...
```

- `BetContext` bundles what a decision can use: `odds`, `bankroll`, `model_probs`, `prediction`, `date`, `history` (the `BetHistory` ledger).
- Selectors: `HighestProbability`, `Prediction`, `Favourite`, `HighestEV(min_ev)`, `HighestKelly(min_kelly, min_prob)`.
- Sizers: `Amount(x)`, `Fraction(f)`, `Kelly(fraction, max_bet)`, `UnitPlan(unit, plan)`, `EdgeProportional(...)`.
- Each sizer and selector reports its own extra info (`kelly_fraction_i`, `ev_i`, …) so results keep their columns.

```python
strategy = btb.Strategy(select=btb.HighestEV(min_ev=0.03), size=btb.Kelly(fraction=0.5, max_bet=0.05))
```

**Hard requirement: backward compatibility.**

- Every existing class stays a class with the same constructor and the same results: `FixedStake`, `KellyCriterion`, `ValueBet`, `UnitStake`, `OddsFilter`, `RiskLimits`. Internally they become preset combinations, e.g. `ValueBet(min_ev, amount=10)` = `Strategy(HighestEV(min_ev), Amount(10))`.
- A user can still write a strategy as **one class** by subclassing `BaseStrategy` and implementing the three methods, exactly as documented today. The toolkit is an additional, simpler path, not a replacement.
- Wrappers (`OddsFilter`, `RiskLimits`) keep working around both old-style and toolkit strategies.
- Acceptance: the full current test suite passes unchanged, and `tests/integration/test_metrics_reference.py` produces identical results for all preset classes.

**Effort.** Medium. Mostly moving existing logic into selectors and sizers.

---

## 🟢 Kelly for simultaneous events

**Why.** Matches on the same date are settled one after another, so the stake of a later match already uses the result of an earlier one, which is look-ahead. Independent single-bet Kelly on many simultaneous matches also over-bets in total.

**Design.**

- Engine: optional `settle_simultaneous=True` on backtests. All bets of one date (or time slot) are sized from the bankroll at the start of the slot and settled together.
- Sizer: `SimultaneousKelly(fraction, max_total)` scales the individual Kelly stakes so their sum stays under `max_total` of the bankroll. An exact multi-bet Kelly optimization is a possible later refinement.

**Effort.** Medium; the engine change needs care around `bt_starting_bankroll` per row.

---

## 🟢 Edge-proportional sizing

**Why.** A simple alternative to Kelly: stake grows with the edge `p * odds - 1`, with less variance than full Kelly.

**Design.** Sizer `EdgeProportional(base_fraction, max_fraction)`: stake = `bankroll * min(base_fraction * edge, max_fraction)`. Fits the toolkit directly; until then it is a small standalone class.

**Effort.** Small.

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
