# Plan: an event-driven core shared by backtest and live

> **Status: APPROVED IN PART (2026-10-11).** Owner decisions are recorded in §11. Authorised now: **M0–M3
> only** (then stop, report, wait). M4+ is not started; M8 needs separate approval. The rest of this
> document is the original proposal and is unchanged except §9.3 (new evidence) and §11 (decisions).
>
> Honest scope notes. (1) Nothing here promises better trading results; the goal is *testability and
> parity*. (2) The signed Binance connector has still never touched the real API (HTTP 451 from the build
> environment), so every claim about venue behaviour below is "per the local copies of the official docs",
> not "observed". (3) Nothing here adds a safety gate, flag or confirmation step; where a migration step
> would normally want a switch, the plan uses *code-level* sequencing and `git revert` instead.

## 0. How this plan was produced (evidence, constraints respected)

| What | How | Touched |
|---|---|---|
| Docs and code read | README, `docs/DECISIONS.md`, `RISKS_AND_LIMITATIONS.md`, `SECURITY.md`, `VERIFICATION.md`, `ARCHITECTURE.md`; `src/` mapped with `grep`/reads | read-only |
| Time sources | `grep` for `time.time`, `monotonic`, `sleep`, `datetime.now` over `src/` | read-only |
| NautilusTrader facts | PyPI JSON metadata for `nautilus_trader` + the README shipped in it (fetched 2026-10-11 from `pypi.org`) | no install, no import, no exchange host |
| Binance order-type facts | the local copies of the official docs saved earlier in this session (`rest-api.md`, `filters.md`, `enums.md`, `errors.md`, `user-data-stream.md`) | read-only, no network |
| Gate-redundancy evidence | **mutation experiments in a throw-away copy of the repo** under the session scratchpad (the real working tree was never edited); each experiment disables one gate and runs the suite | working tree: `git status` clean throughout |
| Never done | contact `api.binance.com` / `testnet.binance.vision` / any exchange host; run `live-check`, `arm-live` or anything that can send an order; read or print `.env`, keys or secrets; add a dependency; implement anything | — |

---

## 1. (a) Current architecture map

### 1.1 Where each thing happens today

| Concern | Where (file:line at HEAD `0019cdc`) | Notes |
|---|---|---|
| **Market data enters (live/paper)** | `runner.py:59` `BinanceMarketFeed` → `data/market.py:140` `fetch_latest_closed`, `:147` `fetch_quote` → `exchange/client.py` (public `GET`s) | Pull-based: the loop *asks* once per cycle. `MarketFeed` Protocol: `runner.py:51` |
| **Market data enters (backtest)** | `data/market.py:119` `fetch_range` / `:167` `load_csv` → `cli.py:138` `_backtest` → `backtest/study.py:116` `run_study`, `:233` `run_walk_forward` → `backtest/engine.py:68` `run_backtest` | Candles only; no quotes |
| **Validation of data** | `data/market.py:65` `check_candles`, `:88` `validate_live_candles` (live), `runner.py:815` call site | Gaps, duplicates, staleness |
| **Strategy decision** | `strategy/base.py:23` `Strategy.on_candles(candles, in_position) -> Signal`; `strategy/sma_cross.py:49` | **Already pure**: no I/O, no clock, no exchange import. Called from `runner.py:955` (live) and `backtest/engine.py:183` (backtest) |
| **"In position" input to the strategy** | live: `runner.py` `_in_position` ← ledger/balances (sellable free qty); backtest: `engine.py:97` `holding()` ← simulated `base` | **Different sources** (see 1.3, gap P2) |
| **Protective decision (stop-loss)** | live: `runner.py:888` `_protect` (+ `:916` `_stop_key`) from *quote bid*; backtest: `engine.py:172` inside-candle `c.low <= stop_price` | **Two independent implementations**, different data |
| **Orders built** | entry: `runner.py:984` `_submit_entry` (spread guard → `RiskManager.size_entry` `risk/manager.py:209` → `check_order` `:283` → `OrderIntent`); exit: `:1023` `_submit_exit`; flatten: `:409` `_flatten`. Backtest: `engine.py:152` calls `size_entry` only, then does its own bookkeeping | `OrderIntent` is `execution/order_manager.py:50` |
| **Orders sent** | `OrderManager.submit` `execution/order_manager.py:182` (write-ahead → `Broker.place_order`) → `execution/live.py:121` `LiveBroker` / `execution/paper.py:112` `PaperBroker` → `exchange/client.py` | The `Broker` Protocol is `execution/broker.py:12` |
| **Order lifecycle / reconcile** | `order_manager.py:422` `reconcile_pending`, `:309` `_apply`, `:371` `_requery_lost`; state in `storage.py` | Polling-based; no user-data stream |
| **Bot-owned accounting** | `ledger.py` (`Ledger`, `owned_base`), `runner.py:488` `_check_ownership` | Not present in the backtest engine |
| **Gates / halts** | `safety.py:120` `check_live_gate`; `app.py:214` `make_gate_check`; `runner.py:304` `_check_gate`, `:329` `_entry_guard`; `killswitch.py` | Live only |
| **Time is read** | `clock.py` (the **only** place that calls `time.time/monotonic/sleep`); `Clock` Protocol injected into `TradingLoop`, `OrderManager`, `PaperBroker`, `RiskManager`, `KillSwitch`, `BinanceSpotClient`, `BinanceMarketFeed`; `engine.py:67` uses `FakeClock` + `set_ms(c.open_time_ms)`. CLI one-offs: `cli.py:79,153,402,431,454,455,490,491` (`SystemClock().now_ms()` for timestamps) | `grep` found **zero** direct wall-clock calls in `src/` outside `clock.py` |
| **Cadence** | `runner.py:204` `run()`: fixed-cadence poll loop (`poll_seconds`) calling `run_cycle` (`:272`) | The "event loop" is a timer |
| **One decision per candle** | `runner.py:932` `_signals`: `kv lastc:<sym>` + per-process `_warmed` set | De-duplication is bolted onto the poll model |

### 1.2 The cycle today (what an event-driven core must reproduce, in this order)

`run_cycle` (`runner.py:272`) → kill check → `_check_gate` → `_cycle` (`:729`): health probe → rules refresh
→ `reconcile_pending` → per-symbol quotes → open-order check → `_snapshot` (ledger/ownership blockers) →
daily-loss update → **protection first** (`_protect` per symbol) → **then** persist (`risk_state`, equity) →
**then** candle fetch/validate/signals (`_signals`) → error accounting. This ordering is a *safety property*
(D16, D17: protection must not depend on candles or on persistence). It is the main thing a naive
"everything is an event" rewrite could silently lose; §2.4 pins it with explicit priority classes.

### 1.3 Parity gaps that exist **today** (why backtest ≠ live even for identical candles)

| # | Gap | Where |
|---|---|---|
| P1 | Backtest never exercises `OrderManager`, `Store`, `Ledger`, `check_order`, the entry guards (spread, signal age, blockers) or the order state machine | `engine.py` vs `runner.py` |
| P2 | `in_position` is computed from different state (simulated balance vs ledger-owned sellable qty) | `engine.py` `holding()` vs `runner._in_position` |
| P3 | Stop-loss: candle-low model vs live bid-poll model; stop price derived from simulated vs actual fill | `engine.py` vs `runner._protect` |
| P4 | Execution timing: backtest fills at *next candle open*; live fills at the poll that sees the closed candle (≤ `poll_seconds` + age guard) | `engine.py` docstring vs `_signals` |
| P5 | Warm start, signal-age guard, one-decision-per-candle bookkeeping exist only in the live loop | `runner._signals` |
| P6 | Daily-loss halt: auto-reset each UTC day in backtest, sticky + human reset live (documented simplification) | `engine.py` vs D23 |
| P7 | Fees/slippage use the same *formulas* (shared by design) but are applied in two code paths | `engine.py` vs `paper.py` |
| P8 | No recorded-session replay exists: a paper/live session cannot be re-run offline to check "would the backtest have decided the same?" | — |

---

## 2. (b) Events and the single ordered stream

### 2.1 Event types (all immutable dataclasses with `Decimal`, no floats — D3)

Common header on every event: `seq` (int, assigned once, monotonic), `ts_event_ms` (when it happened on the
venue/in the world), `ts_init_ms` (when *our process* received/created it — this is what makes the live
"signal age" guard replayable), `source` (str).

| Family | Events | Produced by |
|---|---|---|
| **Market data** | `CandleClosed(symbol, interval, candle)` — closed candles only, never the forming one; `QuoteUpdated(symbol, bid, ask)`; later optional `BookDepth` | `LiveSource` (poll today, WebSocket later) / `ReplaySource` |
| **Order updates** | `OrderSubmitted`, `OrderAccepted`, `OrderPartiallyFilled`, `OrderFilled`, `OrderCanceled`, `OrderExpired(reason)`, `OrderRejected`, `OrderOutcomeUnknown`, `OrderLost`, `OrderLostResolved`; each carries the existing `OrderRecord` snapshot + `Fill`s. 1:1 with today's `OrderStatus` (`models.py:22`) | `ExecutionService` (wraps `OrderManager`) |
| **Account / position** | `BalanceChanged`, `PositionChanged(symbol, owned_qty, avg_cost)`, `OwnershipDiscrepancy` / `OwnershipCleared` (today's blockers) | `Ledger`/ownership component |
| **Timers** | `TimerFired(name)`: `PollTick`, `RulesRefreshDue`, `ReconcileDue`, `LostRequeryDue`, `DayRollover(utc_day)`, `KeyRecheckDue`, `LostRequery`. Created via `Clock.set_timer` (§4) | the `Clock` |
| **Lifecycle** | `EngineStarted`, `WarmupComplete(symbol)`, `KillSwitchTripped(reason)`, `GateClosed(reason)` / `GateOpened`, `RulesChanged(symbols)`, `DataUnusable(symbol, why)` (today's `no_signal`), `EngineStopping(why)` | engine / gate / data validator |

**Events vs commands.** Strategies and components emit **commands** (`SubmitIntent`, `CancelOrder`); only the
`ExecutionService` turns them into venue calls, and results come back **as events**. This keeps one-way
data flow: *events in, intents out*.

### 2.2 The single ordered stream

One `EventBus` (single-threaded, synchronous dispatch). Total order key:

```
(ts_event_ms, class_priority, seq)
class_priority:  0 lifecycle < 1 timers < 2 order updates < 3 account < 4 QuoteUpdated < 5 CandleClosed
```

* `seq` is assigned **by the bus at enqueue** in live, and read from the file in replay → replay reproduces the
  exact order.
* Handlers are called in a fixed registration order per event type; handlers may enqueue further events
  (processed after the current one, FIFO within the same key) — no re-entrancy.
* **Recording:** every event that passes through the bus can be serialised (JSON Lines, versioned schema,
  numbers as strings). This is the fixture format (§5.3).

### 2.3 Handler graph (what replaces `_cycle`)

```
Lifecycle/Kill/Gate ─┐
Timers ──────────────┤
Order updates ───────┼─► Ledger/Ownership ─► RiskGuard (blockers, daily loss) ─┐
Account ─────────────┘                                                        │
QuoteUpdated ─► ProtectionPolicy (software stop) ─► [commands]                │
CandleClosed ─► DataValidator ─► Strategy.on_event ─► [strategy Intents]      │
                                  └─► EntryFilters (spread, signal age, blockers) ─► Sizer ─► RiskGate
                                                                              └─► ExecutionService
Persistence (risk_state, equity) is subscribed with a LATER registration slot than ProtectionPolicy.
```

### 2.4 Safety properties the stream must preserve (pinned as tests *before* anything moves — see M0)

1. Protection (quote-driven stop) is dispatched before any candle-driven decision of the same timestamp
   (class_priority 4 < 5) and **does not read candles** (D16).
2. Persistence runs after protection (a full disk cannot skip a stop; `runner.py:797`).
3. Write-ahead before send; deterministic `client_order_id` from `intent_key` (D7); **no automatic resend**
   (D6); `OrderOutcomeUnknown` → reconcile, never re-submit.
4. Entries only through the last-moment guard (gate + risk state, D21); exits stay possible (D18).
5. Ownership: nothing sold or cancelled that is not provably the bot's (D19, D20).
6. One decision per closed candle; per-process warm-up blocks *entries* only (D28/D34).

---

## 3. (c) Strategy interface

### 3.1 Target contract

```python
# botux/strategy/api.py   (no imports from exchange/, execution/, storage, runner, httpx, time)
class Strategy(Protocol):
    name: str
    warmup: int

    def on_event(self, event: Event, ctx: StrategyContext) -> Sequence[Intent]: ...


@dataclass(frozen=True)
class Intent:  # strategy-level, NOT an order
    kind: Literal["ENTER_LONG", "EXIT_LONG"]
    symbol: str
    # stable key, same scheme as today: f"{strategy}:{symbol}:{candle.open_time_ms}:{side}"
    intent_key: str
    reason: str


class StrategyContext(Protocol):  # read-only
    def now_ms(self) -> int: ...  # event time, from the injected Clock
    def candles(self, symbol: str, n: int) -> Sequence[Candle]: ...  # closed candles only
    def position(self, symbol: str) -> PositionView: ...  # owned qty, avg cost (ledger-derived)
```

* **No quantity, price or order type in an `Intent`.** Sizing, caps, spread, age and blockers are *downstream*
  (they depend on account state and venue rules, not on the signal). `Intent` → `OrderIntent`
  (`order_manager.py:50`) is done by one shared `IntentToOrder` function used in both modes.
* **v1 stays valid.** `CandleStrategyAdapter` wraps today's `on_candles(candles, in_position) -> Signal`, so
  `SmaCross` and `tests/test_backtest.py` (49 tests) do not change in M3.
* **Purity is enforced by a *test*, not a runtime gate:** an AST scan asserts that modules under
  `botux/strategy/` import nothing from `botux.exchange`, `botux.execution`, `botux.storage`, `botux.runner`,
  `botux.app`, `httpx`, `socket`, `os`, `subprocess`, and never call `time.*` / `datetime.now/utcnow`.
  (A test that fails CI is not a new safety gate in the sense of §"Constraints": it adds no flag, no
  confirmation step, and cannot affect a running bot.)
* Protective logic (software stop) is **not** a strategy: it is `ProtectionPolicy`, a separate component fed by
  `QuoteUpdated` and position state, so a strategy can neither disable nor override it.

---

## 4. (d) Clock abstraction

Today: `Clock` (`clock.py:9`) = `now_ms()`, `monotonic()`, `sleep()`; `SystemClock`, `FakeClock`; injected
nearly everywhere; **no direct `time.time()` outside `clock.py`** (verified). What is missing for an event core:

| Need | Addition (additive, backwards compatible) |
|---|---|
| Timers as events | `Clock.set_timer(name, interval_ms, *, start_ms=None) -> None` / `cancel_timer(name)`; fires `TimerFired` into the bus. Live: scheduled against `monotonic()`; replay/backtest: timers are *events in the fixture* or derived from event time (no real sleeping) |
| Event-time clock | `TestClock` = today's `FakeClock` + `advance_to(ts_event_ms)`. The dispatcher sets the clock to each event's `ts_event_ms` **before** dispatch, in both backtest and replay (exactly what `engine.py:134` does with `clock.set_ms`) |
| No blocking in the core | `sleep()` is unavailable to handlers (the poll loop's pause becomes a timer). `_sleep_interruptibly` (`runner.py:217`) lives only in the live driver |
| Strategy time | via `ctx.now_ms()` only. A CI test greps the strategy package and the whole `src/` (excluding `clock.py` and the CLI one-liners at `cli.py:79…491`) for `time.time|time.monotonic|time.sleep|datetime.now|datetime.utcnow|perf_counter` — **today this list is empty**, so the test would pass immediately and guard the future |
| Live skew | `BinanceMarketFeed.now_ms()` (`runner.py:67`) already adds the measured server offset; keep that as the live `ts_event` base so signal-age decisions are identical to today's |

---

## 5. (e) Adapter boundaries

### 5.1 Interfaces

```python
class EventSource(Protocol):  # produces events into the bus
    def start(self, bus: EventBus, clock: Clock) -> None: ...
    def stop(self) -> None: ...


class ExecutionVenue(Protocol):  # == today's Broker (execution/broker.py:13), unchanged
    place_order / get_order / cancel_order / open_orders / order_fills / balances / update_rules
```

### 5.2 Concrete adapters

| Adapter | Mode | What it does | Reuses |
|---|---|---|---|
| `ReplaySource` | backtest, parity tests, "re-run a paper session" | reads a fixture (JSONL) or CSV candles; emits events with recorded `ts_event`/`ts_init`/`seq`; sets the `TestClock` | `data/market.py:167 load_csv`, `check_candles` |
| `LivePollSource` | paper + live | on `PollTick`: pulls quotes then candles through the existing `MarketFeed` (public GETs only) and emits `QuoteUpdated`/`CandleClosed`; de-dupes by `open_time_ms` (replaces `lastc` bookkeeping); emits `DataUnusable` instead of raising | `BinanceMarketFeed`, `validate_live_candles`; **no new endpoint** |
| `SimVenue` | backtest, paper | `PaperBroker` fed by a *candle-derived* quote (open for entries; stop touch rule from `engine.py`), plus a `FastSim` profile (no write-ahead, no SQLite) for study grids | `execution/paper.py`, cost model |
| `BinanceVenue` | live | today's `LiveBroker` unchanged behind the same interface | `execution/live.py`, `exchange/client.py` allow-list untouched |
| `SessionExporter` (dev tool) | any | builds a fixture **after the fact** from a paper/live session's SQLite (`orders`, `fills`, `events`, `equity`) + the candles/quotes fetched for that period; **nothing is added to the live path** and no secrets are read | `storage.py` readers |
| (later, optional, separate approval) `LiveStreamSource` | live | WebSocket kline/bookTicker/user-data-stream as an *additional* source; polling `get_order` reconcile stays authoritative (absence of a push is not proof of non-execution — ARCHITECTURE.md "why polling") | — |

### 5.3 Fixture format (versioned, text, diff-able)

JSON Lines, one event per line: `{"v":1,"seq":…,"ts_event":…,"ts_init":…,"type":"CandleClosed","symbol":"BTCUSDT",…}`; decimals as strings;
header line with `schema_version`, symbol rules snapshot (tick/step/min-notional), strategy params, risk
settings, and the pinned-portfolio script for P1 (below). Contains **public market data and our own
order/fill records only** — no keys, no account ids; `GLOBAL_REDACTOR` runs over the export as defence in depth.

---

## 6. (f) Parity definition

**Setup.** A fixture `F` (ordered event list) + fixed strategy params + fixed risk settings. Two *compositions* run `F`:

* **Backtest composition:** `ReplaySource` + `TestClock` + `SimVenue` (+ `IntentToOrder`, `Sizer`, `ProtectionPolicy`).
* **Live composition:** `LivePollSource` + `ExecutionService`/`OrderManager` + `BinanceVenue` code paths, driven **without network**
  by recorded HTTP transcripts through the existing `httpx.MockTransport` fakes (`tests/conftest.py:FakeExchange`,
  `tests/test_live_loop_e2e.py:VenueState`) and an injected clock. Same strategy object, same event order.

**Intent record** (the unit of comparison):
`(trigger_seq, symbol, kind ∈ {ENTER, EXIT, PROTECT}, side, intent_key, reason)`.

| Level | Must be **identical** (ordered list equality) | Held fixed |
|---|---|---|
| **P0 — strategy parity** (the headline) | the sequence of strategy `Intent`s | position-state events come **from the fixture** (scripted fills), so both modes see the same `in_position` |
| **P1 — sized parity** | P0 + `(quantity, order_type, limit_price)` of the resulting `OrderIntent` | a *pinned portfolio script* (equity / free quote per event) is part of the fixture, so sizing does not depend on simulated fills |
| **P2 — protective parity** | `PROTECT` intents and their trigger event | fixture supplies the quote stream; entry price comes from the scripted fill |

**NOT part of equality (reported as metrics instead):** fill price, filled quantity, partial-fill pattern,
fee amount/asset, venue order ids, acknowledgement timestamps, latency, slippage.
**Fill-quality report** (per run and per symbol), computed by joining intents to fills by `intent_key`:

| Metric | Definition |
|---|---|
| `slippage_bps` | side-signed `(fill_avg − reference_price)/reference_price × 10⁴` (reference = quote at intent time) |
| `fee_dev_bps` | actual commission − modelled commission, in bps of notional (BNB-paid fees reported separately; not valued, per D13-era note) |
| `latency_ms` | first fill `ts_event` − intent `ts_event` |
| `fill_ratio`, `partial_rate`, `reject_rate`, `unknown_rate`, `lost_rate` | counts from the order-update stream |
| `next_open_gap_bps` | backtest-only: difference between live decision price and the next-candle-open assumption (quantifies gap P4) |
| `unpinned_divergence_count` | informational: intents that differ when fills are **not** pinned (real/simulated fills feed back into sizing/stops). A metric, **never** a pass/fail criterion |

**Why three levels.** Equality of *sized* intents is only meaningful if the account state is identical; that
state depends on fills, which are explicitly excluded. P0 is the strict, always-on contract; P1/P2 are the
same contract with the account/quote state pinned by the fixture.

**Tests.** (1) Golden: `SmaCross` on a committed candle fixture → expected intent list (checked into git).
(2) Property (hypothesis): random candle paths + random pinned fills → backtest ≡ live composition at P0.
(3) After any paper run: `export-session` → replay → compare against what the session actually did
(closes gap P8; the strongest realistic parity check available without touching the exchange).

**Honest limits.** Anything that depends on wall-clock arrival (age guard, warm start) is only reproducible
because `ts_init_ms` is recorded; a fixture without `ts_init` replays with latency 0. Live-only behaviours
(rate limits, 5XX, partial fills) are outside P0–P2 and stay covered by the existing live-composition tests.

---

## 7. (g) Migration steps (smallest and safest first; each independently testable and revertible)

No step changes live behaviour until M6; M0–M5 are additive. Sizes: S ≈ <½ day, M ≈ 1–2 days, L ≈ 3+ days.
There are **no new config flags**: sequencing is by merged code and `git revert`.

| Step | Change | Acceptance (independently testable) | Size |
|---|---|---|---|
| **M0 Characterise** | Tests only. (a) Golden intent sequences from today's `TradingLoop` on the existing scenarios (`tests/test_runner.py:SCENARIO`, e2e `VenueState`). (b) Run today's `run_backtest` and the paper loop on the *same* candles and record the P0 divergence as a baseline number. (c) Time a 2-year hourly backtest (baseline for performance risk) | suite green; baseline numbers committed in `docs/VERIFICATION.md` | S |
| **M1 Events + bus** | `botux/events.py` (dataclasses) and `EventBus` with the ordering key of §2.2. No wiring | property tests: total order, determinism, FIFO within key, no re-entrancy; (de)serialisation round-trip | S |
| **M2 Clock timers** | additive `set_timer/cancel_timer`, `TestClock.advance_to`; AST test "no wall-clock reads outside `clock.py`" | new clock tests; full suite unchanged | S |
| **M3 Strategy v2 + adapter** | `strategy/api.py`, `CandleStrategyAdapter`, purity AST test. `SmaCross` untouched | `tests/test_backtest.py` (49) unchanged and green; purity test green | S |
| **M4 ReplaySource + fixtures + SessionExporter** | JSONL schema v1; CSV→events; export from a paper session SQLite | round-trip test; exported paper session replays to the same candles/quotes | M |
| **M5 Engine in "FastSim" profile = new `run_backtest`** | dispatcher + `ProtectionPolicy` + `Sizer` + `SimVenue`. Old `engine.py` kept **side by side**; new one must reproduce its `BacktestResult` (equity curve, trades) **exactly** on all current backtest/study tests | golden equality old vs new on every test in `test_backtest.py` + a randomized candle property; then old engine removed in a *separate* commit | L |
| **M6 Shadow run (tests/paper only)** | `LivePollSource` + new strategy path run **in parallel** with the current loop in paper; the new path's intents are logged/compared, never executed. Exercised in tests and in a manual paper session; no change to what the bot does | P0 parity on recorded paper sessions; zero unexplained divergences over N sessions (N chosen by you) | M |
| **M7 Cut over signals** | the event core produces entry/exit intents; `runner` still owns protection, reconcile, gate, persistence | all 195-ish loop tests green via the `TradingLoop` facade; P0 golden still green | M |
| **M8 Cut over protection, reconcile, timers, lifecycle** | one handler group at a time (protection → reconcile/lost-requery → rules refresh → kill/gate), each with the §2.4 property tests green *before and after* | per-group: existing safety tests + mutation spot-checks of the same mutants listed in `VERIFICATION.md` | L |
| **M9 Fill-quality report + parity CI** | metrics of §6 in `backtest` CLI output; parity suite wired into `make check` | report generated from a paper session | S |

Stop points: after **M3** you already have purity + clock guarantees; after **M5** a shared engine for
backtests; after **M7** signals share code in both modes; M8 is the only step that moves safety-critical
ordering and should be approved separately.

---

## 8. (h) Risks and what could break existing tests

### 8.1 Test coupling (measured at HEAD)

| Coupling | Count | Consequence |
|---|---|---|
| `run_cycle()`/`.cycle(` calls | **194 calls in 11 test files** | Keep `TradingLoop.run_cycle` and the `Harness` (`tests/test_runner.py:71`) as a **facade** until M8; adapt the Harness once rather than 194 call sites |
| Private `loop._…` attributes used by tests | `_position_qty` 10, `_kill` 10, `_strategy` 6, `_rules_loader` 4, `_gate_check` 4, `_s` 3, `_ledger` 3, `_open_positions_text` 2, `_om` 2, `_notifier` 2, `_max_signal_age_ms` 2, `_last_positions` 2, `_flatten_with_retries` 2, `_snapshot`/`_rules_next_try`/`_owned` 1 each | Moving logic out of `TradingLoop` (M7/M8) breaks these first. Mitigation: keep thin delegating properties on the facade through M8, remove last |
| Backtest/study tests | `tests/test_backtest.py`: 49 tests incl. mutation-hardened ones | M5's exact-equality requirement is the guard; any numeric drift = stop |
| Property/invariant tests | `test_loop_invariants.py`, `test_books_property.py` | become the first consumers of the new bus (good: they run the whole loop under random events) |

### 8.2 Risks

| Risk | Why it matters | Mitigation |
|---|---|---|
| **Re-ordering safety-critical steps** (protection-first, persist-after-protection) | the whole point of D16/D17; an "all events equal" bus would lose it | class priorities (§2.2) + explicit tests in M0 *before* refactoring; M8 moves one group at a time |
| **Two engines coexisting** (M5–M7) | double maintenance, silent drift | exact-equality tests; time-box the overlap; delete old engine in its own commit |
| **Performance** | `Decimal` + SQLite per candle could make study grids (many backtests) slow | `FastSim` profile without write-ahead/SQLite for grids; measure in M0; full-stack profile only for parity/final candidate runs |
| **Fixture realism** | a fixture is only as good as what was recorded; exported paper sessions are *simulated* venue behaviour, not Binance | label fixtures by origin (`paper-sim` / `scripted`); do not claim venue parity from them; real-venue fixtures only exist after a real run (out of scope) |
| **Hidden state in the strategy API** | `ctx.candles()` could leak the forming candle | the bus only carries `CandleClosed`; test that no event with `close_time > now` reaches `Strategy` |
| **Scope creep into safety code** | `safety.py`, `client.py` allow-list, kill switch must not change | explicitly out of scope of M0–M9; diff review rule: those files stay untouched |
| **Recorder/exporter leaking data** | fixtures could contain sensitive values | exporter reads only the SQLite + public data, never `.env`; redactor pass; fixtures committed only if produced from paper/simulated runs |
| **Replay of `ts_init`** | without it, the age guard is untestable | recorded from M4 on; older fixtures default to latency 0 and say so in the header |

---

## 9. Step 2 — Evaluations (no implementation)

### 9.1 Native exchange-side stop (STOP_LOSS / STOP_LOSS_LIMIT) vs the current software stop

**What the local copy of the official docs says (verified in `rest-api.md`/`filters.md`/`enums.md`/`errors.md`):**
`STOP_LOSS` executes a **MARKET** order when `stopPrice` is met; `STOP_LOSS_LIMIT` needs `timeInForce`,
`quantity`, `price`, `stopPrice` (rest-api.md:2174-2175); a SELL stop must be **below** the market
(rest-api.md:2197-2198); a stop that is already past the last traded price is rejected with *"Order would
trigger immediately"* (errors.md:390); stops count against the **`MAX_NUM_ALGO_ORDERS`** filter (example value 5,
filters.md:229-237); order lists exist (OCO/OTO/OTOCO/OPO) but the OTO *working* order must be `LIMIT`/`LIMIT_MAKER`
and the pending order is placed only after the working order is **fully filled** (rest-api.md:3359-3368) — so a
**MARKET entry cannot carry an attached OTO stop**; updates arrive as `executionReport` over the user-data stream
(user-data-stream.md:74) but our design polls (D5). **Not verified (451, no network):** trigger price source
semantics, which stop types BTCUSDT allows today (`SymbolRules.order_types` exists, `exchange/rules.py:36`, so the bot
can read it at runtime), fill behaviour in a gap.

| Aspect | Software stop (today: `runner._protect`, D9/D16) | Exchange-side stop |
|---|---|---|
| **Bot down / crashed / network cut / host asleep / IP banned (418/429) / DB unwritable** | **No protection at all** (documented, `RISKS_AND_LIMITATIONS.md`). `flatten_on_kill=false` leaves the position bare | **Protects while the bot is down**, if the order is live and the stop was not rejected. Biggest advantage |
| Latency | ≥ one poll (`poll_seconds`, default 30 s); price seen only at poll instants | matching-engine side, continuous |
| Gap / crash | market sell after trigger, slippage unbounded | `STOP_LOSS` (market): same slippage. `STOP_LOSS_LIMIT`: if price gaps below `price`, the order **rests unfilled and the coins stay locked** — position neither protected nor sellable until the market returns or a human acts |
| Placement failures | none (nothing is placed in advance) | rejection at placement (filters, `MAX_NUM_ALGO_ORDERS`, "would trigger immediately" when price already moved), unknown outcome on timeout (handled by our UNKNOWN→reconcile machinery) |
| Window after entry | stop armed from the *actual* average cost on the next cycle | the stop can only be placed **after** the BUY fill is known (qty net of base-asset fee, price from fills) → a gap during which only a software stop can protect |
| Cost/limits | none | extra order + cancel per position; counts toward order-rate and `MAX_NUM_ORDERS`/`ALGO` filters |

**What would change in order handling (concrete code/design consequences found by reading the current code):**

1. `_position_qty`/`_in_position` use **free** balance (`runner.py:464` `_position_qty`, `:883` `_in_position`): a resting stop *locks* the coins, so the bot would see
   "no position" and could **re-enter**. Ownership must count `locked` coins that belong to our own stop.
2. `OrderManager.cancel_all_open` (`order_manager.py:430`, used by `_shutdown` and `_on_kill`) would **cancel our own protective stops exactly when the bot stops** — the opposite of the benefit. Shutdown/kill policy (D24) would need to *exclude* protective orders.
3. A strategy EXIT or a software fallback must first **cancel** the stop, confirm it is gone, then sell — a race in which the stop can trigger in between (second SELL → `insufficient balance`). New states: `EXITING`, cancel-then-sell with reconcile.
4. `is_symbol_blocked` (`order_manager.py:146`), `_check_open_orders` (`runner.py:586`), `max_open_orders` (`risk/manager.py:337`) treat open orders as blockers/counters; a *permanent* resting stop must be classified "ours, protective" so it does not block entries or raise the foreign-order blocker.
5. `PaperBroker` supports only `MARKET`/`LIMIT` (D8); it needs stop semantics, and `engine.py` needs a stop model that matches the venue's (gap behaviour of `STOP_LOSS_LIMIT`).
6. Partial fills of the stop, stop `EXPIRED`/cancelled by a human, quantity changes when the user trades the same asset (baseline coins), re-arming after restart from `orders` table.

**Reconciliation needs (new):** a periodic *protection audit* — for every owned position verify that exactly one live stop exists with `qty ≈ owned`, `stopPrice` consistent with policy, and `executedQty` accounted; detect manual cancels; re-arm; and handle "stop filled while we were down" (our deterministic ids + `reconcile_pending` already adopt that fill into the ledger; the position book (`PositionBook`) and blockers need to learn the position is now flat).

**Recommendation:** *do not replace* the software stop. Treat an exchange-side stop as a **second, independent layer** (software stop stays as backstop for the placement window and for stop-rejected cases) and only after (i) the event core exists (M7+), because the audit/reconcile logic is much easier to test as event handlers, (ii) it is exercised against **Testnet from an unblocked environment** — not possible from here — and (iii) you decide on the trade-off `STOP_LOSS` (market, bounded only by slippage) vs `STOP_LOSS_LIMIT` (floor price, risk of being stranded). Until then the hard fact stays in the docs: *if the bot is down, the position has no stop.* **Decision for you.**

### 9.2 Should NautilusTrader be a dependency?

**Verified facts** (PyPI JSON + the README bundled in it, fetched 2026-10-11; nothing installed or imported):

| Item | Value |
|---|---|
| Latest **stable** on PyPI | **1.231.0**, uploaded 2026-08-02 |
| Newer pre-releases | `2.0.0rc2` … `2.0.0rc6` (latest **2026-10-05**); README: *"The legacy Cython v1 core remains supported during the v2 release-candidate phase"*; Rust-native v2 via PyO3 |
| **License** | **LGPL-3.0-or-later** — PyPI `license` field says `LGPL-3.0-or-later`, trove classifier is *GNU Lesser General Public License v3 or later (LGPLv3+)*, and the README states its license checks "enforce LGPL-3.0-or-later compatibility". I did **not** read the `LICENSE` file itself (it lives in a GitHub repository outside this session's scope); confirm the text before any decision. Not legal advice |
| Python | `>=3.12,<3.15` (we run 3.13 — compatible) |
| Hard runtime deps (from `requires_dist`) | `click`, `fsspec==2026.2.0`, `msgspec`, `numpy`, `pandas<4`, `portion`, `pyarrow>=25`, `pytz`, `tqdm`, `tzdata`, `uvloop` (non-Windows) — vs **one** dependency today (`httpx`, D2/D14) |
| Size | binary wheels ≈ 110–190 MB per platform |
| Stability statement (README) | *"still under active development … breaking changes can occur between releases"*; bi-weekly release cadence |
| Binance | listed as an integration with status *stable* (README table) — spot/futures coverage and key types not verified by me |
| Positioning (README) | "single event-driven architecture", deterministic simulation and live execution, "strategies deploy from research to production with no code changes" — i.e. the same goal as this plan |

From general knowledge of its public documentation (**not verified here; re-check names before reuse as
reference**): a message bus + cache + data/risk/execution engines, a `Clock` with a test (event-time) and a live
implementation, strategies that react to `on_bar`/`on_quote_tick`/`on_order_filled`-style callbacks and submit
orders through the engine, `BacktestNode`/`TradingNode` composition, adapter pairs (data client + execution client).
The plan above borrows only the *shape*: one ordered stream, injected clocks, strategy-in/intents-out, adapters at the edge.

**What it would replace if adopted:** `runner.py`, `engine.py`/`study.py`'s engine, `clock.py`, `data/market.py`, `exchange/client.py`
and the Binance order path, `execution/*`, and likely `storage.py`'s order tables.
**What it would *not* replace and we would have to re-implement on top:** the bot-capital ledger and ownership logic (D19/D20),
write-ahead + deterministic ids + no-resend + LOST semantics (D6/D7/D25), the live gate, kill switch and arming file, the endpoint
allow-list (D2: today we can *prove* no withdrawal/transfer/margin endpoint exists in a ~600-line client; we cannot make that claim about
a large third-party Binance adapter without auditing it), `Decimal` arithmetic (D3; Nautilus uses its own fixed-point price/quantity types),
the stricter fee/ownership accounting, and the 595-test safety net, much of which targets our own internals.

**Recommendation: do not add it now; build the in-house core.** Reasons: (1) the problem is small and *we already own the hard
parts* (our strategy interface is already pure, the clock is already injected, the broker boundary already exists); (2) the dependency
footprint goes from 1 to ~10 heavy packages and ~150 MB, in a project whose security story is "tiny auditable surface"; (3) v1→v2 core
transition and "breaking changes between releases" make version pinning a maintenance burden; (4) none of our safety invariants would be
inherited — they would be re-proved against someone else's engine; (5) the licence is workable for use as a library but adds obligations
for the project's still-undecided licence. **Revisit if** you need multi-venue, order-book/L2 backtesting, or tick-level simulation. If ever adopted:
pin an exact version (1.231.0 or the v2 GA), isolate it in an optional extra behind the `EventSource`/`ExecutionVenue` interfaces defined here
(so the parity suite of §6 is the acceptance test), and keep our endpoint allow-list in front of any HTTP it makes. **Needs your approval; nothing was added.**

### 9.3 Safety gates that duplicate each other or add no measurable protection — **for your review; nothing was removed or changed**

Method: (A) read the code paths and note conditions checked in more than one place; (B) **mutation experiments** in a scratch copy: disable one
gate, run the whole suite (scratch copy skips 20 tests that need a git checkout — secret-scan tests only; 575 passed). *SURVIVED* = no test
notices its absence ("no measured protection today", which can mean "untested", not "useless"). *KILLED* = some test pins it.
Experiments D/C/E were repeated with `tests/test_risk.py` excluded, to see whether anything *outside the unit tests* depends on them: **all 544
other tests passed** for each (so those checks bind only in unit tests).

| ID | Gate | Where | Relationship / evidence | My classification |
|---|---|---|---|---|
| G1 | `risk.bot_capital is None` refused in the live gate | `safety.py:129` | **SURVIVED in round 1; now KILLED** (round 2: `TestNoExplicitCapitalRefusesToStart`). Round-1 note: Same requirement enforced again at `TradingLoop.__init__` (`runner.py:139`, `ConfigError`) and `check_funding` (`app.py:269`) | **Duplicate + untested** — loop refuses anyway (different exception). Needs a test or a decision |
| G2 | arming file must carry the confirmation-phrase digest | `safety.py:148` (digest written at `:76`) | **SURVIVED.** The digest is a public constant; anyone who can write the file can compute it; fingerprint + age + interactive prompt already cover accidents (SECURITY.md already calls the file a "speed bump") | **No measured protection** |
| G3 | arming file dated in the future (> 60 s) | `safety.py:158` | KILLED (`test_future_dated_arming_is_refused`) | Marginal (clock-skew / forged file only) |
| G4 | kill switch checked in the gate | `safety.py:127` | KILLED. Also checked at `OrderManager.submit` (`order_manager.py:184`), every `run_cycle` (`runner.py:273`), and `KillSwitch.ensure_writable` at start (`app.py:100`) | **4× overlap** — `submit` is the last line; gate-level adds only start-up refusal |
| G5 | `isinstance(authorization, LiveAuthorization)` | `execution/live.py:113` | KILLED, **by the same single test** as G6 | Overlaps G6/G7 |
| G6 | `client.trading_enabled` in `LiveBroker` | `execution/live.py:115` | KILLED (same test as G5) | Overlaps G5/G7 |
| G7 | endpoint guard: TRADE endpoints need `trading_enabled` | `exchange/client.py:256` | KILLED (separate test) | Keep — the only one at the HTTP layer; G5/G6 are redundant *given* G7 |
| G8 | gate re-check at the start of every cycle | `runner.py:278` | KILLED (one test kills G8 **and** G9) | Per-order (G9) dominates for protection; G8 adds earliest alert/blocker |
| G9 | gate re-check right before every new order (`_entry_guard`) | `runner.py:331` | KILLED | Keep (closes the check-then-order window) |
| G10 | `mode == "live"` **and** `[live].enabled` **and** arming (which itself requires both and is bound to the fingerprint covering both) | `safety.py:123,125`, `:103` | analysis only | Three overlapping "I mean live" signals; `enabled` is a separate deliberate edit but adds no technical protection beyond the fingerprint |
| G11 | position / total-exposure caps re-checked in `check_order` after `size_entry` already applied them | `risk/manager.py:321,327` vs `:237-238` | **KILLED only by `test_risk.py`**; with it excluded, 544 other tests pass for D and C | Defence in depth against a sizing bug; in the loop entries happen only when flat, and with defaults (order 100 < position 200 < total 300, one symbol) these two caps **cannot bind** |
| G12 | `max_open_orders` | `risk/manager.py:337` | **KILLED only by `test_risk.py`**; 544 others pass. `is_symbol_blocked` (`order_manager.py:146-170`) already blocks entries while any order on the symbol is unresolved/LOST; MARKET orders rarely rest | Dominated on single-symbol configs |
| G13 | candle staleness in `validate_live_candles` vs the entry signal-age guard | `data/market.py:101` vs `runner.py:858,960` | analysis only | For *entries* the age guard (≤ min(interval/2, 10 min)) is far stricter; the validator still protects *exits* from stale data |
| G14 | `claim_mode` vs per-mode database filenames | `storage.py:167`; `config.database_path` `:551` | `claim_mode("live")` **KILLED** | Overlap only when `paths.database` is overridden to a shared file |
| G15 | `check_funding` (start-only refusal) vs ownership cash-deficit blocker | `app.py:269` vs `runner.py:488` | analysis only | Partial overlap; start-up vs runtime |
| G16 | `--confirm-checked` / `--confirm-contact-binance` | `cli.py` (`ack-balances`, `reset-daily-loss`, `live-check`) | analysis only | Prevent typos, not scripts (documented); no technical protection against a script |

**Round 2 evidence (decision 5; nothing removed).** After adding the G1 tests (the gate layer *and* the loop layer at `runner.py:139`, which had no test at all, are now each killed by their own test), I mutated more gates in a scratch copy and ran the whole suite:

| Gate | Result | Reading |
|---|---|---|
| `[live].enabled` (`safety.py:125`) | KILLED (2 tests, `test_live_gate.py`) | pinned; overlap with G10 is by design |
| `mode == "live"` (`safety.py:123`) | KILLED (2 tests) | pinned |
| stale-data check in `validate_live_candles` | KILLED (2 tests: `test_backtest.py`, `test_runner.py`) | pinned (so G13 is *not* dead code, only dominated for entries) |
| entry signal-age guard (`runner.py:960`) | KILLED (1 test) | pinned |
| `check_funding` at live start (`app.py:269`) | KILLED (1 test, `test_review_fixes.py`) | pinned (G15 overlap is partial) |
| `OrderManager.submit` kill check (`order_manager.py:184`) | KILLED (2 tests) | pinned — the last line of the 4× kill-switch overlap (G4) |
| `kill.ensure_writable()` at start (`app.py:100`) | **SURVIVED — no test notices** | untested; the CLI `kill` command's write-failure path *is* tested, the start-up refusal is not. Candidate for a test, not for removal |
| `claim_mode("paper")` (`app.py:171`) | KILLED (1 test) | pinned |
| `max_spread_pct` entry guard | KILLED (1 test) | pinned |
| `max_orders_per_day` | KILLED (1 test, `test_risk.py` only) | unit-pinned only, like G11/G12 |

Conclusions for your review: the only gate that is *both* unpinned *and* duplicated was G1 (now pinned); G2 (arming-phrase digest) remains the one with no measured protection; `ensure_writable` at start-up is unpinned but not duplicated; G11/G12 and `max_orders_per_day` bind only in unit tests.

Not duplicates (listed so you can see what I would **keep**): the endpoint allow-list; live-gate fingerprint-vs-disk comparison; API-key
restriction check (start + hourly); ownership ledger/blockers; write-ahead + deterministic ids.

---

## 10. Decisions I need from you (nothing proceeds without them)

1. Approve the **direction** (in-house event core, §2–§7) and the **first step** (M0 + M1–M3 only, all additive).
2. Is **P0 strategy parity** (§6) the right definition of "same results", with fills reported as metrics?
3. **NautilusTrader:** confirm "do not depend on it" (§9.2), or ask for a time-boxed spike behind the interfaces.
4. **Exchange-side stop:** defer until after M7 and until a Testnet-capable environment exists (§9.1)? And `STOP_LOSS` vs `STOP_LOSS_LIMIT` preference.
5. **Gate list (§9.3):** which entries, if any, you want turned into a *test* (G1 is the clearest), investigated further, or consolidated. I changed nothing.
6. Fixtures: OK to commit fixtures derived from *paper/simulated* sessions only?

## 11. Owner decisions (recorded 2026-10-11)

1. **Approved:** M0–M3 only; stop after M3, report, wait. M4 not started. M8 needs separate approval.
2. **Approved:** P0 strategy-intent parity is the definition of "same results". Fills, slippage, fees and latency are metrics only, never compared.
3. **Confirmed:** no `nautilus_trader` or any new dependency.
4. **Approved:** exchange-side stop deferred until after M7 and a Testnet-capable environment; a second layer, never a replacement for the software stop.
5. **G1:** first a test proving the start-up refusal works (done: `tests/test_live_gate.py::TestNoExplicitCapitalRefusesToStart`, 3 tests; both the gate layer and the loop layer are now pinned — see §9.3). **No check is removed in this round.** Other duplicates: investigated further (§9.3, "Round 2 evidence"), nothing removed.
6. **Approved:** fixtures only from paper/simulated sessions. **Before any commit, every fixture is scanned for API keys, account IDs, balances and IP addresses; if anything is found: stop and report.** (Implemented from M0 as `tests/test_fixture_hygiene.py`, which fails on any such pattern in a fixture directory, so the scan is also part of `make check`.)
7. **Priority:** the strategy's out-of-sample backtest with realistic fees and slippage comes before further architecture work — see `docs/BACKTEST_OUT_OF_SAMPLE.md`.

## Appendix A — experiment log (reproducible; all in a scratch copy, no network)

* `git status --short` of the real tree stayed empty during all experiments.
* Baseline in the scratch copy: `575 passed, 20 skipped` (the 20 are secret-scan tests that need a git checkout; real tree: 595 passed).
* One-line mutants F, G, O, H, I, J, K, L, M, N, E, D, C, A applied one at a time; results are those in §9.3
  (SURVIVED: H=G2, J=G1; all others KILLED). D/C/E re-run without `tests/test_risk.py`: 544 passed each.
* PyPI query: `GET https://pypi.org/pypi/nautilus_trader/json` (metadata only).
