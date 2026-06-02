# StratScout — Agent Context Brief

Read this before writing a single line of code.

---

## What StratScout Is

Regime-based ETF rotation strategy, live-trading in a Schwab Roth IRA. Every 14 days the engine re-detects market regime (risk-on / risk-off rising rates / risk-off falling rates) and rotates into the top ETFs by momentum + inverse-vol blend. Parameters are Bayesian-optimized (Optuna TPE) on rolling 12-month windows, scored on Calmar ratio with volatility targeting.

**Live results (Jan 2020 – now, 76 months OOS):**
- Final portfolio on $10k: $76,987 vs SPY $19,916
- Avg monthly return: 3.27% | Calmar: 99.3 | Win rate: 57%
- Uses 5 bps slippage, 7.35% annualized vol target

**Current live params:** `active_strategy.json` — risk-on pool is SOXL/DRN/SILJ (n=3), `combo_alpha=0.8`, stop-loss 6.2%, lockout 25 days.

---

## Two Repos

| Repo | Purpose |
|---|---|
| `C:\Code\algo-trading-schwab` | Private research repo — working engine, live trader, EC2 runner, full StratScout app |
| `https://github.com/HansDandle/strat-scout` (public) | Clean engine + landing page — the product |

**The research repo is where the app lives.** All paths below are relative to `C:\Code\algo-trading-schwab`.

---

## Files to Read First (in order)

1. **`C:\Code\StratScout-audit\README.md`** — full UI spec, data model, API surface, tab-by-tab design. This is the single source of truth for what to build.
2. **`stratscout/api/app.py`** — the FastAPI backend. Already fully implemented: backtest, fuzz, walk-forward, strategies CRUD, preflight, credentials, scheduler, SSE streaming, factor lab. **Do not re-implement what's already here.**
3. **`live_trader.py`** — signals source of truth. Shows how regime detection drives order placement against Schwab.
4. **`active_strategy.json`** — current live params (the strategy that's making real money right now).
5. **`stratscout/engine/backtest/etf.py`** — core regime-rotation logic.

---

## Product Tiers

| Tier | Price | Delivery |
|---|---|---|
| **Signals** | $15–30/mo | Email every 14 days: current regime + target positions |
| **Pro SaaS** | $50–100/mo | Full web UI — all 6 tabs |

---

## Tech Stack

- **Backend:** Python / FastAPI (`stratscout/api/app.py`) — fully functional, no skeleton
- **Frontend:** Vite + React + TypeScript + Tailwind + Plotly (`stratscout/web/`)
- **DB:** Local SQLite (`stratscout.db`) — strategies, fuzz runs, walk-forward runs, preflight audit
- **Data:** Local feather files under `data/{daily,15min,smallcap,options}/`
- **Broker:** Charles Schwab (Schwab OAuth, `schwab_auth.py`)
- **Credentials:** OS keychain (Windows Credential Locker) via `keyring`

---

## UI: 6-Tab Flow (fully spec'd in README.md)

1. **Onboarding** — pick template (ETF Balanced, ETF Defensive, Small-cap Momentum)
2. **Settings** — connect brokers + download data
3. **Analyze** — single backtest + baseline chart (SPY/QQQ/TQQQ/TLT/GLD)
4. **Find** — fuzzer leaderboard, persistent runs, goal presets
5. **Walk-forward** — rolling OOS validation, month-by-month grid
6. **Live** — trade mode toggle (Off/Paper/Live), preflight gate, order log

**Build priority for v1:** Leaderboard (Find tab) → Strategy Inspector (Analyze) → Live Status

---

## Full API Surface (already implemented)

```
GET  /health
POST /backtest           — ETF or smallcap
POST /baselines          — buy-and-hold NAV overlays
GET  /data/inventory
GET  /data/categories
POST /data/suggest-fuzz-window
POST /data/suggest-walk-forward
POST /data/download
GET  /settings/credentials
PUT  /settings/credentials
DELETE /settings/credentials/{provider}/{field}
POST /settings/credentials/{provider}/test
POST /fuzz               — runs + persists, returns leaderboard
GET  /fuzz/runs
GET  /fuzz/runs/{id}
DELETE /fuzz/runs/{id}
PATCH /fuzz/runs/{id}    — relabel
GET  /fuzz/leaderboard   — cross-run all-time top
GET  /strategies
POST /strategies
GET  /strategies/{id}
PATCH /strategies/{id}
DELETE /strategies/{id}
GET  /strategies/{id}/preflight
GET  /strategies/{id}/walk-forward/latest
GET  /strategies/{id}/orders
POST /strategies/{id}/run-now  — dry/paper/live
POST /walk-forward
POST /walk-forward/stream      — SSE progress
GET  /schedule
POST /schedule
DELETE /schedule
GET  /factors
GET  /factors/survivors
POST /factors/download
```

---

## v1 Scope — What TO Build

1. **Auth** — email + password or magic link (Clerk or simple JWT)
2. **Stripe** — two tiers: Signals ($15–30/mo) and Pro ($50–100/mo)
3. **Signals delivery** — cron job every 14 days: call the engine, email current regime + target positions to Signals subscribers
4. **Three UI screens (priority order):**
   - Find tab (leaderboard) → connects to `/fuzz`, `/fuzz/runs`, `/fuzz/leaderboard`
   - Analyze tab (strategy inspector) → connects to `/backtest`, `/baselines`, `/strategies`
   - Live Status → connects to `/strategies/{id}/run-now`, `/strategies/{id}/orders`, `/schedule`

---

## What NOT to Build Yet

- Multi-broker support (Schwab only for now)
- User-run walk-forwards exposed publicly (EC2 cost risk)
- Mobile app
- Multi-tenant Postgres swap-in
- Tauri desktop shell
- Options strategies

---

## Data Model (SQLite tables)

```
strategies          id, name, kind, params JSON, trade_mode, archived, notes
walk_forward_runs   id, strategy_id, month rows JSON, summary, final_equity
preflight_checks    audit trail per evaluation
fuzz_runs           id, timestamp, label, goal, window, top_score, elapsed
fuzz_results        run_id, score, metrics, params JSON
```

---

## Dev Quick-Start

```powershell
# Terminal 1 — FastAPI on 127.0.0.1:8765
cd C:\Code\algo-trading-schwab
python -m stratscout.api.app

# Terminal 2 — React UI on 127.0.0.1:5173
cd C:\Code\algo-trading-schwab\stratscout\web
npm install  # first time
npm run dev
```

Tests: `python -m pytest stratscout/tests` (66 passing)

---

## Key Architecture Decisions Already Made

- The FastAPI service trusts an `X-User-Id` header from upstream in web mode — auth/billing sit in front of it, not inside it
- CORS is wide-open in desktop mode (`STRATSCOUT_MODE=desktop`), restricted by `STRATSCOUT_CORS_ORIGINS` in web mode
- Fuzz runs save EVERY trial result (not just top 100) — the leaderboard is recomputed from SQLite
- Walk-forward streams progress via SSE (`/walk-forward/stream`) — big runs take minutes
- `combo_alpha ≈ 0.4–0.5` is empirically robust across all configs; current live uses 0.8 (optimizer choice)
- Vol targeting (7.35% ann. target) dramatically outperforms binary stop-loss — both are still active
