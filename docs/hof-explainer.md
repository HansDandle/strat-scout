# Hall of Fame Seeding — How It Works & Overfitting Risk

## What is HoF seeding?

Every 14-day optimization period, the optimizer runs hundreds of random parameter trials against trailing 12-month data. Without seeding, each period starts from scratch — pure random search.

With HoF seeding enabled (`--hof-seeds N`), a small number of historically proven parameter sets are injected as fixed trials *before* the random search begins. These seeds come from past periods that performed well in similar market conditions.

As the run progresses, each completed period writes its winning params back to the HoF in real time — so seeds for period 10 include what actually worked in periods 1-9 of the same run, not just prior runs.

## How are seeds selected?

Seeds are matched by:
1. **Actual regime** — AGG/BIL and TLT/BIL ratios determine whether you're in risk-on, risk-off rising rates, or risk-off falling rates. Seeds from the same regime are prioritized.
2. **Performance quality** — only periods with val_return > 2% and Calmar > 0.8 qualify. Marginal wins don't seed forward.
3. **Recency** — seeds are sorted by Calmar ratio descending so the best risk-adjusted performers come first.

All seeds are strictly date-gated: `WHERE month < current_period`. No future data can leak in.

## Does this cause overfitting?

It can — and here's exactly how to think about it:

**The guardrails that keep it honest:**
- Seeds only count toward ~8% of trials (25 out of 300). The other 275 are pure random, keeping the search space wide.
- Seeds compete on merit — they're scored against the same 3 training sub-windows as every random trial. If last period's params don't generalize, they lose to random params and don't get used.
- Regime gating prevents rising-rate params from bleeding into risk-on periods and vice versa.
- The quality filter (`val_return > 2.0`, `Calmar > 0.8`) means only genuinely good periods seed forward.

**The real risk — consecutive similar regimes:**
If you're in risk-on for 10 straight periods, the same params keep winning and keep getting seeded. The optimizer reinforces a narrow corridor of params that worked in the streak. When the regime flips, it may be slower to adapt because it hasn't been forced to explore aggressively.

This is the subtler form of overfitting — not cheating on data, but converging prematurely on a local optimum during a persistent regime.

**Mitigation in progress:** Capping how many times the same param fingerprint can appear in the HoF within a rolling window, so diversity is preserved even during long streaks.

## Empirical result

Tested over the hardest 3-year window in our backtest (2021–2023, which includes the choppy post-COVID regime and the 2022 rate shock):

| | HoF Seeded (25/300 trials) | Baseline (no seeds) |
|--|--|--|
| Median NAV | $33,408 | $13,067 |
| Median CAGR | +48.9% | +9.3% |
| Median MaxDD | 40.4% | 30.9% |
| Median Calmar | 1.21 | 0.32 |

2.6× higher median NAV, 3.8× better Calmar over the period where the strategy historically struggles most. MaxDD is modestly wider (+9.5pp) — the seeds are more aggressive, which costs some drawdown in exchange for significantly higher returns.

## Should I turn it on?

Default is off (`--hof-seeds 0`). Turn it on if:
- You have at least 1-2 full runs worth of HoF data (run `build_hof.py` first)
- You're willing to accept slightly wider drawdowns for meaningfully higher returns
- You understand the consecutive-regime risk and will monitor for param convergence

Recommended value: `--hof-seeds 20` to `--hof-seeds 30`. Above 50 starts to crowd out random exploration.
