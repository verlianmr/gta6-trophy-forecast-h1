# GTA VI trophy rarity forecast

[![GTA VI trophy rarity forecast](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/verlianmr/gta6-trophy-forecast-h1/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/verlianmr/gta6-trophy-forecast-h1/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01)

An entry for [make.fun's GTA VI trophy rarity forecast](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
for every trophy of *Grand Theft Auto VI* on PlayStation 5, five quantiles of its earn rate on Exophase at
2026-11-26T00:00Z — a week after launch — from the reads the house recorded up to the deadline.

`python3 main.py`, Python 3.13, standard library only, no network, deterministic.

## The problem in one paragraph

An Exophase earn rate is the percent of the players Exophase tracks for the game who have earned a trophy. It
is a ratio, and in launch week both terms move: players sync their profiles in as they buy the game, and the
players already tracked keep playing. The rate therefore grows, decelerating, and the recording is a
staircase — Exophase only changes a number when it refreshes its counts, so consecutive reads repeat and then
jump. Launch night makes the point: 638 players tracked and the page unchanged for hours, then a refresh to
3,868 players with *every* rate but the 100% one **lower** than before. A forecast that assumes rates only
rise is wrong about the first day; a forecast that assumes the last read is the answer is wrong about the
week.

## The model

**Scale.** Rates are bounded and multiplicative, so everything happens in the logit, `L = ln(p / (1 - p))` —
the same scale the scorer uses. A rate that multiplies by *k* moves a fixed distance, and a rate near 100
cannot overshoot. Rates are clamped to 0.01–99.99 before the logit, exactly as the scorer clamps them.

**Growth.** `L(t) = a + b · ln(1 + t)`, with *t* in hours since 2026-11-19T00:00Z. A launch week's rates grow
close to a power law in time (earners and the tracked pool both grow like powers of *t*, at different
exponents), and `ln(1 + t)` is the power law's linear form. Growth decelerates because the pool matures and
because each trophy approaches its ceiling; the logit link handles the ceiling for free.

**A robust per-trophy fit.** The staircase is first collapsed to its change points — the first read at each
new level, at most 40 of them — which turns ~250 reads into ~15 genuinely distinct observations. The slope is
the **weighted median of all pairwise slopes** (Theil–Sen), not a least-squares line: one refresh glitch must
not tilt a trophy's whole week. Reads are weighted by recency with a 24-hour half-life and a 15% floor, so the
last two days of a three-day history drive the slope while launch night still counts.

**Hierarchical pooling.** One staircase cannot pin down a slope — at 638 tracked players a rate of 0.63% is
four people, and one more earner moves it by 25%. Fifty-five trophies share one progression curve, so each
slope is shrunk onto a prior built two ways and averaged:

* *by tier* (platinum / gold / silver / bronze, which is also the points band 300 / 90 / 30 / 15), with the
  tier pools themselves shrunk towards the all-trophy pool so the platinum — a tier of one — borrows the
  field's curve;
* *by level*: a slope in logit units is not the same size everywhere. A trophy at 60% is near its ceiling and
  gains a few points a day; a trophy at 0.05% multiplies. Regressing the fitted slopes on the trophies' own
  last levels captures that, and it is the only thing that can inform a trophy so rare its staircase is
  mostly counting noise.

Shrinkage is inverse-variance (empirical Bayes): a precisely measured slope keeps its own weight, a noisy one
is pulled onto the prior. Slopes are clamped non-negative — a trophy's rate does not fall over a week — and
the forecast always starts from the trophy's own last observed rate, so a quiet trophy cannot drift off its
reading.

**Spread.** Three independent parts, added in quadrature, then `q_τ = logistic(L̂ + z_τ · σ)`:

| part | what it prices in |
|---|---|
| `0.30 · √(days ahead)`, floored at 0.12 | the curve's *shape* may be wrong: growth decelerates, but whether it decelerates like `ln(1+t)` or faster is not knowable from 2.5 days of launch data. Dominant at 4.5 days. |
| `se(b̂) · dx` | the fitted slope's standard error, times the extrapolation distance, inflated 1.6× because a run of identical staircase reads is not a run of independent observations |
| `0.8 · residual` + `1/√earners` | the trophy's own reads wander around its curve (refresh jumps), and the anchor itself is a count: a rate backed by 4 earners is worth less than one backed by 400 |

**Edge cases.** A trophy no read carries, a case with fewer than two reads, or a series that never moves falls
back to the house baseline's persistence with its random-walk spread. Duplicate timestamps collapse to the
last read, null rates drop out per trophy, 0.00 and 100.00 clamp to 0.01 and 99.99, a trophy already at the
top is pinned, and the output row is clamped, rounded to 4 decimals and forced non-decreasing before it is
written.

## Backtest

`backtest.py` cuts a case's history at several checkpoints, forecasts from each cut, and scores against the
read that follows — pinball loss on the logit scale at the five levels, plus the 90% coverage and the typical
miss, exactly as the rules define them.

**What the practice cases can and cannot show.** Both practice cases are cut from launch night and every read
in their histories is the same unchanged page: the whole history is one flat staircase. On a flat history
every model collapses to persistence and every within-history cut scores ~0, because the reads it predicts
are identical to the reads it fitted. Their answers (the refreshed page 90 minutes later) are not in the file
at all. The practice is a smoke test that the program reads a case and writes a valid file — nothing more.

**What carries the calibration** is a synthetic launch week built to look like the real recording: a tracked
pool growing as a power law (638 at launch night, ~4,000 four hours later, ~90,000 at the checkpoint), a
per-trophy curve that is a probit in log time (slow at launch, saturating later) — deliberately *not* the
model's functional form — Exophase refreshing the whole page on its own clock, binomial counting noise, and
scenarios where the world changes regime after the checkpoint (saturates, re-accelerates) or a refresh dilutes
every rate as launch night's first one did. Mean over 8 scenarios × 2 seeds × 9 cut/horizon cells:

| model | mean pinball loss | 90% coverage | typical miss (pts) |
|---|---|---|---|
| persistence (the house baseline) | 0.509 | 0.87 | 7.02 |
| per-trophy curve, no pooling, no floor | 0.153 | 0.64 | 2.32 |
| **final model** | **0.148** | **1.00** | **2.27** |

At the final's own shape — fit on 59.5 h, score the read at 168 h — the model loses **0.17** in the
well-behaved scenarios against **0.57** for persistence, and holds coverage at 0.98–1.00 in every one.

Two honest caveats. The coverage of 1.00 says the synthetic world is *smoother* than a real launch week: it
over-covers, and a floor tuned only on it would be near zero. The floor is kept at 0.30 per √day — inside the
range the rules suggest — because in the scenarios where the world changes regime after the checkpoint no
spread saves the median, and a floor halves the loss there (0.63 → 0.44) for ~0.02 in the easy ones. The
second caveat: this is a test against my priors, not against the world. It can show the machinery is sound,
that pooling beats a per-trophy fit and that the floor is in the right range. It cannot show that a launch
week's rates really grow like this.

## Run it

```sh
# one case at a time, as the grader does
mkdir -p /tmp/out
MAKEFUN_INPUT_DIR=/path/to/kit/input MAKEFUN_OUTPUT_DIR=/tmp/out python3 main.py

# the backtest: practice cases + the synthetic family
python3 backtest.py /path/to/kit/input/p00.in /path/to/kit/input/p01.in

# sweep the spread floor and the extrapolation scale
python3 backtest.py --tune --no-synthetic
```

`main.py` reads every `*.in` in `$MAKEFUN_INPUT_DIR` (default `/input`) in name order and writes one
`<id>.out` per case to `$MAKEFUN_OUTPUT_DIR` (default `/output`):

```json
{"trophies":[{"id":1,"q":[0.0208,0.0467,0.0818,0.1433,0.3207]},{"id":2,"q":[0.7776,1.6001,2.6299,4.2934,8.5154]}]}
```

One row per trophy of the case, ids unique, five finite non-decreasing numbers in 0–100 at the levels
0.05 / 0.25 / 0.5 / 0.75 / 0.95. A final-sized case (359 reads × 55 trophies) runs in **0.10 s** and writes
**2.9 KB**, against an 8 s and 65,536-byte budget.

## Files

| file | what it is |
|---|---|
| `main.py` | the entrypoint: `parse_case`, `series_for_trophy`, `fit_growth`, `pool_slopes`, `forecast_quantiles`, `write_output`, and a `pinball_loss` helper |
| `backtest.py` | checkpoint backtests, the synthetic launch-week simulator, and the constant sweep |
| `LICENSE` | MIT |

`main.ts` is deliberately absent: the rules take one entrypoint at the repository root, not both.
