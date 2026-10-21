# Results — rung 1

6 seeds, 80 ES epochs, 3-way routing, chance = 0.333. `RESULTS.json` has
the per-seed numbers. Verdicts against `PREREGISTER.md`.

## Scoreboard (true routing accuracy, mean ± 95% CI over 6 seeds)

| arm | true acc | decisive rate | corr(endo, true) |
|---|---|---|---|
| `A1_fixed` (hand-set weights, frozen) | 0.502 [0.434, 0.569] | 1.00 | −0.085 [−0.171, 0.000] |
| `A2_endo` (ES on endogenous reward) | 0.349 [0.293, 0.404] | 1.00 | +0.002 [−0.109, +0.113] |
| `A3_endo_restr` (+ prune/spawn) | 0.375 [0.325, 0.424] | 1.00 | −0.018 [−0.092, +0.057] |
| `A0_oracle` (ES on the label) | 0.437 [0.408, 0.466] | 1.00 | −0.114 [−0.182, −0.045] |

## Verdicts

| check | predicted | verdict | note |
|---|---|---|---|
| `premise_endo_tracks_truth` | DEAD/marginal | **DEAD** | `corr(endo, true)` CI includes 0 for every arm. A decisively-wrong route scores `+1.3` exactly like a decisively-right one, and `decisive = 1.00` always, so the endogenous signal has neither discriminative content nor variance. |
| `H_A2_beats_A1` | REFUTED/marginal | **REFUTED** | 0.349 vs 0.502. Endogenous ES makes it *worse* than the hand-set weights. |
| `H_A2_below_oracle` | CONFIRMED | **CONFIRMED** but meaningless — 0.349 vs 0.437, both near chance. |
| `H_A3_no_better_than_A2` | CONFIRMED | **REFUTED (nominally)** — 0.375 > 0.349, but the CIs overlap heavily and both are near chance; this is noise, not restructuring helping. |

## Honest read — scaffolded, but the test is substrate-limited

**Two things are clear:**

1. **The endogenous signal carries no information about correctness.** Across
   ~12 configurations in the tuning pass and all 4 arms here,
   `corr(endogenous reward, true-correct)` sat in `[−0.20, +0.11]` with the
   CI always spanning 0. "A route won the spiking race decisively" simply
   does not track "the right route won" for this program.

2. **The substrate can't route the task even with an oracle.** `A0_oracle`
   (ES on true labels, and in the tuning pass up to 400 epochs / signed
   weights / longer race) never clears ~0.5–0.6 on a 3-way route. The
   first-past-the-post WTA race over rate-coded input is close to "route
   with the largest `Σ weight × input-rate` wins" — a near-degenerate
   discriminator for similar non-negative patterns — and ES on the noisy
   discrete race reward degrades even a good hand-set init.

**So the premise ("a language that DL's itself from its own execution
trace") is not supported here — but this rung cannot conclusively kill it**,
because the ceiling it is measured against is itself near chance. What it
*can* say: the specific endogenous proxy chosen ("decisive single winner")
is a dead end, consistent with the portfolio base rate for elegant
self-referential mechanisms (`oscillator_memory`, `catq_ternary_llm`, the
SSHNode rung — all lost to a simple baseline).

## What a conclusive rung 1b would need

- A substrate a label-trained oracle can actually solve to ~0.85+: a hidden
  layer, signed weights that let a route be *inhibited* by wrong-pattern
  senses, or a learned readout instead of first-past-the-post.
- An endogenous signal with real variance — e.g. a task with a genuine
  ambiguous fraction so "indecisive" decisions exist, or a
  self-consistency signal (does re-running the same input give the same
  route?) rather than "was the winner decisive".
- Compared against `spiking_orchestrator`'s existing promotion ledger on
  the same task, per the original rung design.

Whether to build 1b is a real call: the base rate and this rung's flat
`corr ≈ 0` both argue the line closes here.

## Files

`sdl.py` (the DSL — clean, works: compiles to a real `pyspike.Net`,
weights + prune/spawn exposed), `benchmark.py` (4 arms, ES trainer,
correlation diagnostic), `PREREGISTER.md`, `RESULTS.json`.
