# Pre-registration — a language that DL's itself (rung 1)

Predictions were stated before `benchmark.py` was built (see the design
message that spawned this rung). Difficulty tuning was done before the
scored multi-seed run, per honest-benchmark discipline.

## The question

`sdl` (`sdl.py`) is a ~5-construct DSL whose runtime **is** a Spikeling
spiking net. A program declares senses, routes, a dense sense→route wire
block, and winner-take-all inhibition, and compiles to a real
`pyspike.Net` → `SpikelingRuntime`. `decide(x)` runs a first-past-the-post
spiking race and returns a `Trace`.

Can the program improve its own routing from an **endogenous** reward —
`+1` if exactly one route won the race, `−1` otherwise, plus a decisiveness
term (how far ahead the winner was) — **with no labels ever seen**?

## Arms (same program, same held-out eval set)

| id | training signal | topology |
|---|---|---|
| `A1_fixed` | none — weights hand-set to the task's route patterns | frozen |
| `A2_endo` | ES on the endogenous reward only | frozen |
| `A3_endo_restr` | `A2` + prune/spawn driven by the same endogenous reward | self-restructuring |
| `A0_oracle` | ES on the **true** reward (route == label) | frozen — ceiling for this substrate |

## Primary diagnostic

Pearson `corr(endogenous reward, true-correctness)` per decision on the
eval set. **If its CI includes ~0 the premise is dead** regardless of arm
accuracies — the endogenous signal would carry no information about
whether the route was right.

## Predicted verdicts

- **`premise_endo_tracks_truth`** — predict **DEAD or marginal**. "A route
  won decisively" is scored `+1.3` whether the route is right or wrong; a
  confidently-wrong route looks identical to a confidently-right one. Only
  a genuinely-ambiguous-input fraction could give the decisiveness term
  discriminative variance.
- **`H_A2_beats_A1`** — predict **REFUTED or marginal**. Endogenous ES
  barely moves accuracy above the hand-set baseline.
- **`H_A2_below_oracle`** — predict **CONFIRMED**. The label-trained
  ceiling is clearly above the endogenous arm.
- **`H_A3_no_better_than_A2`** — predict **CONFIRMED**. Restructuring driven
  by a weak proxy adds noise, not signal — same as the SSHNode rung and the
  base rate for elegant meta-mechanisms in this portfolio
  (`oscillator_memory`, `catq_ternary_llm`, the SSH node).

## Falsification / success bars

- Premise **ALIVE** (`corr` CI lower bound > 0.05) **and** `A3 > A2 > A1`
  with non-overlapping CIs → this line is worth a rung 2 (real substrate,
  compare against `spiking_orchestrator`'s promotion ledger).
- Otherwise → lean-negative, and the line closes the way the photonic and
  SSHNode lines did.

## Discovered limitation (tuning pass, before the scored run)

The first-past-the-post WTA-race substrate is **too weak to route this
task even with an oracle**: `A0_oracle` tops out near chance-to-0.6 on a
3-way route (chance 0.333) across ~12 weight-scale / window / signed-vs-
unsigned configurations. So this rung can measure "does the endogenous
signal train the program" only relative to a ceiling that is itself low.
A conclusive test of the *premise* needs a substrate a label-trained
oracle can actually solve (hidden layer, learned readout, or an easier
task) — recorded here as rung 1b, contingent on rung 1's result being
interesting enough to justify it.
