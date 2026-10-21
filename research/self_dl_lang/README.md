# self_dl_lang — a DSL whose runtime is a spiking net, asked to train itself

**Question.** Can a program written in a tiny DSL improve its own behaviour
from an *endogenous* signal derived from its own execution trace — no
labels — where the DSL's runtime is a real Spikeling spiking net?

Rung 1 tests the smallest version: a 3-way router. The endogenous signal
is "did exactly one route win the spiking race, decisively." Arms compare
that against a frozen hand-set baseline, a self-restructuring variant, and
a label-trained oracle ceiling.

## Files

| file | what |
|---|---|
| `sdl.py` | the DSL: `sense` / `route` / `wire` / `wta` / `window`. `compile_source(text)` → `Program` wrapping a real `pyspike.Net` → `SpikelingRuntime`. `Program.decide(x)` runs a first-past-the-post spiking WTA race and returns a `Trace`. `.weights`, `.prune_dead_routes`, `.spawn_hidden` for the outer loop. |
| `PREREGISTER.md` | predictions, written before the benchmark. |
| `benchmark.py` | 4 arms (fixed / endo-trained / endo-trained+restructure / oracle), ES trainer, endogenous-vs-true correlation diagnostic. |
| `RESULTS.md` + `RESULTS.json` | outcome. |

## Run

```bash
python benchmark.py                    # 6 seeds
python benchmark.py --seeds 2 --epochs 40
python benchmark.py --smoke
```

## Outcome (see RESULTS.md)

Scaffolded and runnable, but **substrate-limited**: the first-past-the-post
WTA-race substrate can't route the task even with an oracle (label-trained
ES tops out near chance), so the rung can't conclusively judge the premise.
What is clear: the endogenous "decisive winner" signal has **zero
correlation** with routing correctness (`corr ∈ [−0.20, +0.11]`, CI spans
0) at every setting — a decisively-wrong route scores the same as a
decisively-right one. Lean-negative, consistent with the portfolio base
rate for elegant self-referential mechanisms. `RESULTS.md` lists what a
conclusive rung 1b would require.
