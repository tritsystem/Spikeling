# Pre-registration — Izhikevich & AdEx as first-class `.spk` neuron models

**Date:** 2026-09-10   **Method:** gbranaa4-hue (pre-register acceptance
bars *before* running; report negatives exactly like positives; RED/GREEN
reproduction; 3-way verdicts).

## The gap being closed

`NeuronDef.neuron_type` has accepted `"Izhikevich"` and `"AdEx"` since the
type field was added, and `core/stdlib/neurons.spk` documents presets for
them — but the runtime's `SpikelingRuntime.__init__` put every non-Resonator
neuron into a plain `NeuronState` and ran the LIF `stimulate()` path on it.
A `.spk` author writing `type=Izhikevich` silently got a relabelled LIF: no
`u` recovery variable, no quadratic term, no bursting, no adaptation. The C
code generator emitted the same flat `Neuron` struct for all of them.
`pyspike_neuron_models.py` carries a *standalone*, self-tested Izhikevich /
AdEx implementation that was never wired into the DSL.

This change makes the two models real through the Python runtime and adds a
generated-C parity path, using `pyspike_neuron_models.py` as the frozen
behavioural reference.

## What is being built (task A)

1. `core/runtime/neuron_models.py` — `IzhikevichModel` / `AdExModel`, each
   with `reset() -> state`, `step(state, drive) -> (state, fired)`, and C
   emit hooks. The `step()` math is a line-for-line port of the verified
   classes in `pyspike_neuron_models.py` (same sub-step sizes: 0.5 ms for
   Izhikevich, 0.1 ms for AdEx; same reset rules).
2. `core/runtime/runtime.py` — `NeuronState` gains `model` / `model_state` /
   `syn_current`; `SpikelingRuntime(ast, dt_ms=1.0)` builds a model for
   `type in {Izhikevich, AdEx}`; `stimulate()` and `tick()` delegate to
   `model.step`; `_fire()` skips the membrane reset for model neurons (they
   self-reset); `_deliver_spike()` injects synaptic charge as `syn_current`
   for a model downstream instead of touching `membrane_potential`.
3. `core/compiler/compiler.py` — `NeuronDef.params` (a `{k: v}` bag parsed
   off trailing `key=value` tokens on the `neuron` line); `NEURON_RE`
   extended to capture them; `CCodeGenerator.generate_model_sim()` emits a
   standalone `spikeling_model_sim.c` whose `main()` runs one model neuron
   under constant drive and prints spike times.
4. `core/stdlib/neurons.spk` fixed to real grammar; two example nets added.

**The LIF path is not edited.** `model is None` → the exact pre-existing
lines run.

## Acceptance bars (verdict = met / not-met / unreachable)

**Bar 1 — LIF zero-regression.**
Every `.spk` in `core/examples/` that parses today still parses, and the
Python runtime produces a **byte-identical spike log** to a pre-change
baseline for a fixed stimulation script (per-example seeded drive sequence,
2000 ticks). Captured as a golden file committed alongside the harness.
*Not-met if any example's spike log differs in any tick or neuron name.*

**Bar 2 — Izhikevich fidelity (chattering signature).**
A `.spk` net `neuron rs ... type=Izhikevich preset=regular_spiking` and
`neuron ch ... type=Izhikevich preset=chattering`, both driven at constant
`drive` for 300 ms at `dt_ms=0.5`, reproduce the signature asserted by
`_selftest_izhikevich_chattering_bursts()` in `pyspike_neuron_models.py`:
- `len(ch.spikes) > 5`, and
- `ISI_range_ratio(ch) > 1.5 × ISI_range_ratio(rs)` where
  `ISI_range_ratio = max(ISI) / min(ISI)`.
Additionally the runtime's `ch` spike-time list must match a direct
`IzhikevichNeuron(preset="chattering")` reference fed the identical current
**exactly** (same float math, same sub-stepping). *Not-met if the signature
fails or the runtime path diverges from the reference by ≥ 1 spike.*

**Bar 3 — AdEx fidelity (spike-frequency adaptation).**
A `.spk` net `neuron a ... type=AdEx` under constant drive for 500 ms at
`dt_ms=0.1` shows ISI growth: `ISI[-1] > ISI[0] × 1.2` (matches
`_selftest_adex_shows_spike_frequency_adaptation()`), while a `type=LIF`
neuron under a drive tuned to the same initial rate does **not**
(`|ISI[-1] − ISI[0]| < ISI[0] × 0.15`). *Not-met if AdEx fails to adapt or
LIF spuriously adapts.*

**Bar 4 — generated-C parity (Izhikevich).**
`CCodeGenerator.generate_model_sim()` for the chattering net, compiled with
`gcc` and run for 300 ms at `dt_ms=0.5` under the same constant drive,
produces spike times within **±1 tick** of the Python runtime for every
spike, and the **same spike count**. *Not-met if the count differs or any
spike is off by > 1 tick. Unreachable if no C compiler is present* (report
as unreachable, not as a pass).

## RED / GREEN

- **RED (pre-change):** `type=Izhikevich` net driven at constant current
  produces LIF-shaped regular firing — no bursts; `ISI_range_ratio ≈ 1`.
  `type=AdEx` shows no ISI growth. Recorded before editing the runtime.
- **GREEN (post-change):** Bars 2–4 as above.

## Out of scope (named, not done)

- Verilog / GDScript backends for the new models (the cross-backend
  conformance harness — task B — is scaffolded to accept them but only
  Python + C are wired now).
- `trainer.py` interpreting a model neuron's biophysical `membrane_potential`
  scale (no example uses a model neuron in a trained net yet).
- Synaptic-current calibration: `syn_current += weight × 50.0` reuses the
  LIF constant; whether 50.0 is the right pA-scale mapping for Izhikevich
  is left for a follow-up (documented in `neuron_models.py`).
