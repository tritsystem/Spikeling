# Results — Izhikevich & AdEx as first-class `.spk` neuron models

**Date:** 2026-09-10   Companion to `PRE_REGISTRATION_neuron_models.md`.
Reproduce: `python core/tests/test_neuron_model_conformance.py` (exit 0).

## Verdict table

| Bar | Claim | Verdict | Evidence |
|---|---|---|---|
| 1 | LIF zero-regression | **MET** | 4 example nets (`sound_localizer` 15, `signal_detector` 970, `word_learner` 808, `agent_brain` 445 spikes) produce a **byte-identical** spike log to a pre-change golden captured by `git stash`-ing the runtime/compiler edits. `CCodeGenerator.generate()` emits **byte-identical** C for all 5 examples before/after. |
| 2 | Izhikevich fidelity | **MET** | `.spk` `type=Izhikevich preset=chattering` @ dt=0.5, drive=10.0, 300 ms → 27 spikes (>5), ISI-range-ratio **19.60** vs `regular_spiking` **1.84** (19.60 > 1.5×1.84). Runtime spike times are an **exact** match (27/27, bit-identical) to `pyspike_neuron_models.IzhikevichNeuron("chattering")` fed I=10.0. |
| 3 | AdEx fidelity | **MET** | `.spk` `type=AdEx` @ dt=0.1, drive=50.0, 500 ms → 22 spikes, ISI **16.2 → 22.4 ms** (grows > 1.2×). Exact match (22/22) to `pyspike_neuron_models.AdExNeuron`. Runtime `type=LIF` under matched constant drive: ISI **3.20 → 3.20 ms** (flat, < 0.15 drift). |
| 4 | Generated-C parity (Izhikevich) | **MET** | `CCodeGenerator.generate_model_sim()` for the chattering net, compiled `gcc -O2 -std=c11`, run 300 ms @ dt=0.5: 27 spikes, **max per-spike deviation 0.000 ms** vs the Python runtime (tolerance was ±0.5 ms / ±1 tick). Bit-exact. |

**4 MET / 0 NOT-MET / 0 UNREACHABLE.**

## RED / GREEN

- **RED (pre-change, captured by stash):** `type=Izhikevich` and `type=AdEx`
  ran the LIF `stimulate()` path — regular firing, `ISI_range_ratio ≈ 1`, no
  bursts, no adaptation. A relabelled LIF.
- **GREEN (post-change):** the verdict table above.

## What shipped

| File | Change |
|---|---|
| `core/runtime/neuron_models.py` | **new.** `IzhikevichModel` / `AdExModel` — `reset()`, `step(state, drive) → (state, fired)`, C emit hooks. `step()` math is a line-for-line port of the verified `pyspike_neuron_models.py` classes (`v ** 2` kept, not `v * v`, to stay bit-identical). `build_model()`, `emit_model_sim_c()`. |
| `core/runtime/runtime.py` | `NeuronState` gains `model` / `model_state` / `syn_current`. `SpikelingRuntime(ast, dt_ms=1.0)`. `stimulate()` / `tick()` delegate to `model.step` for model neurons. `_fire()` skips the membrane reset for them (they self-reset v). `_deliver_spike()` injects synaptic charge as `syn_current` (a model takes a current, not a membrane bump). **Every model branch is `if n.model is not None`; the LIF lines are untouched.** |
| `core/compiler/compiler.py` | `NeuronDef.params` (a `{k: v}` bag). `NEURON_RE` extended to capture trailing `key=value` tokens (stops at `# comment`; empty for plain LIF). `SpikelingParser._parse_kv()`. `CCodeGenerator.generate_model_sim()` — standalone `spikeling_model_sim.c`. `generate()` (the LIF hardware path) not modified. |
| `core/stdlib/neurons.spk` | rewritten to real, parseable grammar (was `neuron_type …`, which the parser never accepted). |
| `core/examples/izhikevich_chattering.spk`, `core/examples/adex_adaptation.spk` | new example nets. |
| `core/tests/test_neuron_model_conformance.py`, `core/tests/_conformance_drive.py`, `core/tests/golden/*.json` | the conformance harness + shared deterministic driver + pre-change goldens. |

## Honest limits (also in the pre-registration)

- **Verilog / GDScript backends** for the new models are not wired — the
  harness is structured to add backends but only proves Python ↔ C today.
- **Synaptic-current scale**: `syn_current += weight × 50.0` reuses the LIF
  constant. Whether 50.0 is the right mapping onto Izhikevich's / AdEx's
  own current units is a calibration question left for a follow-up; the
  multi-neuron model path works and is deterministic, but no example
  exercises a *trained* model-neuron net yet.
- `trainer.py` reads a neuron's `membrane_potential` on a 0..threshold
  scale; for a model neuron that field now carries a biophysical voltage
  (≈ −65..30 mV). No example trains a model neuron, so nothing breaks, but
  a mixed LIF+Izhikevich trained net would need `trainer.py` to learn the
  per-neuron scale.
