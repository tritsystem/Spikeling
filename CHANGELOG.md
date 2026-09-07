# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_Nothing yet._

## [1.0.2] — 2026-09-06

### Fixed

- `pyproject.toml`'s `py-modules` list was still missing `pyspike_neuron_models`
  (the standalone, independently-verified Izhikevich/AdEx implementations) —
  added it alongside the three modules already listed in 1.0.1.
- `Net.neuron()`'s live-mode branch (used after `build_live()`) always created
  a plain `NeuronState` in `self._live_rt.neurons` regardless of the declared
  `type`, so a `type="Resonator"` neuron added *after* `build_live()` never
  landed in `self._live_rt.resonators` — and `step_resonators()` only ever
  iterates `.resonators.values()`, so such a neuron could never fire no
  matter what it was driven with. The batch `build()` path
  (`SpikelingRuntime.__init__`) already routed by `neuron_type` correctly;
  live mode had silently diverged from it. Found while building a
  heterogeneous-neuron-dynamics demo in the sibling `spikegate` project (a
  live-constructed Resonator produced zero detections regardless of input —
  not a tuning problem, the neuron was never actually a Resonator at
  runtime). Fixed by routing on `type` in the live branch too, deriving
  `coupling` the same way the batch path does
  (`DEFAULT_RESONATOR_BASE_GAIN * omega**2`) when not given explicitly.
  Regression test: `test_live_resonator_routing.py` (routing + real
  on-frequency detection + off-frequency rejection, all driven through the
  actual live-mode code path, not the batch one).

## [1.0.1] — 2026-09-06

### Fixed

- `pyproject.toml` only packaged `core*`; a real `pip install` of `spikeling`
  did not ship the loose root-level modules (`pyspike.py`, `pyspike_causal.py`,
  `pyspike_surrogate_gradient.py`) that every downstream consumer
  (`spiking_orchestrator.py`, `arch-finder`, the new `spikegate` package)
  actually imports. Added `[tool.setuptools] py-modules = [...]` for those
  three files — additive only, no existing `sys.path.insert(...)` call site
  changes behavior.

## [1.0.0] — 2026-09-03

First tagged release. The `.spk` language and the four backends have been stable
in use for months; this tags that state and adds packaging.

### Language & runtime (`core/`)

- `.spk` DSL: `neuron` / `connect` / `action` / `refractory` / `learn=STDP`.
- Neuron types: **LIF**, **Izhikevich**, **AdEx**, **Resonator** (a damped
  oscillator that responds only near its own frequency — a frequency-domain
  primitive, benchmarked vs Goertzel/FFT in `resonator-prototype/`).
- Interactive Python runtime with STDP learning; `python -m core` /
  `python -m core path/to/net.spk`.

### Backends — one `.spk`, four targets

- **C** — generated for embedded / production.
- **Verilog** — generated for FPGA / hardware simulation, with a testbench
  (`sdk-verilog/`).
- **GDScript** — drop a spiking "mind" into a Godot game (`godot-runtime/`,
  `godot-plugin/`); an FPS with SNN-driven enemy AI in `fps-game/`.

### Around the runtime

- Hardware **sensor-adapter layer** (acoustic / telemetry / video confirmed
  against real devices).
- **MCP server** (`mcp_server.py`) exposing the runtime as tools.
- `ai-apps/` — Ollama/RAG assistants built around Spikeling.
- **Agent-orchestration-as-SNN** experiment — routing and lateral inhibition
  work; the scheduling/arbitration variants were **falsified as differentiators**
  against classical baselines (greedy graph colouring, online colouring). Kept in
  the repo and reported as negatives.

### Packaging (new)

- `pyproject.toml` — installs as `spikeling`, exposes a `spikeling` console
  script. This is a thin wrapper; `python -m core` remains the canonical
  interface.
- `.github/workflows/test.yml` — runtime smoke + pure-logic parity tests on
  push/PR (py3.10, 3.12).
- Fixed a `UnicodeEncodeError` on non-UTF-8 consoles in `core/__main__.py`.
- `package.json` renamed to `spikeling-render-pipeline` and marked `private` — it
  is the dev-only React screenshot toolchain, not the runtime.

[Unreleased]: https://github.com/tritsystem/Spikeling/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/tritsystem/Spikeling/releases/tag/v1.0.0
