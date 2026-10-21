# 🧠 Spikeling — Neuromorphic AI Runtime

A domain-specific language and runtime for spiking neural networks.  
Write networks in `.spk` files. Run them in Python. Compile them to C for production.

---

## Project Structure

```
spikeling/
├── __main__.py               # Entry point
├── compiler/
│   └── compiler.py           # .spk parser + C code generator
├── runtime/
│   └── runtime.py            # Python runtime: LIF neurons, STDP, spike dispatch
├── stdlib/
│   └── neurons.spk           # Standard neuron type presets
└── examples/
    └── sound_localizer.spk   # Binaural sound localizer demo
```

---

## The DSL

```spk
# my_network.spk

neuron LeftMic  threshold=110 leak=5  type=LIF
neuron RightMic threshold=110 leak=5  type=LIF
neuron Motor    threshold=80  leak=3  type=LIF

connect LeftMic  -> Motor weight=0.8
connect RightMic -> Motor weight=0.8

action LeftMic  -> [SOUND_LOCALIZED_LEFT]
action RightMic -> [SOUND_LOCALIZED_RIGHT]
action Motor    -> [MOTOR_FIRE]

refractory=400ms
learn=STDP rate=0.01
```

### Directives

| Directive | Description |
|---|---|
| `neuron <name> threshold=<int> leak=<int> [type=LIF]` | Define a neuron |
| `neuron <name> threshold=<int> leak=<int> type=Izhikevich [preset=<regime>] [gain=<float>]` | Two-variable cortical model (bursts, adaptation) |
| `neuron <name> threshold=<int> leak=<int> type=AdEx [gain=<float>] [b=<float>]` | Adaptive-exponential model (spike-frequency adaptation) |
| `connect <src> -> <dst> weight=<float>` | Add a weighted synapse |
| `action <neuron> -> [<COMMAND>]` | Map a spike to a named command |
| `refractory=<int>ms` | Global refractory period |
| `learn=STDP rate=<float>` | Enable STDP learning |

### Neuron types (stdlib/neurons.spk)

| Type | Description |
|---|---|
| `LIF` | Leaky Integrate-and-Fire — fast, standard |
| `Izhikevich` | Cortical neuron model — burst firing, adaptation |
| `AdEx` | Adaptive Exponential — best realism/speed balance |

---

## Running

```bash
# Run default example (interactive)
python -m spikeling

# Run your own network
python -m spikeling path/to/my_network.spk

# Compile only, no interactive loop
python -m spikeling path/to/my_network.spk --no-interactive
```

### Interactive controls

| Key | Action |
|---|---|
| `a` / `←` | Stimulate first neuron |
| `d` / `→` | Stimulate second neuron |
| `q` / `ESC` | Quit |

---

## What the compiler emits

Running any `.spk` file produces two C files next to it:

**`spikeling_hw.h`** — structs, constants, function declarations  
**`spikeling_hw.c`** — neuron table, synapse table, `spikeling_tick()`, `spikeling_stdp_update()`

Compile for production:

```bash
gcc -O2 -o my_network main.c spikeling_hw.c -lm
```

---

## How it actually works

### Leaky Integrate-and-Fire (LIF)

Each neuron maintains a membrane potential `V`. Each tick:

1. **Leak**: `V -= leak` (potential decays toward rest)
2. **Input drive**: `V += stimulus`
3. **Threshold check**: if `V >= threshold` → fire, reset `V = 0`
4. **Refractory**: neuron is silent for `refractory_ms` after firing

### STDP Learning

Weights update based on the timing between pre- and post-synaptic spikes:

```
Δw = rate × exp(-|dt| / 20ms)

dt > 0  (pre before post) → strengthen (LTP)
dt < 0  (pre after post)  → weaken     (LTD)
```

Weights are clamped to `[0.0, 1.0]`.

### Spike propagation

When a neuron fires, it walks all synapses where it is the source and adds `weight × 50.0` to each downstream neuron's membrane potential. This can trigger cascade firing.

---

## What's different from the original C project

| Original | Spikeling |
|---|---|
| Random weights, never updated | STDP learning, weights change over time |
| Hash function dressed as a network | Real LIF dynamics with leak and threshold |
| Windows-only (`msvcrt`) | Cross-platform (Windows + Unix/Mac) |
| Monolithic C files | Modular: DSL → compiler → Python runtime → C output |
| `leak` parsed but ignored | Leak applied every tick |
| No synapse propagation | Weighted synapses, cascade firing |

---

## Roadmap

- [x] `type=Izhikevich` runtime implementation — real (v, u) dynamics, ported
      from the verified `pyspike_neuron_models.py`; presets `regular_spiking` /
      `chattering` / `fast_spiking` / `intrinsically_bursting`. See
      `core/runtime/neuron_models.py`, `core/PRE_REGISTRATION_neuron_models.md`.
- [x] `type=AdEx` runtime implementation — real (v, w) dynamics with
      spike-frequency adaptation; same reference + pre-registration.
      Cross-backend (Python vs generated-C) parity proven by
      `core/tests/test_neuron_model_conformance.py`.
- [ ] Verilog / GDScript backends for Izhikevich & AdEx
- [ ] Multi-layer network examples
- [ ] Audio input driver (mic → spike encoder)
- [ ] Spike train visualiser
- [ ] Weight persistence (save/load trained networks)
