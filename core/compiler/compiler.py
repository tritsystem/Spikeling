"""
spikeling/compiler/compiler.py
==============================
Spikeling DSL Compiler

Parses .spk source files and emits:
  - A SpikelingNetwork object (for the Python runtime)
  - A .c / .h file pair (for production C compilation)

DSL Grammar:
  neuron <name> threshold=<int> leak=<int> [type=LIF|Izhikevich|AdEx]
  connect <src> -> <dst> weight=<float>
  action <neuron> -> [<COMMAND>]
  refractory=<int>ms
  learn=STDP rate=<float>
"""

import re
import os
import math
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  AST Nodes
# ─────────────────────────────────────────────

@dataclass
class NeuronDef:
    name: str
    threshold: int
    leak: int
    neuron_type: str = "LIF"
    # Resonator-only fields (None for LIF/Izhikevich/AdEx neurons).
    # A Resonator is a damped harmonic oscillator tuned to `freq_hz`,
    # used for frequency-selective detection (see
    # resonator-prototype/accuracy_benchmark.py for why this beats a
    # naive amplitude threshold at telling specific signals apart).
    freq_hz: Optional[float] = None
    damping: Optional[float] = None
    coupling: Optional[float] = None


@dataclass
class ConnectionDef:
    src: str
    dst: str
    weight: float
    delay_ms: float = 0.0  # see runtime.py's Synapse.delay_ms -- default 0.0 == instant, unchanged behavior


@dataclass
class ActionDef:
    neuron: str
    command: str


@dataclass
class SpikelingAST:
    neurons: list[NeuronDef]         = field(default_factory=list)
    connections: list[ConnectionDef] = field(default_factory=list)
    actions: list[ActionDef]         = field(default_factory=list)
    refractory_ms: int               = 0
    learn_rule: Optional[str]        = None
    learn_rate: float                = 0.01


# ─────────────────────────────────────────────
#  Parser
# ─────────────────────────────────────────────

class SpikelingParser:
    """Tokenises and parses a .spk source string into a SpikelingAST."""

    NEURON_RE     = re.compile(r"neuron\s+(\w+)\s+threshold=(\d+)\s+leak=(\d+)(?:\s+type=(\w+))?")
    # Resonator neurons use a different parameter set (no threshold/leak --
    # they're driven continuously, not pulsed): freq_hz + damping, with an
    # optional explicit coupling override (auto-derived from freq if omitted,
    # see runtime.ResonatorState — gain falls off at higher frequencies
    # otherwise, see benchmarks/ for why that matters).
    RESONATOR_RE  = re.compile(r"neuron\s+(\w+)\s+type=Resonator\s+freq=([\d.]+)\s+damping=([\d.]+)(?:\s+coupling=([\d.]+))?")
    # weight may be NEGATIVE now: a negative weight is an INHIBITORY synapse --
    # when src fires it DRAINS the target's membrane potential instead of adding
    # to it, so one neuron can veto/suppress another (see runtime propagation).
    CONNECT_RE    = re.compile(r"connect\s+(\w+)\s*->\s*(\w+)\s+weight=(-?[\d.]+)")
    ACTION_RE     = re.compile(r"action\s+(\w+)\s*->\s*\[(\w+)\]")
    REFRACTORY_RE = re.compile(r"refractory=(\d+)ms")
    LEARN_RE      = re.compile(r"learn=(\w+)\s+rate=([\d.]+)")

    def parse(self, source: str) -> SpikelingAST:
        ast = SpikelingAST()
        errors = []

        for lineno, raw_line in enumerate(source.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("neuron"):
                m = self.NEURON_RE.match(line)
                if m:
                    name, thresh, leak, ntype = m.groups()
                    ast.neurons.append(NeuronDef(
                        name=name,
                        threshold=int(thresh),
                        leak=int(leak),
                        neuron_type=ntype or "LIF"
                    ))
                else:
                    rm = self.RESONATOR_RE.match(line)
                    if rm:
                        name, freq, damping, coupling = rm.groups()
                        ast.neurons.append(NeuronDef(
                            name=name,
                            threshold=0,   # unused for Resonator; kept for C codegen compatibility
                            leak=0,        # unused for Resonator
                            neuron_type="Resonator",
                            freq_hz=float(freq),
                            damping=float(damping),
                            coupling=float(coupling) if coupling else None,
                        ))
                    else:
                        errors.append(f"Line {lineno}: malformed neuron — '{line}'")

            elif line.startswith("connect"):
                m = self.CONNECT_RE.match(line)
                if m:
                    src, dst, weight = m.groups()
                    ast.connections.append(ConnectionDef(src, dst, float(weight)))
                else:
                    errors.append(f"Line {lineno}: malformed connect — '{line}'")

            elif line.startswith("action"):
                m = self.ACTION_RE.match(line)
                if m:
                    neuron, command = m.groups()
                    ast.actions.append(ActionDef(neuron, command))
                else:
                    errors.append(f"Line {lineno}: malformed action — '{line}'")

            elif line.startswith("refractory"):
                m = self.REFRACTORY_RE.match(line)
                if m:
                    ast.refractory_ms = int(m.group(1))
                else:
                    errors.append(f"Line {lineno}: malformed refractory — '{line}'")

            elif line.startswith("learn"):
                m = self.LEARN_RE.match(line)
                if m:
                    ast.learn_rule = m.group(1)
                    ast.learn_rate = float(m.group(2))
                else:
                    errors.append(f"Line {lineno}: malformed learn — '{line}'")

            else:
                errors.append(f"Line {lineno}: unknown directive — '{line}'")

        if errors:
            raise SyntaxError("Spikeling parse errors:\n" + "\n".join(errors))

        self._validate(ast)
        return ast

    def _validate(self, ast: SpikelingAST):
        """Semantic checks — names must exist before they're referenced."""
        names = {n.name for n in ast.neurons}
        for c in ast.connections:
            if c.src not in names:
                raise NameError(f"connect: unknown neuron '{c.src}'")
            if c.dst not in names:
                raise NameError(f"connect: unknown neuron '{c.dst}'")
        for a in ast.actions:
            if a.neuron not in names:
                raise NameError(f"action: unknown neuron '{a.neuron}'")


# ─────────────────────────────────────────────
#  C Code Generator
# ─────────────────────────────────────────────

C_HEADER_TEMPLATE = """\
/* Auto-generated by Spikeling Compiler — DO NOT EDIT */
#ifndef SPIKELING_HW_H
#define SPIKELING_HW_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── Neuron ── */
typedef struct {{
    const char* name;
    uint32_t    threshold;
    uint32_t    leak;
    float       membrane_potential;
    float       last_spike_time;
    uint32_t    fire_count;
}} Neuron;

/* ── Synapse ── */
typedef struct {{
    uint32_t src_idx;
    uint32_t dst_idx;
    float    weight;
}} Synapse;

/* ── Resonator ──
 * A damped harmonic oscillator tuned to freq_hz, for frequency-selective
 * detection (e.g. telling a specific tone/signature apart from other
 * tones + noise) -- not pulsed like Neuron, continuously driven via
 * spikeling_resonator_step(). See runtime/runtime.py's ResonatorState
 * for the Python reference implementation this mirrors; gain is
 * normalized by coupling ~ omega^2 so higher-frequency channels in the
 * same bank don't respond weaker than low-frequency ones (measured in
 * resonator-prototype/resonator_bank.py). */
typedef struct {{
    const char* name;
    float       freq_hz;
    float       damping;
    float       coupling;
    float       threshold;   /* RMS-amplitude level that counts as "detected" */
    float       gate_threshold; /* |x| below this: skip the x*x energy multiply */
    float       energy_time_constant; /* seconds -- see spikeling_resonator_step()'s
                                          comment for why this must scale with dt,
                                          not be a fixed per-step decay constant */
    float       x;
    float       v;
    float       energy_ema;  /* mean-square amplitude, exponential moving average */
}} Resonator;

/* ── Network dimensions ── */
#define NEURON_COUNT     {neuron_count}
#define SYNAPSE_COUNT    {synapse_count}
#define RESONATOR_COUNT  {resonator_count}
#define REFRACTORY_MS    {refractory_ms}
#define LEARN_RATE       {learn_rate}f

/* ── Neuron table ── */
extern Neuron     neurons[NEURON_COUNT];
extern Synapse    synapses[SYNAPSE_COUNT];
extern Resonator  resonators[RESONATOR_COUNT];

/* ── Spike dispatch ── */
void spikeling_tick(uint32_t neuron_idx, float current_time_ms);
void spikeling_stdp_update(uint32_t pre_idx, uint32_t post_idx, float dt);

/* Advance one Resonator by one timestep with a shared drive signal
 * (e.g. an audio sample). Returns 1 the instant it crosses its
 * detection threshold (edge-triggered, like a neuron firing), else 0. */
int spikeling_resonator_step(uint32_t resonator_idx, float drive, float dt, float current_time_ms);

#endif /* SPIKELING_HW_H */
"""

C_SOURCE_TEMPLATE = """\
/* Auto-generated by Spikeling Compiler — DO NOT EDIT */
#include "spikeling_hw.h"
#include <math.h>

/* ── Neuron definitions ── */
Neuron neurons[NEURON_COUNT] = {{
{neuron_rows}
}};

/* ── Synapse definitions ── */
Synapse synapses[SYNAPSE_COUNT] = {{
{synapse_rows}
}};

/* ── Action dispatch table ── */
static const char* action_table[NEURON_COUNT] = {{
{action_rows}
}};

/* ── Resonator definitions ── */
Resonator resonators[RESONATOR_COUNT] = {{
{resonator_rows}
}};

/* ── Resonator action dispatch table ── */
static const char* resonator_action_table[RESONATOR_COUNT] = {{
{resonator_action_rows}
}};

/* ── Spike tick ── */
void spikeling_tick(uint32_t idx, float current_time_ms) {{
    Neuron* n = &neurons[idx];
    float elapsed = current_time_ms - n->last_spike_time;

    if (elapsed < REFRACTORY_MS) {{
        return;  /* Refractory lockout */
    }}

    /* Leak */
    n->membrane_potential -= n->leak;
    if (n->membrane_potential < 0) n->membrane_potential = 0;

    /* Threshold check */
    n->membrane_potential += n->threshold + 1;  /* simulated input drive */
    if (n->membrane_potential >= n->threshold) {{
        n->membrane_potential = 0;
        n->last_spike_time    = current_time_ms;
        n->fire_count++;

        /* Emit action */
        if (action_table[idx]) {{
            printf("[SPIKE] %s -> %s\\n", n->name, action_table[idx]);
        }}

        /* Propagate to downstream synapses */
        for (int s = 0; s < SYNAPSE_COUNT; s++) {{
            if (synapses[s].src_idx == idx) {{
                neurons[synapses[s].dst_idx].membrane_potential +=
                    synapses[s].weight * 50.0f;
            }}
        }}
    }}
}}

/* ── Resonator step ──
 * SAMPLE-RATE BUG, FOUND AND FIXED: the energy EMA's decay rate must be
 * derived from dt and energy_time_constant, not a fixed per-step
 * constant -- a fixed 0.01f implicitly assumes whatever dt it was tuned
 * at (here, 2.5e-5s / 40kHz). Run this at a different sample rate (e.g.
 * 1MHz, needed to accurately resolve a 40kHz signal -- see
 * a separate 40kHz resonator sample-rate test) without rescaling,
 * and the averaging window silently shrinks far below this resonator's
 * own settling time, which collapsed Python-side detection accuracy
 * from ~99% to ~45%/0%-recall before this was caught and fixed there
 * first (see runtime.ResonatorState.step's docstring for the full
 * story). Mirrored here for parity. */
int spikeling_resonator_step(uint32_t idx, float drive, float dt, float current_time_ms) {{
    Resonator* r = &resonators[idx];
    float omega = 2.0f * (float)M_PI * r->freq_hz;
    float accel = -(omega * omega) * r->x - 2.0f * r->damping * omega * r->v;
    accel += r->coupling * drive;
    r->v += accel * dt;
    r->x += r->v * dt;

    float alpha = dt / r->energy_time_constant;
    if (alpha > 1.0f) alpha = 1.0f;

    float was_above = sqrtf(r->energy_ema) >= r->threshold;
    /* Amplitude gating: skip the x*x multiply when |x| is below the
     * noise floor (see runtime.ResonatorState.step's docstring for the
     * full rationale + benchmarks/BENCHMARKS.md entry #6). Measured to
     * be a wash on a CPU (~1.0x, modern multiply units are already
     * pipelined/cheap) -- kept here only for behavioral parity with the
     * Python runtime, not as a performance claim for this backend. */
    if (fabsf(r->x) >= r->gate_threshold) {{
        r->energy_ema += alpha * (r->x * r->x - r->energy_ema);
    }} else {{
        r->energy_ema -= alpha * r->energy_ema;
    }}
    float now_above = sqrtf(r->energy_ema) >= r->threshold;

    if (now_above && !was_above) {{
        if (resonator_action_table[idx]) {{
            printf("[SPIKE] %s -> %s  (t=%.2fms)\\n", r->name, resonator_action_table[idx], current_time_ms);
        }}
        return 1;
    }}
    return 0;
}}

/* ── STDP weight update ── */
void spikeling_stdp_update(uint32_t pre_idx, uint32_t post_idx, float dt) {{
    float delta = LEARN_RATE * expf(-fabsf(dt) / 20.0f);
    for (int s = 0; s < SYNAPSE_COUNT; s++) {{
        if (synapses[s].src_idx == pre_idx && synapses[s].dst_idx == post_idx) {{
            synapses[s].weight += (dt > 0) ? delta : -delta * 0.5f;
            if (synapses[s].weight > 1.0f) synapses[s].weight = 1.0f;
            if (synapses[s].weight < 0.0f) synapses[s].weight = 0.0f;
        }}
    }}
}}
"""


class CCodeGenerator:
    """Emits a .h + .c file pair from a SpikelingAST."""

    # Mirrors runtime.DEFAULT_RESONATOR_BASE_GAIN -- duplicated rather than
    # imported because the C backend is meant to be usable standalone
    # (see Spikeling-Project/sdk-verilog), without a dependency on the
    # Python runtime package. Keep these two constants in sync by hand.
    DEFAULT_RESONATOR_BASE_GAIN = 4.0e-4
    DEFAULT_RESONATOR_THRESHOLD = 0.0008
    DEFAULT_RESONATOR_GATE_THRESHOLD = 0.00024
    DEFAULT_RESONATOR_ENERGY_TIME_CONSTANT = 0.0025  # seconds; see spikeling_resonator_step()

    def generate(self, ast: SpikelingAST, output_dir: str = "."):
        lif_neurons = [n for n in ast.neurons if n.neuron_type != "Resonator"]
        resonator_neurons = [n for n in ast.neurons if n.neuron_type == "Resonator"]
        neuron_index = {n.name: i for i, n in enumerate(lif_neurons)}

        # Header
        header = C_HEADER_TEMPLATE.format(
            neuron_count     = len(lif_neurons),
            synapse_count    = len(ast.connections),
            resonator_count  = max(1, len(resonator_neurons)),
            refractory_ms    = ast.refractory_ms,
            learn_rate       = ast.learn_rate,
        )

        # Neuron rows (LIF/Izhikevich/AdEx only -- Resonators have their own table)
        neuron_rows = []
        for n in lif_neurons:
            neuron_rows.append(
                f'    {{"{n.name}", {n.threshold}, {n.leak}, 0.0f, 0.0f, 0}}'
            )

        # Synapse rows (Resonators don't currently participate in synapses)
        synapse_rows = []
        for c in ast.connections:
            synapse_rows.append(
                f"    {{{neuron_index[c.src]}, {neuron_index[c.dst]}, {c.weight}f}}"
            )

        # Action rows (NULL if no action defined for that neuron)
        action_map = {a.neuron: a.command for a in ast.actions}
        action_rows = []
        for n in lif_neurons:
            cmd = action_map.get(n.name)
            action_rows.append(f'    "{cmd}"' if cmd else "    NULL")

        # Resonator rows -- auto-derive coupling from frequency if the
        # .spk file didn't specify one explicitly (same gain-normalization
        # as runtime.SpikelingRuntime.__init__).
        resonator_rows = []
        resonator_action_rows = []
        if resonator_neurons:
            for n in resonator_neurons:
                omega = 2 * math.pi * n.freq_hz
                coupling = n.coupling if n.coupling is not None else self.DEFAULT_RESONATOR_BASE_GAIN * (omega ** 2)
                resonator_rows.append(
                    f'    {{"{n.name}", {n.freq_hz}f, {n.damping}f, {coupling}f, '
                    f'{self.DEFAULT_RESONATOR_THRESHOLD}f, {self.DEFAULT_RESONATOR_GATE_THRESHOLD}f, '
                    f'{self.DEFAULT_RESONATOR_ENERGY_TIME_CONSTANT}f, 0.0f, 0.0f, 0.0f}}'
                )
                cmd = action_map.get(n.name)
                resonator_action_rows.append(f'    "{cmd}"' if cmd else "    NULL")
        else:
            # RESONATOR_COUNT is clamped to >= 1 for valid C array syntax
            # even with no Resonator neurons in this network -- this is an
            # inert placeholder entry that can never fire (threshold
            # unreachable) and has no action.
            resonator_rows.append('    {"_none_", 1.0f, 1.0f, 0.0f, 1e9f, 1e9f, 1.0f, 0.0f, 0.0f, 0.0f}')
            resonator_action_rows.append("    NULL")

        source = C_SOURCE_TEMPLATE.format(
            neuron_rows  = ",\n".join(neuron_rows) if neuron_rows else "    /* none */",
            synapse_rows = ",\n".join(synapse_rows) if synapse_rows else "    /* none */",
            action_rows  = ",\n".join(action_rows) if action_rows else "    /* none */",
            resonator_rows = ",\n".join(resonator_rows),
            resonator_action_rows = ",\n".join(resonator_action_rows),
        )

        h_path = os.path.join(output_dir, "spikeling_hw.h")
        c_path = os.path.join(output_dir, "spikeling_hw.c")

        with open(h_path, "w", encoding="utf-8") as f:
            f.write(header)
        with open(c_path, "w", encoding="utf-8") as f:
            f.write(source)

        print(f"[compiler] emitted {h_path}")
        print(f"[compiler] emitted {c_path}")
        return h_path, c_path


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def compile_file(spk_path: str, output_dir: str = ".") -> SpikelingAST:
    """Parse a .spk file and emit C output. Returns the AST for the Python runtime."""
    with open(spk_path, encoding="utf-8") as f:
        source = f.read()

    parser    = SpikelingParser()
    ast       = parser.parse(source)
    codegen   = CCodeGenerator()
    codegen.generate(ast, output_dir=output_dir)
    return ast


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "examples/sound_localizer.spk"
    compile_file(path, output_dir=".")
