"""
spikeling/runtime/neuron_models.py
==================================
First-class neuron models for the `.spk` runtime and code generators.

Before this, `type=Izhikevich` / `type=AdEx` parsed but the runtime ran LIF
dynamics on them regardless -- anyone specifying those types silently got a
relabeled LIF (see `pyspike_neuron_models.py`, which noted the gap and gave a
verified *standalone* implementation but did not wire it into the DSL). This
module wires them in, using that standalone code as the behavioural reference.

A model is a small object with:
  * ``reset()``            -> a fresh state dict
  * ``step(state, drive)`` -> ``(new_state, fired: bool)``   (advances dt_ms)
  * ``c_fields()``         -> C struct field declarations for this model's state
  * ``c_step(var, drive)`` -> C statements advancing ``var`` by one tick,
                              setting ``int fired``

``state["v"]`` is always the readable membrane potential (used by the UI /
STDP timing). LIF stays inline in ``runtime.py`` (the default, zero-risk
path); Resonator stays in its own ``ResonatorState``. This module owns
Izhikevich and AdEx.

Reference dynamics (identical to ``pyspike_neuron_models.py``):

  Izhikevich (2003):  v' = 0.04 v^2 + 5 v + 140 - u + I ;  u' = a (b v - u)
                      v >= 30  ->  v <- c ,  u <- u + d
  AdEx (Brette & Gerstner 2005):
      v' = (-gL (v-EL) + gL DeltaT exp((v-VT)/DeltaT) - w + I) / C
      w' = (a (v-EL) - w) / tau_w
      v >= 0  ->  v <- Vreset ,  w <- w + b
"""
from __future__ import annotations

import math


class IzhikevichModel:
    """Two-variable spiking model. `preset` selects a named cortical firing
    regime from Izhikevich (2003) Table 1 (regular_spiking / chattering /
    fast_spiking / intrinsically_bursting). `gain` maps a `.spk` `drive`
    unit onto the model's own current scale (a constant drive ~50 * gain
    lands in a firing regime). `dt_ms` is the tick length; the model
    sub-steps at 0.5 ms internally for numerical stability."""

    PRESETS = {
        "regular_spiking":        dict(a=0.02, b=0.2, c=-65.0, d=8.0),
        "chattering":             dict(a=0.02, b=0.2, c=-50.0, d=2.0),
        "fast_spiking":           dict(a=0.10, b=0.2, c=-65.0, d=2.0),
        "intrinsically_bursting": dict(a=0.02, b=0.2, c=-55.0, d=4.0),
    }

    def __init__(self, preset: str = "regular_spiking", gain: float = 0.2,
                 dt_ms: float = 1.0, v0: float = -65.0):
        if preset not in self.PRESETS:
            raise ValueError(f"Izhikevich preset '{preset}' unknown; "
                             f"choose from {sorted(self.PRESETS)}")
        p = self.PRESETS[preset]
        self.a, self.b, self.c, self.d = p["a"], p["b"], p["c"], p["d"]
        self.gain, self.dt_ms, self.v0 = float(gain), float(dt_ms), float(v0)

    def reset(self) -> dict:
        return {"v": self.v0, "u": self.b * self.v0}

    def step(self, state: dict, drive: float) -> tuple[dict, bool]:
        I = drive * self.gain
        v, u = state["v"], state["u"]
        substeps = max(1, int(self.dt_ms / 0.5))
        sub_dt = self.dt_ms / substeps
        fired = False
        for _ in range(substeps):
            # `v ** 2` (not `v * v`) to stay bit-identical to the reference
            # IzhikevichNeuron.step in pyspike_neuron_models.py -- the
            # conformance harness (Bar 2) asserts an exact spike-time match.
            dv = 0.04 * v ** 2 + 5.0 * v + 140.0 - u + I
            du = self.a * (self.b * v - u)
            v += dv * sub_dt
            u += du * sub_dt
            if v >= 30.0:
                v = self.c
                u += self.d
                fired = True
        state["v"], state["u"] = v, u
        return state, fired

    # -- C backend --------------------------------------------------------
    def c_fields(self) -> list[str]:
        return ["double v;", "double u;"]

    def c_init(self, var: str) -> list[str]:
        return [f"{var}.v = {self.v0};", f"{var}.u = {self.b} * {self.v0};"]

    def c_step(self, var: str, drive_expr: str) -> list[str]:
        subs = max(1, int(self.dt_ms / 0.5))
        return [
            f"{{ double I = ({drive_expr}) * {self.gain};",
            f"  double sub_dt = {self.dt_ms} / {subs}.0;",
            f"  for (int _s = 0; _s < {subs}; ++_s) {{",
            f"    double dv = 0.04*{var}.v*{var}.v + 5.0*{var}.v + 140.0 - {var}.u + I;",
            f"    double du = {self.a}*({self.b}*{var}.v - {var}.u);",
            f"    {var}.v += dv * sub_dt;  {var}.u += du * sub_dt;",
            f"    if ({var}.v >= 30.0) {{ {var}.v = {self.c}; {var}.u += {self.d}; fired = 1; }}",
            f"  }} }}",
        ]


class AdExModel:
    """Adaptive exponential integrate-and-fire. The adaptation variable `w`
    accumulates on each spike and suppresses later firing -- spike-frequency
    adaptation, which LIF cannot produce. Defaults are the Brette & Gerstner
    (2005) regular-spiking set; `gain` maps `.spk` drive onto the pA current
    scale, `b` is the per-spike adaptation increment."""

    def __init__(self, gain: float = 8.0, dt_ms: float = 1.0, b: float = 60.0,
                 C=200.0, gL=10.0, EL=-70.0, VT=-50.0, DeltaT=2.0,
                 tau_w=30.0, a=2.0, Vreset=-58.0, v_peak=0.0):
        self.gain, self.dt_ms, self.b = float(gain), float(dt_ms), float(b)
        self.C, self.gL, self.EL, self.VT, self.DeltaT = C, gL, EL, VT, DeltaT
        self.tau_w, self.a, self.Vreset, self.v_peak = tau_w, a, Vreset, v_peak

    def reset(self) -> dict:
        return {"v": self.EL, "w": 0.0}

    def step(self, state: dict, drive: float) -> tuple[dict, bool]:
        I = drive * self.gain
        v, w = state["v"], state["w"]
        substeps = max(1, int(self.dt_ms / 0.1))
        sub_dt = self.dt_ms / substeps
        fired = False
        for _ in range(substeps):
            exp_term = self.DeltaT * math.exp(min(50.0, (v - self.VT) / self.DeltaT))
            dv = (-self.gL * (v - self.EL) + self.gL * exp_term - w + I) / self.C
            dw = (self.a * (v - self.EL) - w) / self.tau_w
            v += dv * sub_dt
            w += dw * sub_dt
            if v >= self.v_peak:
                v = self.Vreset
                w += self.b
                fired = True
        state["v"], state["w"] = v, w
        return state, fired

    # -- C backend --------------------------------------------------------
    def c_fields(self) -> list[str]:
        return ["double v;", "double w;"]

    def c_init(self, var: str) -> list[str]:
        return [f"{var}.v = {self.EL};", f"{var}.w = 0.0;"]

    def c_step(self, var: str, drive_expr: str) -> list[str]:
        subs = max(1, int(self.dt_ms / 0.1))
        return [
            f"{{ double I = ({drive_expr}) * {self.gain};",
            f"  double sub_dt = {self.dt_ms} / {subs}.0;",
            f"  for (int _s = 0; _s < {subs}; ++_s) {{",
            f"    double et = {self.DeltaT} * exp(fmin(50.0, ({var}.v - {self.VT})/{self.DeltaT}));",
            f"    double dv = (-{self.gL}*({var}.v - {self.EL}) + {self.gL}*et - {var}.w + I) / {self.C};",
            f"    double dw = ({self.a}*({var}.v - {self.EL}) - {var}.w) / {self.tau_w};",
            f"    {var}.v += dv*sub_dt;  {var}.w += dw*sub_dt;",
            f"    if ({var}.v >= {self.v_peak}) {{ {var}.v = {self.Vreset}; {var}.w += {self.b}; fired = 1; }}",
            f"  }} }}",
        ]


MODELS = {"Izhikevich": IzhikevichModel, "AdEx": AdExModel}

# numeric params each model constructor accepts off the `.spk` line
_NUMERIC_PARAMS = {
    "IzhikevichModel": {"gain", "b", "v0"},
    "AdExModel": {"gain", "b"},
}


def build_model(neuron_type: str, params: dict, dt_ms: float = 1.0):
    """`params`: the trailing `key=value` tokens parsed off the `.spk` neuron
    line (already without `type=`). `dt_ms`: the runtime tick length -- baked
    into the model so the Python runtime and the generated C use one dt."""
    cls = MODELS[neuron_type]
    allowed = _NUMERIC_PARAMS[cls.__name__]
    kw: dict = {"dt_ms": float(dt_ms)}
    for k, v in (params or {}).items():
        if k in ("dt", "dt_ms"):
            kw["dt_ms"] = float(v)
        elif k == "preset":
            kw["preset"] = v
        elif k in allowed:
            kw[k] = float(v)
    return cls(**kw)


# ---------------------------------------------------------------------------
#  Standalone C emitter -- one model neuron under constant drive, prints spike
#  times. Used by the cross-backend conformance harness to prove the generated
#  C matches the Python runtime spike-for-spike. Deliberately NOT part of the
#  spikeling_hw.c/.h hardware path (that stays LIF-only, zero regression).
# ---------------------------------------------------------------------------
def emit_model_sim_c(model, name: str, drive: float, total_ms: float) -> str:
    """`model` is an IzhikevichModel or AdExModel instance."""
    fields = "\n    ".join(model.c_fields())
    init = "\n    ".join(model.c_init("s"))
    step = "\n        ".join(model.c_step("s", "DRIVE"))
    steps = int(round(total_ms / model.dt_ms))
    return f"""\
/* Auto-generated by Spikeling CCodeGenerator.generate_model_sim() -- DO NOT EDIT.
 * Standalone parity harness for neuron '{name}' ({type(model).__name__}).
 * Prints one spike time (ms) per line. */
#include <stdio.h>
#include <math.h>

#define DRIVE   ({drive})
#define DT_MS   ({model.dt_ms})
#define NSTEPS  {steps}

typedef struct {{
    {fields}
}} State;

int main(void) {{
    State s;
    {init}
    for (int t = 0; t < NSTEPS; ++t) {{
        int fired = 0;
        {step}
        /* same convention as the Python runtime: the spike is stamped with
         * the time of the stimulate() call that produced it (step index t),
         * not t+1 -- keeps C and Python spike times on the same phase. */
        if (fired) printf("%.6f\\n", t * DT_MS);
    }}
    return 0;
}}
"""
