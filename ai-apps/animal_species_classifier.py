#!/usr/bin/env python
"""
animal_species_classifier.py
==============================
Driver for animal_detector_gate.spk -- loads the config-driven network
(see animal_acoustic_config.py / animal_gate_spk_builder.py), drives the
Resonator (acoustic) channels with a signal, drives the LIF
(environmental-plausibility + region-prior) channels from real-valued
readings, and combines both into a per-class classification.

WHY THIS FILE EXISTS SEPARATELY FROM THE .spk NETWORK: Resonator neurons
in this runtime dispatch through the action/handler path but do NOT
participate in `connect`-based synapse propagation (real constraint, see
animal_gate_spk_builder.py's module docstring for the code-level reason).
So acoustic detections and environmental gating are two populations of the
SAME SpikelingRuntime/AST, composed here via registered handlers -- not
two separate networks, just cross-population logic that has to live in
Python because the DSL doesn't support wiring it directly (yet).

HONESTY NOTE: classify()'s output is a set of (class, passed_gates: bool)
pairs from RAW RESONATOR ENERGY THRESHOLD CROSSINGS -- there is no
trained classifier here, no confusion-matrix accuracy claim, nothing
validated against a real recording of any animal. See
test_animal_gate_synthetic.py for what was actually tested (synthetic
tones/chirps only) and its actual measured results.
"""
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from compiler.compiler import SpikelingParser  # noqa: E402
from runtime.runtime import SpikelingRuntime    # noqa: E402

from animal_acoustic_config import CLASSES, ENV_GATE_RULES, REGION_PRIOR_STUB  # noqa: E402

SPK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "animal_detector_gate.spk")

# LIF gate/region neurons are threshold=40 leak=5 (see the builder) --
# "condition true" is encoded as a sustained rate-coded drive comfortably
# above threshold on the very next stimulate() call (a single pulse, since
# threshold=40 <= one drive of 50 as used throughout this project, e.g.
# runtime.py's own _run_benchmark_tick default drive=50.0); "condition
# false" is simply never stimulated, so the neuron sits at 0 and never
# fires. This matches the existing rate/spike-coding convention in this
# runtime rather than inventing a new one.
GATE_DRIVE_TRUE = 50.0


def _band_neuron_name(cls, band) -> str:
    return cls.label if len(cls.bands) == 1 else f"{cls.label}_{band.band_label}"


def _evaluate_env_rule(rule, month: int, temperature_c: float, humidity_pct: float) -> bool:
    """Generic dispatch over EnvGateRule.kind -- NOT bat-specific. A future
    rule kind (e.g. a bird class's "dawn_chorus_hours") would add one more
    branch here, not a parallel bat-shaped code path."""
    if rule.kind == "active_months":
        return month in rule.params["active_months"]
    if rule.kind == "min_temperature_c":
        return temperature_c >= rule.params["min_temperature_c"]
    if rule.kind == "max_humidity_pct":
        return humidity_pct <= rule.params["max_humidity_pct"]
    raise ValueError(f"Unknown EnvGateRule kind: {rule.kind!r}")


@dataclass
class AnimalDetectorGate:
    rt: SpikelingRuntime = field(init=False)
    _band_fired: dict = field(init=False, default_factory=dict)   # class.label -> set of band_labels fired since last poll()
    _gate_state: dict = field(init=False, default_factory=dict)   # EnvGateRule.label -> bool
    _t_ms: float = field(init=False, default=0.0)

    def __post_init__(self):
        with open(SPK_PATH) as f:
            ast = SpikelingParser().parse(f.read())
        self.rt = SpikelingRuntime(ast)
        self._gate_state = {rule.label: False for rule in ENV_GATE_RULES}
        self._register_handlers()

    def _register_handlers(self):
        for c in CLASSES:
            for band in c.bands:
                name = _band_neuron_name(c, band)
                cmd = f"DETECT_{name.upper()}"

                def handler(class_label=c.label, band_label=band.band_label):
                    self._band_fired.setdefault(class_label, set()).add(band_label)

                self.rt.register_handler(cmd, handler)

        for rule in ENV_GATE_RULES:
            cmd = f"ENV_GATE_{rule.label.upper()}"

            def handler(rule_label=rule.label):
                self._gate_state[rule_label] = True

            self.rt.register_handler(cmd, handler)

        # RegionPrior_* action handlers exist (see the generated .spk) but
        # currently have no effect on classify() beyond confirming the
        # channel fires when driven -- REGION_PRIOR_STUB is neutral (1.0)
        # for every class, see that dict's docstring in the config module.

    # ── Acoustic drive ──────────────────────────────────────────────
    def step_acoustic(self, drive: float, dt: float):
        self._t_ms += dt * 1000.0
        self.rt.step_resonators(drive=drive, dt=dt, current_time_ms=self._t_ms)

    # ── Environmental drive ─────────────────────────────────────────
    def set_environment(self, month: int, temperature_c: float, humidity_pct: float,
                         lat: Optional[float] = None, lon: Optional[float] = None):
        """Evaluates every ENV_GATE_RULES entry against the given reading
        (generic dispatch, see _evaluate_env_rule) and stimulates the
        corresponding Gate<rule.label> LIF neuron when true. Also
        stimulates every RegionPrior_<class> neuron (geolocation IS a real
        wired input channel -- see the config's REGION_PRIOR_STUB
        docstring for why its current *effect* on classify() is neutral).
        lat/lon are accepted and recorded for a future real regional-prior
        lookup; not used for anything yet (no real regional dataset
        available -- see REGION_PRIOR_STUB)."""
        self._t_ms += 1.0
        self.lat, self.lon = lat, lon
        for rule in ENV_GATE_RULES:
            ok = _evaluate_env_rule(rule, month, temperature_c, humidity_pct)
            self._gate_state[rule.label] = ok
            if ok:
                self.rt.stimulate(f"Gate{rule.label}", self._t_ms, drive=GATE_DRIVE_TRUE)
        for c in CLASSES:
            self.rt.stimulate(f"RegionPrior_{c.label}", self._t_ms, drive=GATE_DRIVE_TRUE)

    # ── Classification ──────────────────────────────────────────────
    def poll(self) -> dict:
        """Returns {class.label: {"bands_fired": set(...), "genus_level": bool,
        "gates_passed": bool, "region_prior": float, "display_name": str}}
        for every class that has fired at least one band since the last
        poll(), then clears the fired-band record (edge-triggered, like a
        single detection event)."""
        out = {}
        for c in CLASSES:
            fired = self._band_fired.get(c.label)
            if not fired:
                continue
            applicable_rules = [r for r in ENV_GATE_RULES if c.label in r.applies_to]
            gates_passed = all(self._gate_state.get(r.label, False) for r in applicable_rules)
            out[c.label] = {
                "display_name": c.display_name,
                "bands_fired": set(fired),
                "genus_level": c.genus_level,
                "gates_passed": gates_passed,
                "region_prior": REGION_PRIOR_STUB.get(c.label, 1.0),
            }
        self._band_fired = {}
        return out


if __name__ == "__main__":
    print("This module is a driver library -- see test_animal_gate_synthetic.py "
          "for a runnable synthetic-tone test.")
