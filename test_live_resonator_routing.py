#!/usr/bin/env python
"""
test_live_resonator_routing.py — regression test for a real bug found and
fixed 2026-09-06: `Net.neuron()`'s live-mode branch (used after
`build_live()`) always created a plain `NeuronState` in `self._live_rt.neurons`
regardless of `type`, so a `type="Resonator"` neuron added AFTER
`build_live()` never landed in `self._live_rt.resonators` — and
`step_resonators()` only ever iterates `.resonators.values()`, so such a
neuron could never fire no matter what it was driven with. Not a tuning
problem: the neuron was never actually a Resonator at runtime.

Found while building a heterogeneous-neuron-dynamics demo in the sibling
`spikegate` project (a Resonator constructed after `build_live()` produced
zero detections regardless of input). The batch `build()` path
(`SpikelingRuntime.__init__`) already routed by `neuron_type` correctly —
live mode had silently diverged from it.

Three checks, each falsifiable:
  A. ROUTING    -- a live-constructed Resonator lands in .resonators, not
                    .neurons (this is what was actually broken).
  B. DETECTION  -- a live-constructed Resonator, driven with a real
                    on-frequency sine wave (its own coupling auto-derived,
                    not hand-tuned), actually fires via the normal
                    action/handler dispatch path.
  C. SELECTIVITY -- the same construction, driven with an off-frequency
                    tone of equal amplitude, does NOT fire — proving it's
                    genuinely frequency-selective in live mode too, not
                    just amplitude-triggered.

    python test_live_resonator_routing.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pyspike import Net  # noqa: E402

SR = 40_000
DT = 1.0 / SR
N_SAMPLES = 2000
TARGET_HZ = 440.0
DISTRACTOR_HZ = 2000.0


def _build_and_drive(drive_hz: float) -> dict:
    net = Net(refractory_ms=5.0)
    rt = net.build_live()  # empty live runtime FIRST -- neuron() must come after
    r = net.neuron("R", type="Resonator", freq_hz=TARGET_HZ, damping=0.02)
    routed_as_resonator = "R" in rt.resonators
    routed_as_plain_neuron = "R" in rt.neurons

    fired = []
    net.action(r)(lambda: fired.append(True))
    for i in range(N_SAMPLES):
        t = i * DT
        rt.step_resonators(drive=math.sin(2 * math.pi * drive_hz * t), dt=DT, current_time_ms=t)

    return {
        "routed_as_resonator": routed_as_resonator,
        "routed_as_plain_neuron": routed_as_plain_neuron,
        "fire_count": len(fired),
        "final_rms": math.sqrt(rt.resonators["R"].energy_ema) if routed_as_resonator else None,
    }


def run() -> None:
    print("=" * 78)
    print("  LIVE-MODE RESONATOR ROUTING -- regression test")
    print("=" * 78)

    print("\n  A. ROUTING (on-target drive, just checking placement)")
    on_target = _build_and_drive(TARGET_HZ)
    ok_a = on_target["routed_as_resonator"] and not on_target["routed_as_plain_neuron"]
    print(f"     'R' in rt.resonators: {on_target['routed_as_resonator']} (expected True)")
    print(f"     'R' in rt.neurons:    {on_target['routed_as_plain_neuron']} (expected False)")
    print(f"     [{'PASS' if ok_a else 'FAIL'}] live-constructed Resonator routed to the right dict")

    print(f"\n  B. DETECTION (driven at its own tuned frequency, {TARGET_HZ}Hz)")
    ok_b = on_target["fire_count"] > 0
    print(f"     fire count: {on_target['fire_count']} (expected > 0)")
    print(f"     final RMS energy: {on_target['final_rms']:.6f}")
    print(f"     [{'PASS' if ok_b else 'FAIL'}] on-frequency drive actually fires the live neuron")

    print(f"\n  C. SELECTIVITY (driven at a distractor frequency, {DISTRACTOR_HZ}Hz, same amplitude)")
    off_target = _build_and_drive(DISTRACTOR_HZ)
    ok_c = off_target["fire_count"] == 0
    print(f"     fire count: {off_target['fire_count']} (expected 0)")
    print(f"     final RMS energy: {off_target['final_rms']:.6f}")
    print(f"     [{'PASS' if ok_c else 'FAIL'}] off-frequency drive of equal amplitude does NOT fire it")

    print()
    all_ok = ok_a and ok_b and ok_c
    print(f"  OVERALL: {'PASS' if all_ok else 'FAIL'} -- live-mode Resonator construction is "
          f"correctly routed, fires on-frequency, and stays frequency-selective.")
    if not all_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    run()
