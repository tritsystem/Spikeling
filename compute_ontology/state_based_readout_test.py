#!/usr/bin/env python
"""
state_based_readout_test.py — pushing further on the real, honest gap
left open in heterogeneous_ontology_test.py: Part 1's raw-spike-COUNT
readout couldn't see per-type specialization (everything fired on
everything once properly driven). This tests whether reading each
type's own documented INTERNAL STATE SIGNATURE -- the same thing
Tribe's real, shipped mechanics (BurstTrauma, BetrayalFatigue) actually
read, not spike count -- recovers real specialization.

PRE-REGISTERED (stated before running):
- Resonator: sustained mean sqrt(energy_ema) over the FULL anomaly
  window will be measurably higher for the frequency anomaly than for
  magnitude/burst/repetition. Mechanism: a single pulse's resonant
  response is transient and decays within a few energy_time_constants
  (~0.0025s x a few = ~10-25ms); a real periodic drive keeps rebuilding
  energy for the whole window. A peak-detection readout (Part 1) can't
  see this difference; an averaged-over-window readout should.
- Izhikevich: ISI coefficient of variation (max_isi/min_isi, same
  metric already validated in pyspike_neuron_models.py's own
  chattering-burst self-test) will be measurably higher for the burst
  anomaly (4 tight 0.1s pulses -> clustered spikes) than for the
  magnitude anomaly (continuous plateau -> roughly uniform ISI under
  constant current).
- AdEx: relative decline in per-pulse response from the FIRST half of
  the anomaly window to the SECOND half will be measurably larger for
  the repetition anomaly (8 discrete pulses) than for the burst anomaly
  (4 discrete pulses) -- more repeats, more real fatigue accumulation,
  per AdEx's own w-accumulates-per-spike mechanism.
DISCONFIRM: if these state-based metrics show no better discrimination
than Part 1's raw counts -- report that honestly too, not force a story.
"""

import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heterogeneous_ontology_test import (
    ResonatorNeuron, ANOMALY_GENERATORS, SCALE, DT_FOR_KIND, native_dt,
    RESONATOR_THRESHOLD, RESONATOR_GATE, RESONATOR_TAU, TARGET_FREQ,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspike_neuron_models import IzhikevichNeuron, AdExNeuron  # noqa: E402


def run_one_resonator(seed, anomaly_name, damping=0.05):
    gen = ANOMALY_GENERATORS[anomaly_name]
    rng = random.Random(seed + 999)
    r = ResonatorNeuron(freq_hz=TARGET_FREQ, damping=damping, coupling=SCALE["resonator"],
                         threshold=RESONATOR_THRESHOLD, gate_threshold=RESONATOR_GATE,
                         energy_time_constant=RESONATOR_TAU)
    dt = DT_FOR_KIND["resonator"]
    t_start, total_time = 2.0, 5.0
    energy_samples = []
    t = 0.0
    while t < total_time:
        raw = gen(t, t_start, rng)
        r.step(raw * SCALE["resonator"], dt, t)
        if t >= t_start:  # only sample during/after anomaly onset
            energy_samples.append(math.sqrt(r.energy_ema))
        t += dt
    return sum(energy_samples) / len(energy_samples)  # mean sustained energy


def run_one_izhikevich(seed, anomaly_name, preset="regular_spiking"):
    gen = ANOMALY_GENERATORS[anomaly_name]
    rng = random.Random(seed + 999)
    n = IzhikevichNeuron(preset=preset)
    dt = DT_FOR_KIND["izhikevich"]
    step_dt = native_dt("izhikevich", dt)
    t_start, total_time = 2.0, 5.0
    t = 0.0
    while t < total_time:
        raw = gen(t, t_start, rng)
        n.step(raw * SCALE["izhikevich"], step_dt, t)
        t += dt
    isis = [n.spike_log[i + 1] - n.spike_log[i] for i in range(len(n.spike_log) - 1)]
    if len(isis) < 2 or min(isis) <= 0:
        return 0.0, len(n.spike_log)
    return max(isis) / min(isis), len(n.spike_log)  # ISI CV proxy, spike count


def run_one_adex(seed, anomaly_name, tau_w=600.0, b=60.0):
    gen = ANOMALY_GENERATORS[anomaly_name]
    rng = random.Random(seed + 999)
    n = AdExNeuron(tau_w=tau_w, b=b)
    dt = DT_FOR_KIND["adex"]
    step_dt = native_dt("adex", dt)
    t_start, total_time = 2.0, 5.0
    mid = t_start + 1.5  # midpoint of the 3s anomaly window
    early_fires, late_fires = 0, 0
    t = 0.0
    while t < total_time:
        raw = gen(t, t_start, rng)
        fired = n.step(raw * SCALE["adex"], step_dt, t)
        if fired and t_start <= t < mid:
            early_fires += 1
        elif fired and t >= mid:
            late_fires += 1
        t += dt
    total = early_fires + late_fires
    if total == 0:
        return 0.0, 0
    decline = (early_fires - late_fires) / total  # +1 = all early, -1 = all late, 0 = flat
    return decline, total


if __name__ == "__main__":
    N_TRIALS = 15
    ANOMALIES = ["magnitude", "burst", "repetition", "frequency"]

    print("=" * 90)
    print("Resonator: mean SUSTAINED energy over full window (not peak/threshold-crossing)")
    print("=" * 90)
    for anomaly in ANOMALIES:
        vals = [run_one_resonator(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        mean_v = sum(vals) / len(vals)
        print(f"  {anomaly:<12} mean sustained sqrt(energy_ema) = {mean_v:.6f}  "
              f"(threshold for reference: {RESONATOR_THRESHOLD})")

    print("\n" + "=" * 90)
    print("Izhikevich: ISI coefficient-of-variation proxy (max_isi/min_isi)")
    print("=" * 90)
    for anomaly in ANOMALIES:
        results = [run_one_izhikevich(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        cvs = [r[0] for r in results]
        counts = [r[1] for r in results]
        mean_cv = sum(cvs) / len(cvs)
        mean_count = sum(counts) / len(counts)
        print(f"  {anomaly:<12} mean ISI-CV-proxy = {mean_cv:6.2f}   (mean spike count = {mean_count:.1f})")

    print("\n" + "=" * 90)
    print("AdEx: relative decline in firing rate, first half of window vs second half")
    print("=" * 90)
    for anomaly in ANOMALIES:
        results = [run_one_adex(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        declines = [r[0] for r in results]
        totals = [r[1] for r in results]
        mean_decline = sum(declines) / len(declines)
        mean_total = sum(totals) / len(totals)
        print(f"  {anomaly:<12} mean decline = {mean_decline:+.3f}   (mean total spikes = {mean_total:.1f})")

    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
