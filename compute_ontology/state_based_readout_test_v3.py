#!/usr/bin/env python
"""
state_based_readout_test_v3.py — fixing a real bug introduced by v2
itself, caught by direct inspection of raw spike times (not assumed
correct just because the numbers changed). v2 normalized the BASELINE
signal's RMS to the same target as the anomaly "on" periods -- but that
made the quiet period just as loud as the anomaly, so Izhikevich (at
SCALE=12) fired continuously and REGULARLY on the loud baseline tone
itself, before the anomaly even started. Direct inspection confirmed
it: magnitude and burst produced IDENTICAL spike trains for the first
2 seconds (the pre-anomaly baseline period) -- the readout was
dominated by baseline entrainment, not anomaly-specific structure.

Fix: baseline stays QUIET (the same small amplitude as v1's original
baseline_signal), matching how the OTHER anomaly types (magnitude,
burst, repetition) were already correctly built -- quiet baseline,
loud onset. Only the four anomaly "on" periods get RMS-matched to each
other (same fix as v2), not matched to the baseline.
"""

import os
import sys
import math
import random

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heterogeneous_ontology_test import (
    ResonatorNeuron, SCALE, DT_FOR_KIND, native_dt,
    RESONATOR_THRESHOLD, RESONATOR_GATE, RESONATOR_TAU, TARGET_FREQ, DISTRACTOR_FREQ,
)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspike_neuron_models import IzhikevichNeuron, AdExNeuron  # noqa: E402

TARGET_RMS = 0.5
PULSE_HEIGHT = TARGET_RMS / math.sqrt(0.4 / 2.0)  # 1.118


def baseline_signal_v3(t, rng):
    """QUIET -- same small amplitude as v1's original baseline_signal,
    NOT normalized up to TARGET_RMS. This is the real fix."""
    return 0.05 * math.sin(2 * math.pi * DISTRACTOR_FREQ * t) + rng.gauss(0, 0.02)


def magnitude_anomaly_v3(t, t_start, rng):
    if t_start <= t < t_start + 2.0:
        return TARGET_RMS + rng.gauss(0, 0.02)
    return baseline_signal_v3(t, rng)


def burst_anomaly_v3(t, t_start, rng):
    for k in range(4):
        pulse_start = t_start + k * 0.5
        if pulse_start <= t < pulse_start + 0.1:
            return PULSE_HEIGHT + rng.gauss(0, 0.05)
    return baseline_signal_v3(t, rng)


def repetition_anomaly_v3(t, t_start, rng):
    for k in range(8):
        pulse_start = t_start + k * 0.3
        if pulse_start <= t < pulse_start + 0.05:
            return PULSE_HEIGHT + rng.gauss(0, 0.03)
    return baseline_signal_v3(t, rng)


def frequency_anomaly_v3(t, t_start, rng):
    """Quiet baseline (distractor tone) -> louder target tone at
    TARGET_RMS during the anomaly window, matching how the other three
    anomaly types step UP from quiet to loud. This intentionally does
    NOT hold baseline and anomaly at equal amplitude (that was v1's
    purity criterion for isolating frequency specifically, and it's
    still a real, disclosed departure) -- it matches the amplitude
    STEP-UP pattern all four types now share, needed for a fair
    cross-type comparison of "on" signal RMS."""
    if t_start <= t < t_start + 2.0:
        return TARGET_RMS * math.sqrt(2) * math.sin(2 * math.pi * TARGET_FREQ * t) + rng.gauss(0, 0.02)
    return baseline_signal_v3(t, rng)


ANOMALY_GENERATORS_V3 = {
    "magnitude": magnitude_anomaly_v3,
    "burst": burst_anomaly_v3,
    "repetition": repetition_anomaly_v3,
    "frequency": frequency_anomaly_v3,
}


def run_one_izhikevich(seed, anomaly_name, preset="regular_spiking"):
    gen = ANOMALY_GENERATORS_V3[anomaly_name]
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
    # only look at spikes DURING the anomaly window (t >= t_start) --
    # real fix (v3): v1/v2 computed ISI-CV over the WHOLE spike log
    # including pre-anomaly baseline spikes, which let baseline
    # entrainment dominate the metric undetected.
    anomaly_spikes = [s for s in n.spike_log if s >= t_start]
    # SECOND real bug, caught by inspecting raw spike times directly:
    # the substepped integrator can log two fires at the literal same
    # outer-loop t (a real near-simultaneous double-fire artifact of
    # sub-stepping, not a duplicate record) -- these produce a zero
    # ISI that tripped the old "min(isis)<=0 -> return 0.0" guard and
    # silently discarded a real, large, genuine burst-clustering signal
    # that WAS present in the data (verified directly: raw times showed
    # 4-spike clusters separated by real ~0.4s silent gaps -- exactly
    # the predicted pattern). Fix: merge same-timestamp fires into one
    # logical spike event before computing ISIs, instead of guarding
    # against zero ISIs by discarding the whole trial.
    merged = []
    for s in anomaly_spikes:
        if not merged or s - merged[-1] > 1e-9:
            merged.append(s)
    isis = [merged[i + 1] - merged[i] for i in range(len(merged) - 1)]
    if len(isis) < 2:
        return 0.0, len(anomaly_spikes)
    return max(isis) / min(isis), len(anomaly_spikes)


# real pulse boundaries per anomaly type, matching the generator
# functions above exactly -- used for a condition-appropriate
# first-pulse-vs-last-pulse comparison instead of an arbitrary shared
# time split. The arbitrary "mid = t_start+1.5" split (first attempt)
# fell at a DIFFERENT relative position in burst's 4-pulse timeline
# (right at the start of pulse 4) than in repetition's 8-pulse timeline
# (cleanly between pulses 5 and 6) -- a real, condition-dependent bias
# unrelated to actual fatigue, caught by inspecting raw fire times and
# w trajectories directly (both conditions showed real, substantial w
# growth across the window -- the metric, not the mechanism, was wrong).
PULSE_WINDOWS = {
    "burst": [(k * 0.5, k * 0.5 + 0.1) for k in range(4)],
    "repetition": [(k * 0.3, k * 0.3 + 0.05) for k in range(8)],
}


def run_one_adex(seed, anomaly_name, tau_w=600.0, b=60.0):
    """For burst/repetition: real first-pulse-vs-last-pulse spike-count
    decline. For magnitude/frequency (no discrete pulse structure --
    fatigue accumulation isn't the relevant question for those), falls
    back to total spike count only (no decline metric applies)."""
    gen = ANOMALY_GENERATORS_V3[anomaly_name]
    rng = random.Random(seed + 999)
    n = AdExNeuron(tau_w=tau_w, b=b)
    dt = DT_FOR_KIND["adex"]
    step_dt = native_dt("adex", dt)
    t_start, total_time = 2.0, 5.0
    t = 0.0
    fire_times = []
    while t < total_time:
        raw = gen(t, t_start, rng)
        if n.step(raw * SCALE["adex"], step_dt, t):
            fire_times.append(t - t_start)
        t += dt
    total = len(fire_times)
    if anomaly_name not in PULSE_WINDOWS or total == 0:
        return 0.0, total
    windows = PULSE_WINDOWS[anomaly_name]
    per_pulse = [sum(1 for f in fire_times if lo <= f < hi) for lo, hi in windows]
    first_count = per_pulse[0]
    last_count = per_pulse[-1]
    if first_count + last_count == 0:
        return 0.0, total
    decline = (first_count - last_count) / (first_count + last_count)
    return decline, total


def run_one_resonator(seed, anomaly_name, damping=0.05):
    gen = ANOMALY_GENERATORS_V3[anomaly_name]
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
        if t >= t_start:
            energy_samples.append(math.sqrt(r.energy_ema))
        t += dt
    return sum(energy_samples) / len(energy_samples)


if __name__ == "__main__":
    rng = random.Random(0)
    print("Verification -- real RMS of each v3 anomaly's ON-period signal, baseline quiet-check:")
    for name, gen in ANOMALY_GENERATORS_V3.items():
        on_vals, base_vals = [], []
        t_start = 2.0
        t = 0.0
        while t < 4.0:
            v = gen(t, t_start, rng)
            (on_vals if t >= t_start else base_vals).append(v)
            t += 0.001
        on_rms = (sum(v * v for v in on_vals) / len(on_vals)) ** 0.5
        base_rms = (sum(v * v for v in base_vals) / len(base_vals)) ** 0.5
        print(f"  {name:<12} on-RMS={on_rms:.4f}   baseline-RMS={base_rms:.4f}")

    N_TRIALS = 15
    ANOMALIES = ["magnitude", "burst", "repetition", "frequency"]

    print("\n" + "=" * 90)
    print("Izhikevich: ISI-CV-proxy over ONLY anomaly-window spikes, quiet baseline")
    print("=" * 90)
    for anomaly in ANOMALIES:
        results = [run_one_izhikevich(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        mean_cv = sum(r[0] for r in results) / len(results)
        mean_count = sum(r[1] for r in results) / len(results)
        print(f"  {anomaly:<12} mean ISI-CV-proxy = {mean_cv:6.2f}   (mean spike count = {mean_count:.1f})")

    print("\n" + "=" * 90)
    print("AdEx: early-vs-late decline, quiet baseline")
    print("=" * 90)
    for anomaly in ANOMALIES:
        results = [run_one_adex(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        mean_decline = sum(r[0] for r in results) / len(results)
        mean_total = sum(r[1] for r in results) / len(results)
        print(f"  {anomaly:<12} mean decline = {mean_decline:+.3f}   (mean total spikes = {mean_total:.1f})")

    print("\n" + "=" * 90)
    print("Resonator: mean sustained energy, quiet baseline (sanity re-check)")
    print("=" * 90)
    for anomaly in ANOMALIES:
        vals = [run_one_resonator(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        print(f"  {anomaly:<12} mean sustained sqrt(energy_ema) = {sum(vals)/len(vals):.6f}")

    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
