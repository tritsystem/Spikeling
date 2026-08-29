#!/usr/bin/env python
"""
state_based_readout_test_v2.py — confound-controlled re-test. The v1
state-based readout test (state_based_readout_test.py) came back with
ALL THREE hypotheses disconfirmed, in the OPPOSITE direction predicted.
Before accepting that as a real result, checked for the same class of
confound already caught once elsewhere this session (the symmetry-
breaking test's v1 magnitude confound): measured the real RMS amplitude
of each anomaly type's "on" signal and found a 27x unmatched spread
(magnitude=0.49, burst=1.10, repetition=0.37, frequency=0.04) -- the
anomaly amplitudes were tuned independently per type for the ORIGINAL
raw-spike-count test, never matched for a fair cross-type comparison.

This file builds RMS-amplitude-normalized versions of all four anomaly
generators (same real temporal/spectral SHAPE as the originals, common
target RMS) and re-runs the same three state-based readouts against
them. The original generators in heterogeneous_ontology_test.py are
left untouched -- Part B's already-logged 100%-vs-50% headline result
stays valid; this is a separate, disclosed variant built for this one
follow-up question.

PRE-REGISTERED (same three hypotheses as v1, now confound-controlled):
- Resonator: mean sustained sqrt(energy_ema) higher for frequency than
  the other three.
- Izhikevich: ISI-CV-proxy higher for burst than for magnitude.
- AdEx: relative early-vs-late decline larger for repetition than burst.
DISCONFIRM, same as before: if the pattern still doesn't hold once
amplitude is genuinely controlled, report that as the real, final,
double-confirmed negative for this readout family -- not chased further.
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

TARGET_RMS = 0.5  # common target, close to magnitude's original real RMS (0.49)


def baseline_signal_v2(t, rng):
    return TARGET_RMS * math.sqrt(2) * math.sin(2 * math.pi * DISTRACTOR_FREQ * t) + rng.gauss(0, 0.02)


def magnitude_anomaly_v2(t, t_start, rng):
    if t_start <= t < t_start + 2.0:
        return TARGET_RMS + rng.gauss(0, 0.02)  # constant signal: RMS == value
    return baseline_signal_v2(t, rng)


def burst_anomaly_v2(t, t_start, rng):
    """4 pulses of 0.1s each within a 2s window (0.4s total 'on' time,
    matching v1's real timing) -- pulse height set so this segment's own
    RMS over the full 2s anomaly window (including silent gaps) equals
    TARGET_RMS: height * sqrt(0.4/2.0) = TARGET_RMS -> height = TARGET_RMS/sqrt(0.2)."""
    height = TARGET_RMS / math.sqrt(0.4 / 2.0)
    for k in range(4):
        pulse_start = t_start + k * 0.5
        if pulse_start <= t < pulse_start + 0.1:
            return height + rng.gauss(0, 0.05)
    return baseline_signal_v2(t, rng)


def repetition_anomaly_v2(t, t_start, rng):
    """8 pulses of 0.05s each (0.4s total 'on' time, same as burst's
    total 'on' time by construction -- only repeat COUNT differs, 8 vs
    4, holding total on-time and RMS equal), same RMS-matching formula."""
    height = TARGET_RMS / math.sqrt(0.4 / 2.0)
    for k in range(8):
        pulse_start = t_start + k * 0.3
        if pulse_start <= t < pulse_start + 0.05:
            return height + rng.gauss(0, 0.03)
    return baseline_signal_v2(t, rng)


def frequency_anomaly_v2(t, t_start, rng):
    if t_start <= t < t_start + 2.0:
        return TARGET_RMS * math.sqrt(2) * math.sin(2 * math.pi * TARGET_FREQ * t) + rng.gauss(0, 0.02)
    return baseline_signal_v2(t, rng)


ANOMALY_GENERATORS_V2 = {
    "magnitude": magnitude_anomaly_v2,
    "burst": burst_anomaly_v2,
    "repetition": repetition_anomaly_v2,
    "frequency": frequency_anomaly_v2,
}


def verify_rms_matched():
    rng = random.Random(0)
    print("Verification -- real RMS of each v2 anomaly's 'on' signal (target: all ~equal):")
    for name, gen in ANOMALY_GENERATORS_V2.items():
        vals = []
        t_start = 2.0
        t = 0.0
        while t < 4.0:
            v = gen(t, t_start, rng)
            if t >= t_start:
                vals.append(v)
            t += 0.001
        rms = (sum(v * v for v in vals) / len(vals)) ** 0.5
        print(f"  {name:<12} RMS={rms:.4f}")
    print()


def run_one_resonator(seed, anomaly_name, damping=0.05):
    gen = ANOMALY_GENERATORS_V2[anomaly_name]
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


def run_one_izhikevich(seed, anomaly_name, preset="regular_spiking"):
    gen = ANOMALY_GENERATORS_V2[anomaly_name]
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
    return max(isis) / min(isis), len(n.spike_log)


def run_one_adex(seed, anomaly_name, tau_w=600.0, b=60.0):
    gen = ANOMALY_GENERATORS_V2[anomaly_name]
    rng = random.Random(seed + 999)
    n = AdExNeuron(tau_w=tau_w, b=b)
    dt = DT_FOR_KIND["adex"]
    step_dt = native_dt("adex", dt)
    t_start, total_time = 2.0, 5.0
    mid = t_start + 1.5
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
    return (early_fires - late_fires) / total, total


if __name__ == "__main__":
    verify_rms_matched()

    N_TRIALS = 15
    ANOMALIES = ["magnitude", "burst", "repetition", "frequency"]

    print("=" * 90)
    print("Resonator: mean sustained energy, RMS-matched anomalies")
    print("=" * 90)
    for anomaly in ANOMALIES:
        vals = [run_one_resonator(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        print(f"  {anomaly:<12} mean sustained sqrt(energy_ema) = {sum(vals)/len(vals):.6f}")

    print("\n" + "=" * 90)
    print("Izhikevich: ISI-CV-proxy, RMS-matched anomalies")
    print("=" * 90)
    for anomaly in ANOMALIES:
        results = [run_one_izhikevich(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        mean_cv = sum(r[0] for r in results) / len(results)
        mean_count = sum(r[1] for r in results) / len(results)
        print(f"  {anomaly:<12} mean ISI-CV-proxy = {mean_cv:6.2f}   (mean spike count = {mean_count:.1f})")

    print("\n" + "=" * 90)
    print("AdEx: early-vs-late decline, RMS-matched anomalies")
    print("=" * 90)
    for anomaly in ANOMALIES:
        results = [run_one_adex(seed=s * 137, anomaly_name=anomaly) for s in range(N_TRIALS)]
        mean_decline = sum(r[0] for r in results) / len(results)
        mean_total = sum(r[1] for r in results) / len(results)
        print(f"  {anomaly:<12} mean decline = {mean_decline:+.3f}   (mean total spikes = {mean_total:.1f})")

    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
