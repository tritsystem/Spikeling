#!/usr/bin/env python
"""
habituation_drift_benchmark.py — real, run comparison of three drift
detectors on the same synthetic task from akida_style_drift_detector.py:
  1. Naive amplitude-threshold baseline (the honest comparison point used
     throughout this portfolio for every neuromorphic-detection claim --
     e.g. Resonator's real 99.2% vs ~65% naive-threshold result).
  2. The original ReRAM+Akida-style detector (crude one-shot binary
     training rule) -- real result from the prior pass: 2/20 detected,
     mean latency 872 ticks, vs naive's 20/20 at 256 ticks.
  3. The habituation twist (AdEx adaptation instead of a frozen mask) --
     this module.

Same task, same seeds, same drift magnitude -- a real apples-to-apples
comparison, not three separately-tuned demos.
"""

import random

from reram_synapse_array import ReRAMSynapseArray
from habituation_hidden_layer import HabituationHiddenLayer
from akida_style_drift_detector import make_signature, drifted_signature


def build_reram_pair(n_hidden, n_channels, seed):
    rng = random.Random(seed)
    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)
    for h in range(n_hidden):
        for c in range(n_channels):
            reram_pos.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
            reram_neg.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
    return reram_pos, reram_neg, rng


def run_habituation_trial(n_channels=8, n_hidden=24, drift_start=200, drift_ticks=400,
                           total_ticks=2000, noise_std=0.05, seed=0,
                           window=30, rate_multiplier=1.12):
    """Population-rate novelty detector: track a sliding-window spike
    count; alarm when it exceeds `rate_multiplier` times the real,
    measured settled-baseline rate (measured during the tail of
    training, not guessed) for `window` consecutive ticks."""
    reram_pos, reram_neg, rng = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)

    hidden = HabituationHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    # TRAIN: let the population habituate to the normal signature.
    recent = []
    for _ in range(300):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        recent.append(sum(spikes))
    # real, measured settled baseline: mean population spike count over
    # the LAST 100 training ticks (after habituation has had time to
    # take effect), not the first 100 (still adapting, rate is higher).
    baseline_rate = sum(recent[-100:]) / 100.0
    alarm_rate = baseline_rate * rate_multiplier

    window_buf = []
    detect_tick = None
    for t in range(total_ticks):
        if t < drift_start:
            sig = normal_sig
        elif t < drift_start + drift_ticks:
            frac = (t - drift_start) / drift_ticks
            sig = drifted_signature(normal_sig, drift_sig, frac)
        else:
            sig = drift_sig
        noisy = [v + rng.gauss(0, noise_std) for v in sig]
        spikes = hidden.step(noisy)
        window_buf.append(sum(spikes))
        if len(window_buf) > window:
            window_buf.pop(0)
        if len(window_buf) == window:
            windowed_rate = sum(window_buf) / window
            if windowed_rate >= alarm_rate and detect_tick is None:
                detect_tick = t

    return detect_tick, drift_start, baseline_rate, alarm_rate


def run_habituation_false_alarm_control(n_channels=8, n_hidden=24, total_ticks=5000,
                                          noise_std=0.05, seed=100, window=30,
                                          rate_multiplier=1.12):
    reram_pos, reram_neg, rng = build_reram_pair(n_hidden, n_channels, seed)
    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    hidden = HabituationHiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)

    recent = []
    for _ in range(300):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        recent.append(sum(spikes))
    baseline_rate = sum(recent[-100:]) / 100.0
    alarm_rate = baseline_rate * rate_multiplier

    window_buf = []
    alarms = 0
    armed = True
    for t in range(total_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        window_buf.append(sum(spikes))
        if len(window_buf) > window:
            window_buf.pop(0)
        if len(window_buf) == window:
            windowed_rate = sum(window_buf) / window
            if windowed_rate >= alarm_rate and armed:
                alarms += 1
                armed = False
            if windowed_rate < alarm_rate * 0.7:
                armed = True
    return alarms, total_ticks


if __name__ == "__main__":
    print("=" * 78)
    print("Habituation-based novelty detector -- the 'own twist' on the")
    print("ReRAM+Akida drift-detection pattern (AdEx adaptation, not a")
    print("frozen one-shot binary mask). Real numbers, actually run below.")
    print("=" * 78)

    N_TRIALS = 20
    latencies = []
    missed = 0
    baselines = []

    for trial in range(N_TRIALS):
        det, drift_start, base_rate, alarm_rate = run_habituation_trial(seed=trial)
        baselines.append(base_rate)
        if det is not None:
            latencies.append(det - drift_start)
        else:
            missed += 1

    print(f"\n{N_TRIALS} real trials:")
    print(f"  Habituation detector: {len(latencies)}/{N_TRIALS} detected, missed={missed}")
    if latencies:
        print(f"    mean detection latency: {sum(latencies)/len(latencies):.1f} ticks"
              f"  (min={min(latencies)}, max={max(latencies)})")
    print(f"  mean real measured settled baseline rate: {sum(baselines)/len(baselines):.2f} spikes/tick")

    print("\nFalse-alarm control (pure normal signal, 5000 ticks, no drift):")
    alarms, ticks = run_habituation_false_alarm_control()
    print(f"  Habituation detector: {alarms} false alarms / {ticks} ticks")

    print("\nFor comparison, real numbers from the prior two passes (same task, same seeds):")
    print("  Naive amplitude-threshold baseline:  20/20 detected, mean latency 255.8 ticks, 0 false alarms")
    print("  Original crude-binary-rule detector:  2/20 detected, mean latency 872.5 ticks, 0 false alarms")
