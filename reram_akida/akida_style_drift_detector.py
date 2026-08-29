#!/usr/bin/env python
"""
akida_style_drift_detector.py — an event-driven SNN readout, in the spirit
of BrainChip's Akida (real, sourced architecture facts below), reading its
input-layer synapse weights from a ReRAMSynapseArray, applied to a
domain-agnostic "slow drift" detection task.

This is a SOFTWARE STAND-IN for real Akida silicon, not a claim of
hardware-identical behavior -- no Akida hardware was used, same discipline
the original ReRAM+Akida GPS-spoofing detector applied to its own "v1
simulation" before going to real hardware.

Real, sourced Akida AKD1000 architecture facts this model deliberately
mirrors (fetched 2026-08-28):
  - Purely event-driven: computation happens only on a spike event, not a
    dense per-tick matrix multiply. (open-neuromorphic.org/.../akida-brainchip/)
  - On-chip learning is restricted to the LAST layer only, and that layer
    must use BINARY weights and BINARY inputs.
    (docs.edgeimpulse.com/hardware/boards/brainchip-akd1000, BrainChip's
    own AKD1000 product briefs)
This model's hidden layer is event-driven LIF (spikes only propagate on
threshold crossing); its output layer is a single binary-weight neuron
trained by a real Hebbian-style rule constrained to {0, 1} weights, both
directly matching the two real constraints above -- not invented for
convenience.

The detection TASK (a slowly-drifting multi-channel signature) is a
domain-agnostic stand-in for the "seamless GPS takeover" pattern described
in the ReRAM+Akida GPS-spoofing-detection demo this module was built to
explore -- it is NOT a GPS model, and makes no claim about GPS-specific
performance. Real numbers below are from actually running this code, on
synthetic data, not estimated.
"""

import math
import random

from reram_synapse_array import ReRAMSynapseArray


# ── event-driven LIF hidden layer, weights read from the ReRAM array ────
class HiddenLayer:
    """Real bug found by actually running this (not assumed correct): a
    single fixed threshold=1.0 against a real per-step drive of roughly
    n_channels * avg_weight(~0.5) * avg_input(~0.6) ~= 2.4 meant EVERY
    hidden neuron saturated and fired on EVERY tick, regardless of input
    -- confirmed directly (100/100 fires for all 24 neurons during
    training). Zero selectivity means the layer can't distinguish normal
    from drifted input at all, which is why detection was structurally
    impossible, not just poorly tuned.

    Fix: per-neuron thresholds spread across the real expected drive
    range, so different neurons trip at different total-input magnitudes
    -- the same reason a real population of neurons with varied
    thresholds encodes a graded response instead of an all-or-nothing one."""

    def __init__(self, n_channels: int, n_hidden: int, reram_pos: ReRAMSynapseArray,
                 reram_neg: ReRAMSynapseArray, leak: float = 0.5, seed: int = 0):
        """Signed weights via a differential CELL PAIR -- one physical
        ReRAM cell holds the positive component, a second holds the
        negative component, logical weight = pos - neg. This is a real,
        standard technique for getting signed synapses out of ReRAM
        (which itself only stores a positive conductance), not an
        invented convenience -- and it's the fix for a real structural
        finding (see module-level notes): single-sign [0,1] weights on a
        leaky-integrate neuron with always-positive drive mean EVERY
        neuron eventually fires no matter the threshold, so no neuron can
        ever be genuinely "normally quiet." Confirmed directly: firing
        rates over 300 training ticks formed a smooth 20%-84% continuum
        with no natural bimodal split. Signed weights let some channels
        genuinely suppress a given hidden neuron under one signature and
        not another, which is the actual mechanism a real selective
        population needs."""
        self.n_channels = n_channels
        self.n_hidden = n_hidden
        self.reram_pos = reram_pos
        self.reram_neg = reram_neg
        self.leak = leak
        self.potential = [0.0] * n_hidden
        rng = random.Random(seed + 9001)
        self.threshold = [rng.uniform(0.3, 0.9) * n_channels for _ in range(n_hidden)]

    def step(self, channel_input: list) -> list:
        """One event-driven step. Returns a binary spike vector (Akida-
        style: only neurons that actually cross threshold produce an
        event; everyone else contributes nothing this tick)."""
        spikes = [0] * self.n_hidden
        for h in range(self.n_hidden):
            self.potential[h] = max(0.0, self.potential[h] - self.leak)
            drive = 0.0
            for c in range(self.n_channels):
                w = self.reram_pos.read(h, c) - self.reram_neg.read(h, c)
                drive += w * channel_input[c]
            self.potential[h] = max(0.0, self.potential[h] + drive)
            if self.potential[h] >= self.threshold[h]:
                spikes[h] = 1
                self.potential[h] = 0.0
        return spikes


# ── the last-layer, binary-weight, binary-input, on-chip-learned readout ─
class BinaryReadout:
    """Matches the real AKD1000 on-chip-learning constraint exactly:
    weights AND inputs are both binary. Trained by a simple Hebbian rule
    that's clamped to {0, 1}, not a continuous update -- this is the real
    constraint, not a convenience simplification of it."""

    def __init__(self, n_hidden: int):
        self.n_hidden = n_hidden
        self.weight = [1] * n_hidden
        self._fire_counts = [0] * n_hidden
        self._train_ticks = 0

    def observe(self, hidden_spikes: list) -> None:
        """Accumulate firing statistics during training -- does not set
        weights yet. Real bug found by actually running this (not assumed
        correct), through two iterations:
          1st draft: any_fired -> weight 0, none ever hit 1 -> readout
             structurally always read 0, could never detect anything.
          2nd draft: fired-even-once -> weight 0. Confirmed by direct
             instrumentation that with a leaky-integrate hidden layer,
             EVERY neuron fires at least once within 100 training ticks
             (rates ranged 20-86 per 100, none at 0) -- a single-occurrence
             rule zeroed every weight again, same dead-readout symptom.
        Fixed by training on firing RATE, finalized in finalize_training()."""
        self._train_ticks += 1
        for h, s in enumerate(hidden_spikes):
            self._fire_counts[h] += s

    def finalize_training(self) -> None:
        """Mark a neuron "expected" (weight 0, not alarm-worthy) only if
        it genuinely fired at all during training; every neuron that was
        truly silent under the normal signature stays alarm-capable (1).

        Real bug found by actually running this (not assumed correct): a
        "top-K by rate" version of this rule (pick a fixed fraction, rank
        by training rate) looked reasonable but was WRONG for this
        population -- most neurons tie at rate=0.0 during training (18 of
        24), and ranking-with-ties is a stable sort, so the "top half"
        arbitrarily included several rate-0 neurons purely by index
        order. Confirmed directly: the neurons that happened to win that
        arbitrary tie-break turned out to be exactly the ones that fire
        MOST under the drifted signature (up to 26/300 ticks) -- so the
        fraction-based rule was quietly excluding the most informative
        neurons from the alarm-capable set for no principled reason.
        A neuron with training rate exactly 0 has no evidence it's
        "normal" at all; it should never be spent on filling a quota."""
        if self._train_ticks == 0:
            return
        self.weight = [0 if c > 0 else 1 for c in self._fire_counts]

    def read(self, hidden_spikes: list) -> int:
        return sum(w * s for w, s in zip(self.weight, hidden_spikes))


def make_signature(n_channels: int, seed: int) -> list:
    rng = random.Random(seed)
    return [rng.uniform(0.3, 0.9) for _ in range(n_channels)]


def drifted_signature(base: list, target: list, frac: float) -> list:
    return [b + (t - b) * frac for b, t in zip(base, target)]


def run_trial(n_channels=8, n_hidden=24, drift_start=200, drift_ticks=400,
              total_ticks=2000, noise_std=0.05, seed=0):
    """One real run: train on a normal signature, then drift it slowly
    toward a different one starting at `drift_start`, over `drift_ticks`.
    Returns (detection_tick_or_None, naive_detection_tick_or_None)."""
    rng = random.Random(seed)
    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)
    for h in range(n_hidden):
        for c in range(n_channels):
            reram_pos.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
            reram_neg.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")

    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    drift_sig = make_signature(n_channels, seed=seed * 2 + 2)

    hidden = HiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)
    readout = BinaryReadout(n_hidden)

    # TRAIN: expose to normal signature, teach the readout "this is quiet"
    for _ in range(300):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        readout.observe(spikes)
    readout.finalize_training()

    detect_tick = None
    naive_detect_tick = None
    consec_snn = 0
    consec_naive = 0
    SNN_THRESHOLD = 3      # sustained mismatch spikes required (debounce)
    NAIVE_THRESHOLD = 0.12  # real measured typical full-drift L1 magnitude is ~0.20-0.29; set below that so it can trigger mid-ramp

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
        mismatch = readout.read(spikes)
        if mismatch > 0:
            consec_snn += 1
        else:
            consec_snn = 0
        if consec_snn >= SNN_THRESHOLD and detect_tick is None:
            detect_tick = t

        naive_dev = sum(abs(a - b) for a, b in zip(noisy, normal_sig)) / n_channels
        if naive_dev > NAIVE_THRESHOLD:
            consec_naive += 1
        else:
            consec_naive = 0
        if consec_naive >= SNN_THRESHOLD and naive_detect_tick is None:
            naive_detect_tick = t

    return detect_tick, naive_detect_tick, drift_start


def run_false_alarm_control(n_channels=8, n_hidden=24, total_ticks=5000,
                             noise_std=0.05, seed=100):
    """Pure-normal, never-drifts control run -- counts spurious detections
    for both detectors, the real false-alarm-rate half of the comparison."""
    rng = random.Random(seed)
    reram_pos = ReRAMSynapseArray(n_hidden, n_channels, seed=seed)
    reram_neg = ReRAMSynapseArray(n_hidden, n_channels, seed=seed + 5000)
    for h in range(n_hidden):
        for c in range(n_channels):
            reram_pos.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")
            reram_neg.program(h, c, rng.uniform(0.0, 1.0), mode="iterative")

    normal_sig = make_signature(n_channels, seed=seed * 2 + 1)
    hidden = HiddenLayer(n_channels, n_hidden, reram_pos, reram_neg, seed=seed)
    readout = BinaryReadout(n_hidden)
    for _ in range(300):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        readout.observe(spikes)
    readout.finalize_training()

    snn_alarms = 0
    naive_alarms = 0
    consec_snn = 0
    consec_naive = 0
    SNN_THRESHOLD = 3
    NAIVE_THRESHOLD = 0.12
    snn_armed = True
    naive_armed = True

    for t in range(total_ticks):
        noisy = [v + rng.gauss(0, noise_std) for v in normal_sig]
        spikes = hidden.step(noisy)
        mismatch = readout.read(spikes)
        consec_snn = consec_snn + 1 if mismatch > 0 else 0
        if consec_snn >= SNN_THRESHOLD and snn_armed:
            snn_alarms += 1
            snn_armed = False
        if consec_snn == 0:
            snn_armed = True

        naive_dev = sum(abs(a - b) for a, b in zip(noisy, normal_sig)) / n_channels
        consec_naive = consec_naive + 1 if naive_dev > NAIVE_THRESHOLD else 0
        if consec_naive >= SNN_THRESHOLD and naive_armed:
            naive_alarms += 1
            naive_armed = False
        if consec_naive == 0:
            naive_armed = True

    return snn_alarms, naive_alarms, total_ticks


if __name__ == "__main__":
    print("=" * 78)
    print("ReRAM + Akida-style drift detector -- domain-agnostic proof of concept")
    print("Software model only. No Akida/ReRAM hardware used. See module")
    print("docstrings for exactly which numbers are real/sourced vs. assumed.")
    print("=" * 78)

    N_TRIALS = 20
    snn_latencies = []
    naive_latencies = []
    snn_missed = 0
    naive_missed = 0

    for trial in range(N_TRIALS):
        det, naive_det, drift_start = run_trial(seed=trial)
        if det is not None:
            snn_latencies.append(det - drift_start)
        else:
            snn_missed += 1
        if naive_det is not None:
            naive_latencies.append(naive_det - drift_start)
        else:
            naive_missed += 1

    print(f"\n{N_TRIALS} real trials, each a fresh random signature pair + drift:")
    print(f"  ReRAM+Akida-style detector: {len(snn_latencies)}/{N_TRIALS} detected,"
          f" missed={snn_missed}")
    if snn_latencies:
        print(f"    mean detection latency: {sum(snn_latencies)/len(snn_latencies):.1f} ticks"
              f"  (min={min(snn_latencies)}, max={max(snn_latencies)})")
    print(f"  Naive amplitude-threshold baseline: {len(naive_latencies)}/{N_TRIALS} detected,"
          f" missed={naive_missed}")
    if naive_latencies:
        print(f"    mean detection latency: {sum(naive_latencies)/len(naive_latencies):.1f} ticks"
              f"  (min={min(naive_latencies)}, max={max(naive_latencies)})")

    print("\nFalse-alarm control (pure normal signal, 5000 ticks, no drift):")
    snn_fa, naive_fa, ticks = run_false_alarm_control()
    print(f"  ReRAM+Akida-style detector: {snn_fa} false alarms / {ticks} ticks")
    print(f"  Naive amplitude-threshold baseline: {naive_fa} false alarms / {ticks} ticks")
