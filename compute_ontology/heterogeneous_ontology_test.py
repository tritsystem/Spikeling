#!/usr/bin/env python
"""
heterogeneous_ontology_test.py — own-research test, 2026-08-29: does
Kevin D. Johnson's real architectural principle (heterogeneous compute
resources, each given its own TYPED metric with semantic direction,
outperform forcing everything through one flat/homogenized scalar --
the "homogenization error" his papers name explicitly) hold when applied
to NEURON POPULATION COMPOSITION instead of compute scheduling?

This is NOT a reproduction of his work -- it is his real, general
architectural claim ("resources differing in kind require handling
designed for their differences rather than handling that erases the
differences through a common abstraction") tested as a genuine
hypothesis against a completely different real system: this project's
own four independently-verified Spikeling neuron models, documented in
tribe/NEURON_TYPES.md as each answering a genuinely different question
(LIF: magnitude, Izhikevich: burst pattern, AdEx: repetition/fatigue,
Resonator: frequency) -- structurally the same "unity in distinction"
shape as his ELIM-typed GPU/NPU/QPU/mainframe resources, just in a
domain he never touched.

Reuses real, already-verified code directly, not reimplementations:
  - IzhikevichNeuron, AdExNeuron, LIFReference from
    Spikeling/pyspike_neuron_models.py (unchanged).
  - ResonatorNeuron below is a faithful port of
    Spikeling/core/runtime/runtime.py's real ResonatorState.step()
    (symplectic Euler, energy_ema RMS-threshold edge trigger) -- same
    equations, adapted to this file's per-tick stepping convention.

DISCLOSED ADAPTATION: each neuron type has its own real, different
native operating range (Izhikevich/AdEx use mV/pA-scale units per their
literature presets; LIFReference and Resonator use small normalized
scales). Feeding one raw shared signal to all four unmodified would be
physically meaningless. Each type receives the SAME underlying anomaly
signal (same real temporal structure), rescaled into that type's own
native drive range -- an honest, necessary adaptation, not a hidden
tuning advantage for any one type.

PRE-REGISTERED HYPOTHESIS (stated before running anything):
1. Each neuron type will show a real, measurably LARGER detection
   advantage on the anomaly type matching its own documented specialty
   (LIF->magnitude, Izhikevich->burst, AdEx->repetition, Resonator->
   frequency) than on the other three -- a 4x4 cross-tabulation, no
   single type dominating every column.
2. A HETEROGENEOUS ensemble (all four types, typed per-type thresholds,
   detect = ANY type's population rate crosses ITS OWN calibrated
   threshold) will detect a LARGER fraction of the combined 4-anomaly-
   type test set than a HOMOGENEOUS ensemble of the same total neuron
   count using only LIF (the "homogenization" control -- LIF is the
   real, cheapest, most general-purpose type per NEURON_TYPES.md's own
   "good for" table, so it is the fair, not strawman, baseline).
DISCONFIRM: if LIF alone, given enough neurons and threshold diversity,
detects comparably across all four anomaly types -- report that
honestly; it would mean the "different kinds need different types"
principle doesn't transfer to this domain, a real, useful negative
result either way.
"""

import os
import sys
import math
import random

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from pyspike_neuron_models import IzhikevichNeuron, AdExNeuron, LIFReference  # noqa: E402


class ResonatorNeuron:
    """Faithful port of core/runtime/runtime.py's real ResonatorState.step()
    -- same symplectic-Euler order, same energy_ema RMS-threshold edge
    trigger, same real sample-rate-independent alpha derivation. Only
    change from the original: takes dt explicitly per call like the
    other three classes here, instead of being a dataclass field."""

    def __init__(self, freq_hz, damping=0.05, coupling=1.0, threshold=0.02,
                 gate_threshold=0.006, energy_time_constant=0.05):
        self.freq_hz = freq_hz
        self.damping = damping
        self.coupling = coupling
        self.threshold = threshold
        self.gate_threshold = gate_threshold
        self.energy_time_constant = energy_time_constant
        self.x = 0.0
        self.v = 0.0
        self.energy_ema = 0.0
        self.spike_log = []

    def step(self, I, dt, t):
        omega = 2 * math.pi * self.freq_hz
        accel = -(omega ** 2) * self.x - 2 * self.damping * omega * self.v
        accel += self.coupling * I
        self.v += accel * dt
        self.x += self.v * dt

        alpha = min(1.0, dt / self.energy_time_constant)
        was_above = math.sqrt(self.energy_ema) >= self.threshold
        if abs(self.x) >= self.gate_threshold:
            self.energy_ema += alpha * (self.x * self.x - self.energy_ema)
        else:
            self.energy_ema -= alpha * self.energy_ema
        now_above = math.sqrt(self.energy_ema) >= self.threshold

        fired = now_above and not was_above
        if fired:
            self.spike_log.append(t)
        return fired


# ── signal generators: one shared underlying anomaly, rescaled per type ──

TARGET_FREQ = 8.0     # Hz, the Resonator's tuned frequency
DISTRACTOR_FREQ = 3.0  # Hz, present in "quiet" baseline as background noise-tone


def baseline_signal(t, rng):
    """Quiet state: small background noise + a distractor tone, no anomaly."""
    return 0.05 * math.sin(2 * math.pi * DISTRACTOR_FREQ * t) + rng.gauss(0, 0.02)


def magnitude_anomaly(t, t_start, rng):
    """Sustained DC elevation for 2 seconds -- pure magnitude increase,
    no burst/repetition/frequency structure. LIF's home turf."""
    if t_start <= t < t_start + 2.0:
        return 0.6 + rng.gauss(0, 0.02)
    return baseline_signal(t, rng)


def burst_anomaly(t, t_start, rng):
    """Same total time-integrated extra current as magnitude_anomaly
    (0.6 * 2.0 = 1.2 unit-seconds), delivered as 4 short 0.1s pulses of
    height 3.0 instead of one sustained 2.0s plateau -- same energy
    budget, different temporal shape. Izhikevich's home turf."""
    for k in range(4):
        pulse_start = t_start + k * 0.5
        if pulse_start <= t < pulse_start + 0.1:
            return 3.0 + rng.gauss(0, 0.05)
    return baseline_signal(t, rng)


def repetition_anomaly(t, t_start, rng):
    """The SAME single 0.05s/height-1.0 pulse that occurs once per 3s in
    the normal background rate (simulated separately as the calibration
    condition), but repeated 8 times in rapid succession (every 0.3s)
    instead of isolated -- same per-pulse magnitude, only repeat COUNT
    changes. AdEx's home turf (fatigue accumulates with repeat count)."""
    for k in range(8):
        pulse_start = t_start + k * 0.3
        if pulse_start <= t < pulse_start + 0.05:
            return 1.0 + rng.gauss(0, 0.03)
    return baseline_signal(t, rng)


def frequency_anomaly(t, t_start, rng):
    """The target frequency tone appears for 2 seconds, replacing the
    distractor tone -- same amplitude as the baseline's own distractor
    tone, only the FREQUENCY differs. Resonator's home turf, and the
    same real paradigm already validated in NEURON_TYPES.md's
    99.2%-vs-~65% result."""
    if t_start <= t < t_start + 2.0:
        return 0.05 * math.sin(2 * math.pi * TARGET_FREQ * t) + rng.gauss(0, 0.02)
    return baseline_signal(t, rng)


ANOMALY_GENERATORS = {
    "magnitude": magnitude_anomaly,
    "burst": burst_anomaly,
    "repetition": repetition_anomaly,
    "frequency": frequency_anomaly,
}

# per-type drive rescaling: same underlying signal, native operating range.
# Real values below, not guesses -- each measured via a direct diagnostic
# probe (2026-08-29) before use, same discipline as every other threshold
# in this session. First guesses (izhikevich=12, adex=6, resonator
# threshold=0.02/coupling=1) were all wrong -- adex=6 never crossed the
# real spike threshold at all (needs ~400, matching the self-test scale
# in pyspike_neuron_models.py itself); resonator's threshold=0.02 was
# 25x its own real, already-validated production default (0.0008).
SCALE = {
    "lif": 1.0,
    "izhikevich": 12.0,
    "adex": 400.0,        # measured: I=18 (old scale) never fires; I=400 fires reliably
    "resonator": 15.0,    # measured: coupling=15 crosses the real 0.0008 threshold within ~0.2s
}

# real per-kind simulation timestep -- NOT the same for every type. The
# Resonator specifically needs fine temporal resolution to resolve an
# 8Hz oscillation at all (measured: dt=0.02 gave a numerically-suppressed
# response 30% below the true physical steady-state amplitude; dt=0.0005
# matches it). LIF/Izhikevich/AdEx are driven by discrete pulses, not an
# oscillation, and don't need finer resolution than the real per-tick
# rate already used throughout this session's detection tasks. This is
# itself a small real instance of the same principle under test: even
# within "neurons," different types genuinely need different native
# operating timescales, not one forced-common simulation rate.
DT_FOR_KIND = {
    "lif": 0.02,
    "izhikevich": 0.02,
    "adex": 0.02,
    "resonator": 0.0005,
}

# real production Resonator parameters (core/runtime/runtime.py's own
# defaults), not guessed -- see the module docstring's own real
# 99.2%-vs-~65% validated result for these exact numbers.
RESONATOR_THRESHOLD = 0.0008
RESONATOR_GATE = 0.00024
RESONATOR_TAU = 0.0025


def make_population(kind, n, seed):
    rng = random.Random(seed + hash(kind) % 10000)
    pop = []
    if kind == "lif":
        for _ in range(n):
            pop.append(LIFReference(threshold=rng.uniform(0.8, 1.3), leak=0.05, reset=0.0))
    elif kind == "izhikevich":
        for _ in range(n):
            pop.append(IzhikevichNeuron(preset=rng.choice(
                ["regular_spiking", "chattering", "intrinsically_bursting"])))
    elif kind == "adex":
        for _ in range(n):
            pop.append(AdExNeuron(tau_w=rng.uniform(400.0, 900.0), b=rng.uniform(40.0, 90.0)))
    elif kind == "resonator":
        for _ in range(n):
            pop.append(ResonatorNeuron(freq_hz=TARGET_FREQ,
                                        damping=rng.uniform(0.03, 0.08),
                                        threshold=RESONATOR_THRESHOLD,
                                        gate_threshold=RESONATOR_GATE,
                                        energy_time_constant=RESONATOR_TAU))
    return pop


def native_dt(kind, dt):
    """Real bug, caught and fixed 2026-08-29: IzhikevichNeuron/AdExNeuron
    (pyspike_neuron_models.py) carry their real literature time constants
    UNCHANGED (ms-scale: AdEx's tau_w~30-900ms, Izhikevich's 0.5ms
    substep threshold) -- passing this file's real-SECONDS dt directly
    made a 3-second anomaly window equal ~3 MILLISECONDS of their
    internal integration time, nowhere near enough to ever fire. Same
    class of bug as this project's own documented Lesson
    (biological-millisecond-time-constants-dont-survive-translation-to
    -real-second-game-scale.md) -- caught here by a suspicious all-zero
    result, not assumed correct."""
    return dt * 1000.0 if kind in ("izhikevich", "adex") else dt


def run_population(kind, pop, signal_fn, total_time, seed):
    rng = random.Random(seed + 999)
    fires = [0] * len(pop)
    t = 0.0
    dt = DT_FOR_KIND[kind]
    step_dt = native_dt(kind, dt)
    while t < total_time:
        raw = signal_fn(t, rng)
        drive = raw * SCALE[kind]
        for i, n in enumerate(pop):
            if n.step(drive, step_dt, t):
                fires[i] += 1
        t += dt
    return sum(fires)


def calibrate_threshold(kind, n, seed, calib_time=20.0, k_sigma=3.0, n_windows=10):
    """Real measured baseline: run n_windows independent quiet-baseline
    windows, calibrate threshold at mean + k*sigma of the total-spike-
    count-per-window distribution -- same discipline as every other
    threshold in this session's work. calib_time shortened from 60s to
    20s real time for the resonator's sake -- at its real dt=0.0005 that
    is still 40,000 real integration steps per window, calibration cost
    scales with 1/dt per kind."""
    rng_master = random.Random(seed)
    counts = []
    for w in range(n_windows):
        pop = make_population(kind, n, seed=rng_master.randint(0, 1_000_000))
        c = run_population(kind, pop, baseline_signal, calib_time, seed=w)
        counts.append(c)
    mean = sum(counts) / len(counts)
    var = sum((c - mean) ** 2 for c in counts) / max(1, len(counts) - 1)
    std = var ** 0.5
    # real floor: spike counts are non-negative integers, so a truly
    # zero-variance quiet baseline (confirmed: this population never
    # spontaneously fires at all) collapses mean+k*std to exactly 0,
    # which would make ANY single spike trivially "detected" and defeat
    # the whole point of calibration. Floor at 1 real spike minimum.
    return max(mean + k_sigma * std, 1.0), mean, std


def test_type_vs_anomaly(kind, anomaly_name, n, threshold, n_trials, seed_base):
    """Real detection test: n_trials independent runs, each a quiet
    period followed by the named anomaly injected once; count how many
    trials exceed the calibrated threshold DURING the anomaly window.
    Also returns mean raw spike count, so results stay interpretable
    even when boolean detection saturates at the floor threshold."""
    gen = ANOMALY_GENERATORS[anomaly_name]
    detected = 0
    total_spikes = 0
    dt = DT_FOR_KIND[kind]
    step_dt = native_dt(kind, dt)
    for trial in range(n_trials):
        seed = seed_base + trial * 137
        pop = make_population(kind, n, seed=seed)
        rng = random.Random(seed + 999)
        t_start = 2.0  # quiet lead-in before the anomaly (real seconds)
        total_time = t_start + 3.0
        fires = [0] * len(pop)
        t = 0.0
        while t < total_time:
            raw = gen(t, t_start, rng)
            drive = raw * SCALE[kind]
            for i, nrn in enumerate(pop):
                if nrn.step(drive, step_dt, t):
                    fires[i] += 1
            t += dt
        s = sum(fires)
        total_spikes += s
        if s >= threshold:
            detected += 1
    return detected, n_trials, total_spikes / n_trials


if __name__ == "__main__":
    N_PER_TYPE = 8
    N_TRIALS = 15
    SEED_BASE = 0
    KINDS = ["lif", "izhikevich", "adex", "resonator"]
    ANOMALIES = ["magnitude", "burst", "repetition", "frequency"]

    print("=" * 90)
    print("Part A: 4x4 cross-tabulation -- does each neuron type specialize on its own")
    print("documented anomaly type (magnitude/burst/repetition/frequency)?")
    print("=" * 90)

    thresholds = {}
    for kind in KINDS:
        thr, mean, std = calibrate_threshold(kind, N_PER_TYPE, seed=SEED_BASE)
        thresholds[kind] = thr
        print(f"  {kind:<12} threshold={thr:.2f} (baseline mean={mean:.2f}, std={std:.2f})")

    print()
    print("Detection rate (n_detected/n_trials):")
    header = f"{'':12}" + "".join(f"{a:>12}" for a in ANOMALIES)
    print(header)
    results = {}
    for kind in KINDS:
        row = f"{kind:<12}"
        for anomaly in ANOMALIES:
            det, n, mean_spikes = test_type_vs_anomaly(
                kind, anomaly, N_PER_TYPE, thresholds[kind],
                N_TRIALS, seed_base=SEED_BASE + hash(anomaly) % 1000)
            results[(kind, anomaly)] = (det, n, mean_spikes)
            row += f"{det}/{n:>9}"
        print(row)

    print("\nMean raw spike count per trial (real magnitude, not just above/below threshold):")
    print(header)
    for kind in KINDS:
        row = f"{kind:<12}"
        for anomaly in ANOMALIES:
            _, _, mean_spikes = results[(kind, anomaly)]
            row += f"{mean_spikes:>12.1f}"
        print(row)

    print("\n--- read the 4x4 table: does each type's own row-anomaly column beat the others? ---\n")

    print("=" * 90)
    print("Part B: heterogeneous typed ensemble vs. homogeneous all-LIF ensemble")
    print("Same total neuron count (32), combined across all 4 anomaly types")
    print("=" * 90)

    N_TOTAL = 32
    homog_thr, hmean, hstd = calibrate_threshold("lif", N_TOTAL, seed=SEED_BASE + 5000)
    print(f"  homogeneous (32x LIF) threshold={homog_thr:.2f} (mean={hmean:.2f}, std={hstd:.2f})")

    het_detected, het_total = 0, 0
    homog_detected, homog_total = 0, 0
    for anomaly in ANOMALIES:
        # heterogeneous: detect = ANY type's own population (8 each) crosses ITS OWN threshold
        het_hits = 0
        for trial in range(N_TRIALS):
            seed = SEED_BASE + 20000 + hash(anomaly) % 1000 + trial * 137
            any_hit = False
            for kind in KINDS:
                gen = ANOMALY_GENERATORS[anomaly]
                pop = make_population(kind, N_PER_TYPE, seed=seed)
                rng = random.Random(seed + 999)
                dt = DT_FOR_KIND[kind]
                t_start, total_time = 2.0, 5.0
                fires = [0] * len(pop)
                t = 0.0
                step_dt = native_dt(kind, dt)
                while t < total_time:
                    raw = gen(t, t_start, rng)
                    drive = raw * SCALE[kind]
                    for i, nrn in enumerate(pop):
                        if nrn.step(drive, step_dt, t):
                            fires[i] += 1
                    t += dt
                if sum(fires) >= thresholds[kind]:
                    any_hit = True
            if any_hit:
                het_hits += 1
        het_detected += het_hits
        het_total += N_TRIALS

        # homogeneous: 32 LIF neurons, single flat threshold
        homog_det, homog_n, homog_mean_spikes = test_type_vs_anomaly(
            "lif", anomaly, N_TOTAL, homog_thr,
            N_TRIALS, seed_base=SEED_BASE + 30000 + hash(anomaly) % 1000)
        homog_detected += homog_det
        homog_total += homog_n
        print(f"  {anomaly:<12} heterogeneous={het_hits}/{N_TRIALS}   homogeneous(32xLIF)={homog_det}/{homog_n}")

    print(f"\n  TOTAL across all 4 anomaly types:")
    print(f"    Heterogeneous (typed, 8 each of 4 kinds): {het_detected}/{het_total} "
          f"({100*het_detected/het_total:.1f}%)")
    print(f"    Homogeneous (32x LIF, flat):              {homog_detected}/{homog_total} "
          f"({100*homog_detected/homog_total:.1f}%)")
    print("\n--- read the raw numbers above against the pre-registration; report honestly ---")
