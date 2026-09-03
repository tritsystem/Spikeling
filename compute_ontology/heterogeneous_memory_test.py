#!/usr/bin/env python
"""
heterogeneous_memory_test.py -- own-research follow-up, 2026-08-30, to
heterogeneous_ontology_test.py's real, confirmed result (heterogeneous
typed ensemble: 100% detection across 4 anomaly types; homogeneous 32x
LIF: 50%, missing repetition/frequency entirely).

That test was DETECTION (binary: anomaly present or not). This test is
MEMORY: can a population's spike activity support genuine associative
PATTERN COMPLETION -- recovering a full stored pattern from a corrupted
(partially-missing) cue -- and does the heterogeneous composition help
there too, or was the earlier result specific to detection?

Design, reusing the SAME real, already-verified neuron models/scales/
generators from heterogeneous_ontology_test.py (not reimplemented):
  - A "pattern" = a specific non-empty subset of the 4 channels
    (magnitude, burst, repetition, frequency) firing SIMULTANEOUSLY.
    15 possible non-empty subsets from 4 binary channels.
  - A "cue" = the same pattern with ONE active channel silently DROPPED
    (a genuinely missing piece of information, not just added noise).
  - The population's real spike-count vector (32-dim: 8 neurons x 4
    types, or 32x LIF for the homogeneous control) is the population
    code.
  - A trained linear (ridge regression) readout maps population code ->
    4 real-valued channel-presence scores, thresholded at 0.5.
  - RECALL = can the readout, given only the CORRUPTED cue's population
    response, still output the TRUE FULL pattern's 4 bits (including the
    one that was dropped from the actual input)? That's genuine pattern
    completion, not just "what channels are literally present."

PRE-REGISTERED HYPOTHESIS: a heterogeneous population (8 each of
LIF/Izhikevich/AdEx/Resonator) will show higher bit-level recall
accuracy on corrupted cues than a homogeneous population of the same
total size (32x LIF), because each channel's presence is most reliably
signaled by its own specialist type (per the already-confirmed 4x4
cross-tab), so the readout has a cleaner, more separable signal to
learn the cross-channel associations FROM in the first place.
DISCONFIRM: if homogeneous LIF, given the same total neuron count and a
trained linear readout, recalls comparably or better -- report that
honestly, it would mean detection-level specialization doesn't
transfer to associative recall.
"""

import os
import sys
import random
import itertools

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heterogeneous_ontology_test import (   # noqa: E402
    ANOMALY_GENERATORS, SCALE, DT_FOR_KIND, make_population, native_dt,
)

CHANNELS = ["magnitude", "burst", "repetition", "frequency"]
KINDS = ["lif", "izhikevich", "adex", "resonator"]
N_PER_TYPE = 8
WINDOW = 2.4          # real seconds -- covers the longest channel (repetition, 8*0.3=2.4s)
LEAD_IN = 1.0          # quiet seconds before the pattern window


def pattern_signal(active_channels, t, t_start, rng):
    """Sum of the SAME real generators from heterogeneous_ontology_test.py,
    one per active channel, all sharing t_start -- a real, literal
    multi-channel co-occurrence, not a synthetic combined waveform."""
    if not (t_start <= t < t_start + WINDOW):
        # still want baseline noise outside the pattern window
        from heterogeneous_ontology_test import baseline_signal
        return baseline_signal(t, rng)
    total = 0.0
    from heterogeneous_ontology_test import baseline_signal
    base = baseline_signal(t, rng)
    contributed = False
    for ch in active_channels:
        gen = ANOMALY_GENERATORS[ch]
        val = gen(t, t_start, rng)
        # each generator returns baseline_signal outside ITS OWN active
        # sub-window even inside [t_start, t_start+WINDOW) (e.g. magnitude
        # is only active for the first 2.0s of the 2.4s window) -- only
        # count its contribution above baseline, then add baseline once.
        total += (val - base)
        contributed = True
    return base + total if contributed else base


def run_pop_on_pattern(kind_list_and_pops, active_channels, seed):
    """kind_list_and_pops: list of (kind, population) pairs (heterogeneous
    passes all 4; homogeneous passes just [("lif", 32-neuron pop)]).
    Returns a flat spike-count feature vector, ORDER FIXED across calls
    (concatenated per kind in KINDS order, or single block for homog)."""
    rng = random.Random(seed + 999)
    total_time = LEAD_IN + WINDOW
    feature = []
    for kind, pop in kind_list_and_pops:
        dt = DT_FOR_KIND[kind]
        step_dt = native_dt(kind, dt)
        fires = [0] * len(pop)
        t = 0.0
        # separate rng stream per kind would decorrelate types unrealistically;
        # use ONE shared signal realization across all kinds in this trial,
        # matching heterogeneous_ontology_test.py's own convention (same
        # underlying signal, rescaled per type) -- re-seed rng identically
        # per kind so every type sees the identical noise realization.
        kind_rng = random.Random(seed + 999)
        t = 0.0
        while t < total_time:
            raw = pattern_signal(active_channels, t, LEAD_IN, kind_rng)
            drive = raw * SCALE[kind]
            for i, n in enumerate(pop):
                if n.step(drive, step_dt, t):
                    fires[i] += 1
            t += dt
        feature.extend(fires)
    return feature


def make_heterogeneous_pops(seed):
    return [(kind, make_population(kind, N_PER_TYPE, seed=seed)) for kind in KINDS]


def make_homogeneous_pop(seed):
    return [("lif", make_population("lif", N_PER_TYPE * len(KINDS), seed=seed))]


# Mixed ensemble, testing the working explanation from the pure-heterogeneous
# result: specialists win at low P (clean per-channel signal) but collapse
# below chance at high P (confident-wrong completions when their own
# channel's evidence is missing). Homogeneous LIF degrades more gracefully
# instead of catastrophically. Hypothesis: a smaller specialist CORE (still
# giving each channel its own clean detector) plus a larger generic-LIF
# REGULARIZER block might keep the low-P advantage while damping the high-P
# collapse. Same total count (32) as both pure conditions for a fair
# comparison -- 4 of each specialist type (16) + 16 additional plain LIF.
N_SPECIALIST_MIX = 4
N_REGULARIZER_LIF = 16


def make_mixed_pops(seed):
    pops = [(kind, make_population(kind, N_SPECIALIST_MIX, seed=seed)) for kind in KINDS]
    pops.append(("lif", make_population("lif", N_REGULARIZER_LIF, seed=seed + 777)))
    return pops


ALL_PATTERNS = [c for r in range(1, len(CHANNELS) + 1)
                for c in itertools.combinations(CHANNELS, r)]  # 15 non-empty subsets


def bits_for(active_channels):
    return [1.0 if ch in active_channels else 0.0 for ch in CHANNELS]


# ---------------------------------------------------------------------
# Ridge regression (real, pivoted -- reusing the same discipline as
# spin_wave_selection_rule_test.py's own ridge_regression, not a new
# untested implementation).
# ---------------------------------------------------------------------
def standardize(X, mean=None, std=None):
    if mean is None:
        mean = [sum(col) / len(col) for col in zip(*X)]
    if std is None:
        std = []
        for j, m in enumerate(mean):
            var = sum((row[j] - m) ** 2 for row in X) / len(X)
            std.append(var ** 0.5 if var > 1e-12 else 1.0)
    Xs = [[(row[j] - mean[j]) / std[j] for j in range(len(row))] for row in X]
    return Xs, mean, std


def ridge_regression(X, Y, lam=1.0):
    """X: list of feature rows, Y: list of target rows (multi-output).
    Returns weight matrix W (n_features+1 x n_outputs, bias appended as
    last row) via normal equations with partial pivoting."""
    n, p = len(X), len(X[0])
    Xb = [row + [1.0] for row in X]  # bias column
    p1 = p + 1
    # X^T X + lam*I (don't regularize bias row/col)
    XtX = [[sum(Xb[k][i] * Xb[k][j] for k in range(n)) for j in range(p1)] for i in range(p1)]
    for i in range(p):
        XtX[i][i] += lam
    n_out = len(Y[0])
    XtY = [[sum(Xb[k][i] * Y[k][o] for k in range(n)) for o in range(n_out)] for i in range(p1)]

    # Gaussian elimination with partial pivoting, solve XtX * W = XtY
    A = [row[:] + XtY[i] for i, row in enumerate(XtX)]
    for col in range(p1):
        piv = max(range(col, p1), key=lambda r: abs(A[r][col]))
        if abs(A[piv][col]) < 1e-12:
            continue
        A[col], A[piv] = A[piv], A[col]
        pivval = A[col][col]
        A[col] = [v / pivval for v in A[col]]
        for r in range(p1):
            if r != col:
                factor = A[r][col]
                if factor != 0:
                    A[r] = [A[r][c] - factor * A[col][c] for c in range(len(A[r]))]
    W = [A[i][p1:] for i in range(p1)]
    return W


def predict(W, X):
    preds = []
    for row in X:
        rowb = row + [1.0]
        preds.append([sum(rowb[i] * W[i][o] for i in range(len(rowb))) for o in range(len(W[0]))])
    return preds


def bit_accuracy(preds, true_bits):
    correct, total = 0, 0
    for p_row, t_row in zip(preds, true_bits):
        for p, t in zip(p_row, t_row):
            pred_bit = 1.0 if p >= 0.5 else 0.0
            correct += int(pred_bit == t)
            total += 1
    return correct / total


def run_capacity_test(P, pop_factory, seed_base, n_reps_train=25, n_reps_test=20, lam=2.0):
    """Real associative-memory-capacity design: store a FIXED set of P
    patterns (drawn once, held fixed for this run), train the readout on
    repeated CLEAN presentations of exactly those P patterns (different
    noise realizations each rep -- so it's real signal averaging, not one
    single degenerate example per pattern), then test dropped-channel
    recall on FRESH corrupted presentations of the SAME P patterns.
    This is the correct analogue of Hopfield-style P-pattern capacity
    testing, not generalization across the full combinatorial space."""
    rng = random.Random(seed_base)
    # only patterns with >=2 active channels can have one dropped and still
    # leave something to recover FROM -- of the 15 non-empty subsets, 4 are
    # singletons (11 usable). Sample directly from the usable pool instead
    # of sampling-then-swapping (that approach ran out of replacements at
    # high P -- real bug, caught via a real IndexError, not assumed fixed).
    usable = [p for p in ALL_PATTERNS if len(p) >= 2]
    stored = rng.sample(usable, min(P, len(usable)))

    X_train, Y_train = [], []
    rep_seed = 0
    for pat in stored:
        for r in range(n_reps_train):
            seed = seed_base + 1000 + rep_seed
            rep_seed += 1
            pops = pop_factory(seed)
            feat = run_pop_on_pattern(pops, pat, seed)
            X_train.append(feat)
            Y_train.append(bits_for(pat))

    Xs, mean, std = standardize(X_train)
    W = ridge_regression(Xs, Y_train, lam=lam)
    train_acc = bit_accuracy(predict(W, Xs), Y_train)

    X_test, Y_test_true, dropped_idx_list = [], [], []
    rep_seed = 0
    for pat in stored:
        for r in range(n_reps_test):
            dropped = rng.choice(pat)
            cue = tuple(ch for ch in pat if ch != dropped)
            seed = seed_base + 50000 + rep_seed
            rep_seed += 1
            pops = pop_factory(seed)
            feat = run_pop_on_pattern(pops, cue, seed)
            X_test.append(feat)
            Y_test_true.append(bits_for(pat))
            dropped_idx_list.append(CHANNELS.index(dropped))

    Xs_test, _, _ = standardize(X_test, mean=mean, std=std)
    test_preds = predict(W, Xs_test)
    dropped_bit_correct = sum(
        int((1.0 if p_row[didx] >= 0.5 else 0.0) == t_row[didx])
        for p_row, t_row, didx in zip(test_preds, Y_test_true, dropped_idx_list)
    )
    dropped_bit_total = len(dropped_idx_list)
    return train_acc, dropped_bit_correct / dropped_bit_total, len(stored)


def mean_ci95(vals):
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    se = (var / n) ** 0.5
    return m, 1.96 * se  # normal approximation, matches this project's established CI convention


if __name__ == "__main__":
    import time
    SEED_BASE = 0
    P_VALUES = [1, 2, 3, 4, 6, 8, 10, 11]
    N_REPLICATIONS = 8   # INDEPENDENT reps: different seed_base -> different STORED
                          # pattern SET each time, not just different noise on one
                          # fixed set -- tests robustness to pattern-set choice too,
                          # same "never trust a small-sample effect without a CI"
                          # lesson already learned the hard way in oscillator-memory.

    print("=" * 90)
    print(f"Real associative-memory capacity test, {N_REPLICATIONS} independent replications")
    print("per P (different stored pattern set + different noise each rep) -> real 95% CIs.")
    print("=" * 90)

    raw = {}  # (label, P) -> list of recall values across replications
    t0 = time.time()
    for label, pop_factory in [("HETEROGENEOUS", make_heterogeneous_pops),
                                ("HOMOGENEOUS (32x LIF)", make_homogeneous_pop)]:
        print(f"\n--- {label} ---", flush=True)
        for P in P_VALUES:
            recalls = []
            for rep in range(N_REPLICATIONS):
                seed_base = SEED_BASE + hash((label, P, rep)) % 1_000_000
                _, recall, _ = run_capacity_test(P, pop_factory, seed_base,
                                                  n_reps_train=15, n_reps_test=15)
                recalls.append(recall)
            raw[(label, P)] = recalls
            m, ci = mean_ci95(recalls)
            print(f"  P={P:>3}  dropped_bit_recall = {m*100:5.1f}% +/- {ci*100:4.1f}%  "
                  f"(n={N_REPLICATIONS} reps, per-rep: {[f'{r*100:.0f}' for r in recalls]})"
                  f"  [{time.time()-t0:.0f}s elapsed]", flush=True)

    print("\n" + "=" * 90)
    print("Side by side, mean +/- 95% CI (dropped-channel recall %, chance = 50%):")
    print("=" * 90)
    print(f"{'P':>4} {'heterogeneous':>22} {'homogeneous':>22}")
    for P in P_VALUES:
        m1, ci1 = mean_ci95(raw[("HETEROGENEOUS", P)])
        m2, ci2 = mean_ci95(raw[("HOMOGENEOUS (32x LIF)", P)])
        s1 = f"{m1*100:5.1f}% +/- {ci1*100:4.1f}%"
        s2 = f"{m2*100:5.1f}% +/- {ci2*100:4.1f}%"
        overlap = "" if (m1+ci1 < m2-ci2 or m2+ci2 < m1-ci1) else "  (CIs overlap)"
        print(f"{P:>4} {s1:>22} {s2:>22}{overlap}")
