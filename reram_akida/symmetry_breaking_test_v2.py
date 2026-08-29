#!/usr/bin/env python
"""
symmetry_breaking_test_v2.py — properly-designed follow-up. Two real
fixes over the confound-controlled v1:

1. ADDITIVE metric only (drift_rate - normal_rate), never a fractional/
   relative change -- v1's relative metric was dominated by a single
   outlier from a near-zero denominator and gave an untrustworthy mean.
2. N=100 paired trials (not 15) with a real paired t-test -- matching
   the actual rigor the user's own quasicrystal work used (paired t
   p~1e-13, per reservoir_quasicrystal_finding's cross-validation), not
   just eyeballing two means from a noisy small sample.

Paired design: for each seed, BOTH the symmetric (magnitude-matched) and
broken-symmetry conditions are measured on the exact same task instance
(same normal/drift signatures, same noise draws) -- so the comparison is
within-pair, which is the correct design for a small, noisy per-trial
effect (same reasoning as any paired study: it cancels per-trial task
variance that would otherwise swamp the signal).
"""

import math
import random

from symmetry_breaking_test import build_symmetric_reram_pair, build_broken_symmetry_reram_pair
from symmetry_breaking_test_controlled import mean_abs_drive, measure_differential_response_scaled


def paired_t_test(diffs):
    """Real paired t-test, no scipy dependency -- t = mean(diff) /
    (std(diff)/sqrt(n)), two-tailed p from the normal approximation
    (fine for n=100; not claiming exact-t-distribution precision, this
    is a real, honest approximation, not dressed up as more than it is)."""
    n = len(diffs)
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    std = math.sqrt(var)
    se = std / math.sqrt(n)
    t = mean / se if se > 0 else float("inf")
    # normal-approximation two-tailed p-value from |t|
    z = abs(t)
    p_approx = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
    return mean, std, t, p_approx


if __name__ == "__main__":
    print("=" * 78)
    print("Symmetry-breaking theorem test v2 -- additive metric, N=100 paired trials")
    print("=" * 78)

    N_TRIALS = 100
    n_channels, n_hidden = 8, 24

    sym_rates_normal, sym_rates_drift = [], []
    brk_rates_normal, brk_rates_drift = [], []
    paired_diffs = []  # (broken_additive - symmetric_additive) per trial

    for seed in range(N_TRIALS):
        from akida_style_drift_detector import make_signature
        normal_sig = make_signature(n_channels, seed=seed * 2 + 1)

        rp_sym, rn_sym = build_symmetric_reram_pair(n_hidden, n_channels, seed)
        rp_brk, rn_brk = build_broken_symmetry_reram_pair(n_hidden, n_channels, seed)

        sym_drive = mean_abs_drive(rp_sym, rn_sym, n_hidden, n_channels, normal_sig)
        brk_drive = mean_abs_drive(rp_brk, rn_brk, n_hidden, n_channels, normal_sig)
        gain = brk_drive / sym_drive if sym_drive > 0 else 1.0

        nr_s, dr_s, _ = measure_differential_response_scaled(
            rp_sym, rn_sym, n_channels, n_hidden, seed, input_gain=gain)
        nr_b, dr_b, _ = measure_differential_response_scaled(
            rp_brk, rn_brk, n_channels, n_hidden, seed, input_gain=1.0)

        sym_rates_normal.append(nr_s)
        sym_rates_drift.append(dr_s)
        brk_rates_normal.append(nr_b)
        brk_rates_drift.append(dr_b)

        sym_additive = dr_s - nr_s
        brk_additive = dr_b - nr_b
        paired_diffs.append(brk_additive - sym_additive)

    sym_additives = [d - n for d, n in zip(sym_rates_drift, sym_rates_normal)]
    brk_additives = [d - n for d, n in zip(brk_rates_drift, brk_rates_normal)]

    sym_silent = sum(1 for n, d in zip(sym_rates_normal, sym_rates_drift) if n == 0 and d == 0)
    brk_silent = sum(1 for n, d in zip(brk_rates_normal, brk_rates_drift) if n == 0 and d == 0)

    print(f"\n{N_TRIALS} real paired trials, magnitude-equalized, additive metric only:")
    print(f"  Symmetric:   mean additive change = {sum(sym_additives)/N_TRIALS:+.4f}"
          f"   ({sym_silent}/{N_TRIALS} trials completely silent in both conditions)")
    print(f"  Broken:      mean additive change = {sum(brk_additives)/N_TRIALS:+.4f}"
          f"   ({brk_silent}/{N_TRIALS} trials completely silent in both conditions)")

    mean_diff, std_diff, t, p = paired_t_test(paired_diffs)
    print(f"\nPaired t-test (broken_additive - symmetric_additive per trial):")
    print(f"  mean paired difference = {mean_diff:+.4f}  (std={std_diff:.4f})")
    print(f"  t = {t:.3f}, two-tailed p (normal approximation) = {p:.4g}")
    print(f"  {'SIGNIFICANT at p<0.05' if p < 0.05 else 'NOT significant at p<0.05'}"
          f" -- {'broken symmetry shows a real, larger drift response' if (p < 0.05 and mean_diff > 0) else ('symmetric shows a real, larger drift response' if (p < 0.05 and mean_diff < 0) else 'no real, resolvable difference at this sample size')}")
