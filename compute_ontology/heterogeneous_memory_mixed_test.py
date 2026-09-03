#!/usr/bin/env python
"""
heterogeneous_memory_mixed_test.py -- follow-up to
heterogeneous_memory_test.py's real, replicated crossover finding
(pure heterogeneous wins recall at P=2, pure homogeneous wins at P>=8,
see vault/Research/heterogeneous-memory-capacity-crossover-finding.md).

Tests the working explanation directly: does a MIXED ensemble (a smaller
specialist core, still giving each channel its own clean detector, plus a
larger generic-LIF regularizer block) capture the low-P specialist
advantage while damping the high-P confident-wrong collapse?

Kept as a separate script (not overwriting heterogeneous_memory_test.py)
so the original, already-logged, already-vault-cited result stays
reproducible on its own -- same convention as state_based_readout_test
v1/v2/v3 being kept side by side rather than replaced.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heterogeneous_memory_test import (   # noqa: E402
    run_capacity_test, mean_ci95,
    make_heterogeneous_pops, make_homogeneous_pop, make_mixed_pops,
    N_SPECIALIST_MIX, N_REGULARIZER_LIF,
)

if __name__ == "__main__":
    SEED_BASE = 0
    # the key diagnostic points from the pure-condition sweep: P=2 (pure
    # heterogeneous's clear win), P=4/6 (the indistinguishable middle),
    # P=8/11 (pure homogeneous's clear win) -- not the full 1..11 sweep,
    # to keep runtime bounded while still covering the real crossover zone.
    P_VALUES = [2, 4, 6, 8, 11]
    N_REPLICATIONS = 8

    print("=" * 90)
    print(f"Mixed ensemble ({N_SPECIALIST_MIX} of each specialist type + "
          f"{N_REGULARIZER_LIF} regularizer LIF, {4*N_SPECIALIST_MIX + N_REGULARIZER_LIF} total)")
    print("vs. the two pure conditions, at the real crossover diagnostic points.")
    print("=" * 90)

    conditions = [
        ("HETEROGENEOUS (pure)", make_heterogeneous_pops),
        ("MIXED (specialist core + LIF regularizers)", make_mixed_pops),
        ("HOMOGENEOUS (pure, 32x LIF)", make_homogeneous_pop),
    ]

    raw = {}
    t0 = time.time()
    for label, pop_factory in conditions:
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
                  f"[{time.time()-t0:.0f}s elapsed]", flush=True)

    print("\n" + "=" * 90)
    print("Side by side, mean +/- 95% CI (dropped-channel recall %, chance = 50%):")
    print("=" * 90)
    header = f"{'P':>4}" + "".join(f"{label[:22]:>26}" for label, _ in conditions)
    print(header)
    for P in P_VALUES:
        row = f"{P:>4}"
        for label, _ in conditions:
            m, ci = mean_ci95(raw[(label, P)])
            row += f"{m*100:>18.1f}% +/-{ci*100:4.1f}%"
        print(row)

    print("\nReal test of the hypothesis: does MIXED beat pure-heterogeneous's high-P")
    print("collapse while staying close to pure-heterogeneous's low-P advantage?")
    for P in P_VALUES:
        mh, ch = mean_ci95(raw[("HETEROGENEOUS (pure)", P)])
        mm, cm = mean_ci95(raw[("MIXED (specialist core + LIF regularizers)", P)])
        mo, co = mean_ci95(raw[("HOMOGENEOUS (pure, 32x LIF)", P)])
        print(f"  P={P:>3}: het={mh*100:.1f}%  mixed={mm*100:.1f}%  homog={mo*100:.1f}%")
