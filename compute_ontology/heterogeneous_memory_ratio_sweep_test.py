#!/usr/bin/env python
"""
heterogeneous_memory_ratio_sweep_test.py -- follow-up rung to
heterogeneous_memory_mixed_test.py's real disconfirm (see
FINDINGS_mixed_ensemble.md / vault/Research/heterogeneous-memory-capacity-
crossover-finding.md, "Update 2026-09-04"): a 16-specialist/16-LIF mixed
ensemble did NOT rescue the high-P associative-memory collapse -- it tracked
pure heterogeneous's collapse almost exactly (P=11: 34.0% vs 33.6%, both far
below homogeneous's 51.7%).

That leaves two live, untested explanations open:
  (a) dilution IS the right idea but 16/16 wasn't enough LIF -- the rescue
      is dose-dependent on specialist:LIF ratio, and a smaller specialist
      core would show more rescue.
  (b) the collapse is a hard floor once ANY meaningful specialist fraction
      is present -- dilution by count doesn't touch it regardless of ratio,
      because the specialists' confident-wrong completions dominate the
      readout's learned weights out of proportion to their population share.

PRE-REGISTERED HYPOTHESIS (stated before running anything): if (a), high-P
recall should trend MONOTONICALLY with specialist fraction across the full
axis --
    0 specialist (32 LIF, pure homogeneous)          -- already measured, 51.7% @ P=11
    8 specialist (2/type, 24 LIF)                     -- NEW, this run
    16 specialist (4/type, 16 LIF)                    -- already measured, 34.0% @ P=11
    24 specialist (6/type, 8 LIF)                      -- NEW, this run
    32 specialist (8/type, 0 LIF, pure heterogeneous)  -- already measured, 33.6% @ P=11
i.e. 8-specialist should sit meaningfully ABOVE 16-specialist's 34.0% (closer
to homogeneous), and 24-specialist should sit close to or below 16-specialist
(closer to pure heterogeneous).

If (b), 8-specialist will ALSO collapse to roughly the same ~33-38% band as
16/24/32-specialist at high P, despite having 3x fewer specialists than the
16/16 condition and only 1/4 the specialists of pure heterogeneous --
falsifying dose-dependence and supporting a hard-floor / disproportionate-
readout-influence explanation instead.

DESIGN: same methodology as the prior two rungs for direct comparability --
same diagnostic P points (2, 4, 6, 8, 11), same 8 independent replications
per point (different stored pattern SET per rep), same real 95% CIs (normal
approximation), same ridge-regression dropped-channel-recall readout, same
n_reps_train=15/n_reps_test=15 -- reusing run_capacity_test/mean_ci95/
make_population unchanged, not reimplemented.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heterogeneous_ontology_test import make_population  # noqa: E402
from heterogeneous_memory_test import run_capacity_test, mean_ci95, KINDS  # noqa: E402


def make_ratio_pops_factory(n_specialist_per_type, n_lif):
    """Returns a pop_factory(seed) closure for a given specialist:LIF split.
    Same total-neuron-count discipline as the prior rungs would require if
    the total were fixed at 32 -- here it stays fixed at 32 for every ratio
    tested (4*n_specialist_per_type + n_lif == 32) so this is a fair sweep
    along one axis, not a confound with population size."""
    total = 4 * n_specialist_per_type + n_lif
    assert total == 32, f"ratio must keep total at 32, got {total}"

    def factory(seed):
        pops = [(kind, make_population(kind, n_specialist_per_type, seed=seed))
                for kind in KINDS]
        if n_lif > 0:
            pops.append(("lif", make_population("lif", n_lif, seed=seed + 777)))
        return pops
    return factory


if __name__ == "__main__":
    SEED_BASE = 0
    P_VALUES = [2, 4, 6, 8, 11]
    N_REPLICATIONS = 8

    # (label, n_specialist_per_type, n_lif) -- only the two NEW ratios; the
    # 0/32, 16/16, and 32/0 points already have real measured data from the
    # prior two rungs and are not re-run here (see combine step at the end).
    RATIOS = [
        ("8-SPECIALIST (2/type) + 24-LIF", 2, 24),
        ("24-SPECIALIST (6/type) + 8-LIF", 6, 8),
    ]

    print("=" * 90)
    print("Specialist:LIF ratio sweep -- dose-dependence test for the mixed-ensemble disconfirm")
    print("=" * 90)

    raw = {}
    t0 = time.time()
    for label, n_spec, n_lif in RATIOS:
        factory = make_ratio_pops_factory(n_spec, n_lif)
        print(f"\n--- {label} ---", flush=True)
        for P in P_VALUES:
            recalls = []
            for rep in range(N_REPLICATIONS):
                seed_base = SEED_BASE + hash((label, P, rep)) % 1_000_000
                _, recall, _ = run_capacity_test(P, factory, seed_base,
                                                  n_reps_train=15, n_reps_test=15)
                recalls.append(recall)
            raw[(label, P)] = recalls
            m, ci = mean_ci95(recalls)
            print(f"  P={P:>3}  dropped_bit_recall = {m*100:5.1f}% +/- {ci*100:4.1f}%  "
                  f"[{time.time()-t0:.0f}s elapsed]", flush=True)

    # Reference points already measured in prior rungs (not re-run here) --
    # hardcoded from FINDINGS_mixed_ensemble.md / the crossover vault finding,
    # for the combined table only.
    REFERENCE = {
        ("0-SPECIALIST (32-LIF, pure homogeneous)", 2): (82.1, 4.1),
        ("0-SPECIALIST (32-LIF, pure homogeneous)", 4): (55.2, 5.6),
        ("0-SPECIALIST (32-LIF, pure homogeneous)", 6): (48.3, 6.1),
        ("0-SPECIALIST (32-LIF, pure homogeneous)", 8): (52.5, 5.5),
        ("0-SPECIALIST (32-LIF, pure homogeneous)", 11): (51.7, 2.3),
        ("16-SPECIALIST (4/type) + 16-LIF", 2): (83.8, 3.6),
        ("16-SPECIALIST (4/type) + 16-LIF", 4): (52.3, 8.4),
        ("16-SPECIALIST (4/type) + 16-LIF", 6): (43.8, 7.8),
        ("16-SPECIALIST (4/type) + 16-LIF", 8): (38.5, 7.7),
        ("16-SPECIALIST (4/type) + 16-LIF", 11): (34.0, 2.8),
        ("32-SPECIALIST (8/type, pure heterogeneous)", 2): (86.2, 6.2),
        ("32-SPECIALIST (8/type, pure heterogeneous)", 4): (47.3, 11.5),
        ("32-SPECIALIST (8/type, pure heterogeneous)", 6): (44.2, 7.2),
        ("32-SPECIALIST (8/type, pure heterogeneous)", 8): (34.5, 6.0),
        ("32-SPECIALIST (8/type, pure heterogeneous)", 11): (33.6, 2.6),
    }

    order = [
        "0-SPECIALIST (32-LIF, pure homogeneous)",
        "8-SPECIALIST (2/type) + 24-LIF",
        "16-SPECIALIST (4/type) + 16-LIF",
        "24-SPECIALIST (6/type) + 8-LIF",
        "32-SPECIALIST (8/type, pure heterogeneous)",
    ]

    print("\n" + "=" * 110)
    print("FULL RATIO SWEEP, mean +/- 95% CI (dropped-channel recall %, chance = 50%)")
    print("(0/32 and 16/16 and 32/0 rows are reference data from prior rungs, not re-run here)")
    print("=" * 110)
    header = f"{'P':>4}" + "".join(f"{lbl.split(' ')[0]:>20}" for lbl in order)
    print(header)
    for P in P_VALUES:
        row = f"{P:>4}"
        for lbl in order:
            if (lbl, P) in REFERENCE:
                m, ci = REFERENCE[(lbl, P)]
            else:
                m, ci = mean_ci95(raw[(lbl, P)])
                m, ci = m * 100, ci * 100
            row += f"{m:>15.1f}%+-{ci:3.1f}%"
        print(row)
