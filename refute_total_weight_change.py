#!/usr/bin/env python3
"""
REFUTE total_weight_change on the STDP rate×tau sweep.

Independent second check via DoWhy (placebo treatment, random common cause,
data subset) — same x|zs as ADJUST already used, different estimator + 3
real perturbation tests. Runs standalone so we don't need the copilot loop
to happen to call REFUTE this time.
"""

import csv, os, statistics, math, json
import numpy as np
import pandas as pd
from dowhy import CausalModel

CSV = os.path.join(
    os.path.dirname(__file__),
    "..", "Spikeling", "stdp_rate_tau_sweep.csv",
)
OUT = os.path.join(
    os.path.dirname(__file__),
    "ledger_stdp_rate_tau_refute.txt",
)

data = []
with open(CSV, newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        data.append({
            "rate": float(row["rate"]),
            "tau": float(row["tau"]),
            "max_weight": float(row["max_weight"]),
            "min_weight": float(row["min_weight"]),
            "weight_spread": float(row["weight_spread"]),
            "total_weight_change": float(row["total_weight_change"]),
            "accuracy": float(row["accuracy"]),
        })
df = pd.DataFrame(data)
assert len(df) == 25, f"expected 25 rows, got {len(df)}"

xs = ["rate", "tau", "max_weight", "min_weight", "weight_spread", "total_weight_change"]
zs = [c for c in xs if c != "total_weight_change"]   # everything except the tested x

out = []
def log(s=""):
    out.append(s)
    print(s)

log(f"REFUTE: total_weight_change -> accuracy, n={len(df)}")
log(f"confounders (same set ADJUST used): {zs}")
log("")

try:
    cm = CausalModel(
        data=df,
        treatment="total_weight_change",
        outcome="accuracy",
        common_causes=zs,
    )
    identified = cm.identify_effect(proceed_when_unidentifiable=True)
    est = cm.estimate_effect(identified, method_name="backdoor.linear_regression")
    orig = float(est.value)
    log(f"original estimate (backdoor.linear_regression, independent of ADJUST's fit): {orig:+.4f}")
    log("")

    placebo = cm.refute_estimate(identified, est, method_name="placebo_treatment_refuter", placebo_type="permute")
    rcc     = cm.refute_estimate(identified, est, method_name="random_common_cause")
    subset  = cm.refute_estimate(identified, est, method_name="data_subset_refuter", subset_fraction=0.8)

    p_new   = float(placebo.new_effect)
    rcc_new = float(rcc.new_effect)
    sub_new = float(subset.new_effect)

    scale = abs(orig) if abs(orig) > 1e-9 else 1e-9
    collapsed    = abs(p_new)   < 0.1 * scale
    stable_rcc   = abs(rcc_new - orig) < 0.2 * scale
    stable_sub   = abs(sub_new - orig) < 0.2 * scale
    n_pass = sum([collapsed, stable_rcc, stable_sub])

    log(f"  placebo treatment: new effect {p_new:+.4f} -- "
        + ("collapsed toward 0, as expected for a real effect." if collapsed
           else "did NOT collapse -- suspicious; estimate may reflect fitting procedure.")
        + "")
    log(f"  random common cause: new effect {rcc_new:+.4f} -- "
        + ("stable." if stable_rcc else "changed notably -- sensitive to irrelevant confounder.")
        + "")
    log(f"  data subset (80%): new effect {sub_new:+.4f} -- "
        + ("stable." if stable_sub else "changed notably -- may be driven by influential points.")
        + "")
    log(f"[{n_pass}/3 refutation checks consistent with a real, stable effect]")
except Exception as e:
    log(f"REFUTE error: {e}")

with open(OUT, "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
print(f"\nWrote {OUT}")
