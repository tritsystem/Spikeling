#!/usr/bin/env python3
"""
pyspike_stdp_rate_tau_sweep.py — Vary STDP rate & tau, measure discrimination
accuracy on the jet-engine task. Output CSV for methodlm causal analysis.

Reuses pyspike_jet_engine_stdp.py / pyspike_jet_engine_train.py framework:
  - compile jet_engine.spk once
  - for each (rate, tau): expose to 40 sustained trials via real STDPLearner,
    then evaluate on 16 held-out trials (8 sustained, 8 blip) with copied weights.
"""

import os
import sys
import tempfile
import csv

import numpy as np

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, SPIKELING_ROOT)

from compiler.compiler import compile_file
from runtime.runtime import SpikelingRuntime, STDPLearner
from pyspike_jet_engine_train import (
    SPK_PATH, make_curriculum, DRIVE_LOW, DRIVE_HIGH,
    T_SUSTAINED, T_WINDOW,
)
from pyspike_jet_engine_stdp import (
    run_trial, evaluate_runtime, fresh_runtime,
)

# ── compile once, reuse AST ──────────────────────────────────────────────
ast = compile_file(SPK_PATH, output_dir=tempfile.mkdtemp(prefix="sweep_ast_"))

# baseline weights from a fresh runtime (no STDP exposure)
baseline_rt = SpikelingRuntime(ast)
baseline_weights = [syn.weight for syn in baseline_rt.synapses]


def make_runtime_from_ast():
    return SpikelingRuntime(ast)


def evaluation_factory(post_weights):
    """Return a callable that builds a fresh runtime and copies post-exposure
    weights in — same shape as fresh_runtime() but with weights pre-set."""
    def factory():
        rt = make_runtime_from_ast()
        for syn, w in zip(rt.synapses, post_weights):
            syn.weight = w
        return rt
    return factory


# ── sweep grid ────────────────────────────────────────────────────────────
RATES = [0.005, 0.01, 0.02, 0.05, 0.1]
TAUS  = [5.0, 10.0, 20.0, 40.0, 80.0]

N_EXPOSURES = 40
EVAL_SEED_BASE = 5000

rows = []

for rate in RATES:
    for tau in TAUS:
        # --- exposure phase: real STDP, 40 sustained trials ---
        rt = SpikelingRuntime(ast)
        rt.learner = STDPLearner(rate=rate, tau=tau)

        rng = np.random.default_rng(hash((rate, tau, 0)) % 2**32)
        for _ in range(N_EXPOSURES):
            drive = rng.uniform(DRIVE_LOW, DRIVE_HIGH)
            run_trial(rt, drive, sustained=True)

        post_weights = [syn.weight for syn in rt.synapses]

        # --- evaluation: 16 held-out trials, fresh runtimes, copied weights ---
        eval_seed = EVAL_SEED_BASE + int(rate * 1000) + int(tau * 10)
        eval_trials = make_curriculum(n_sustained=8, n_blip=8, seed=eval_seed)
        factory = evaluation_factory(post_weights)
        acc = evaluate_runtime(factory, eval_trials)

        # weight stats
        max_w = max(post_weights)
        min_w = min(post_weights)
        spread = max_w - min_w
        total_change = sum(abs(p - b) for p, b in zip(post_weights, baseline_weights))

        row = {
            "rate": rate,
            "tau": tau,
            "accuracy": round(acc, 4),
            "max_weight": round(max_w, 4),
            "min_weight": round(min_w, 4),
            "weight_spread": round(spread, 4),
            "total_weight_change": round(total_change, 4),
        }
        rows.append(row)
        print(f"rate={rate:.3f} tau={tau:5.1f}  acc={acc:.1%}  "
              f"max_w={max_w:.3f}  spread={spread:.3f}  Δ={total_change:.3f}")

# ── write CSV ─────────────────────────────────────────────────────────────
out_path = os.path.join(SPIKELING_ROOT, "stdp_rate_tau_sweep.csv")
with open(out_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print(f"\nWrote {len(rows)} rows to {out_path}")
print("\nColumn meanings for methodlm:")
print("  rate, tau  — STDP hyperparameters (what we varied)")
print("  accuracy   — discrimination accuracy on held-out trials (target)")
print("  max_weight, min_weight, weight_spread — synaptic weight stats after exposure")
print("  total_weight_change — L1 distance from baseline weights")
