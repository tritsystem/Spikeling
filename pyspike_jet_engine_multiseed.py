#!/usr/bin/env python
"""pyspike_jet_engine_multiseed.py -- the rigorous check the single-run
STDP-vs-backprop-vs-hand-tuned comparison needed: N independent seed pairs
(exposure/training seed x held-out evaluation seed), mean + 95% CI for all
three conditions, same statistical discipline used everywhere else
tonight (the feedback-robustness confound check, the topological-phononics
multi-seed work) -- a 16-trial single-run gap of one trial (93.8% vs
100%) is not trustworthy on its own.
"""
import os
import sys
import math
import statistics

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, SPIKELING_ROOT)

import tempfile
import numpy as np

from compiler.compiler import compile_file
from runtime.runtime import SpikelingRuntime, STDPLearner

from pyspike_jet_engine_train import (
    SPK_PATH, make_curriculum, evaluate as evaluate_native,
    native_adam_train_curriculum, compile_native,
)
from pyspike_jet_engine_stdp import (
    build_runtime_with_stdp, run_trial, evaluate_runtime, fresh_runtime,
)

N_SEEDS = 10


def ci95(vals):
    m = statistics.mean(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    se = s / math.sqrt(len(vals))
    return m, m - 1.96 * se, m + 1.96 * se


def run_one_seed_pair(exposure_seed, eval_seed):
    held_out_trials = make_curriculum(n_sustained=8, n_blip=8, seed=eval_seed)
    train_trials = make_curriculum(n_sustained=10, n_blip=10, seed=exposure_seed)

    # 1. hand-tuned baseline (deterministic given eval seed, no training randomness)
    acc_handtuned = evaluate_native(compile_native(SPK_PATH)[0], held_out_trials)

    # 2. backprop: random init seeded by exposure_seed, native training
    np.random.seed(exposure_seed)
    net_bp, _ = compile_native(SPK_PATH)
    net_bp.weight = np.random.uniform(0.3, 1.5, size=net_bp.n_synapses)
    native_adam_train_curriculum(net_bp, train_trials, margin=1.0, epochs=15, lr=0.03)
    acc_backprop = evaluate_native(net_bp, held_out_trials)

    # 3. STDP: unsupervised exposure seeded by exposure_seed, real runtime
    exposure_runtime, _ = build_runtime_with_stdp(rate=0.02, tau=20.0)
    rng = np.random.default_rng(exposure_seed)
    for _ in range(40):
        drive_level = rng.uniform(60.0, 250.0)
        run_trial(exposure_runtime, drive_level, sustained=True)
    stdp_weights = [syn.weight for syn in exposure_runtime.synapses]
    acc_stdp = evaluate_runtime(fresh_runtime, held_out_trials, learner=stdp_weights)

    return acc_handtuned, acc_backprop, acc_stdp


if __name__ == "__main__":
    print("=" * 78)
    print(f"  MULTI-SEED CHECK: hand-tuned vs backprop vs STDP, N={N_SEEDS} seed pairs")
    print("=" * 78)

    handtuned_accs, backprop_accs, stdp_accs = [], [], []
    for i in range(N_SEEDS):
        exposure_seed = 1000 + i
        eval_seed = 2000 + i
        h, b, s = run_one_seed_pair(exposure_seed, eval_seed)
        handtuned_accs.append(h)
        backprop_accs.append(b)
        stdp_accs.append(s)
        print(f"  seed pair {i+1}/{N_SEEDS} (exposure={exposure_seed}, eval={eval_seed}): "
              f"hand-tuned={h:.1%}  backprop={b:.1%}  STDP={s:.1%}")

    print()
    print("=" * 78)
    print("RESULTS (mean, 95% CI over the seed pairs above)")
    print("=" * 78)
    for label, accs in [("hand-tuned", handtuned_accs), ("backprop", backprop_accs), ("STDP", stdp_accs)]:
        m, lo, hi = ci95(accs)
        print(f"  {label:12s} mean={m:.1%}  95% CI=[{lo:.1%}, {hi:.1%}]  raw={[f'{a:.0%}' for a in accs]}")

    # pairwise win-rate: how often did STDP beat / tie / lose to hand-tuned and backprop
    stdp_vs_hand = [s - h for s, h in zip(stdp_accs, handtuned_accs)]
    stdp_vs_bp = [s - b for s, b in zip(stdp_accs, backprop_accs)]
    print()
    print(f"STDP vs hand-tuned: wins={sum(1 for d in stdp_vs_hand if d>0)}  "
          f"ties={sum(1 for d in stdp_vs_hand if d==0)}  losses={sum(1 for d in stdp_vs_hand if d<0)}  "
          f"mean diff={statistics.mean(stdp_vs_hand):+.1%}")
    print(f"STDP vs backprop:   wins={sum(1 for d in stdp_vs_bp if d>0)}  "
          f"ties={sum(1 for d in stdp_vs_bp if d==0)}  losses={sum(1 for d in stdp_vs_bp if d<0)}  "
          f"mean diff={statistics.mean(stdp_vs_bp):+.1%}")

    m_h, lo_h, hi_h = ci95(handtuned_accs)
    m_s, lo_s, hi_s = ci95(stdp_accs)
    overlap = not (lo_s > hi_h or lo_h > hi_s)
    print()
    if overlap:
        print("VERDICT: STDP vs hand-tuned CIs overlap -- the single-run 100% vs 93.8% gap "
              "does NOT hold up as a robust difference across seeds. Real, but noise-level.")
    else:
        winner = "STDP" if m_s > m_h else "hand-tuned"
        print(f"VERDICT: real separation across seeds, {winner} is robustly higher.")
