#!/usr/bin/env python
"""Deploys ONE individually-verified STDP-derived weight set into the
production jet_engine_spike_pipeline.spk.

CORRECTED from a first, flawed attempt: averaging weight VECTORS across 10
independent STDP runs and deploying that average scored only 50.0% (coin
flip) on a fresh held-out set -- naive parameter-space averaging across
independent runs of a nonlinear, threshold-gated network does not preserve
performance the way averaging a linear model's weights might. Caught by
actually testing the deployed result, not assumed safe. Individual
(non-averaged) runs were then verified: 5 different exposure seeds, tested
on a completely fresh 30-trial held-out set never used in the multi-seed
check, ALL scored 93.3%, consistently beating the hand-tuned baseline's
90.0%. Deploying exposure_seed=1000's actual, individually-tested weights.

Static deployment only: weights fixed after this, no learn=STDP directive,
no ongoing live adaptation in production (explicit user choice).
"""
import os
import sys

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, SPIKELING_ROOT)

import numpy as np
from compiler.compiler import compile_file
import tempfile

from pyspike_jet_engine_stdp import build_runtime_with_stdp, run_trial

SPK_PATH = os.path.join(SPIKELING_ROOT, "ai-apps", "jet_engine_spike_pipeline.spk")
DEPLOY_SEED = 1000  # individually verified: 93.3% on a fresh 30-trial held-out set

exposure_runtime, ast = build_runtime_with_stdp(rate=0.02, tau=20.0)
rng = np.random.default_rng(DEPLOY_SEED)
for _ in range(40):
    drive_level = rng.uniform(60.0, 250.0)
    run_trial(exposure_runtime, drive_level, sustained=True)

weights = [syn.weight for syn in exposure_runtime.synapses]
synapse_labels = [(c.src, c.dst) for c in ast.connections]
print(f"deploying exposure_seed={DEPLOY_SEED} weights:")
for (s, d), w in zip(synapse_labels, weights):
    print(f"  {s:14s} -> {d:14s}  weight={w:.4f}")

with open(SPK_PATH, encoding="utf-8") as f:
    lines = f.readlines()

weight_by_pair = {pair: w for pair, w in zip(synapse_labels, weights)}

out_lines = []
for line in lines:
    stripped = line.strip()
    if stripped.startswith("connect "):
        parts = stripped.split()
        src, dst = parts[1], parts[3]
        if (src, dst) in weight_by_pair:
            new_w = weight_by_pair[(src, dst)]
            comment = ""
            if "#" in line:
                comment = "  #" + line.split("#", 1)[1].rstrip("\n")
            out_lines.append(f"connect {src} -> {dst} weight={new_w:.4f}{comment}\n")
            continue
    out_lines.append(line)

with open(SPK_PATH, "w", encoding="utf-8") as f:
    f.writelines(out_lines)

print(f"\nDeployed to {SPK_PATH}")
print(f"Original hand-tuned backup remains at {SPK_PATH}.backup_before_stdp")
