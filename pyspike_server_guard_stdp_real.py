#!/usr/bin/env python
"""pyspike_server_guard_stdp_real.py -- STDP trained on the REAL
server-guard distribution, not the synthetic 60-250 drive range that
caused the deployed weights to fail on real data (25.0% -- see
pyspike_server_guard_real_curriculum.py). Same STDP mechanism, same
runtime, only the exposure signal changes: real historical windows
instead of a generic synthetic range.

REAL train/held-out split, not everything used for both: the 39 real
sustained windows are split, most for exposure, a genuine held-out subset
never seen during STDP exposure for evaluation. Small real sample
disclosed, not padded out: only 5 real blip windows exist in the actual
history, used for evaluation only (too few to meaningfully split further).
"""
import os
import sys

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, SPIKELING_ROOT)

import tempfile
import numpy as np
from compiler.compiler import compile_file
from runtime.runtime import SpikelingRuntime, STDPLearner

from pyspike_server_guard_real_curriculum import (
    load_real_values, extract_real_windows, run_window, evaluate,
    SPK_PATH, BEACON_SCALE,
)

BACKUP_SPK = SPK_PATH + ".backup_before_stdp"


def compile_stdp_runtime(rate=0.02, tau=20.0, spk_path=BACKUP_SPK):
    """Starts from the ORIGINAL hand-tuned weights (the ones that scored
    97.7% on real data) and applies STDP on top -- refining a working
    baseline via real exposure, not starting from scratch."""
    ast = compile_file(spk_path, output_dir=tempfile.mkdtemp(prefix="stdp_real_"))
    runtime = SpikelingRuntime(ast)
    runtime.learner = STDPLearner(rate=rate, tau=tau)
    return runtime, ast


if __name__ == "__main__":
    print("=" * 78)
    print("  STDP TRAINED ON REAL SERVER-GUARD DATA (not synthetic)")
    print("=" * 78)

    values = load_real_values()
    sustained_windows, blip_windows = extract_real_windows(values)
    print(f"real sustained windows: {len(sustained_windows)}, real blip windows: {len(blip_windows)}")

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(sustained_windows))
    n_train = 30
    train_windows = [sustained_windows[i] for i in idx[:n_train]]
    held_out_sustained = [sustained_windows[i] for i in idx[n_train:]]
    print(f"real train (exposure) windows: {len(train_windows)}, "
          f"held-out sustained: {len(held_out_sustained)}, held-out blip (all): {len(blip_windows)}\n")

    # STDP exposure: REAL sustained windows only (matches the real-world
    # case that matters -- the network needs to recognize REAL sustained
    # beacon activity, exposure IS the real signal, no synthetic substitute)
    exposure_runtime, ast = compile_stdp_runtime()
    for w in train_windows:
        run_window(exposure_runtime, w, BEACON_SCALE)

    weights = [syn.weight for syn in exposure_runtime.synapses]
    print("weights after REAL-data STDP exposure (vs original hand-tuned):")
    for c, w in zip(ast.connections, weights):
        print(f"  {c.src:14s} -> {c.dst:14s}  {c.weight:.3f} -> {w:.3f}")

    # write a candidate .spk for evaluation (not deployed yet)
    with open(BACKUP_SPK, encoding="utf-8") as f:
        lines = f.readlines()
    weight_by_pair = {(c.src, c.dst): w for c, w in zip(ast.connections, weights)}
    out_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("connect "):
            parts = stripped.split()
            src, dst = parts[1], parts[3]
            if (src, dst) in weight_by_pair:
                out_lines.append(f"connect {src} -> {dst} weight={weight_by_pair[(src, dst)]:.4f}\n")
                continue
        out_lines.append(line)
    CANDIDATE_SPK = os.path.join(SPIKELING_ROOT, "ai-apps", "jet_engine_spike_pipeline_stdp_realtrained.spk")
    with open(CANDIDATE_SPK, "w", encoding="utf-8") as f:
        f.writelines(out_lines)
    print(f"\ncandidate weights written to {CANDIDATE_SPK} (NOT deployed)\n")

    print(f"--- EVALUATION on held-out real data (never seen during exposure) ---")
    acc_candidate = evaluate(held_out_sustained, blip_windows, BEACON_SCALE,
                              "real-data-trained STDP candidate", spk_path=CANDIDATE_SPK, verbose=True)

    acc_handtuned_heldout = evaluate(held_out_sustained, blip_windows, BEACON_SCALE,
                                      "hand-tuned (same held-out set)", spk_path=BACKUP_SPK, verbose=False)

    print(f"\n{'='*78}")
    print("COMPARISON on the SAME held-out real data:")
    print(f"  hand-tuned (original):              {acc_handtuned_heldout:.1%}")
    print(f"  STDP trained on REAL data (candidate): {acc_candidate:.1%}")
    print("=" * 78)
