#!/usr/bin/env python
"""pyspike_unique_ips_stdp_real.py -- same careful methodology as the
beacon-detection fix, applied to net.unique_remote_ips: real historical
windows (not synthetic), a SCALE derived from this channel's own real
observed range (not reused from beacon's, which is a different
magnitude), real train/held-out split, multi-split validation before
trusting anything, NOT deployed to production without explicit
confirmation.

SCALE derivation, ATTEMPT 1 (measured, then found wrong by testing):
SCALE=2.0 put baseline drive at 71.4 -- looked safely below threshold=80
on paper, but real blip windows still false-positive-ignited 10/10 times,
identically for both hand-tuned and STDP, across all 5 splits (28.6% both
conditions). Diagnosis: this channel's real baseline is NEVER near zero
(mean=35.72, min=12 -- unlike beacon detection's near-zero quiet
baseline), so even "quiet" padding around a blip accumulates toward
threshold within 1-2 ticks via LIF integration (71.4 + 71.4 - leak(5) =
137.8, clears 80 by tick 2 regardless of the actual burst). SCALE=2.0 was
too aggressive for a channel that never truly goes quiet.

SCALE derivation, ATTEMPT 2 (corrected): SCALE=1.0 puts baseline drive at
35.7 (needs several ticks of PURE baseline to ever accumulate to
threshold, not 1-2) and elevated-threshold drive at 53.6 (still needs ~2
ticks to clear 80, but sustained elevation gets there fast; a single-tick
blip won't). Re-validated empirically below, not assumed correct just
because the arithmetic looks better this time.
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
    load_real_values, extract_real_windows, run_window, evaluate, SPK_PATH,
)

CHANNEL = "net.unique_remote_ips"
SCALE = 1.0  # corrected from 2.0 -- see module docstring, ATTEMPT 2
ELEVATED_THRESHOLD = 53.58  # mean + std, measured directly from this channel's real data
BACKUP_SPK = SPK_PATH + ".backup_before_stdp"  # original hand-tuned weights, known-good starting point


def train_and_validate_split(sustained_windows, blip_windows, split_seed, n_train=None, verbose=False):
    rng = np.random.default_rng(split_seed)
    n_total_sustained = len(sustained_windows)
    if n_train is None:
        n_train = max(1, int(n_total_sustained * 0.7))  # 70/30 split, real small-sample discipline
    idx = rng.permutation(n_total_sustained)
    train_windows = [sustained_windows[i] for i in idx[:n_train]]
    held_out_sustained = [sustained_windows[i] for i in idx[n_train:]]

    ast = compile_file(BACKUP_SPK, output_dir=tempfile.mkdtemp(prefix="ips_stdp_"))
    runtime = SpikelingRuntime(ast)
    runtime.learner = STDPLearner(rate=0.02, tau=20.0)
    for w in train_windows:
        run_window(runtime, w, SCALE)
    weights = [syn.weight for syn in runtime.synapses]

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
                out_lines.append(f"connect {src} -> {dst} weight={weight_by_pair[(src,dst)]:.4f}\n")
                continue
        out_lines.append(line)
    cand_path = os.path.join(SPIKELING_ROOT, "ai-apps", f"_tmp_ips_split{split_seed}.spk")
    with open(cand_path, "w", encoding="utf-8") as f:
        f.writelines(out_lines)

    acc_stdp = evaluate(held_out_sustained, blip_windows, SCALE,
                         f"STDP (real, split={split_seed})", spk_path=cand_path, verbose=verbose)
    acc_hand = evaluate(held_out_sustained, blip_windows, SCALE,
                         f"hand-tuned (split={split_seed})", spk_path=BACKUP_SPK, verbose=False)
    os.remove(cand_path)
    return acc_stdp, acc_hand, len(held_out_sustained), len(blip_windows)


if __name__ == "__main__":
    print("=" * 78)
    print(f"  {CHANNEL}: STDP trained on REAL data, multi-split validation")
    print("=" * 78)

    values = load_real_values(channel=CHANNEL)
    sustained_windows, blip_windows = extract_real_windows(
        values, elevated_threshold=ELEVATED_THRESHOLD, sustained_min_len=15,
        blip_max_len=8, sustained_window=20, pad=5)
    print(f"real sustained windows: {len(sustained_windows)}, real blip windows: {len(blip_windows)}")
    print(f"SCALE={SCALE}, elevated_threshold={ELEVATED_THRESHOLD}\n")

    if len(sustained_windows) < 3 or len(blip_windows) < 1:
        print("Too few real windows to validate meaningfully -- stopping honestly rather than "
              "forcing a result from an inadequate sample.")
        sys.exit(1)

    results = []
    for split_seed in [1, 2, 3, 4, 5]:
        acc_stdp, acc_hand, n_held_sus, n_blip = train_and_validate_split(
            sustained_windows, blip_windows, split_seed, verbose=(split_seed == 1))
        results.append((split_seed, acc_stdp, acc_hand))
        print(f"  split_seed={split_seed} (held-out: {n_held_sus} sustained + {n_blip} blip): "
              f"STDP(real)={acc_stdp:.1%}  hand-tuned={acc_hand:.1%}")

    print()
    print("=" * 78)
    print("SUMMARY across 5 real train/held-out splits:")
    wins = sum(1 for _, s, h in results if s > h)
    ties = sum(1 for _, s, h in results if s == h)
    losses = sum(1 for _, s, h in results if s < h)
    mean_stdp = np.mean([s for _, s, h in results])
    mean_hand = np.mean([h for _, s, h in results])
    print(f"  STDP(real) mean={mean_stdp:.1%}  hand-tuned mean={mean_hand:.1%}")
    print(f"  wins={wins}  ties={ties}  losses={losses}")
    print("=" * 78)
