#!/usr/bin/env python
"""
verify_nonneg_layernorm_fix.py -- verifies the LayerNorm fix added to
ReservoirAttentionReadout in pyspike_reservoir_attention_hybrid.py
(2026-07-31 patch, see the class docstring/comment there for the full
diagnosis) actually closes the gap between the original signed [-1,1]
task (NMSE 0.118) and the amplitude-only [0,1] task variant used by
pyspike_reservoir_attention_acoustic_groundtruth.py (NMSE 0.72,
undertrained).

Ruled out via a pure-numpy physics replay (no training needed): marker
contrast between the two tasks is nearly identical (~3.28 vs ~3.36), so
this script exists to confirm the actual fix (normalizing reservoir_states
before attention, since the nonneg task's states are ~30% smaller in
absolute magnitude at the same fixed weight-init/lr) rather than just
re-running with more epochs, which would leave any FUTURE task variant
with a different input scale exposed to the same undertraining.

Run:
    pip install torch --index-url https://download.pytorch.org/whl/cpu
    python3 verify_nonneg_layernorm_fix.py

Does NOT need audio hardware/sounddevice -- make_task_nonneg is copied
in locally rather than importing the acoustic groundtruth script.
"""
import numpy as np
import torch

from pyspike_reservoir_attention_hybrid import (
    DEVICE, T, K_LAG, M_RESERVOIR, ReservoirBank, ReservoirAttentionReadout,
    run_stage, make_task,
)


def make_task_nonneg(rng, n_samples):
    """Copied from pyspike_reservoir_attention_acoustic_groundtruth.py
    (not imported directly -- that file imports sounddevice and assumes
    Windows-specific audio device indices that won't exist on every
    machine this verification script might run on)."""
    u = rng.uniform(0.0, 1.0, size=(n_samples, T)).astype(np.float32)
    marker_pos = rng.integers(5, T - K_LAG - 1, size=n_samples)
    for i in range(n_samples):
        u[i, marker_pos[i]] = 2.5
    target = np.array([u[i, marker_pos[i] + K_LAG] for i in range(n_samples)], dtype=np.float32)
    return u, target, marker_pos


def eval_task(label, task_fn, expected_before):
    rng = np.random.default_rng(0)
    train_u, train_y, _ = task_fn(rng, 800)
    test_u, test_y, _ = task_fn(rng, 200)

    reservoir = ReservoirBank(M_RESERVOIR).to(DEVICE)
    FEAT = 2 * M_RESERVOIR

    torch.manual_seed(0)
    model = ReservoirAttentionReadout(FEAT, use_ternary=False, use_spiking=False).to(DEVICE)
    nmse = run_stage(f"{label} (WITH LayerNorm fix)", model, reservoir, train_u, train_y, test_u, test_y)

    print(f"  reference (pre-fix, recorded 2026-07-31): NMSE = {expected_before:.4f}")
    improved = nmse < expected_before
    print(f"  {'IMPROVED' if improved else 'DID NOT IMPROVE'} "
          f"({expected_before:.4f} -> {nmse:.4f}, "
          f"{(1 - nmse/expected_before)*100:+.1f}% change)\n")
    return nmse


if __name__ == "__main__":
    print("=" * 78)
    print("VERIFYING: does normalizing reservoir_states fix the nonneg-task gap?")
    print("=" * 78)

    print("\n--- Original signed [-1,1] task (sanity check: fix shouldn't hurt this) ---")
    nmse_signed = eval_task("signed task", make_task, expected_before=0.118)

    print("--- Nonneg amplitude-only [0,1] task (the one that was undertrained) ---")
    nmse_nonneg = eval_task("nonneg task", make_task_nonneg, expected_before=0.72)

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  signed task NMSE : {nmse_signed:.4f}  (reference 0.118, pre-fix)")
    print(f"  nonneg task NMSE : {nmse_nonneg:.4f}  (reference 0.72, pre-fix)")
    if nmse_nonneg < 0.2:
        print("\n  PASS -- nonneg task now trains close to the signed task's level.")
        print("  Safe to re-run pyspike_reservoir_attention_acoustic_groundtruth.py")
        print("  for a real, trustworthy ground-truth NMSE on real audio hardware.")
    else:
        print("\n  Still elevated -- LayerNorm alone did not fully close the gap.")
        print("  Next: lr sweep specifically for the nonneg task, or check init")
        print("  scale of Wk/Wv/Wo relative to LayerNorm's unit-variance output.")
