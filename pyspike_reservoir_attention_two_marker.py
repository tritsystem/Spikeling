#!/usr/bin/env python
"""
pyspike_reservoir_attention_two_marker.py -- generalization test for the
velocity-feature + lr fix verified in pyspike_reservoir_attention_hybrid.py.

That fix solved a SINGLE-marker task (find the one outlier, report K
steps later). This is a genuinely harder variant: TWO distinct markers
(+3.0 and -3.0) appear in every sequence at random positions; the model
must report the value K steps after the +3.0 marker SPECIFICALLY, not
just after "any" outlier. This requires content-based SELECTION by
identity, not just outlier detection -- a real test of whether the
velocity fix generalizes or was a single-task fluke.

Reuses ReservoirBank, ternary_ste, spike, and ReservoirAttentionReadout
UNCHANGED from pyspike_reservoir_attention_hybrid.py -- if the same
architecture (no modification) solves this harder task, that is a real
generalization result, not a retuned one.
"""
import numpy as np
import torch
import torch.nn.functional as F

from pyspike_reservoir_attention_hybrid import (
    DEVICE, T, K_LAG, M_RESERVOIR, ReservoirBank, ReservoirAttentionReadout,
    LinearBaseline, run_stage,
)


def make_two_marker_task(rng, n_samples):
    """u: (n_samples, T). Two markers per sequence: +3.0 (the TARGET
    marker) and -3.0 (a distractor), at random distinct positions, both
    far enough from the edges that +K_LAG stays in-bounds. target =
    value K_LAG steps after the +3.0 marker specifically."""
    u = rng.uniform(-1, 1, size=(n_samples, T)).astype(np.float32)
    pos_lo, pos_hi = 5, T - K_LAG - 1
    target_pos = rng.integers(pos_lo, pos_hi, size=n_samples)
    distractor_pos = rng.integers(pos_lo, pos_hi, size=n_samples)
    # resample any collisions so the two markers never land on the same tick
    collide = target_pos == distractor_pos
    while collide.any():
        distractor_pos[collide] = rng.integers(pos_lo, pos_hi, size=collide.sum())
        collide = target_pos == distractor_pos

    for i in range(n_samples):
        u[i, target_pos[i]] = 3.0
        u[i, distractor_pos[i]] = -3.0
    target = np.array([u[i, target_pos[i] + K_LAG] for i in range(n_samples)], dtype=np.float32)
    return u, target, target_pos


if __name__ == "__main__":
    print("=" * 78)
    print("  GENERALIZATION TEST: two-marker selection-by-identity")
    print(f"  (same architecture as pyspike_reservoir_attention_hybrid.py, unmodified)")
    print("=" * 78)

    rng = np.random.default_rng(0)
    train_u, train_y, _ = make_two_marker_task(rng, 800)
    test_u, test_y, test_marker_pos = make_two_marker_task(rng, 200)

    reservoir = ReservoirBank(M_RESERVOIR).to(DEVICE)
    FEAT = 2 * M_RESERVOIR

    print("\nSTAGE 1: linear baseline")
    torch.manual_seed(0)
    baseline = LinearBaseline(FEAT).to(DEVICE)
    nmse1 = run_stage("linear readout (baseline)", baseline, reservoir, train_u, train_y, test_u, test_y, is_linear_baseline=True)

    print("\nSTAGE 2: full-precision two-hop attention")
    torch.manual_seed(0)
    attn_model = ReservoirAttentionReadout(FEAT, use_ternary=False, use_spiking=False).to(DEVICE)
    nmse2 = run_stage("attention (full precision)", attn_model, reservoir, train_u, train_y, test_u, test_y)
    with torch.no_grad():
        test_states = reservoir(torch.tensor(test_u, device=DEVICE))
        _, attn1 = attn_model(test_states)
        found_pos = attn1.argmax(dim=-1).cpu().numpy()
        loc_acc = (found_pos == test_marker_pos).mean() * 100
        print(f"  hop-1 CORRECT-marker localization accuracy: {loc_acc:.2f}%")

    print("\nSTAGE 3: ternary (lr=0.003, the fix verified on the single-marker task)")
    torch.manual_seed(0)
    ternary_model = ReservoirAttentionReadout(FEAT, use_ternary=True, use_spiking=False).to(DEVICE)
    nmse3 = run_stage("attention (ternary QAT)", ternary_model, reservoir, train_u, train_y, test_u, test_y, lr=0.003)

    print("\nSTAGE 4: ternary + spiking")
    torch.manual_seed(0)
    full_model = ReservoirAttentionReadout(FEAT, use_ternary=True, use_spiking=True).to(DEVICE)
    nmse4 = run_stage("attention (ternary + spiking)", full_model, reservoir, train_u, train_y, test_u, test_y, lr=0.003)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"  Stage 1 (linear, no attention):   NMSE = {nmse1:.4f}")
    print(f"  Stage 2 (+ attention, fp32):       NMSE = {nmse2:.4f}")
    print(f"  Stage 3 (+ ternary):               NMSE = {nmse3:.4f}")
    print(f"  Stage 4 (+ ternary + spiking):     NMSE = {nmse4:.4f}")
