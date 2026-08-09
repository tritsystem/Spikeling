#!/usr/bin/env python
"""
pyspike_cascade_gradient.py — first real multi-neuron step toward wiring
Forward-Forward into runtime.py for real: can gradients correctly flow
THROUGH a synapse cascade (neuron A fires -> injects into neuron B -> B
may fire too), matching runtime.py's actual _fire() propagation rule?

Deliberately isolates ONE new piece of complexity at a time (same
discipline as the whole FF debugging arc): this file does NOT yet combine
cascading with the FF goodness objective. It reuses the SIMPLER,
already-trusted target-spike-count objective from
pyspike_surrogate_gradient.py's own verification 2, now applied to
neuron B's output while training BOTH neurons' weights (w_A directly,
w_AB through the cascade) -- if backprop-through-cascade is correct,
gradient descent should train A's firing pattern to make B hit a target,
even though B receives no direct external input at all.

TWO NEURONS, refractory_ms=0, matching runtime.py's REAL rules exactly:

  A (input neuron, direct external drive x[t] . w_A):
    v_leaked_A = leak_toward_zero(v_prev_A, leak_A)
    v_pre_A    = v_leaked_A + drive_A[t]
    spike_A[t] = Heaviside(v_pre_A - threshold_A)
    v_next_A   = v_pre_A * (1 - spike_A[t])                    # hard reset

  B (downstream, synapse-only -- no direct external drive, matching how
     a real hidden/output-layer neuron in a multi-layer .spk network only
     receives synaptic input, never stimulate() directly):
    v_leaked_B = leak_toward_zero(v_prev_B, leak_B)             # ticks every step
    inject_B   = spike_A[t] * w_AB * 50.0                       # ONLY if A fired
    v_pre_B    = max(-threshold_B, v_leaked_B + inject_B)       # runtime.py's real clamp
    spike_B[t] = Heaviside(v_pre_B - threshold_B)
    v_next_B   = 0.0 if spike_B[t] else v_pre_B                 # hard reset on fire

Two verifications, same discipline as every prior script:
  1. Correctness: analytic gradient (w.r.t. BOTH w_A and w_AB) of a smooth
     two-neuron loss matches finite-differencing of that same smooth loss.
  2. Does it learn: gradient descent (hard forward, surrogate backward)
     trains A's firing pattern so B's spike COUNT approaches a target,
     even though B never receives external input directly -- proof the
     gradient genuinely propagates through the cascade, not just to A.

    python pyspike_cascade_gradient.py
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def surrogate_derivative(v_minus_threshold, k=4.0):
    s = sigmoid(k * v_minus_threshold)
    return k * s * (1.0 - s)


def leak_toward_zero(v, leak, smooth=False, beta_smooth=8.0):
    if not smooth:
        if v > 0.0:
            return max(0.0, v - leak)
        return min(0.0, v + leak)
    return np.logaddexp(0.0, beta_smooth * (v - leak)) / beta_smooth


def d_leak_toward_zero(v, leak):
    if v > leak:
        return 1.0
    if v >= 0.0:
        return 0.0
    if v > -leak:
        return 0.0
    return 1.0


def d_leak_toward_zero_smooth(v, leak, beta_smooth=8.0):
    return sigmoid(beta_smooth * (v - leak))


def floor_clamp(v, floor, smooth=False, beta_smooth=8.0):
    """runtime.py's max(-threshold, x) clamp on the cascade injection."""
    if not smooth:
        return max(floor, v)
    # smooth max(floor, v) = floor + softplus(beta*(v-floor))/beta
    return floor + np.logaddexp(0.0, beta_smooth * (v - floor)) / beta_smooth


def d_floor_clamp(v, floor):
    return 1.0 if v > floor else 0.0


def d_floor_clamp_smooth(v, floor, beta_smooth=8.0):
    return sigmoid(beta_smooth * (v - floor))


K = 4.0


def forward(x, w_A, w_AB, leak_A, leak_B, threshold_A, threshold_B, W_SYN, smooth_forward=False):
    """x: (T, n_inputs). w_A: (n_inputs,). w_AB, leak_*, threshold_*: scalars.
    Returns (spikes_A, spikes_B, cache)."""
    T = x.shape[0]
    v_prev_A, v_prev_B = 0.0, 0.0
    spikes_A = np.zeros(T)
    spikes_B = np.zeros(T)
    v_pre_A_hist = np.zeros(T)
    v_pre_B_hist = np.zeros(T)
    v_leaked_B_hist = np.zeros(T)
    v_prev_A_hist = np.zeros(T)   # state ENTERING each tick, needed for leak's own gradient
    v_prev_B_hist = np.zeros(T)

    for t in range(T):
        v_prev_A_hist[t] = v_prev_A
        v_prev_B_hist[t] = v_prev_B

        v_leaked_A = leak_toward_zero(v_prev_A, leak_A, smooth=smooth_forward)
        v_pre_A = v_leaked_A + float(x[t] @ w_A)
        v_pre_A_hist[t] = v_pre_A
        vtA = v_pre_A - threshold_A
        sA = sigmoid(K * vtA) if smooth_forward else (1.0 if vtA >= 0.0 else 0.0)
        spikes_A[t] = sA
        v_next_A = v_pre_A * (1.0 - sA)

        v_leaked_B = leak_toward_zero(v_prev_B, leak_B, smooth=smooth_forward)
        v_leaked_B_hist[t] = v_leaked_B
        inject_B = sA * w_AB * W_SYN
        v_pre_B_unclamped = v_leaked_B + inject_B
        v_pre_B = floor_clamp(v_pre_B_unclamped, -threshold_B, smooth=smooth_forward)
        v_pre_B_hist[t] = v_pre_B
        vtB = v_pre_B - threshold_B
        sB = sigmoid(K * vtB) if smooth_forward else (1.0 if vtB >= 0.0 else 0.0)
        spikes_B[t] = sB
        v_next_B = v_pre_B * (1.0 - sB)

        v_prev_A, v_prev_B = v_next_A, v_next_B

    cache = {
        "x": x, "w_A": w_A, "w_AB": w_AB, "leak_A": leak_A, "leak_B": leak_B,
        "threshold_A": threshold_A, "threshold_B": threshold_B, "W_SYN": W_SYN,
        "spikes_A": spikes_A, "spikes_B": spikes_B,
        "v_pre_A": v_pre_A_hist, "v_pre_B": v_pre_B_hist,
        "v_prev_A": v_prev_A_hist, "v_prev_B": v_prev_B_hist,
        "smooth_forward": smooth_forward,
    }
    return spikes_A, spikes_B, cache


def backward(cache, d_loss_d_spikes_B):
    """d_loss_d_spikes_B: (T,) external gradient w.r.t. spikes_B only (B is
    the one with a loss in this test; A has none directly -- its gradient
    must arrive ENTIRELY through the cascade, which is exactly what this
    verifies). Returns (d_w_A, d_w_AB)."""
    x = cache["x"]
    w_A, w_AB = cache["w_A"], cache["w_AB"]
    leak_A, leak_B = cache["leak_A"], cache["leak_B"]
    threshold_A, threshold_B = cache["threshold_A"], cache["threshold_B"]
    W_SYN = cache["W_SYN"]
    spikes_A, spikes_B = cache["spikes_A"], cache["spikes_B"]
    v_pre_A, v_pre_B = cache["v_pre_A"], cache["v_pre_B"]
    v_prev_A, v_prev_B = cache["v_prev_A"], cache["v_prev_B"]
    smooth = cache["smooth_forward"]
    leak_deriv = d_leak_toward_zero_smooth if smooth else d_leak_toward_zero
    floor_deriv = d_floor_clamp_smooth if smooth else d_floor_clamp

    T = x.shape[0]
    d_w_A = np.zeros_like(w_A)
    d_w_AB = 0.0
    d_vnext_A, d_vnext_B = 0.0, 0.0

    for t in reversed(range(T)):
        # ---- B's local gradients ----
        vtB = v_pre_B[t] - threshold_B
        d_sB_d_vpreB = surrogate_derivative(vtB, K)
        d_loss_d_vpreB = d_loss_d_spikes_B[t] * d_sB_d_vpreB
        if t + 1 < T:
            d_vnextB_d_vpreB = (1.0 - spikes_B[t]) - v_pre_B[t] * d_sB_d_vpreB
            d_loss_d_vpreB += d_vnext_B * d_vnextB_d_vpreB

        # v_pre_B = floor_clamp(v_leaked_B + inject_B, -threshold_B)
        v_leaked_B_t = leak_toward_zero(v_prev_B[t], leak_B, smooth=smooth)
        pre_clamp_B = v_leaked_B_t + spikes_A[t] * w_AB * W_SYN
        d_clamp = floor_deriv(pre_clamp_B, -threshold_B)
        d_loss_d_preclamp_B = d_loss_d_vpreB * d_clamp

        # path 1: -> inject_B -> w_AB and -> spikes_A[t] (SAME-tick cross-neuron path)
        d_w_AB += d_loss_d_preclamp_B * spikes_A[t] * W_SYN
        d_loss_d_sA_same_tick = d_loss_d_preclamp_B * w_AB * W_SYN

        # path 2: -> v_leaked_B -> v_prev_B[t] (B's own recurrence)
        d_leakB = leak_deriv(v_prev_B[t], leak_B)
        d_vnext_B = d_loss_d_preclamp_B * d_leakB

        # ---- A's local gradients: external loss (none) + same-tick cascade path + A's own recurrence ----
        vtA = v_pre_A[t] - threshold_A
        d_sA_d_vpreA = surrogate_derivative(vtA, K)
        d_loss_d_vpreA = d_loss_d_sA_same_tick * d_sA_d_vpreA   # ONLY path into A this tick (no external loss on A)
        if t + 1 < T:
            d_vnextA_d_vpreA = (1.0 - spikes_A[t]) - v_pre_A[t] * d_sA_d_vpreA
            d_loss_d_vpreA += d_vnext_A * d_vnextA_d_vpreA
        d_w_A += d_loss_d_vpreA * x[t]
        d_leakA = leak_deriv(v_prev_A[t], leak_A)
        d_vnext_A = d_loss_d_vpreA * d_leakA

    return d_w_A, d_w_AB


def _selftest_backprop_matches_finite_difference():
    rng = np.random.default_rng(0)
    N = 6
    T = 10
    x = rng.normal(0, 1.0, size=(T, N))
    w_A = rng.normal(0.3, 0.15, size=N)
    w_AB = 0.8
    leak_A, leak_B = 0.05, 0.05
    threshold_A, threshold_B = 1.0, 1.0
    W_SYN = 50.0

    def loss_fn(w_A_, w_AB_):
        _, sB, _ = forward(x, w_A_, w_AB_, leak_A, leak_B, threshold_A, threshold_B, W_SYN, smooth_forward=True)
        target = np.ones(T) * 0.5
        return float(np.sum((sB - target) ** 2))

    spikes_A, spikes_B, cache = forward(x, w_A, w_AB, leak_A, leak_B, threshold_A, threshold_B, W_SYN, smooth_forward=True)
    target = np.ones(T) * 0.5
    d_loss_d_sB = 2 * (spikes_B - target)
    analytic_dwA, analytic_dwAB = backward(cache, d_loss_d_sB)

    eps = 1e-5
    numeric_dwA = np.zeros(N)
    for i in range(N):
        wp, wm = w_A.copy(), w_A.copy()
        wp[i] += eps
        wm[i] -= eps
        numeric_dwA[i] = (loss_fn(wp, w_AB) - loss_fn(wm, w_AB)) / (2 * eps)
    numeric_dwAB = (loss_fn(w_A, w_AB + eps) - loss_fn(w_A, w_AB - eps)) / (2 * eps)

    max_diff_A = float(np.max(np.abs(analytic_dwA - numeric_dwA)))
    diff_AB = float(abs(analytic_dwAB - numeric_dwAB))
    print(f"    d_w_A   analytic: {np.round(analytic_dwA, 5)}")
    print(f"    d_w_A   numeric:  {np.round(numeric_dwA, 5)}")
    print(f"    max abs diff (w_A):  {max_diff_A:.2e}")
    print(f"    d_w_AB  analytic: {analytic_dwAB:.6f}  numeric: {numeric_dwAB:.6f}  diff: {diff_AB:.2e}")
    ok = max_diff_A < 1e-3 and diff_AB < 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient through the 2-neuron cascade matches "
          f"finite-difference (both w_A and w_AB, including the cross-neuron path)")


def _selftest_training_converges():
    rng = np.random.default_rng(1)
    N = 8
    T = 25
    x = (rng.random((T, N)) < 0.35).astype(float)
    leak_A, leak_B = 0.05, 0.05
    threshold_A, threshold_B = 1.0, 1.0
    W_SYN = 50.0
    w_A = rng.normal(0.15, 0.08, size=N)
    w_AB = 0.5
    lr = 0.02
    target_rate = 0.4
    target_count = round(target_rate * T)

    def loss_of(spikes_B):
        return (float(spikes_B.sum()) - target_count) ** 2

    spikes_A0, spikes_B0, _ = forward(x, w_A, w_AB, leak_A, leak_B, threshold_A, threshold_B, W_SYN, smooth_forward=False)
    loss0 = loss_of(spikes_B0)
    countB0 = int(spikes_B0.sum())
    countA0 = int(spikes_A0.sum())

    for step in range(400):
        spikes_A, spikes_B, cache = forward(x, w_A, w_AB, leak_A, leak_B, threshold_A, threshold_B, W_SYN, smooth_forward=False)
        d_loss_d_sB = np.full(T, 2.0 * (spikes_B.sum() - target_count))
        d_w_A, d_w_AB = backward(cache, d_loss_d_sB)
        w_A = w_A - lr * d_w_A
        w_AB = w_AB - lr * d_w_AB

    spikes_A_f, spikes_B_f, _ = forward(x, w_A, w_AB, leak_A, leak_B, threshold_A, threshold_B, W_SYN, smooth_forward=False)
    loss_final = loss_of(spikes_B_f)
    countB_f = int(spikes_B_f.sum())
    countA_f = int(spikes_A_f.sum())

    print(f"    initial: A fired {countA0}/{T}, B fired {countB0}/{T} (target~{target_count}), loss={loss0:.3f}")
    print(f"    final:   A fired {countA_f}/{T}, B fired {countB_f}/{T} (target~{target_count}), loss={loss_final:.3f}")
    print(f"    final w_AB={w_AB:.4f} (started 0.5)")
    ok = loss_final < loss0 * 0.5 and abs(countB_f - target_count) <= abs(countB0 - target_count)
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient descent through the cascade trains A's firing "
          f"pattern to move B's spike count toward the target -- B never receives external input "
          f"directly, so this only works if the cross-neuron gradient path is real")


if __name__ == "__main__":
    print("=" * 78)
    print("  CASCADE GRADIENT -- backprop through a real 2-neuron synapse cascade")
    print("  (runtime.py's actual propagation rule: fire -> inject downstream -> maybe fire)")
    print("=" * 78)
    _selftest_backprop_matches_finite_difference()
    print()
    _selftest_training_converges()
