#!/usr/bin/env python
"""
pyspike_cascade_converging.py — third multi-neuron step: two upstream
neurons (B, C) converging onto ONE downstream neuron D (the mirror of
pyspike_cascade_branching.py's fan-out case).

A REAL WRINKLE THIS CASE HAS THAT FAN-OUT DIDN'T: runtime.py's _fire()
propagates along ONE firing neuron's synapses at a time, sequentially --
it is NOT "sum all incoming injections, then check threshold once."
Within one tick, if B is processed before C (matching how a real caller
would sequentially stimulate() B then C), D's membrane potential can
already cross threshold and RESET from B's injection alone, before C's
injection is even applied -- and `fire_count` genuinely increments per
firing EVENT, so D can fire twice in a single tick if both injections
independently cross threshold. This file models that faithfully, not a
simplified simultaneous-sum version, because the whole point of this
series is matching runtime.py's REAL mechanics, not a convenient
approximation of them.

TWO NEURONS -> ONE, refractory_ms=0, sequential order B-then-C within
each tick:

  B, C: independent input neurons, each own direct external drive
        (same LIF dynamics as every prior script).
  D:    downstream, receives B's injection FIRST (may fire + reset),
        THEN C's injection into whatever D's state now is (may fire
        again). D's spike COUNT for tick t = (did B's injection fire D)
        + (did C's injection fire D) -- 0, 1, or 2, matching real
        fire_count semantics exactly, not a per-tick boolean.

Two verifications, same discipline as every prior script:
  1. Correctness: analytic gradient (w_B, w_C, w_BD, w_CD) vs finite-
     difference on a smooth loss, INCLUDING the two-stage sequential
     injection/reset within a single tick.
  2. Does it learn: gradient descent trains BOTH upstream neurons'
     firing patterns (and both synapse weights) to move D's total
     spike-event count toward a target.

    python pyspike_cascade_converging.py
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def surrogate_derivative(v_minus_threshold, k):
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
    if not smooth:
        return max(floor, v)
    return floor + np.logaddexp(0.0, beta_smooth * (v - floor)) / beta_smooth


def d_floor_clamp(v, floor):
    return 1.0 if v > floor else 0.0


def d_floor_clamp_smooth(v, floor, beta_smooth=8.0):
    return sigmoid(beta_smooth * (v - floor))


# DIAGNOSED already in pyspike_cascade_branching.py: K must be scaled to
# the real injection magnitude (weight*W_SYN), not borrowed from the
# smaller-scale single-neuron scripts, or synapse-weight gradients
# underflow to exactly zero. Reusing that fix here directly.
K = 0.1
W_SYN = 50.0


def forward(x_B, x_C, w_B, w_C, w_BD, w_CD, leak, threshold, smooth_forward=False):
    """x_B, x_C: (T, n_inputs) each. Returns (spikes_B, spikes_C,
    d_events (T,) -- D's spike-event COUNT per tick, 0/1/2 -- cache)."""
    T = x_B.shape[0]
    v_prev = {"B": 0.0, "C": 0.0, "D": 0.0}
    hist = {n: {"spikes": np.zeros(T), "v_pre": np.zeros(T), "v_prev": np.zeros(T)} for n in "BC"}
    # D has TWO sub-events per tick (after B's injection, after C's injection)
    hist_D = {
        "v_prev": np.zeros(T),            # D's state entering the tick (before B's injection)
        "v_pre_afterB": np.zeros(T),      # D's clamped potential right after B's injection
        "spike_afterB": np.zeros(T),
        "v_afterB_reset": np.zeros(T),    # D's state after B's sub-event resolves
        "v_pre_afterC": np.zeros(T),
        "spike_afterC": np.zeros(T),
    }
    d_events = np.zeros(T)

    for t in range(T):
        for n, x in (("B", x_B), ("C", x_C)):
            hist[n]["v_prev"][t] = v_prev[n]
            v_leaked = leak_toward_zero(v_prev[n], leak, smooth=smooth_forward)
            w = w_B if n == "B" else w_C
            v_pre = v_leaked + float(x[t] @ w)
            hist[n]["v_pre"][t] = v_pre
            vt = v_pre - threshold
            s = sigmoid(K_NEURON * vt) if smooth_forward else (1.0 if vt >= 0.0 else 0.0)
            hist[n]["spikes"][t] = s
            v_prev[n] = v_pre * (1.0 - s)

        sB, sC = hist["B"]["spikes"][t], hist["C"]["spikes"][t]

        # ---- D, sub-event 1: B's injection ----
        hist_D["v_prev"][t] = v_prev["D"]
        v_leaked_D = leak_toward_zero(v_prev["D"], leak, smooth=smooth_forward)
        pre_afterB = v_leaked_D + sB * w_BD * W_SYN
        v_afterB = floor_clamp(pre_afterB, -threshold, smooth=smooth_forward)
        hist_D["v_pre_afterB"][t] = v_afterB
        vtB = v_afterB - threshold
        sD_B = sigmoid(K * vtB) if smooth_forward else (1.0 if vtB >= 0.0 else 0.0)
        hist_D["spike_afterB"][t] = sD_B
        v_after_reset = v_afterB * (1.0 - sD_B)
        hist_D["v_afterB_reset"][t] = v_after_reset

        # ---- D, sub-event 2: C's injection, into whatever D is NOW ----
        pre_afterC = v_after_reset + sC * w_CD * W_SYN   # no separate leak call between sub-events (same tick)
        v_afterC = floor_clamp(pre_afterC, -threshold, smooth=smooth_forward)
        hist_D["v_pre_afterC"][t] = v_afterC
        vtC = v_afterC - threshold
        sD_C = sigmoid(K * vtC) if smooth_forward else (1.0 if vtC >= 0.0 else 0.0)
        hist_D["spike_afterC"][t] = sD_C
        v_prev["D"] = v_afterC * (1.0 - sD_C)

        d_events[t] = sD_B + sD_C

    cache = {"x_B": x_B, "x_C": x_C, "w_B": w_B, "w_C": w_C, "w_BD": w_BD, "w_CD": w_CD,
              "leak": leak, "threshold": threshold, "hist": hist, "hist_D": hist_D,
              "smooth_forward": smooth_forward}
    return hist["B"]["spikes"], hist["C"]["spikes"], d_events, cache


K_NEURON = 4.0  # upstream neurons' OWN surrogate -- their driving current is a normal small-scale dot product, K=4 is fine there; only the SYNAPSE-injection-scale nonlinearities need K=0.1


def backward(cache, d_loss_d_devents):
    """d_loss_d_devents: (T,) external gradient w.r.t. D's per-tick event
    count. Returns (d_w_B, d_w_C, d_w_BD, d_w_CD)."""
    x_B, x_C = cache["x_B"], cache["x_C"]
    w_B, w_C, w_BD, w_CD = cache["w_B"], cache["w_C"], cache["w_BD"], cache["w_CD"]
    leak, threshold = cache["leak"], cache["threshold"]
    hist, hist_D, smooth = cache["hist"], cache["hist_D"], cache["smooth_forward"]
    leak_deriv = d_leak_toward_zero_smooth if smooth else d_leak_toward_zero
    floor_deriv = d_floor_clamp_smooth if smooth else d_floor_clamp

    T = x_B.shape[0]
    d_w_B, d_w_C = np.zeros_like(w_B), np.zeros_like(w_C)
    d_w_BD, d_w_CD = 0.0, 0.0
    d_vnext = {"B": 0.0, "C": 0.0, "D": 0.0}

    for t in reversed(range(T)):
        # d_events[t] = sD_B[t] + sD_C[t] -- external gradient splits equally onto both sub-events
        d_loss_d_sDB_ext = d_loss_d_devents[t]
        d_loss_d_sDC_ext = d_loss_d_devents[t]

        # ---- sub-event 2 (C's injection) backward ----
        v_afterC = hist_D["v_pre_afterC"][t]
        sD_C = hist_D["spike_afterC"][t]
        d_sDC_d_vafterC = surrogate_derivative(v_afterC - threshold, K)
        d_loss_d_vafterC = d_loss_d_sDC_ext * d_sDC_d_vafterC
        if t + 1 < T:
            d_vnextD_d_vafterC = (1.0 - sD_C) - v_afterC * d_sDC_d_vafterC
            d_loss_d_vafterC += d_vnext["D"] * d_vnextD_d_vafterC

        v_after_reset = hist_D["v_afterB_reset"][t]
        sC = hist["C"]["spikes"][t]
        pre_clampC = v_after_reset + sC * w_CD * W_SYN
        d_clampC = floor_deriv(pre_clampC, -threshold)
        d_loss_d_preclampC = d_loss_d_vafterC * d_clampC

        d_w_CD += d_loss_d_preclampC * sC * W_SYN
        d_loss_d_sC_from_D = d_loss_d_preclampC * w_CD * W_SYN
        # gradient flowing back into v_after_reset (sub-event 1's output)
        d_loss_d_vafterreset = d_loss_d_preclampC * 1.0  # d(pre_clampC)/d(v_after_reset) = 1

        # ---- sub-event 1 (B's injection) backward ----
        v_afterB = hist_D["v_pre_afterB"][t]
        sD_B = hist_D["spike_afterB"][t]
        d_sDB_d_vafterB = surrogate_derivative(v_afterB - threshold, K)
        # v_after_reset = v_afterB * (1 - sD_B) -- product rule, same shape as every prior reset
        d_vafterreset_d_vafterB = (1.0 - sD_B) - v_afterB * d_sDB_d_vafterB
        d_loss_d_vafterB = d_loss_d_sDB_ext * d_sDB_d_vafterB + d_loss_d_vafterreset * d_vafterreset_d_vafterB

        v_prev_D_t = hist_D["v_prev"][t]
        sB = hist["B"]["spikes"][t]
        v_leaked_D_t = leak_toward_zero(v_prev_D_t, leak, smooth=smooth)
        pre_clampB = v_leaked_D_t + sB * w_BD * W_SYN
        d_clampB = floor_deriv(pre_clampB, -threshold)
        d_loss_d_preclampB = d_loss_d_vafterB * d_clampB

        d_w_BD += d_loss_d_preclampB * sB * W_SYN
        d_loss_d_sB_from_D = d_loss_d_preclampB * w_BD * W_SYN

        d_leakD = leak_deriv(v_prev_D_t, leak)
        d_vnext["D"] = d_loss_d_preclampB * d_leakD

        # ---- B's and C's own local gradients (external loss only via D) ----
        for name, d_loss_d_s_from_D, w_local, x_local, dwacc in (
            ("B", d_loss_d_sB_from_D, w_B, x_B, "B"), ("C", d_loss_d_sC_from_D, w_C, x_C, "C")
        ):
            v_pre = hist[name]["v_pre"][t]
            s = hist[name]["spikes"][t]
            d_s_d_vpre = surrogate_derivative(v_pre - threshold, K_NEURON)
            d_loss_d_vpre = d_loss_d_s_from_D * d_s_d_vpre
            if t + 1 < T:
                d_vnext_d_vpre = (1.0 - s) - v_pre * d_s_d_vpre
                d_loss_d_vpre += d_vnext[name] * d_vnext_d_vpre
            if name == "B":
                d_w_B += d_loss_d_vpre * x_B[t]
            else:
                d_w_C += d_loss_d_vpre * x_C[t]
            d_leak_local = leak_deriv(hist[name]["v_prev"][t], leak)
            d_vnext[name] = d_loss_d_vpre * d_leak_local

    return d_w_B, d_w_C, d_w_BD, d_w_CD


def _selftest_backprop_matches_finite_difference():
    rng = np.random.default_rng(0)
    N, T = 5, 10
    x_B = rng.normal(0, 1.0, size=(T, N))
    x_C = rng.normal(0, 1.0, size=(T, N))
    w_B = rng.normal(0.3, 0.15, size=N)
    w_C = rng.normal(0.3, 0.15, size=N)
    w_BD, w_CD = 0.8, -0.5
    leak, threshold = 0.05, 1.0

    def loss_fn(w_B_, w_C_, w_BD_, w_CD_):
        _, _, d_events, _ = forward(x_B, x_C, w_B_, w_C_, w_BD_, w_CD_, leak, threshold, smooth_forward=True)
        target = np.ones(T) * 0.7
        return float(np.sum((d_events - target) ** 2))

    sB, sC, d_events, cache = forward(x_B, x_C, w_B, w_C, w_BD, w_CD, leak, threshold, smooth_forward=True)
    target = np.ones(T) * 0.7
    d_loss_d_devents = 2 * (d_events - target)
    a_dwB, a_dwC, a_dwBD, a_dwCD = backward(cache, d_loss_d_devents)

    eps = 1e-5
    n_dwB = np.zeros(N)
    for i in range(N):
        wp, wm = w_B.copy(), w_B.copy()
        wp[i] += eps
        wm[i] -= eps
        n_dwB[i] = (loss_fn(wp, w_C, w_BD, w_CD) - loss_fn(wm, w_C, w_BD, w_CD)) / (2 * eps)
    n_dwC = np.zeros(N)
    for i in range(N):
        wp, wm = w_C.copy(), w_C.copy()
        wp[i] += eps
        wm[i] -= eps
        n_dwC[i] = (loss_fn(w_B, wp, w_BD, w_CD) - loss_fn(w_B, wm, w_BD, w_CD)) / (2 * eps)
    n_dwBD = (loss_fn(w_B, w_C, w_BD + eps, w_CD) - loss_fn(w_B, w_C, w_BD - eps, w_CD)) / (2 * eps)
    n_dwCD = (loss_fn(w_B, w_C, w_BD, w_CD + eps) - loss_fn(w_B, w_C, w_BD, w_CD - eps)) / (2 * eps)

    diffB = float(np.max(np.abs(a_dwB - n_dwB)))
    diffC = float(np.max(np.abs(a_dwC - n_dwC)))
    diffBD = float(abs(a_dwBD - n_dwBD))
    diffCD = float(abs(a_dwCD - n_dwCD))
    print(f"    max abs diff w_B:  {diffB:.2e}   w_C: {diffC:.2e}")
    print(f"    w_BD analytic={a_dwBD:.6f} numeric={n_dwBD:.6f} diff={diffBD:.2e}")
    print(f"    w_CD analytic={a_dwCD:.6f} numeric={n_dwCD:.6f} diff={diffCD:.2e}")
    ok = diffB < 1e-3 and diffC < 1e-3 and diffBD < 1e-3 and diffCD < 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient through a CONVERGING cascade (B -> D <- C), "
          f"including the sequential two-sub-event same-tick reset, matches finite-difference "
          f"on all four weights")


def _selftest_training_converges():
    rng = np.random.default_rng(1)
    N, T = 8, 25
    x_B = (rng.random((T, N)) < 0.35).astype(float)
    x_C = (rng.random((T, N)) < 0.35).astype(float)
    leak, threshold = 0.05, 1.0
    w_B = rng.normal(0.15, 0.08, size=N)
    w_C = rng.normal(0.15, 0.08, size=N)
    w_BD, w_CD = 0.5, 0.5
    lr = 0.001
    target_total = 20   # D's total spike-EVENT count across all T ticks (can exceed T, since D can double-fire)

    def loss_of(d_events):
        return (float(d_events.sum()) - target_total) ** 2

    sB0, sC0, d0, _ = forward(x_B, x_C, w_B, w_C, w_BD, w_CD, leak, threshold, smooth_forward=False)
    loss0 = loss_of(d0)
    total0 = int(d0.sum())

    for step in range(500):
        sB, sC, d_events, cache = forward(x_B, x_C, w_B, w_C, w_BD, w_CD, leak, threshold, smooth_forward=False)
        d_loss_d_devents = np.full(T, 2.0 * (d_events.sum() - target_total))
        d_w_B, d_w_C, d_w_BD, d_w_CD = backward(cache, d_loss_d_devents)
        w_B = w_B - lr * d_w_B
        w_C = w_C - lr * d_w_C
        w_BD = w_BD - lr * d_w_BD
        w_CD = w_CD - lr * d_w_CD

    sB_f, sC_f, d_f, _ = forward(x_B, x_C, w_B, w_C, w_BD, w_CD, leak, threshold, smooth_forward=False)
    loss_final = loss_of(d_f)
    total_f = int(d_f.sum())

    print(f"    initial: D total spike-events={total0} (target {target_total}), loss={loss0:.3f}")
    print(f"    final:   D total spike-events={total_f} (target {target_total}), loss={loss_final:.3f}")
    ok = loss_final < loss0 * 0.5 and abs(total_f - target_total) < abs(total0 - target_total)
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient descent through a converging cascade (two independent "
          f"upstream neurons, sequential same-tick injection into one downstream neuron) trains D's "
          f"total spike-event count toward the target")


if __name__ == "__main__":
    print("=" * 78)
    print("  CASCADE CONVERGING -- two upstream neurons feeding one downstream neuron")
    print("  (sequential same-tick injection + possible double-fire, matching runtime.py exactly)")
    print("=" * 78)
    _selftest_backprop_matches_finite_difference()
    print()
    _selftest_training_converges()
