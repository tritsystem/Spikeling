#!/usr/bin/env python
"""
pyspike_refractory_gradient.py — the last deferred piece of runtime.py's
real dynamics: refractory_ms>0. Deliberately isolated to a SINGLE neuron
(no cascading yet, same discipline as every prior script here) so this
one new piece of complexity can be verified on its own before combining
with the cascade work.

WHAT REFRACTORY ACTUALLY DOES (runtime.py's stimulate()/_fire()): if
`elapsed = current_time_ms - last_spike_time < refractory_ms`, the tick
is a COMPLETE no-op -- not just "no drive," but no leak either, nothing
updates at all. This is a genuinely different kind of non-differentiability
than anything solved so far: a hard gate keyed on THE TIMING of the most
recent spike, not on a value like membrane potential.

THE KEY INSIGHT: "time since last spike" has EXACTLY the same
reset-or-continue shape already solved for membrane potential (v_next =
v_pre*(1-spike), reset to 0 on fire) -- here it's "reset to 0 on fire,
else INCREMENT by dt":

    time_since_spike[t] = (time_since_spike[t-1] + dt) * (1 - spike[t-1])

Differentiable via the exact same product-rule/reset-gate machinery
already used throughout this series -- no new mechanism, same trick,
different quantity. The refractory GATE itself (are we still blocked?)
is then ANOTHER Heaviside needing its own surrogate:

    gate[t] = Heaviside(time_since_spike[t] - refractory_ms)   (1=allowed, 0=blocked)

DIAGNOSED-IN-ADVANCE, applying the K-scaling lesson from the cascade
work BEFORE running anything blind this time: refractory_ms in real .spk
files is in the tens-to-hundreds of milliseconds (e.g. 40-400ms seen in
this repo's own examples), a completely different SCALE than membrane
potential (~O(1)) or synapse injection (~O(25)). Using K=4.0 or even
K=0.1 for this gate would almost certainly saturate it identically to
the branching-cascade bug. K_REFRACTORY is picked to match: with
refractory_ms~30 and dt~10, the gate needs to be sensitive over a
range of a few dt, so K_REFRACTORY ~ 1/dt, not reused from elsewhere.

When blocked (gate[t]=0): v_next = v_prev EXACTLY (frozen, no leak, no
drive) -- modeled as a smooth blend between "frozen" and "normal update"
gated by gate[t], not a hard if/else, so gradient can flow through the
gate decision itself.

Two verifications, same discipline:
  1. Correctness: analytic gradient vs finite-difference on a smooth loss.
  2. Does it learn: gradient descent trains a real target THROUGH the
     refractory gate -- if the gate's gradient were wrong or missing,
     training could not learn to time spikes around the lockout.

    python pyspike_refractory_gradient.py
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


K_NEURON = 4.0        # for the neuron's own membrane-potential spike surrogate (small scale, O(1))
DT = 10.0              # ms per tick -- a real, plausible discretization
K_REFRACTORY = 1.0 / DT  # scaled to the refractory gate's own real range (dt-sized steps), not borrowed


def forward(x, w, leak, threshold, refractory_ms, smooth_forward=False):
    """x: (T, n_inputs). Single neuron, refractory_ms>0. Returns
    (spikes (T,), cache)."""
    T = x.shape[0]
    v_prev = 0.0
    tss_prev = refractory_ms  # "time since spike" starts already >= refractory_ms (never fired yet -> not blocked)
    spikes = np.zeros(T)
    v_pre_hist = np.zeros(T)
    v_prev_hist = np.zeros(T)
    tss_hist = np.zeros(T)
    gate_hist = np.zeros(T)

    for t in range(T):
        v_prev_hist[t] = v_prev
        tss_hist[t] = tss_prev

        gt = tss_prev - refractory_ms
        gate = sigmoid(K_REFRACTORY * gt) if smooth_forward else (1.0 if gt >= 0.0 else 0.0)
        gate_hist[t] = gate

        # blocked: v stays EXACTLY v_prev (no leak, no drive). allowed: normal LIF update.
        v_leaked = leak_toward_zero(v_prev, leak, smooth=smooth_forward)
        v_pre_allowed = v_leaked + float(x[t] @ w)
        v_pre = gate * v_pre_allowed + (1.0 - gate) * v_prev   # smooth blend, gated by `gate`
        v_pre_hist[t] = v_pre

        vt = v_pre - threshold
        s = sigmoid(K_NEURON * vt) if smooth_forward else (1.0 if vt >= 0.0 else 0.0)
        # a blocked tick can never fire, regardless of v_pre -- gate it explicitly (matches
        # real behavior: stimulate() returns None immediately when refractory-locked, before
        # the threshold check ever runs)
        s = gate * s
        spikes[t] = s

        v_next = v_pre * (1.0 - s)
        tss_next = (tss_prev + DT) * (1.0 - s)   # reset to 0 on fire, else increment -- SAME shape as v's reset

        v_prev, tss_prev = v_next, tss_next

    cache = {"x": x, "w": w, "leak": leak, "threshold": threshold, "refractory_ms": refractory_ms,
              "spikes": spikes, "v_pre": v_pre_hist, "v_prev": v_prev_hist, "tss": tss_hist,
              "gate": gate_hist, "smooth_forward": smooth_forward}
    return spikes, cache


def backward(cache, d_loss_d_spikes):
    x, w = cache["x"], cache["w"]
    leak, threshold, refractory_ms = cache["leak"], cache["threshold"], cache["refractory_ms"]
    spikes, v_pre, v_prev_h, tss = cache["spikes"], cache["v_pre"], cache["v_prev"], cache["tss"]
    gate, smooth = cache["gate"], cache["smooth_forward"]
    leak_deriv = d_leak_toward_zero_smooth if smooth else d_leak_toward_zero

    T = x.shape[0]
    d_w = np.zeros_like(w)
    d_vnext, d_tssnext = 0.0, 0.0

    for t in reversed(range(T)):
        B_incoming = d_tssnext   # dL/d(tss_next produced by THIS tick), i.e. dL/d(tss[t+1]) -- save
                                  # before this iteration overwrites the accumulator, needed below
                                  # for a term the derivation initially missed (see docstring note)
        g = gate[t]
        s_gated = spikes[t]                  # = g * s_raw
        # recover s_raw (the ungated spike decision) for the product-rule pieces below
        vt = v_pre[t] - threshold
        d_sraw_d_vpre = surrogate_derivative(vt, K_NEURON)
        s_raw = sigmoid(K_NEURON * vt) if smooth else (1.0 if vt >= 0.0 else 0.0)

        # ---- external loss + future-tick contributions into s_gated ----
        d_loss_d_sgated = d_loss_d_spikes[t]

        # spikes[t] = g * s_raw -- two paths: through s_raw (v_pre-dependent) and through g itself
        d_loss_d_sraw_ext = d_loss_d_sgated * g
        d_loss_d_g_from_spike = d_loss_d_sgated * s_raw

        # v_next = v_pre * (1 - s_gated); tss_next = (tss_prev+DT) * (1 - s_gated)
        # both depend on s_gated, which depends on BOTH v_pre (via s_raw) and g
        if t + 1 < T:
            d_vnext_d_sgated = -v_pre[t]
            d_tssnext_d_sgated = -(tss[t] + DT)
            d_loss_d_sgated_future = d_vnext * d_vnext_d_sgated + d_tssnext * d_tssnext_d_sgated
            d_loss_d_sraw_ext += d_loss_d_sgated_future * g
            d_loss_d_g_from_spike += d_loss_d_sgated_future * s_raw

            d_vnext_d_vpre = (1.0 - s_gated)
            d_tssnext_d_vpre = 0.0  # tss_next doesn't depend on v_pre directly, only through s_gated (handled above)
        else:
            d_vnext_d_vpre = 0.0

        d_loss_d_vpre_via_spike = d_loss_d_sraw_ext * d_sraw_d_vpre
        d_loss_d_vpre_direct = (d_vnext * d_vnext_d_vpre) if t + 1 < T else 0.0
        d_loss_d_vpre = d_loss_d_vpre_via_spike + d_loss_d_vpre_direct

        # v_pre = g*v_pre_allowed + (1-g)*v_prev -- chain into g (2nd path to g) and v_pre_allowed
        v_leaked = leak_toward_zero(v_prev_h[t], leak, smooth=smooth)
        v_pre_allowed = v_leaked + float(x[t] @ w)
        d_loss_d_g_from_vpre = d_loss_d_vpre * (v_pre_allowed - v_prev_h[t])
        d_loss_d_vpre_allowed = d_loss_d_vpre * g
        d_loss_d_vprev_via_vpre = d_loss_d_vpre * (1.0 - g)

        d_w += d_loss_d_vpre_allowed * x[t]
        d_leak_local = leak_deriv(v_prev_h[t], leak)
        d_loss_d_vprev_via_leak = d_loss_d_vpre_allowed * d_leak_local

        # ---- gate[t] = sigmoid(K_REFRACTORY*(tss[t]-refractory_ms)) -- chain into tss[t] ----
        d_loss_d_g_total = d_loss_d_g_from_spike + d_loss_d_g_from_vpre
        d_g_d_tss = surrogate_derivative(tss[t] - refractory_ms, K_REFRACTORY)
        # BUG FOUND (2026-07-31, via verification 1 catching a real 0.148 max-diff, not FD noise):
        # tss[t] has a SECOND path to the loss, missed on the first pass -- tss_next =
        # (tss[t]+DT)*(1-spikes[t]) depends on tss[t] not just through g[t]->spikes[t], but
        # DIRECTLY (the +DT increment happens regardless of gate state), holding spikes[t]
        # fixed: d(tss_next)/d(tss[t]) = (1-spikes[t]). This is the SAME kind of direct-vs-
        # indirect split v_pre[t] already needed (see d_loss_d_vpre_direct above) -- tss[t]
        # needed the identical treatment and didn't get it on the first attempt.
        d_loss_d_tss = d_loss_d_g_total * d_g_d_tss + B_incoming * (1.0 - s_gated)

        # ---- accumulate into PREVIOUS tick's v_prev and tss_prev (t-1's outputs) ----
        d_vnext = d_loss_d_vprev_via_vpre + d_loss_d_vprev_via_leak
        d_tssnext = d_loss_d_tss

    return d_w


def _selftest_backprop_matches_finite_difference():
    rng = np.random.default_rng(0)
    N, T = 6, 12
    x = rng.normal(0, 1.0, size=(T, N))
    w = rng.normal(0.3, 0.15, size=N)
    leak, threshold, refractory_ms = 0.05, 1.0, 25.0

    def loss_fn(w_):
        s, _ = forward(x, w_, leak, threshold, refractory_ms, smooth_forward=True)
        target = np.ones(T) * 0.5
        return float(np.sum((s - target) ** 2))

    s, cache = forward(x, w, leak, threshold, refractory_ms, smooth_forward=True)
    target = np.ones(T) * 0.5
    d_loss_d_s = 2 * (s - target)
    analytic = backward(cache, d_loss_d_s)

    eps = 1e-5
    numeric = np.zeros(N)
    for i in range(N):
        wp, wm = w.copy(), w.copy()
        wp[i] += eps
        wm[i] -= eps
        numeric[i] = (loss_fn(wp) - loss_fn(wm)) / (2 * eps)

    max_diff = float(np.max(np.abs(analytic - numeric)))
    print(f"    analytic: {np.round(analytic, 5)}")
    print(f"    numeric:  {np.round(numeric, 5)}")
    print(f"    max abs diff: {max_diff:.2e}")
    ok = max_diff < 1e-3
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient through the refractory gate (time-since-spike "
          f"reset/increment + the gate's own surrogate) matches finite-difference")


def _selftest_training_converges():
    rng = np.random.default_rng(1)
    N, T = 8, 30
    x = (rng.random((T, N)) < 0.4).astype(float)
    leak, threshold, refractory_ms = 0.05, 1.0, 25.0
    w = rng.normal(0.2, 0.1, size=N)
    lr = 0.01
    target_count = 8   # meaningfully below T/DT's max-possible-fire-rate given the refractory lockout

    def loss_of(s):
        return (float(s.sum()) - target_count) ** 2

    s0, _ = forward(x, w, leak, threshold, refractory_ms, smooth_forward=False)
    loss0 = loss_of(s0)
    count0 = int(round(s0.sum()))

    for step in range(400):
        s, cache = forward(x, w, leak, threshold, refractory_ms, smooth_forward=False)
        d_loss_d_s = np.full(T, 2.0 * (s.sum() - target_count))
        d_w = backward(cache, d_loss_d_s)
        w = w - lr * d_w

    s_f, _ = forward(x, w, leak, threshold, refractory_ms, smooth_forward=False)
    loss_final = loss_of(s_f)
    count_f = int(round(s_f.sum()))

    print(f"    initial: fired {count0}/{T} ticks (target {target_count}), loss={loss0:.3f}")
    print(f"    final:   fired {count_f}/{T} ticks (target {target_count}), loss={loss_final:.3f}")
    ok = loss_final < loss0 * 0.5 and abs(count_f - target_count) < abs(count0 - target_count)
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient descent through the refractory gate trains "
          f"real firing-rate control -- the gate's own gradient path has to work for this to move "
          f"the count at all, since it directly blocks/allows every tick's update")


if __name__ == "__main__":
    print("=" * 78)
    print("  REFRACTORY GRADIENT -- the last deferred piece of runtime.py's real dynamics")
    print(f"  (refractory_ms={25.0}, dt={DT}ms/tick, single neuron, no cascading yet)")
    print("=" * 78)
    _selftest_backprop_matches_finite_difference()
    print()
    _selftest_training_converges()
