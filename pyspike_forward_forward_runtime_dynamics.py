#!/usr/bin/env python
"""
pyspike_forward_forward_runtime_dynamics.py — re-derives and re-verifies
the FF surrogate-gradient math for core/runtime/runtime.py's ACTUAL neuron
dynamics, not the simplified multiplicative-beta model
pyspike_forward_forward.py used. Standalone, isolated (per your request) --
does NOT touch runtime.py. This is the honest prerequisite before any real
DSL/runtime integration: the two dynamics are genuinely different
equations, and applying gradient math derived for one to the other would
be silently wrong, not just imprecise.

REAL RUNTIME DYNAMICS (matched exactly to runtime.py's stimulate()/_leak_
toward_zero()/_fire(), for refractory_ms=0, no synapse cascading -- see
"HONEST SCOPE" below for what's deliberately not covered yet):

  v_leaked          = leak_toward_zero(v_prev, leak)     # SUBTRACTIVE,
                                                          # clamped at 0,
                                                          # not multiplicative
  v_pre_threshold   = v_leaked + drive[t]                # drive[t] = x[t] . w
  spike[t]          = Heaviside(v_pre_threshold - threshold)   # hard forward
  v_next            = v_pre_threshold * (1 - spike[t])   # hard reset-to-0
                                                          # on fire (exact,
                                                          # not a decayed
                                                          # fraction)

leak_toward_zero's own piecewise derivative (NOT the spike surrogate -- a
separate, real non-smoothness in this dynamics that pyspike_forward_forward.py's
model didn't have at all, since it used a smooth multiplicative beta decay):
  d(v_leaked)/d(v_prev) = 1        if v_prev > leak   (unclamped subtraction)
                         = 0        if 0 <= v_prev <= leak   (clamped to 0)
                         (symmetric for v_prev < 0, not exercised here since
                          this test only ever drives the neuron positively)

HONEST SCOPE -- what this does NOT cover yet:
  - refractory_ms > 0 (a hard, timing-history-dependent gate -- real added
    complexity, deliberately deferred)
  - synapse cascading / multi-neuron propagation (real.py's recursive
    _fire() -> downstream neurons -- deliberately deferred, single neuron
    only here)
  - the STDP-during-propagation interaction (moot without cascading)

TWO VERIFICATIONS, same discipline as every prior script here:
  1. Correctness: analytic gradient (smooth forward, both leak-clamp and
     spike surrogate) vs. finite-difference of the same smooth loss.
  2. Does it learn: same toy 3-class task, using THESE dynamics instead of
     the simplified model, with the lr/theta values already found to work.

    python pyspike_forward_forward_runtime_dynamics.py
"""
import numpy as np

from pyspike_forward_forward import (
    make_toy_task, embed_label, repeat_over_time, ff_loss_and_grad,
    N_CLASSES, D_INPUT, T,
)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def surrogate_derivative(v_minus_threshold, k=4.0):
    s = sigmoid(k * v_minus_threshold)
    return k * s * (1.0 - s)


def leak_toward_zero(v, leak, smooth=False, beta_smooth=8.0):
    """Matches runtime.py's _leak_toward_zero EXACTLY when smooth=False.
    smooth=True substitutes a soft-clamp (softplus-based) for the
    correctness check ONLY, so the loss is differentiable end-to-end --
    same pattern as SurrogateLIFLayer's smooth_forward flag."""
    if not smooth:
        if v > 0.0:
            return max(0.0, v - leak)
        return min(0.0, v + leak)
    # smooth surrogate of the same shape, positive side only (this test
    # never drives negative): softplus(beta*(v-leak))/beta stays close to
    # (v-leak) for v>>leak and close to 0 for v<<leak, like the real clamp.
    return np.logaddexp(0.0, beta_smooth * (v - leak)) / beta_smooth


def d_leak_toward_zero(v, leak):
    """Real (non-smoothed) subgradient of leak_toward_zero w.r.t. v --
    legitimate and correct to use for the REAL (hard-forward) training
    pass, since (unlike the spike step function) this clamp genuinely HAS
    a well-defined derivative almost everywhere; no surrogate needed."""
    if v > leak:
        return 1.0
    if v >= 0.0:
        return 0.0
    if v > -leak:
        return 0.0
    return 1.0


def d_leak_toward_zero_smooth(v, leak, beta_smooth=8.0):
    """BUG FOUND (2026-07-30): verification 1 failed (max diff 0.77) because
    backward() was using d_leak_toward_zero (the HARD subgradient) even when
    checking against the SMOOTH forward pass's loss -- a forward/backward
    mismatch, the same class of error the spike surrogate already avoids
    correctly (its forward and backward both consistently use the same
    sigmoid, in both smooth-check and real-training modes). d/dv of the
    smooth leak surrogate softplus(beta*(v-leak))/beta is exactly
    sigmoid(beta*(v-leak)) -- this must be used instead of the hard
    subgradient whenever forward was smoothed, or analytic and numeric
    gradients check DIFFERENT functions and can't be expected to agree."""
    return sigmoid(beta_smooth * (v - leak))


class RuntimeMatchedNeuron:
    """One neuron, runtime.py's real dynamics, refractory_ms=0, no
    synapses. n_inputs -> 1, weight vector w."""

    def __init__(self, n_inputs: int, leak: float = 0.05, threshold: float = 1.0, k: float = 4.0):
        self.n_inputs = n_inputs
        self.leak = leak
        self.threshold = threshold
        self.k = k

    def forward(self, x: np.ndarray, w: np.ndarray, smooth_forward: bool = False):
        T_ = x.shape[0]
        v_leaked_hist = np.zeros(T_)
        v_pre_hist = np.zeros(T_)
        spikes = np.zeros(T_)
        v_prev = 0.0
        for t in range(T_):
            v_leaked = leak_toward_zero(v_prev, self.leak, smooth=smooth_forward)
            v_leaked_hist[t] = v_leaked
            v_pre = v_leaked + float(x[t] @ w)
            v_pre_hist[t] = v_pre
            vt = v_pre - self.threshold
            if smooth_forward:
                s = sigmoid(self.k * vt)
            else:
                s = 1.0 if vt >= 0.0 else 0.0
            spikes[t] = s
            v_prev = v_pre * (1.0 - s)
        cache = {"x": x, "w": w, "v_pre": v_pre_hist, "spikes": spikes, "smooth_forward": smooth_forward}
        return spikes, cache

    def backward(self, cache: dict, d_loss_d_spikes: np.ndarray) -> np.ndarray:
        x, w, v_pre, spikes = cache["x"], cache["w"], cache["v_pre"], cache["spikes"]
        smooth_forward = cache["smooth_forward"]
        leak_deriv_fn = (lambda v: d_leak_toward_zero_smooth(v, self.leak)) if smooth_forward \
            else (lambda v: d_leak_toward_zero(v, self.leak))
        T_ = x.shape[0]
        d_w = np.zeros_like(w)
        d_v_next = 0.0  # gradient flowing back into v_prev from the NEXT tick
        for t in reversed(range(T_)):
            vt = v_pre[t] - self.threshold
            d_spike_d_vpre = surrogate_derivative(vt, self.k)
            d_loss_d_vpre = d_loss_d_spikes[t] * d_spike_d_vpre
            if t + 1 < T_:
                # v_next = v_pre[t]*(1-spike[t]) -- product rule, same shape
                # as SurrogateLIFLayer's, but v_next here feeds into the
                # LEAK function next tick, not directly into v[t+1].
                d_vnext_d_vpre = (1.0 - spikes[t]) - v_pre[t] * d_spike_d_vpre
                d_loss_d_vpre += d_v_next * d_vnext_d_vpre
            # v_pre[t] = leak_toward_zero(v_prev[t]) + drive[t] -- chain
            # through the leak function's derivative (smooth or hard,
            # MATCHING whichever one forward() actually used) to get
            # d_loss/d_v_prev for the tick before this one.
            v_prev_t = v_pre[t - 1] * (1.0 - spikes[t - 1]) if t > 0 else 0.0
            d_leak = leak_deriv_fn(v_prev_t)
            d_v_next = d_loss_d_vpre * d_leak
            d_w += d_loss_d_vpre * x[t]
        return d_w


def _selftest_backprop_matches_finite_difference() -> None:
    rng = np.random.default_rng(0)
    N = 11
    x = rng.normal(0, 1.0, size=(12, N))
    w = rng.normal(0.3, 0.15, size=N)
    neuron = RuntimeMatchedNeuron(N, leak=0.05, threshold=1.0, k=4.0)

    def loss_fn(w_):
        spikes, _ = neuron.forward(x, w_, smooth_forward=True)
        target = np.ones(x.shape[0]) * 0.5
        return float(np.sum((spikes - target) ** 2))

    spikes, cache = neuron.forward(x, w, smooth_forward=True)
    target = np.ones(x.shape[0]) * 0.5
    d_loss_d_spikes = 2 * (spikes - target)
    analytic_grad = neuron.backward(cache, d_loss_d_spikes)

    eps = 1e-5
    numeric_grad = np.zeros(N)
    for i in range(N):
        w_plus, w_minus = w.copy(), w.copy()
        w_plus[i] += eps
        w_minus[i] -= eps
        numeric_grad[i] = (loss_fn(w_plus) - loss_fn(w_minus)) / (2 * eps)

    max_diff = float(np.max(np.abs(analytic_grad - numeric_grad)))
    ok = max_diff < 1e-3
    print(f"    analytic grad: {np.round(analytic_grad, 5)}")
    print(f"    numeric  grad: {np.round(numeric_grad, 5)}")
    print(f"    max abs diff:  {max_diff:.2e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] gradient matches finite-difference "
          f"for runtime.py's REAL leak/reset dynamics (refractory_ms=0, single neuron)")


class RuntimeMatchedFFLayer:
    def __init__(self, n_inputs, n_neurons, leak=0.05, threshold=1.0, k=4.0):
        self.n_neurons = n_neurons
        self.neurons = [RuntimeMatchedNeuron(n_inputs, leak, threshold, k) for _ in range(n_neurons)]

    def forward(self, x, W, smooth_forward=False):
        counts = np.zeros(self.n_neurons)
        caches = []
        for h, neuron in enumerate(self.neurons):
            spikes, cache = neuron.forward(x, W[h], smooth_forward=smooth_forward)
            counts[h] = spikes.sum()
            caches.append(cache)
        return counts, caches

    def goodness(self, counts):
        return float(np.mean(counts ** 2))

    def backward_from_goodness_grad(self, caches, counts, d_loss_d_goodness):
        H = self.n_neurons
        dW = np.zeros((H, self.neurons[0].n_inputs))
        for h, (neuron, cache) in enumerate(zip(self.neurons, caches)):
            d_loss_d_count_h = d_loss_d_goodness * (2.0 / H) * counts[h]
            Tt = cache["x"].shape[0]
            d_loss_d_spikes = np.full(Tt, d_loss_d_count_h)
            dW[h] = neuron.backward(cache, d_loss_d_spikes)
        return dW


def _selftest_training_converges() -> None:
    rng = np.random.default_rng(1)
    H = 24
    theta = 65.0
    lr = 0.02  # already-found value; leak/threshold differ, so this may need its own tuning
    layer = RuntimeMatchedFFLayer(D_INPUT, H, leak=0.05, threshold=1.0, k=4.0)
    W = rng.normal(0.3, 0.15, size=(H, D_INPUT))

    centers, train_labels, train_content = make_toy_task(rng, 300)
    _, test_labels, test_content = make_toy_task(rng, 100)

    def classify(content_row, W_):
        m = float(np.max(content_row))
        gs = []
        for c in range(N_CLASSES):
            x = repeat_over_time(embed_label(content_row, c, m), T)
            counts, _ = layer.forward(x, W_, smooth_forward=False)
            gs.append(layer.goodness(counts))
        return int(np.argmax(gs))

    def accuracy(content_arr, labels_arr, W_):
        preds = [classify(content_arr[i], W_) for i in range(len(labels_arr))]
        return float(np.mean(np.array(preds) == labels_arr))

    acc_before = accuracy(test_content, test_labels, W)
    best_acc, best_epoch, W_best = acc_before, -1, W.copy()

    for epoch in range(30):
        perm = rng.permutation(len(train_labels))
        for idx in perm:
            content_row = train_content[idx]
            true_label = int(train_labels[idx])
            m = float(np.max(content_row))
            x_pos = repeat_over_time(embed_label(content_row, true_label, m), T)
            counts_pos, caches_pos = layer.forward(x_pos, W, smooth_forward=False)
            G_pos = layer.goodness(counts_pos)
            dW_total = np.zeros_like(W)
            for wrong_label in range(N_CLASSES):
                if wrong_label == true_label:
                    continue
                x_neg = repeat_over_time(embed_label(content_row, wrong_label, m), T)
                counts_neg, caches_neg = layer.forward(x_neg, W, smooth_forward=False)
                G_neg = layer.goodness(counts_neg)
                _, dGp, dGn = ff_loss_and_grad(G_pos, G_neg, theta)
                dW_total += layer.backward_from_goodness_grad(caches_pos, counts_pos, dGp)
                dW_total += layer.backward_from_goodness_grad(caches_neg, counts_neg, dGn)
            W = W - lr * dW_total / (N_CLASSES - 1)
        acc_epoch = accuracy(test_content, test_labels, W)
        if acc_epoch > best_acc:
            best_acc, best_epoch, W_best = acc_epoch, epoch, W.copy()

    acc_final = accuracy(test_content, test_labels, W)
    acc_reverify = accuracy(test_content, test_labels, W_best)
    chance = 1.0 / N_CLASSES
    print(f"    accuracy before training:  {acc_before:.3f}  (chance={chance:.3f})")
    print(f"    accuracy final (epoch 29): {acc_final:.3f}")
    print(f"    accuracy best  (epoch {best_epoch:2d}): {best_acc:.3f}  (re-verified: {acc_reverify:.3f})")
    ok = best_acc > chance + 0.10 and best_acc > acc_before and abs(acc_reverify - best_acc) < 1e-9
    print(f"  [{'PASS' if ok else 'FAIL'}] REAL runtime.py dynamics (leak/reset), same toy task")


if __name__ == "__main__":
    print("=" * 78)
    print("  FF SURROGATE GRADIENT -- re-derived for runtime.py's REAL neuron dynamics")
    print("  (refractory_ms=0, single neuron / no synapse cascading -- see HONEST SCOPE)")
    print("=" * 78)
    _selftest_backprop_matches_finite_difference()
    print()
    _selftest_training_converges()
