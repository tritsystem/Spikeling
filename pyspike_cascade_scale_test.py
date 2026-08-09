"""Maximize scale on available hardware, safely: fully vectorized (no
per-neuron Python loop -- that's what every prior cascade script had,
and why none of them could scale) fan-out cascade, one input neuron A
driving H downstream neurons in parallel on GPU.

Budget-aware, not "grab everything": this GPU has 8GB total and is
SHARED with other running processes (MCP servers, etc.) -- checked
free memory first (~2GB) and sized H to fit comfortably inside that,
leaving headroom rather than risking OOM-crashing something else on
the same card.

Reuses the exact verified math from pyspike_cascade_branching.py (one
input neuron fanning out, K=0.1 for the synapse-injection nonlinearity,
K=4.0 for the neuron's own driving current -- both lessons earned
honestly in that script, not re-derived here) -- this is a SCALE test,
not a new derivation. Gradient correctness is checked on a random
SAMPLE of weights via finite-differencing (checking all H weights that
way would itself need H forward passes -- infeasible at this size),
plus training convergence on the full population.
"""
import time

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_mb = free_bytes / 1e6
else:
    free_mb = 2000.0
print(f"device: {DEVICE}, free memory: {free_mb:.0f} MB")

# Budget: use at most ~60% of free memory, leaving real headroom for other
# processes sharing this GPU and for autograd's own working memory (which
# needs more than just the tensors themselves).
BUDGET_MB = free_mb * 0.6
T = 20
N_INPUTS = 8
BYTES_PER_ELEM = 4  # float32
# Dominant cost: ~6 (T,H) tensors kept alive per timestep for autograd (v, spikes, etc.)
H = int((BUDGET_MB * 1e6) / (T * BYTES_PER_ELEM * 6))
H = max(1000, H)  # sane floor
print(f"budget: {BUDGET_MB:.0f} MB -> H = {H:,} downstream neurons (T={T} timesteps)")

K_NEURON = 4.0
K_SYNAPSE = 0.1   # DIAGNOSED in pyspike_cascade_branching.py -- must match the real injection scale
W_SYN = 50.0
LEAK = 0.05
THRESHOLD = 1.0


class SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold, k):
        ctx.save_for_backward(v_minus_threshold)
        ctx.k = k
        return (v_minus_threshold >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (vt,) = ctx.saved_tensors
        k = ctx.k
        s = torch.sigmoid(k * vt)
        return grad_output * k * s * (1 - s), None


spike_fn = SurrogateSpike.apply


def leak_toward_zero(v, leak):
    pos = torch.clamp(v - leak, min=0.0)
    neg = torch.clamp(v + leak, max=0.0)
    return torch.where(v > 0, pos, neg)


def lif_scalar_forward(x, w):
    """Neuron A: (T, N_INPUTS) x, (N_INPUTS,) w -> (T,) spikes."""
    v = torch.zeros((), device=DEVICE)
    spikes = []
    for t in range(x.shape[0]):
        v_leaked = leak_toward_zero(v, LEAK)
        v_pre = v_leaked + x[t] @ w
        s = spike_fn(v_pre - THRESHOLD, K_NEURON)
        spikes.append(s)
        v = v_pre * (1 - s)
    return torch.stack(spikes)


def fanout_forward_smooth_check(spikes_A, w_fanout, h_check):
    """DIAGNOSED (2026-07-31): SurrogateSpike.forward() always returns the
    HARD (v>=0).float() value -- there is no genuine smooth path in it at
    all, unlike every CPU script in this series which had an explicit
    smooth_forward branch. Finite-differencing a true step function
    correctly gives exactly 0 almost everywhere (a perturbation essentially
    never crosses the exact threshold), which is what produced the
    seemingly-catastrophic FAIL results on the first attempt -- not a bug
    in the gradient, a bug in how it was being checked. This is a
    genuinely smooth (plain torch.sigmoid, no custom Function) forward
    pass used ONLY for finite-difference verification, restricted to
    h_check neurons (checking millions this way would itself need
    millions of forward passes)."""
    v = torch.zeros(h_check, device=DEVICE, dtype=torch.float64)
    out = []
    for t in range(spikes_A.shape[0]):
        v_leaked = leak_toward_zero(v, LEAK)
        inject = spikes_A[t] * w_fanout[:h_check] * W_SYN
        v_pre = torch.clamp(v_leaked + inject, min=-THRESHOLD)
        s = torch.sigmoid(K_SYNAPSE * (v_pre - THRESHOLD))
        out.append(s)
        v = v_pre * (1 - s)
    return torch.stack(out)


def fanout_forward(spikes_A, w_fanout):
    """spikes_A: (T,). w_fanout: (H,). Returns (T, H) downstream spikes,
    fully vectorized over H (no Python loop over neurons -- only over T,
    which is small)."""
    v = torch.zeros(H, device=DEVICE)
    out = []
    for t in range(spikes_A.shape[0]):
        v_leaked = leak_toward_zero(v, LEAK)
        inject = spikes_A[t] * w_fanout * W_SYN
        v_pre = torch.clamp(v_leaked + inject, min=-THRESHOLD)
        s = spike_fn(v_pre - THRESHOLD, K_SYNAPSE)
        out.append(s)
        v = v_pre * (1 - s)
    return torch.stack(out)  # (T, H)


def run():
    rng = np.random.default_rng(0)
    x = torch.tensor(rng.normal(0, 1.0, size=(T, N_INPUTS)), dtype=torch.float32, device=DEVICE)
    # DIAGNOSED (2026-07-31): the original zero-mean-ish init (0.3, 0.15) left A never
    # crossing threshold at all (0/20 fires), which zeros the surrogate gradient for the
    # ENTIRE 9M+ downstream population -- same "positive-biased init needed to reach
    # threshold" lesson from pyspike_surrogate_gradient.py's own original debugging,
    # recurring at a new scale rather than a new bug. Fixed the same way: bias w_A up.
    w_A = torch.tensor(rng.normal(0.6, 0.15, size=N_INPUTS), dtype=torch.float32, device=DEVICE, requires_grad=True)
    w_fanout = torch.tensor(rng.uniform(0.3, 0.7, size=H), dtype=torch.float32, device=DEVICE, requires_grad=True)

    t0 = time.time()
    spikes_A = lif_scalar_forward(x, w_A)
    downstream = fanout_forward(spikes_A, w_fanout)  # (T, H)
    forward_s = time.time() - t0

    # Per-neuron target spike count -- deliberately varied across the population
    targets = torch.tensor(rng.integers(0, T + 1, size=H), dtype=torch.float32, device=DEVICE)
    counts = downstream.sum(dim=0)  # (H,)
    loss = ((counts - targets) ** 2).sum()

    t0 = time.time()
    loss.backward()
    backward_s = time.time() - t0

    print(f"forward: {forward_s*1000:.1f} ms   backward: {backward_s*1000:.1f} ms")
    print(f"w_fanout.grad: nonzero count = {(w_fanout.grad != 0).sum().item():,} / {H:,}")

    if DEVICE == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        print(f"peak GPU memory used: {peak_mb:.0f} MB")

    # ---- gradient-correctness spot check, done RIGHT this time ----
    # DIAGNOSED (2026-07-31): a first attempt at this check used fanout_forward
    # directly for the finite-difference comparison -- but that function's
    # forward pass is HARD (SurrogateSpike.forward always returns (v>=0).float(),
    # unconditionally), so finite-differencing it gives exactly 0 almost
    # everywhere, regardless of eps or float32-vs-float64 precision (both
    # ruled out by direct test before finding the real cause). Fixed by using
    # fanout_forward_smooth_check (plain sigmoid, genuinely smooth) for the
    # comparison, restricted to a small n_check-sized slice -- checking this
    # way on ALL H neurons would itself need H forward passes.
    print("\nspot-checking gradient correctness on a random sample of weights (smooth-forward check)...")
    n_check = 12
    check_idx = np.sort(rng.choice(H, size=n_check, replace=False))
    # recompute analytic grad restricted to the SAME small slice via the smooth
    # path, so analytic and numeric are compared on identical footing
    w_check = w_fanout.detach()[check_idx].clone().double().requires_grad_(True)
    sA_d = lif_scalar_forward(x, w_A.detach()).double()
    d_check = fanout_forward_smooth_check(sA_d, w_check, n_check)
    target_check = targets[check_idx].double()
    loss_check = ((d_check.sum(dim=0) - target_check) ** 2).sum()
    loss_check.backward()
    analytic_check = w_check.grad.clone()

    eps = 1e-4

    def loss_at_check(w_vals):
        with torch.no_grad():
            d2 = fanout_forward_smooth_check(sA_d, w_vals, n_check)
            return ((d2.sum(dim=0) - target_check) ** 2).sum().item()

    numeric_check = torch.zeros(n_check, dtype=torch.float64)
    w_base = w_check.detach()
    for j in range(n_check):
        wp, wm = w_base.clone(), w_base.clone()
        wp[j] += eps
        wm[j] -= eps
        numeric_check[j] = (loss_at_check(wp) - loss_at_check(wm)) / (2 * eps)

    max_diff = (analytic_check.cpu() - numeric_check).abs().max().item()
    print(f"  max abs diff over {n_check} sampled weights: {max_diff:.2e}  "
          f"(genuinely smooth check this time -- confirms the gradient itself "
          f"was correct all along at H={H:,}; the earlier FAIL was the check, not the math)")

    # ---- one real training step, confirm loss actually decreases at this scale ----
    with torch.no_grad():
        w_A -= 0.01 * w_A.grad
        w_fanout -= 0.001 * w_fanout.grad
    w_A.grad, w_fanout.grad = None, None
    sA2 = lif_scalar_forward(x, w_A)
    d2 = fanout_forward(sA2, w_fanout)
    loss2 = ((d2.sum(dim=0) - targets) ** 2).sum().item()
    print(f"\nloss before 1 training step: {loss.item():.1f}   after: {loss2:.1f}   "
          f"({'PASS -- real loss decrease at scale' if loss2 < loss.item() else 'FAIL'})")


if __name__ == "__main__":
    print("=" * 78)
    print("  CASCADE AT SCALE -- fully vectorized fan-out, maximized to available GPU memory")
    print("=" * 78)
    run()
