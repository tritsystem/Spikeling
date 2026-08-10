#!/usr/bin/env python
"""pyspike_native_backprop.py -- backprop for a real .spk-compiled network
with ZERO external autodiff dependency: no PyTorch, no autograd library.
Every derivative below is hand-derived and hand-coded, the same discipline
pyspike_surrogate_gradient.py already used for a single neuron (its own
docstring: "no autodiff framework... fully transparent and independently
checkable, not hidden behind a library") -- this extends that same
ownership to a full multi-neuron GRAPH, including the recurrent feedback
loop, instead of outsourcing that step to torch.autograd the way
pyspike_compiler_trainable.py did.

THE MATH, worked by hand (mirrors runtime dynamics, all real ops defined
here, nothing borrowed):

  Forward, per tick t, per neuron n:
    syn_input[n,t]   = sum over synapses (m->n): weight[m->n]*PULSE_SCALE*spike[m,t-1]
    total_input[n,t] = external_drive[n,t] + syn_input[n,t]
    v_reset[n,t]     = v[n,t-1] * (1 - spike[n,t-1])
    v_leaked[n,t]    = sign(v_reset[n,t]) * relu(|v_reset[n,t]| - leak[n])
    v[n,t]           = v_leaked[n,t] + total_input[n,t]
    spike[n,t]       = Heaviside(v[n,t] - threshold[n])         [forward]
    d(spike[n,t])/d(v[n,t]) = k*sigmoid(k*(v-thresh))*(1-sigmoid(...))  [surrogate, backward only]

  Backward, per tick t (REVERSE order, T-1 down to 0), per neuron n:
    dL_dspike[n,t] = (direct loss gradient at this tick, if any)
                     + sum over synapses (n->m): weight[n->m]*PULSE_SCALE * dL_dv[m,t+1]   (cross-neuron path)
                     + (-v[n,t]) * d_leak[n,t+1] * dL_dv[n,t+1]                             (own reset-gate path)
    dL_dv[n,t]     = dL_dspike[n,t] * surrogate_derivative(v[n,t]-threshold[n])
                     + (1-spike[n,t]) * d_leak[n,t+1] * dL_dv[n,t+1]                        (own leak-chain path)
    dL_dweight[m->n] += dL_dv[n,t] * PULSE_SCALE * spike[m,t-1]     (accumulated over all t)

  where d_leak[n,t+1] = 1.0 if |v_reset[n,t+1]| > leak[n] else 0.0 (subgradient of
  the leak's relu kink, 0 at the kink itself).

VALIDATION (not assumed correct): cross-checked against
pyspike_compiler_trainable.py's already-verified PyTorch autograd gradient
on the IDENTICAL real jet-engine topology and inputs, before being trusted
for anything.
"""
import os
import sys

SPIKELING_ROOT = r"C:\Users\gbran\OneDrive\Documents\Spikeling"
sys.path.insert(0, os.path.join(SPIKELING_ROOT, "core"))
sys.path.insert(0, SPIKELING_ROOT)

import tempfile
import numpy as np

from compiler.compiler import compile_file

PULSE_SCALE = 50.0
K_SURROGATE = 0.1  # measured earlier tonight: k=4.0 (tuned for threshold~1.0) vanished
                     # to grad_max=9.7e-25 on this network's threshold=60-120 scale;
                     # k=0.1 carries a real, strong gradient (~19).


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def surrogate_derivative(v_minus_threshold, k=K_SURROGATE):
    s = sigmoid(k * v_minus_threshold)
    return k * s * (1.0 - s)


class NativeSpikingGraph:
    """Same real topology-loading as TrainableSpikingGraph (reads the
    REAL compiled AST -- NeuronDef/ConnectionDef, not reimplemented) but
    the forward/backward math is 100% NumPy, hand-derived, no autodiff."""

    def __init__(self, ast):
        self.names = [n.name for n in ast.neurons]
        self.name_to_idx = {n: i for i, n in enumerate(self.names)}
        self.n_neurons = len(self.names)
        self.threshold = np.array([float(n.threshold) for n in ast.neurons])
        self.leak = np.array([float(n.leak) for n in ast.neurons])

        self.src_idx = np.array([self.name_to_idx[c.src] for c in ast.connections], dtype=int)
        self.dst_idx = np.array([self.name_to_idx[c.dst] for c in ast.connections], dtype=int)
        self.weight = np.array([c.weight for c in ast.connections], dtype=float)
        self.synapse_labels = [(c.src, c.dst) for c in ast.connections]
        self.n_synapses = len(self.synapse_labels)

        # per-destination-neuron list of incoming synapse indices, and
        # per-source-neuron list of outgoing synapse indices -- precomputed
        # once so forward/backward don't rescan all synapses per neuron.
        self.incoming = [[] for _ in range(self.n_neurons)]
        self.outgoing = [[] for _ in range(self.n_neurons)]
        for e, (s, d) in enumerate(zip(self.src_idx, self.dst_idx)):
            self.incoming[d].append(e)
            self.outgoing[s].append(e)

    def forward(self, external_drive):
        """external_drive: (T, n_neurons) numpy array. Returns spikes
        (T, n_neurons) and a cache of every intermediate needed by backward."""
        T = external_drive.shape[0]
        v = np.zeros((T, self.n_neurons))
        v_reset = np.zeros((T, self.n_neurons))
        spike = np.zeros((T, self.n_neurons))
        syn_input = np.zeros((T, self.n_neurons))

        v_prev = np.zeros(self.n_neurons)
        spike_prev = np.zeros(self.n_neurons)

        for t in range(T):
            si = np.zeros(self.n_neurons)
            contrib = self.weight * PULSE_SCALE * spike_prev[self.src_idx]
            np.add.at(si, self.dst_idx, contrib)
            syn_input[t] = si
            total_input = external_drive[t] + si

            vr = v_prev * (1.0 - spike_prev)
            v_reset[t] = vr
            v_leaked = np.sign(vr) * np.clip(np.abs(vr) - self.leak, 0.0, None)

            v[t] = v_leaked + total_input
            spike[t] = (v[t] - self.threshold >= 0.0).astype(float)

            v_prev, spike_prev = v[t], spike[t]

        cache = {"v": v, "v_reset": v_reset, "spike": spike, "external_drive": external_drive}
        return spike, cache

    def backward(self, cache, d_loss_d_spike):
        """d_loss_d_spike: (T, n_neurons), direct loss gradient w.r.t. each
        spike (0 where loss doesn't directly depend on that neuron/tick).
        Returns d_loss/d_weight, shape (n_synapses,) -- hand-derived
        reverse-time, reverse-graph BPTT, no autodiff library."""
        v, v_reset, spike = cache["v"], cache["v_reset"], cache["spike"]
        T = v.shape[0]
        d_weight = np.zeros(self.n_synapses)

        dL_dv_next = np.zeros(self.n_neurons)   # dL/dv[n, t+1], carried backward

        for t in reversed(range(T)):
            # d_leak[n, t+1] subgradient of the leak kink, only meaningful
            # if there IS a t+1 (else no future to chain through)
            if t + 1 < T:
                d_leak_next = (np.abs(v_reset[t + 1]) > self.leak).astype(float)
            else:
                d_leak_next = np.zeros(self.n_neurons)

            v_prev_t = v[t - 1] if t > 0 else np.zeros(self.n_neurons)

            # dL/dspike[n,t]: direct + cross-neuron (via downstream synapses) + own reset-gate path
            dL_dspike = d_loss_d_spike[t].copy()
            for n in range(self.n_neurons):
                for e in self.outgoing[n]:
                    m = self.dst_idx[e]
                    dL_dspike[n] += self.weight[e] * PULSE_SCALE * dL_dv_next[m]
            # own reset-gate path: v_reset[n,t+1] = v[n,t]*(1-spike[n,t]) -> d/d(spike[n,t]) = -v[n,t]
            dL_dspike += (-v[t]) * d_leak_next * dL_dv_next

            dL_dv = dL_dspike * surrogate_derivative(v[t] - self.threshold)
            # own leak-chain path: v_reset[n,t+1] depends on v[n,t] via (1-spike[n,t])
            dL_dv += (1.0 - spike[t]) * d_leak_next * dL_dv_next

            # weight gradient: total_input[n,t] includes weight[m->n]*PULSE_SCALE*spike[m,t-1]
            spike_prev = spike[t - 1] if t > 0 else np.zeros(self.n_neurons)
            for e in range(self.n_synapses):
                m = self.src_idx[e]
                n = self.dst_idx[e]
                d_weight[e] += dL_dv[n] * PULSE_SCALE * spike_prev[m]

            dL_dv_next = dL_dv

        return d_weight


def compile_native(spk_path):
    ast = compile_file(spk_path, output_dir=tempfile.mkdtemp(prefix="native_"))
    return NativeSpikingGraph(ast), ast


def cross_check_against_pytorch():
    """VALIDATION: same real topology, same inputs, same weights -- this
    hand-rolled NumPy gradient must match pyspike_compiler_trainable.py's
    already-verified PyTorch autograd gradient. Not assumed, checked."""
    import torch
    from pyspike_compiler_trainable import compile_trainable

    spk = os.path.join(SPIKELING_ROOT, "ai-apps", "jet_engine_spike_pipeline.spk")

    np.random.seed(0)
    T = 20
    net_np, ast = compile_native(spk)
    w0 = np.random.uniform(0.3, 1.5, size=net_np.n_synapses)
    net_np.weight = w0.copy()

    intake_idx = [net_np.name_to_idx[n] for n in ["intake1", "intake2", "intake3", "intake4"]]
    combustion_idx = net_np.name_to_idx["combustion"]
    drive = np.zeros((T, net_np.n_neurons))
    drive[:, intake_idx] = 80.0

    spikes_np, cache = net_np.forward(drive)
    d_loss_d_spike = np.zeros((T, net_np.n_neurons))
    c_sum = spikes_np[:, combustion_idx].sum()
    # dL/dspike[combustion, t] = 2*(sum - target) for every t (loss = (sum-target)^2)
    d_loss_d_spike[:, combustion_idx] = 2.0 * (c_sum - 3.0)
    grad_np = net_np.backward(cache, d_loss_d_spike)

    # identical setup in PyTorch, same k, same weights, same drive
    torch.manual_seed(0)
    net_pt, _ = compile_trainable(spk)
    with torch.no_grad():
        net_pt.weight.copy_(torch.tensor(w0, dtype=torch.float32))
    drive_pt = torch.tensor(drive, dtype=torch.float32)
    spikes_pt = net_pt(drive_pt)
    c_pt = spikes_pt[:, combustion_idx].sum()
    loss_pt = (c_pt - 3.0) ** 2
    loss_pt.backward()
    grad_pt = net_pt.weight.grad.numpy()

    max_diff = float(np.max(np.abs(grad_np - grad_pt)))
    ok = max_diff < 1e-3
    print("VALIDATION: hand-rolled NumPy backward() vs already-verified PyTorch autograd")
    print(f"  combustion sum: numpy={float(c_sum):.3f}  torch={float(c_pt.detach()):.3f}")
    print(f"  numpy grad: {np.round(grad_np, 5)}")
    print(f"  torch grad: {np.round(grad_pt, 5)}")
    print(f"  max abs diff: {max_diff:.2e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] native NumPy gradient matches PyTorch autograd on the "
          f"real jet-engine topology, including the feedback loop\n")
    return ok, net_np, drive, intake_idx, combustion_idx, w0


def native_adam_train(net, drive, combustion_idx, target=3.0, steps=200, lr=0.05):
    """Hand-rolled Adam, no torch.optim -- full ownership of the training
    loop, not just the gradient. Standard Adam update equations, applied
    directly to net.weight."""
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    m = np.zeros_like(net.weight)
    v_ = np.zeros_like(net.weight)
    losses = []
    for step in range(1, steps + 1):
        spikes, cache = net.forward(drive)
        c_sum = spikes[:, combustion_idx].sum()
        loss = (c_sum - target) ** 2
        losses.append(float(loss))

        T = drive.shape[0]
        d_loss_d_spike = np.zeros((T, net.n_neurons))
        d_loss_d_spike[:, combustion_idx] = 2.0 * (c_sum - target)
        grad = net.backward(cache, d_loss_d_spike)

        m = beta1 * m + (1 - beta1) * grad
        v_ = beta2 * v_ + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** step)
        v_hat = v_ / (1 - beta2 ** step)
        net.weight = net.weight - lr * m_hat / (np.sqrt(v_hat) + eps)
        net.weight = np.clip(net.weight, -3.0, 3.0)
    return losses


if __name__ == "__main__":
    print("=" * 78)
    print("  NATIVE (NO EXTERNAL AUTODIFF) BACKPROP FOR SPIKELING")
    print("=" * 78)
    ok, net, drive_sustained, intake_idx, combustion_idx, w0 = cross_check_against_pytorch()
    if not ok:
        print("Cross-check FAILED -- not proceeding to training on unverified gradients.")
        sys.exit(1)

    T_blip, T_pad = 1, 5
    drive_blip = np.zeros((T_pad + T_blip + T_pad, net.n_neurons))
    drive_blip[T_pad, intake_idx] = 80.0

    net.weight = w0.copy()  # reset to the same random init used in the cross-check
    with_no_grad_c_sustained0 = net.forward(drive_sustained)[0][:, combustion_idx].sum()
    with_no_grad_c_blip0 = net.forward(drive_blip)[0][:, combustion_idx].sum()
    print(f"BEFORE (native, random init): sustained={with_no_grad_c_sustained0:.3f}  blip={with_no_grad_c_blip0:.3f}")

    TARGET = 3.0
    losses = native_adam_train(net, drive_sustained, combustion_idx, target=TARGET, steps=200, lr=0.05)

    c_sustained1 = net.forward(drive_sustained)[0][:, combustion_idx].sum()
    c_blip1 = net.forward(drive_blip)[0][:, combustion_idx].sum()
    print(f"AFTER  (native Adam, 200 steps): sustained={c_sustained1:.3f}  blip={c_blip1:.3f}")
    print(f"loss trajectory (every 40 steps): {[round(l,3) for l in losses[::40]]}")

    # CORRECT criterion (fixed from a first version that compared raw
    # margin before/after, which wrongly failed a run that started with
    # sustained=4.0 -- already above the target=3.0, so training correctly
    # REDUCED it to hit target, which looked like a shrinking margin even
    # though it was a perfect, exact convergence): did it actually converge
    # to the target, and does a clear discrimination margin still exist?
    hit_target = abs(c_sustained1 - TARGET) < 0.5
    clear_margin = (c_sustained1 - c_blip1) > 1.5
    ok_train = hit_target and clear_margin
    print(f"  hit_target={hit_target} (sustained landed at {c_sustained1:.3f}, target {TARGET})  "
          f"clear_margin={clear_margin} (margin {c_sustained1-c_blip1:.3f})")
    print(f"  [{'PASS' if ok_train else 'FAIL'}] fully native (no PyTorch, no autograd library) "
          f"training on the real jet-engine topology converged to the target while keeping "
          f"blip correctly discriminated")
