#!/usr/bin/env python
"""
pyspike_reservoir_attention_heterogeneous.py -- applies the depth-vs-
diversity finding from Documents/phononic-reservoir-cluster/ to this
project's own real ReservoirBank + attention module, at the user's
explicit request ("apply it to an existing Spikeling reservoir-attention
module and measure").

pyspike_reservoir_attention_hybrid.py's ReservoirBank is a SINGLE
homogeneous Fibonacci-coupled damped-oscillator bank (M=48 units,
verified bit-for-bit against core/runtime/runtime.py's real
ResonatorState.step()). This file tests whether replacing it with a
HETEROGENEOUS cluster of 3 differently-coupled sub-banks (Fibonacci +
periodic-SSH + random, 16 units each, same M_total=48) changes the real
marker-relative-recall task's NMSE -- everything else (task, seed, train/
test split, ReservoirAttentionReadout, run_stage's training loop) is
REUSED UNCHANGED from pyspike_reservoir_attention_hybrid.py, so only the
reservoir's internal coupling structure differs between conditions.

PRE-REGISTERED (stated before running): the phononic-reservoir-cluster
finding was "heterogeneity never underperforms a solo specialist, but
whether it beats a HOMOGENEOUS ensemble of matched size depends on
whether the homogeneous type happens to already be a good fit for the
task." This task's original single-Fibonacci bank was hand-picked (not
grid-tuned across structures) -- there is no prior reason to expect
Fibonacci is specifically ill-suited here the way `random(density=0.1)`
was well-suited to NARMA10 in the other experiment. So the PRIOR here is
genuinely uncertain, not skewed toward "heterogeneous wins" -- reported
honestly regardless of which way it comes out.

Three conditions, same M_total=48, same everything else:
  A. ORIGINAL homogeneous Fibonacci-48 (the existing, already-reported
     Stage 1/2 baseline -- re-run here for a same-process comparison,
     not just quoting the old printed numbers).
  B. Homogeneous-diverse-seed: 3x independent Fibonacci-16 sub-banks
     (same coupling TYPE, different random win/seed) -- isolates "does
     just having 3 separate sub-banks help" from "does coupling TYPE
     diversity help."
  C. Heterogeneous: Fibonacci-16 + periodic-SSH-16 + random-16.
"""
import numpy as np
import torch
import torch.nn as nn

from pyspike_reservoir_attention_hybrid import (
    DEVICE, T, K_LAG, M_RESERVOIR, OMEGA, DAMPING, DT, HOLD, RHO, V, W,
    make_task, ReservoirAttentionReadout, LinearBaseline, run_stage, fib_edges,
)


def ssh_periodic_edges(M, seed=0):
    """Same period-2 v/w alternation as topological-phononics/
    fibonacci_connectivity.py's ssh_periodic_K, adapted to this file's
    sparse (src,dst,weight) edge-list convention instead of a dense
    matrix -- a genuinely different topology from the Fibonacci word
    (periodic, not aperiodic), same coupling magnitudes (V, W)."""
    idx = np.arange(M - 1)
    weights = np.array([V if i % 2 == 0 else W for i in idx], dtype=np.float32)
    K_dense = np.zeros((M, M), dtype=np.float32)
    K_dense[idx, idx + 1] = weights
    K_dense[idx + 1, idx] = weights  # symmetrize -- eigvalsh's default UPLO='L' silently
                                      # reads the zero lower triangle otherwise, normalizing
                                      # by ~1e-9 and exploding the weights (caught via NaN output)
    norm = np.max(np.abs(np.linalg.eigvalsh(K_dense))) + 1e-9
    return idx, idx + 1, RHO * weights / norm


def random_edges(M, density=0.15, seed=0):
    """Random sparse graph among M nodes (not restricted to a chain) --
    same density convention as topological-phononics/fibonacci_
    connectivity.py's random_K, adapted to a sparse edge list."""
    rng = np.random.default_rng(seed)
    mask = np.triu(rng.random((M, M)) < density, k=1)
    src, dst = np.nonzero(mask)
    raw_w = rng.normal(0, 1, size=len(src)).astype(np.float32)
    K_dense = np.zeros((M, M), dtype=np.float32)
    K_dense[src, dst] = raw_w
    K_dense[dst, src] = raw_w
    norm = np.max(np.abs(np.linalg.eigvalsh(K_dense))) + 1e-9
    return src, dst, RHO * raw_w / norm


class SubBank(nn.Module):
    """One damped-oscillator sub-reservoir -- the exact same stepping
    physics as ReservoirBank.forward() in the hybrid module (bit-for-bit
    reused equations), factored out so several differently-coupled
    instances can be stepped in lockstep and concatenated."""

    def __init__(self, M, src, dst, weights, seed):
        super().__init__()
        self.M = M
        self.register_buffer("src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(dst, dtype=torch.long))
        self.register_buffer("weights", torch.tensor(weights))
        rng = np.random.default_rng(seed)
        self.register_buffer("win", torch.tensor(rng.uniform(-1, 1, M).astype(np.float32)))

    def step(self, x, v, u_t):
        B = x.shape[0]
        coupling = torch.zeros(B, self.M, device=x.device)
        coupling.index_add_(1, self.src, self.weights * x[:, self.dst])
        coupling.index_add_(1, self.dst, self.weights * x[:, self.src])
        drive = self.win.unsqueeze(0) * u_t + coupling
        accel = -(OMEGA ** 2) * x - 2 * DAMPING * OMEGA * v + drive
        v = v + accel * DT
        x = x + v * DT
        return x, v


class MultiBankReservoir(nn.Module):
    """Steps several SubBanks in lockstep (same u_t, same HOLD/DT
    schedule as the original ReservoirBank), concatenates their [x,v]
    outputs along the feature axis -- the cluster-combination layer,
    same concept as phononic-reservoir-cluster's ClusterReservoir but
    for this project's real resonator-bank substrate instead of tanh
    reservoirs."""

    def __init__(self, banks: list[SubBank]):
        super().__init__()
        self.banks = nn.ModuleList(banks)

    def forward(self, u_batch):
        B, Tlen = u_batch.shape
        xs = [torch.zeros(B, bank.M, device=u_batch.device) for bank in self.banks]
        vs = [torch.zeros(B, bank.M, device=u_batch.device) for bank in self.banks]
        out = []
        for t in range(Tlen):
            u_t = u_batch[:, t:t + 1]
            for _ in range(HOLD):
                for i, bank in enumerate(self.banks):
                    xs[i], vs[i] = bank.step(xs[i], vs[i], u_t)
            tick = torch.cat([torch.cat([xs[i], vs[i]], dim=-1) for i in range(len(self.banks))], dim=-1)
            out.append(tick.clone())
        return torch.stack(out, dim=1)  # (B, T, 2*sum(M_i))


def build_condition(kind: str, m_sub: int = 16, seed_base: int = 100) -> MultiBankReservoir:
    if kind == "homogeneous_fib":
        banks = []
        for i in range(3):
            src, dst, w = fib_edges(m_sub)
            banks.append(SubBank(m_sub, src, dst, w, seed=seed_base + i))
        return MultiBankReservoir(banks)
    if kind == "heterogeneous":
        src1, dst1, w1 = fib_edges(m_sub)
        src2, dst2, w2 = ssh_periodic_edges(m_sub)
        src3, dst3, w3 = random_edges(m_sub, density=0.15, seed=seed_base + 2)
        banks = [
            SubBank(m_sub, src1, dst1, w1, seed=seed_base + 0),
            SubBank(m_sub, src2, dst2, w2, seed=seed_base + 1),
            SubBank(m_sub, src3, dst3, w3, seed=seed_base + 2),
        ]
        return MultiBankReservoir(banks)
    raise ValueError(kind)


def run_condition(name: str, reservoir, feat: int, train_u, train_y, test_u, test_y, seed: int = 0):
    torch.manual_seed(seed)
    baseline = LinearBaseline(feat).to(DEVICE)
    nmse_linear = run_stage(f"{name} -> linear", baseline, reservoir, train_u, train_y, test_u, test_y,
                             is_linear_baseline=True)
    torch.manual_seed(seed)
    attn_model = ReservoirAttentionReadout(feat, use_ternary=False, use_spiking=False).to(DEVICE)
    nmse_attn = run_stage(f"{name} -> attention", attn_model, reservoir, train_u, train_y, test_u, test_y)
    return nmse_linear, nmse_attn


if __name__ == "__main__":
    print("=" * 78)
    print("  HETEROGENEOUS RESERVOIR BANK: applying the depth-vs-diversity")
    print("  finding to the real marker-relative-recall task")
    print("=" * 78)

    rng = np.random.default_rng(0)
    train_u, train_y, _ = make_task(rng, 800)
    test_u, test_y, _ = make_task(rng, 200)

    N_SEEDS = 3
    results = {"A_original_fib48": [], "B_homogeneous_3x16": [], "C_heterogeneous": []}

    for seed in range(N_SEEDS):
        print(f"\n--- seed {seed} ---")

        from pyspike_reservoir_attention_hybrid import ReservoirBank
        reservoir_a = ReservoirBank(M_RESERVOIR).to(DEVICE)
        _, nmse_a = run_condition("A original Fib-48", reservoir_a, 2 * M_RESERVOIR,
                                   train_u, train_y, test_u, test_y, seed=seed)

        reservoir_b = build_condition("homogeneous_fib", m_sub=16, seed_base=seed * 10).to(DEVICE)
        _, nmse_b = run_condition("B homogeneous 3x16", reservoir_b, 2 * 48,
                                   train_u, train_y, test_u, test_y, seed=seed)

        reservoir_c = build_condition("heterogeneous", m_sub=16, seed_base=seed * 10).to(DEVICE)
        _, nmse_c = run_condition("C heterogeneous", reservoir_c, 2 * 48,
                                   train_u, train_y, test_u, test_y, seed=seed)

        results["A_original_fib48"].append(nmse_a)
        results["B_homogeneous_3x16"].append(nmse_b)
        results["C_heterogeneous"].append(nmse_c)

    print("\n" + "=" * 78)
    print("SUMMARY (median NMSE over 3 seeds, attention-readout stage, lower better)")
    print("=" * 78)
    for name, vals in results.items():
        print(f"  {name:25s} median NMSE = {np.median(vals):.4f}   (seeds: {[f'{v:.4f}' for v in vals]})")

    med_a = np.median(results["A_original_fib48"])
    med_b = np.median(results["B_homogeneous_3x16"])
    med_c = np.median(results["C_heterogeneous"])
    print()
    if med_c < med_b * 0.95:
        print("HETEROGENEOUS beats the homogeneous-3x-same-type baseline on this task "
              "(coupling-type diversity itself helps, not just having 3 sub-banks).")
    elif med_c > med_b * 1.05:
        print("HETEROGENEOUS is WORSE than the homogeneous-3x-same-type baseline -- "
              "reported honestly, diversity does not help here.")
    else:
        print("HETEROGENEOUS and homogeneous-3x-same-type are comparable -- coupling-type "
              "diversity is not doing meaningful work on this task either way.")
    if med_b < med_a * 0.95 or med_c < med_a * 0.95:
        print("Splitting into 3 sub-banks (of either kind) beats the original single-48 bank.")
    elif med_b > med_a * 1.05 and med_c > med_a * 1.05:
        print("Splitting into 3 sub-banks is WORSE than the original single-48 bank, "
              "regardless of coupling-type diversity -- reported honestly.")
