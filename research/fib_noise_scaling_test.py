#!/usr/bin/env python3
"""Extends fib_noise_combined_check.py's own identified next step (an M
sweep 24/48/96/192, to test whether the Fibonacci-connectivity cost
penalty keeps shrinking with population size, "suggestive... not
established") -- pushed further using a properly VECTORIZED reservoir
simulator, since the original's per-object ResonatorState.step() Python
loop (M separate objects, called individually every substep) caps out
around M~100-200 before becoming impractically slow.

THE PHYSICS IS UNCHANGED -- verified, not assumed. VectorizedResonatorBank
implements the EXACT SAME equations as core/runtime/runtime.py's
ResonatorState.step() (symplectic Euler: accel = -omega^2*x - 2*damping*
omega*v + coupling*drive; v+=accel*dt; x+=v*dt), just computed as array
ops across all M units at once instead of per-object. Verified to match
the original bit-for-bit (well within float precision) at M=16 before
trusting it at any larger scale.

BOTH K matrices (fib_K, random_degree2_K) are SPARSE by construction --
each is a weighted CHAIN (K[i,i+1]=K[i+1,i]=c, or the same on a permuted
node order) with only M-1 nonzero edges, never O(M^2) despite being
built as dense M x M arrays in the original script. Represented here as
edge lists (O(M) memory) so M can scale far past what a dense matrix
would allow.

add_noise / rec / ridge / nmse are imported UNCHANGED from the original
script -- they operate on the resulting feature matrix, not on the
reservoir's internals, so there's nothing to re-verify there.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fib_noise_combined_check import add_noise, rec, ridge, nmse, fib_word, LONG_WORD  # noqa: E402
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
from runtime.runtime import ResonatorState  # noqa: E402

FREQ = 1.0 / (2 * np.pi)
DAMPING = 0.3
DT = 0.05
HOLD = 20
T = 800
BURN = 150
RHO = 0.7
V, W = 0.4, 1.0
OMEGA = 2 * np.pi * FREQ


def fib_edges(M):
    """Sparse edge list (i, i+1, weight) for the SAME Fibonacci-word
    connectivity as fib_K, RHO-normalized the same way (spectral radius
    of the dense matrix -- computed via the tridiagonal eigenvalues,
    exact, not approximated)."""
    word = LONG_WORD[: M - 1]
    weights = np.array([V if sym == "A" else W for sym in word])
    # spectral radius of a weighted PATH graph = exact via the dense
    # eigval computation on the (still small enough) M for normalization
    # purposes only -- done once, not per-timestep, so cost is fine even
    # at moderate M; for very large M this could be replaced by a
    # tridiagonal eigenvalue solver, not needed at the scale tested here.
    K_dense = np.zeros((M, M))
    idx = np.arange(M - 1)
    K_dense[idx, idx + 1] = weights
    K_dense[idx + 1, idx] = weights
    norm = np.max(np.abs(np.linalg.eigvalsh(K_dense))) + 1e-9
    return idx, idx + 1, RHO * weights / norm


def random_degree2_edges(M, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    weights = rng.choice([V, W], size=M - 1)
    K_dense = np.zeros((M, M))
    a, b = perm[:-1], perm[1:]
    K_dense[a, b] = weights
    K_dense[b, a] = weights
    norm = np.max(np.abs(np.linalg.eigvalsh(K_dense))) + 1e-9
    return a, b, RHO * weights / norm


class VectorizedResonatorBank:
    """Same physics as ResonatorState.step(), vectorized across M units,
    with a SPARSE (edge-list) coupling matrix instead of a dense one."""

    def __init__(self, M, src, dst, weights):
        self.M = M
        self.src, self.dst, self.weights = src, dst, weights
        self.x = np.zeros(M)
        self.v = np.zeros(M)

    def coupling_term(self):
        """K @ x for a sparse symmetric weighted graph, via scatter-add --
        O(edges) = O(M), never the O(M^2) a dense matmul would cost."""
        out = np.zeros(self.M)
        contrib = self.weights * self.x[self.dst]
        np.add.at(out, self.src, contrib)
        contrib2 = self.weights * self.x[self.src]
        np.add.at(out, self.dst, contrib2)
        return out

    def step(self, drive_external):
        drive = drive_external + self.coupling_term()
        accel = -(OMEGA ** 2) * self.x - 2 * DAMPING * OMEGA * self.v + drive
        self.v += accel * DT
        self.x += self.v * DT


def run_vectorized(src, dst, weights, u, win):
    M = len(win)
    bank = VectorizedResonatorBank(M, src, dst, weights)
    X = np.zeros((len(u), 2 * M))
    for t in range(len(u)):
        for _ in range(HOLD):
            bank.step(win * u[t])
        X[t] = np.concatenate([bank.x, bank.x ** 2])
    return X


def make_XY_vectorized(src, dst, weights, M, seed, delay=2):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(T)
    win = rng.uniform(-1, 1, M)
    y = np.zeros(T)
    y[delay:] = u[:-delay]
    return run_vectorized(src, dst, weights, u, win)[BURN:], y[BURN:]


# ---- CORRECTNESS CHECK: vectorized vs. the original per-object ResonatorState, at small M ----
def _verify_matches_original(M=16, seed=0):
    print(f"verifying vectorized bank matches the original ResonatorState object-per-unit "
          f"implementation exactly, at M={M}...")
    src, dst, weights = fib_edges(M)
    K_dense = np.zeros((M, M))
    K_dense[src, dst] = weights
    K_dense[dst, src] = weights

    rng = np.random.default_rng(seed)
    u = rng.standard_normal(50)  # short run, just for the correctness check
    win = rng.uniform(-1, 1, M)

    # original: per-object ResonatorState
    res = [ResonatorState(name=f"r{i}", freq_hz=FREQ, damping=DAMPING, coupling=1.0) for i in range(M)]
    X_orig = np.zeros((len(u), 2 * M))
    for t in range(len(u)):
        for _ in range(HOLD):
            x = np.array([r.x for r in res])
            drive = win * u[t] + K_dense @ x
            for i, r in enumerate(res):
                r.step(float(drive[i]), DT)
        xf = np.array([r.x for r in res])
        X_orig[t] = np.concatenate([xf, xf ** 2])

    # vectorized
    X_vec = run_vectorized(src, dst, weights, u, win)

    max_diff = float(np.max(np.abs(X_orig - X_vec)))
    print(f"  max abs diff (vectorized vs. original ResonatorState physics): {max_diff:.2e}")
    ok = max_diff < 1e-8
    print(f"  [{'PASS' if ok else 'FAIL'}] vectorized bank is numerically identical to the "
          f"original per-object implementation")
    return ok


if __name__ == "__main__":
    print("=" * 78)
    print("  FIBONACCI COST-PENALTY SCALING -- extending the vault's own identified next step")
    print("  (M sweep beyond 192, using a verified-identical vectorized reservoir)")
    print("=" * 78)

    if not _verify_matches_original():
        print("\nABORTING sweep -- vectorized implementation does not match the verified physics.")
        sys.exit(1)
    print()

    M_VALUES = [96, 192, 384, 768, 1536]
    SEEDS = 3  # reduced from the original 5, for total runtime -- flagged, not hidden
    print(f"Sweeping M={M_VALUES}, {SEEDS} seeds each (reduced from the original 5, to keep "
          f"total runtime reasonable -- a real scope reduction, stated plainly)\n")

    results = []
    for M in M_VALUES:
        t0 = time.time()
        ratios = []
        for s in range(SEEDS):
            fsrc, fdst, fw = fib_edges(M)
            rsrc, rdst, rw = random_degree2_edges(M, seed=s)
            Xf, yb = make_XY_vectorized(fsrc, fdst, fw, M, s)
            Xr, _ = make_XY_vectorized(rsrc, rdst, rw, M, s)
            tr = int(0.6 * len(yb))
            f_clean = nmse(Xf[tr:] @ ridge(Xf[:tr], yb[:tr]), yb[tr:])
            r_clean = nmse(Xr[tr:] @ ridge(Xr[:tr], yb[:tr]), yb[tr:])
            ratios.append(f_clean / max(r_clean, 1e-9))
        ratios = np.array(ratios)
        mean_r, sem = ratios.mean(), ratios.std() / np.sqrt(len(ratios))
        elapsed = time.time() - t0
        results.append((M, mean_r, sem))
        print(f"  M={M:5d}: ratio mean={mean_r:.4f}  95% CI=[{mean_r-1.96*sem:.4f}, {mean_r+1.96*sem:.4f}]  "
              f"({elapsed:.1f}s, {SEEDS} seeds)", flush=True)

    print("\n--- Does the Fibonacci cost penalty keep shrinking with population size? ---")
    print(f"  {'M':>6} | {'ratio':>8} | prior context")
    print(f"  {24:>6} | {1.019:>8.3f} | prior work (fib_noise_combined_check.py history)")
    print(f"  {96:>6} | {'~1.001':>8} | keystone-tested (2026-07-15)")
    for M, mean_r, sem in results:
        print(f"  {M:>6} | {mean_r:>8.4f} | this run")
