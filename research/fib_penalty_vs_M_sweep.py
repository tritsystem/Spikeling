#!/usr/bin/env python3
"""Does the Fibonacci-connectivity penalty DECAY with population size M?

WHY THIS IS A NEW CLAIM, not a re-run: prior work measured a real ~1.9% cost at
M=24 (12 seeds, CI [1.012,1.025], excludes 1.0). The M=96 keystone run then came
back at parity (0.991, CI [0.982,1.001]) -- but that was only 5 seeds, and two
points is not a trend. If the penalty genuinely decays with M, that is a NEW
finding and it strengthens the embedded-reservoir spec (bigger population = the
rule gets cheaper, and bigger population is exactly what the noise-margin result
wants). If it does not decay, the honest story is "small constant penalty,
roughly M-independent" -- still fine for the product, but a different claim.

PRE-REGISTERED (before running):
  P3 DECAY: the fib/random recall-NMSE ratio DECREASES with M.
     Operationally: ratio(M=192) < ratio(M=24), AND the M=24 CI sits above the
     M=192 CI (no overlap of the point estimates' direction).
  DISCONFIRM P3: ratio is flat within noise across M (all CIs overlap), or it
     INCREASES with M. Either kills "it gets cheaper at scale" -- report plainly.

  P4 (re-check, 12 seeds this time): at M=96 the ratio CI includes 1.0 (parity).
     The 5-seed keystone said so; this re-tests it at the prior study's seed count.

DISCIPLINE: 12 seeds per M (matches the prior variance check that CAUGHT a false
3-seed "near parity" read). Reuses fib_K / random_degree2_K / run verbatim.
Instrument check prints per-unit cost so we can see if large M is measuring
Python overhead rather than the substrate.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import numpy as np
from runtime.runtime import ResonatorState

FREQ = 1.0 / (2 * np.pi); DAMPING = 0.3; DT = 0.05; HOLD = 20; T = 800; BURN = 150; RHO = 0.7
V, W = 0.4, 1.0

def fib_word(g):
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w
LONG_WORD = fib_word(16)   # 1597 -- covers M up to ~1598

def fib_K(M):
    word = LONG_WORD[: M - 1]
    K = np.zeros((M, M))
    for i, sym in enumerate(word):
        c = V if sym == "A" else W
        K[i, i + 1] = K[i + 1, i] = c
    return RHO * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def random_degree2_K(M, seed=0):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M); vals = rng.choice([V, W], size=M - 1)
    K = np.zeros((M, M))
    for i in range(M - 1):
        a, b = perm[i], perm[i + 1]
        K[a, b] = K[b, a] = vals[i]
    return RHO * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def run(K, u, win):
    M = K.shape[0]
    res = [ResonatorState(name=f"r{i}", freq_hz=FREQ, damping=DAMPING, coupling=1.0) for i in range(M)]
    X = np.zeros((len(u), 2 * M))
    for t in range(len(u)):
        for _ in range(HOLD):
            x = np.array([r.x for r in res])
            drive = win * u[t] + K @ x
            for i, r in enumerate(res):
                r.step(float(drive[i]), DT)
        xf = np.array([r.x for r in res])
        X[t] = np.concatenate([xf, xf ** 2])
    return X

ridge = lambda X, y, lam=1e-2: np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)
nmse = lambda p, y: float(np.mean((p - y) ** 2) / (np.var(y) + 1e-12))

def recall_nmse(K, seed=0, delay=2):
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(T); win = rng.uniform(-1, 1, K.shape[0])
    y = np.zeros(T); y[delay:] = u[:-delay]; yb = y[BURN:]
    X = run(K, u, win)[BURN:]
    return nmse(X @ ridge(X, yb), yb)

MS = [24, 48, 96, 192]
SEEDS = 12
print(f"PENALTY vs POPULATION SIZE -- M sweep {MS}, {SEEDS} seeds each")
print("P3: ratio DECREASES with M | P4: M=96 CI includes 1.0 at 12 seeds\n")
print(f"{'M':>5} | {'fib NMSE':>9} | {'rnd NMSE':>9} | {'ratio':>7} | {'95% CI':>16} | {'ms/unit':>8}")
print("-" * 72)

results = {}
for M in MS:
    t0 = time.time(); ratios = []; fibs = []; rnds = []
    for s in range(SEEDS):
        f = recall_nmse(fib_K(M), seed=s)
        r = recall_nmse(random_degree2_K(M, seed=s), seed=s)
        fibs.append(f); rnds.append(r); ratios.append(f / max(r, 1e-9))
    ratios = np.array(ratios)
    m = ratios.mean(); sem = ratios.std() / np.sqrt(SEEDS)
    lo, hi = m - 1.96 * sem, m + 1.96 * sem
    per_unit = (time.time() - t0) / (SEEDS * 2 * M) * 1000
    results[M] = (m, lo, hi)
    print(f"{M:5d} | {np.mean(fibs):9.4f} | {np.mean(rnds):9.4f} | {m:7.3f} | "
          f"[{lo:6.3f},{hi:6.3f}] | {per_unit:8.2f}", flush=True)

print("\n--- P3: does the penalty DECAY with population size? ---")
m24, lo24, hi24 = results[24]; m192, lo192, hi192 = results[192]
print(f"  M=24  ratio {m24:.3f}  CI [{lo24:.3f},{hi24:.3f}]")
print(f"  M=192 ratio {m192:.3f}  CI [{lo192:.3f},{hi192:.3f}]")
decayed = m192 < m24 and lo24 > hi192
print(f"  ratio(192) < ratio(24)? {m192 < m24}   |  CIs separated (24 above 192)? {lo24 > hi192}")
print(f"  P3 {'MET -- the penalty genuinely shrinks with population (NEW finding)' if decayed else 'NOT MET -- no clean decay; honest read = small, roughly M-independent penalty'}")

print("\n--- P4: is M=96 at parity with 12 seeds? ---")
m96, lo96, hi96 = results[96]
print(f"  M=96 ratio {m96:.3f}  CI [{lo96:.3f},{hi96:.3f}]  -> "
      f"{'includes 1.0 (parity CONFIRMED at 12 seeds)' if lo96 <= 1.0 <= hi96 else 'EXCLUDES 1.0 -- the 5-seed parity read did NOT hold'}")

print("\nmonotone across all M?",
      all(results[MS[i]][0] >= results[MS[i+1]][0] for i in range(len(MS)-1)))
print("(if not monotone, say so -- a 2-point 'trend' is exactly what this run exists to check)")
