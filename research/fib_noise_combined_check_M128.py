#!/usr/bin/env python3
"""Does the M=96 keystone (P1 cost holds + P2 cancellation survives rule-generation)
still hold at M=128 -- the population size PROJECT_embedded_reservoir_sensor_processor.md's
own build order actually asked for, and which was never run?

THE GAP THIS CLOSES: fib_noise_combined_check.py measured P1+P2 together at M=96 (both MET).
The vault spec's honest caveat section says "Tested M=96, not the 128 the spec asked for."
This script closes exactly that gap -- nothing else. If it fails, stage 2+ of the
embedded-reservoir-sensor-processor build plan (device idea #17) do not proceed.

PRE-REGISTERED (before running, identical bars to the M=96 run):
  P1  COST HOLDS: at M=128, per-seed Fibonacci/random recall-NMSE ratio stays small
      (mean ratio <= 1.10). Prior: M=24 was 1.019 CI[1.012,1.025]; M=96 was 0.991 CI[0.982,1.001].
  P2  CANCELLATION SURVIVES RULE-GENERATION: with FIBONACCI connectivity, structured
      (rank-1) noise is still cancelled ~exactly -- structured NMSE within 10% of
      clean -- while RANDOM (full-rank) noise is clearly worse than structured.
  DISCONFIRM P1: ratio mean > 1.10.
  DISCONFIRM P2: structured NMSE > 1.10 * clean under Fibonacci, or structured ~ random.

Reuses fib_K / random_degree2_K / add_noise / rec / run / make_XY VERBATIM from
fib_noise_combined_check.py -- no re-derivation, only M changed (96 -> 128) and the
instrument-check M list extended to include 128 itself (so the bench's own per-unit
cost at the actual tested population is reported, not just interpolated from smaller M).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "core"))
import numpy as np
from runtime.runtime import ResonatorState

# ---- identical constants to fib_noise_combined_check.py ----
FREQ = 1.0 / (2 * np.pi); DAMPING = 0.3; DT = 0.05; HOLD = 20; T = 800; BURN = 150; RHO = 0.7
V, W = 0.4, 1.0

def fib_word(g):
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w
LONG_WORD = fib_word(16)   # length 1597 -- covers M up to ~1598

def fib_K(M):                                   # verbatim
    word = LONG_WORD[: M - 1]
    K = np.zeros((M, M))
    for i, sym in enumerate(word):
        c = V if sym == "A" else W
        K[i, i + 1] = K[i + 1, i] = c
    return RHO * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def random_degree2_K(M, seed=0):                # verbatim
    rng = np.random.default_rng(seed)
    perm = rng.permutation(M)
    vals = rng.choice([V, W], size=M - 1)
    K = np.zeros((M, M))
    for i in range(M - 1):
        a, b = perm[i], perm[i + 1]
        K[a, b] = K[b, a] = vals[i]
    return RHO * K / (np.max(np.abs(np.linalg.eigvalsh(K))) + 1e-9)

def run(K, u, win):                             # verbatim -- real ResonatorState dynamics
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

def add_noise(X, kind, frac, seed):             # verbatim
    r = np.random.default_rng(7000 + seed); nrow, d = X.shape; amp = frac * X.std()
    if kind == "structured":
        c = np.zeros(nrow)
        for t in range(1, nrow): c[t] = 0.9 * c[t - 1] + r.standard_normal()
        return X + amp * np.outer(c / (c.std() + 1e-9), np.ones(d))
    if kind == "random":
        return X + amp * r.standard_normal((nrow, d))
    return X

def rec(X, yb, kind, frac, seed):               # verbatim
    Xn = add_noise(X, kind, frac, seed); tr = int(0.6 * len(yb))
    return nmse(Xn[tr:] @ ridge(Xn[:tr], yb[:tr]), yb[tr:])

def make_XY(K, seed, delay=2):                  # verbatim
    rng = np.random.default_rng(seed)
    u = rng.standard_normal(T); win = rng.uniform(-1, 1, K.shape[0])
    y = np.zeros(T); y[delay:] = u[:-delay]
    return run(K, u, win)[BURN:], y[BURN:]

# ───────────────── INSTRUMENT CHECK ─────────────────
print("INSTRUMENT CHECK -- is the bench measuring the substrate or its own Python overhead?")
for m in (16, 48, 96, 128):
    t0 = time.time(); make_XY(fib_K(m), 0); dt = time.time() - t0
    print(f"  M={m:3d}: one reservoir run = {dt:6.1f}s  ({dt/m*1000:.1f} ms per unit)")
print()

M, SEEDS, FRACS = 128, 5, (0.3, 0.6)
print(f"COMBINED TEST at M={M} (prior work: M=24, M=96 both MET), {SEEDS} seeds")
print("P1 cost holds (<=1.10) | P2 fib still cancels structured noise (within 10% of clean)\n")

ratios, rows = [], []
t0 = time.time()
for s in range(SEEDS):
    Xf, yb = make_XY(fib_K(M), s)
    Xr, _  = make_XY(random_degree2_K(M, seed=s), s)
    tr = int(0.6 * len(yb))
    f_clean = nmse(Xf[tr:] @ ridge(Xf[:tr], yb[:tr]), yb[tr:])
    r_clean = nmse(Xr[tr:] @ ridge(Xr[:tr], yb[:tr]), yb[tr:])
    ratios.append(f_clean / max(r_clean, 1e-9))
    row = {"seed": s, "fib_clean": f_clean, "rnd_clean": r_clean}
    for fr in FRACS:
        row[f"fib_struct_{fr}"] = rec(Xf, yb, "structured", fr, s)
        row[f"fib_rand_{fr}"]   = rec(Xf, yb, "random", fr, s)
    rows.append(row)
    print(f"  seed {s}: fib={f_clean:.4f} rnd={r_clean:.4f} ratio={ratios[-1]:.3f}x  "
          f"| fib+struct(.3)={row['fib_struct_0.3']:.4f} fib+rand(.3)={row['fib_rand_0.3']:.4f}"
          f"  ({time.time()-t0:.0f}s)", flush=True)

ratios = np.array(ratios); mean_r = ratios.mean()
sem = ratios.std() / np.sqrt(len(ratios))
print("\n--- P1: does the Fibonacci cost hold at M=128? ---")
print(f"  ratio mean={mean_r:.3f}  95% CI=[{mean_r-1.96*sem:.3f}, {mean_r+1.96*sem:.3f}]  (M=24 prior: 1.019, M=96 prior: 0.991)")
print(f"  P1 {'MET' if mean_r <= 1.10 else 'NOT MET -- rule-generated connectivity costs real accuracy at scale'}")

print("\n--- P2: does the decoder still cancel structured noise under FIBONACCI connectivity at M=128? ---")
print(f"  {'frac':>5} | {'clean':>8} | {'structured':>10} | {'random':>8} | {'struct/clean':>12}")
p2 = True
for fr in FRACS:
    c = np.mean([r["fib_clean"] for r in rows])
    st = np.mean([r[f"fib_struct_{fr}"] for r in rows])
    rd = np.mean([r[f"fib_rand_{fr}"] for r in rows])
    print(f"  {fr:5.1f} | {c:8.4f} | {st:10.4f} | {rd:8.4f} | {st/max(c,1e-9):12.3f}")
    if st > 1.10 * c or st >= rd: p2 = False
print(f"  P2 {'MET -- cancellation survives rule-generation' if p2 else 'NOT MET -- cancellation does NOT survive rule-generated connectivity'}")

print(f"\nVERDICT: P1={'MET' if mean_r<=1.10 else 'FAILED'}  P2={'MET' if p2 else 'FAILED'}")
print("(this closes the M=128 gap PROJECT_embedded_reservoir_sensor_processor.md's own")
print(" build order flagged as open -- a failure here kills stage 2+ of device idea #17 honestly)")
