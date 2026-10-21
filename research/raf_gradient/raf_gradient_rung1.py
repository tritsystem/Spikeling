"""
Rung 1 — can a bank of the REAL Spikeling ResonatorState oscillators read a
gradient off a hard-threshold readout by resonance?

    python raf_gradient_rung1.py --selftest          # instrument check only, fast
    python raf_gradient_rung1.py                      # one favourable cell (P1)
    python raf_gradient_rung1.py --sweep spacing      # P2
    python raf_gradient_rung1.py --sweep n            # P3
    python raf_gradient_rung1.py --sweep margin       # P4
    python raf_gradient_rung1.py --readout hard       # true step, FD-only ground truth

Pre-registered predictions + kill condition: see PREREGISTER.md. Nothing in
ResonatorState is modified except gate_threshold=0.0 (force the ungated energy
path); the oscillator math and .step() are the shipped code.
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "..", "core"))
from runtime.runtime import ResonatorState, DEFAULT_RESONATOR_BASE_GAIN  # noqa: E402

DT = 1.0 / 40000.0          # Spikeling native sample rate
OUT = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────
#  Target system: a one-layer threshold readout with a known gradient
# ─────────────────────────────────────────────────────────────────────
class Readout:
    def __init__(self, D, margin, beta, kind, seed):
        rng = np.random.default_rng(seed)
        self.x = rng.standard_normal(D)
        self.x /= np.linalg.norm(self.x)
        self.theta0 = 0.05 * rng.standard_normal(D)
        self.beta = beta
        self.kind = kind
        self.target = 0.0
        # place the operating point `margin` logistic-units from the flip:
        # beta * v0 = margin  ->  bias = theta0·x - margin/beta
        self.bias = float(self.theta0 @ self.x - margin / beta)
        self.v0 = float(self.theta0 @ self.x - self.bias)

    def v(self, theta):
        return float(theta @ self.x - self.bias)

    def loss(self, theta):
        v = self.v(theta)
        if self.kind == "soft":
            y = 1.0 / (1.0 + math.exp(-self.beta * v))
        else:                                  # hard step — the real spike
            y = 1.0 if v > 0.0 else 0.0
        return 0.5 * (y - self.target) ** 2

    def surrogate_grad(self, theta):
        """Closed form for the soft readout; zero everywhere for hard."""
        if self.kind != "soft":
            return np.zeros_like(theta)
        v = self.v(theta)
        y = 1.0 / (1.0 + math.exp(-self.beta * v))
        return (y - self.target) * y * (1.0 - y) * self.beta * self.x

    def fd_grad(self, theta, e):
        g = np.zeros_like(theta)
        for i in range(len(theta)):
            tp = theta.copy(); tp[i] += e
            tm = theta.copy(); tm[i] -= e
            g[i] = (self.loss(tp) - self.loss(tm)) / (2.0 * e)
        return g


# ─────────────────────────────────────────────────────────────────────
#  The resonator-bank gradient probe
# ─────────────────────────────────────────────────────────────────────
def make_bank(freqs, damping):
    bank = []
    for f in freqs:
        omega = 2 * math.pi * f
        bank.append(ResonatorState(
            name=f"R{f:.0f}", freq_hz=f, damping=damping,
            coupling=DEFAULT_RESONATOR_BASE_GAIN * (omega ** 2),
            gate_threshold=0.0,                # force ungated energy path
        ))
    return bank


def probe(readout, freqs, A, damping, ticks, drive_gain=None):
    """Dither every parameter at its own f_i, feed the DC-removed loss wiggle as
    one shared drive into a real ResonatorState per parameter, read sqrt(energy_ema).
    Also computes an ideal digital lock-in (signed) as a reference ceiling."""
    D = len(freqs)
    omegas = 2 * math.pi * np.asarray(freqs)
    theta0 = readout.theta0
    settle = ticks // 2

    # ---- warm-up to auto-calibrate drive_gain so amplitudes are sane ----
    dead = False
    if drive_gain is None:
        loss_ema = readout.loss(theta0)
        peaks = []
        for t in range(2000):
            th = theta0 + A * np.sin(omegas * (t * DT))
            L = readout.loss(th)
            loss_ema += 0.01 * (L - loss_ema)
            peaks.append(abs(L - loss_ema))
        wiggle = float(np.percentile(peaks, 90))
        if wiggle < 1e-9:                     # loss never moved: no threshold crossing
            return np.full(D, np.nan), np.full(D, np.nan), 0.0
        drive_gain = min(1e6, 1e-2 / wiggle)

    bank = make_bank(freqs, damping)
    loss_ema = readout.loss(theta0)
    lockin_acc = np.zeros(D)
    n_meas = 0

    for t in range(ticks):
        ts = t * DT
        s = np.sin(omegas * ts)
        th = theta0 + A * s
        L = readout.loss(th)
        loss_ema += 0.01 * (L - loss_ema)
        drive = drive_gain * (L - loss_ema)
        for r in bank:
            r.step(drive, DT)
        if t >= settle:
            lockin_acc += L * s
            n_meas += 1

    res_mag = np.array([math.sqrt(max(r.energy_ema, 0.0)) for r in bank])
    lockin = 2.0 * lockin_acc / (n_meas * A)          # signed, ideal demodulator
    return res_mag, lockin, drive_gain


# ─────────────────────────────────────────────────────────────────────
#  Metrics
# ─────────────────────────────────────────────────────────────────────
def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return float("nan")
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def pearson(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    if np.allclose(a.std(), 0) or np.allclose(b.std(), 0):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def run_cell(D, f_lo, f_hi, margin, *, readout_kind="soft", beta=12.0, A=1e-3,
             damping=0.02, ticks=20000, seed=0):
    rd = Readout(D, margin, beta, readout_kind, seed)
    if readout_kind == "hard" and A == 1e-3:
        A = 0.08                              # sub-threshold 1e-3 dither reads nothing off a step
    freqs = np.logspace(math.log10(f_lo), math.log10(f_hi), D)
    fd_A = rd.fd_grad(rd.theta0, e=A)
    fd_sharp = rd.fd_grad(rd.theta0, e=1e-4)
    surro = rd.surrogate_grad(rd.theta0)
    res_mag, lockin, g = probe(rd, freqs, A, damping, ticks)
    rng = np.random.default_rng(seed + 999)
    randv = rng.standard_normal(D)

    return {
        "D": D, "f_lo": f_lo, "f_hi": f_hi, "spacing_ratio": f_hi / f_lo,
        "margin": margin, "readout": readout_kind, "v0": rd.v0,
        "drive_gain": g,
        "rho_res_vs_absfd":     spearman(res_mag, np.abs(fd_A)),
        "rho_lockin_vs_absfd":  spearman(np.abs(lockin), np.abs(fd_A)),
        "r_lockin_signed_vs_fd": pearson(lockin, fd_A),
        "rho_surrogate_vs_fd":  spearman(surro, fd_A) if readout_kind == "soft" else None,
        "rho_random_vs_fd":     spearman(randv, np.abs(fd_A)),
        "rho_fdA_vs_fdsharp":   spearman(fd_A, fd_sharp),
        "res_mag": res_mag.tolist(), "lockin": lockin.tolist(),
        "fd_A": fd_A.tolist(), "surrogate": surro.tolist(),
        "freqs": freqs.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────
def _print_cell(c):
    print(f"  D={c['D']:2d}  band=[{c['f_lo']:.0f},{c['f_hi']:.0f}]Hz "
          f"(x{c['spacing_ratio']:.1f})  margin={c['margin']}  v0={c['v0']:+.3f}  "
          f"readout={c['readout']}  drive_gain={c['drive_gain']:.1f}")
    print(f"     rho(resonator_mag , |FD|)     = {c['rho_res_vs_absfd']:+.3f}   <- P1 estimator")
    print(f"     rho(ideal_lockin  , |FD|)     = {c['rho_lockin_vs_absfd']:+.3f}   (naive finite-window lock-in, NOT a bound)")
    print(f"     r  (lockin signed , FD)       = {c['r_lockin_signed_vs_fd']:+.3f}   (does sign survive)")
    if c["rho_surrogate_vs_fd"] is not None:
        print(f"     rho(surrogate     , FD)       = {c['rho_surrogate_vs_fd']:+.3f}   <- INSTRUMENT CHECK (>0.95)")
    print(f"     rho(random vec     , |FD|)    = {c['rho_random_vs_fd']:+.3f}   (null anchor ~0)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["spacing", "n", "margin"], default=None)
    ap.add_argument("--readout", choices=["soft", "hard"], default="soft")
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--damping", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=12.0)
    ap.add_argument("--A", type=float, default=1e-3, help="dither amplitude (hard readout auto-bumps 1e-3->0.08)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()

    if args.selftest:
        print("SELFTEST — harness validity (instrument check only)")
        c = run_cell(4, 300, 3000, margin=0.3, readout_kind="soft",
                     beta=args.beta, A=args.A, damping=args.damping, ticks=8000, seed=args.seed)
        _print_cell(c)
        ok = c["rho_surrogate_vs_fd"] is not None and c["rho_surrogate_vs_fd"] > 0.95
        print(f"\n  instrument check {'PASS' if ok else 'FAIL'} "
              f"(rho surrogate-vs-FD = {c['rho_surrogate_vs_fd']:+.3f}, need > 0.95)")
        print(f"  [{time.perf_counter()-t0:.1f}s]")
        sys.exit(0 if ok else 1)

    results = {"meta": {"readout": args.readout, "ticks": args.ticks,
                        "damping": args.damping, "beta": args.beta, "seed": args.seed},
               "cells": []}

    if args.sweep is None:
        print("P1 — most favourable cell")
        c = run_cell(4, 300, 3000, margin=0.3, readout_kind=args.readout,
                     beta=args.beta, A=args.A, damping=args.damping, ticks=args.ticks, seed=args.seed)
        _print_cell(c); results["cells"].append(c)
        tag = f"p1_{args.readout}"

    elif args.sweep == "spacing":
        print("P2 — compress the frequency band (cross-talk)")
        for f_hi in (3000, 1500, 750):
            c = run_cell(8, 300, f_hi, margin=0.3, readout_kind=args.readout,
                         beta=args.beta, A=args.A, damping=args.damping, ticks=args.ticks, seed=args.seed)
            _print_cell(c); results["cells"].append(c); print()
        tag = f"p2_spacing_{args.readout}"

    elif args.sweep == "n":
        print("P3 — grow the parameter count at fixed band (bandwidth saturation)")
        for D in (4, 8, 16, 32):
            c = run_cell(D, 300, 3000, margin=0.3, readout_kind=args.readout,
                         beta=args.beta, A=args.A, damping=args.damping, ticks=args.ticks, seed=args.seed)
            _print_cell(c); results["cells"].append(c); print()
        tag = f"p3_n_{args.readout}"

    elif args.sweep == "margin":
        print("P4 — move away from threshold (is the bank measuring the real slope)")
        for m in (0.3, 1.0, 2.0, 4.0):
            c = run_cell(8, 300, 3000, margin=m, readout_kind=args.readout,
                         beta=args.beta, A=args.A, damping=args.damping, ticks=args.ticks, seed=args.seed)
            _print_cell(c); results["cells"].append(c); print()
        tag = f"p4_margin_{args.readout}"

    fp = OUT / f"results_{tag}.json"
    fp.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {fp.name}   [{time.perf_counter()-t0:.1f}s]")
    print("fill RESULTS.md against PREREGISTER.md's P1–P4 + kill condition.")


if __name__ == "__main__":
    main()
