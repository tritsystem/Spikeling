"""
Rung 1b — phase-preserving RAF readout: recover gradient SIGN from the phase of
the real Spikeling ResonatorState's response, not just |grad| from its energy.

    python raf_gradient_rung1b.py --selftest
    python raf_gradient_rung1b.py                 # Q1 favourable cell
    python raf_gradient_rung1b.py --sweep spacing # Q2
    python raf_gradient_rung1b.py --sweep margin  # Q3 (add --readout hard)
    python raf_gradient_rung1b.py --readout hard --sweep margin

Predictions Q1-Q4 + kill condition: PREREGISTER_1b.md. The ResonatorState is the
shipped class (gate_threshold=0.0 only); we additionally read its .x and .v each
tick and correlate against the known per-channel dither reference.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from raf_gradient_rung1 import (  # reuse rung-1 harness verbatim
    DT, OUT, Readout, make_bank, pearson, spearman,
)


def probe_phase(readout, freqs, A, damping, ticks, drive_gain=None):
    """Same drive path as rung 1, but also accumulate synchronous products of the
    resonator's own x and v against sin/cos references at each channel's f_i."""
    D = len(freqs)
    omegas = 2 * math.pi * np.asarray(freqs)
    theta0 = readout.theta0
    settle = ticks // 2

    if drive_gain is None:
        loss_ema = readout.loss(theta0)
        peaks = []
        for t in range(2000):
            th = theta0 + A * np.sin(omegas * (t * DT))
            L = readout.loss(th)
            loss_ema += 0.01 * (L - loss_ema)
            peaks.append(abs(L - loss_ema))
        wiggle = float(np.percentile(peaks, 90))
        if wiggle < 1e-9:
            nan = np.full(D, np.nan)
            return dict(res_mag=nan, raf_v=nan, raf_iq=nan, loss_lockin=nan, drive_gain=0.0)
        drive_gain = min(1e6, 1e-2 / wiggle)

    bank = make_bank(freqs, damping)
    loss_ema = readout.loss(theta0)
    v_sin = np.zeros(D); x_sin = np.zeros(D); x_cos = np.zeros(D)
    loss_sin = np.zeros(D)
    n = 0

    for t in range(ticks):
        ts = t * DT
        s = np.sin(omegas * ts)
        c = np.cos(omegas * ts)
        th = theta0 + A * s
        L = readout.loss(th)
        loss_ema += 0.01 * (L - loss_ema)
        drive = drive_gain * (L - loss_ema)
        for r in bank:
            r.step(drive, DT)
        if t >= settle:
            vv = np.array([r.v for r in bank])
            xx = np.array([r.x for r in bank])
            v_sin += vv * s
            x_sin += xx * s
            x_cos += xx * c
            loss_sin += L * s
            n += 1

    mag = np.array([math.sqrt(max(r.energy_ema, 0.0)) for r in bank])
    # RAF-v: sign from velocity-reference correlation, magnitude from energy
    raf_v = np.sign(v_sin) * mag
    # RAF-IQ: at resonance the in-phase-with-drive part of displacement is -Q(=x_cos)
    raf_iq = np.sign(-x_cos) * mag
    # ceiling: signed digital lock-in on the loss itself
    loss_lockin = 2.0 * loss_sin / (n * A)
    return dict(res_mag=mag, raf_v=raf_v, raf_iq=raf_iq,
                loss_lockin=loss_lockin, drive_gain=drive_gain)


def sign_agree(est, truth):
    est = np.asarray(est); truth = np.asarray(truth)
    m = np.isfinite(est) & np.isfinite(truth) & (np.abs(truth) > 1e-12)
    if not m.any():
        return float("nan")
    return float(np.mean(np.sign(est[m]) == np.sign(truth[m])))


def run_cell(D, f_lo, f_hi, margin, *, readout_kind="soft", beta=12.0, A=1e-3,
             damping=0.02, ticks=20000, seed=0):
    rd = Readout(D, margin, beta, readout_kind, seed)
    if readout_kind == "hard" and A == 1e-3:
        A = 0.08
    freqs = np.logspace(math.log10(f_lo), math.log10(f_hi), D)
    fd_A = rd.fd_grad(rd.theta0, e=A)
    surro = rd.surrogate_grad(rd.theta0)
    p = probe_phase(rd, freqs, A, damping, ticks)

    return {
        "D": D, "f_lo": f_lo, "f_hi": f_hi, "spacing_ratio": f_hi / f_lo,
        "margin": margin, "readout": readout_kind, "v0": rd.v0, "A": A,
        "drive_gain": p["drive_gain"],
        "sign_agree_raf_v":   sign_agree(p["raf_v"], fd_A),
        "sign_agree_raf_iq":  sign_agree(p["raf_iq"], fd_A),
        "sign_agree_lockin":  sign_agree(p["loss_lockin"], fd_A),
        "r_raf_v_vs_fd":      pearson(p["raf_v"], fd_A),
        "r_raf_iq_vs_fd":     pearson(p["raf_iq"], fd_A),
        "r_lockin_vs_fd":     pearson(p["loss_lockin"], fd_A),
        "rho_mag_vs_absfd":   spearman(p["res_mag"], np.abs(fd_A)),
        "instrument_ok":      bool(np.all(np.sign(fd_A) == np.sign(surro))) if readout_kind == "soft" else None,
        "raf_v": p["raf_v"].tolist(), "fd_A": fd_A.tolist(),
        "loss_lockin": p["loss_lockin"].tolist(), "freqs": freqs.tolist(),
    }


def _print(c):
    print(f"  D={c['D']:2d}  band=[{c['f_lo']:.0f},{c['f_hi']:.0f}] (x{c['spacing_ratio']:.1f})  "
          f"margin={c['margin']}  A={c['A']:.3f}  readout={c['readout']}  drive_gain={c['drive_gain']:.1f}")
    print(f"     sign-agree  RAF-v={c['sign_agree_raf_v']:.3f}   RAF-IQ={c['sign_agree_raf_iq']:.3f}   "
          f"loss-lockin={c['sign_agree_lockin']:.3f}   <- Q1 needs RAF-v >= 0.9")
    print(f"     signed  r   RAF-v={c['r_raf_v_vs_fd']:+.3f}   RAF-IQ={c['r_raf_iq_vs_fd']:+.3f}   "
          f"loss-lockin={c['r_lockin_vs_fd']:+.3f}")
    print(f"     |grad| rho (rung-1 readout, unchanged) = {c['rho_mag_vs_absfd']:+.3f}")
    if c["instrument_ok"] is not None:
        print(f"     instrument check (sign fd == sign surrogate): {'PASS' if c['instrument_ok'] else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["spacing", "margin", "n"], default=None)
    ap.add_argument("--readout", choices=["soft", "hard"], default="soft")
    ap.add_argument("--ticks", type=int, default=20000)
    ap.add_argument("--damping", type=float, default=0.02)
    ap.add_argument("--beta", type=float, default=12.0)
    ap.add_argument("--A", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()
    K = dict(readout_kind=args.readout, beta=args.beta, A=args.A,
             damping=args.damping, ticks=args.ticks, seed=args.seed)

    if args.selftest:
        print("SELFTEST 1b — sign recovery in the favourable cell")
        c = run_cell(8, 300, 3000, 0.3, **{**K, "ticks": 8000})
        _print(c)
        ok = c["sign_agree_raf_v"] >= 0.9 and c["r_raf_v_vs_fd"] >= 0.9
        print(f"\n  Q1 {'PASS' if ok else 'FAIL'}  "
              f"(RAF-v sign-agree={c['sign_agree_raf_v']:.3f} need >=0.9, "
              f"r={c['r_raf_v_vs_fd']:+.3f} need >=0.9)   [{time.perf_counter()-t0:.1f}s]")
        sys.exit(0 if ok else 1)

    res = {"meta": {**{k: v for k, v in K.items()}}, "cells": []}
    if args.sweep is None:
        print("Q1 — favourable cell"); c = run_cell(8, 300, 3000, 0.3, **K); _print(c); res["cells"].append(c)
        tag = f"q1_{args.readout}"
    elif args.sweep == "spacing":
        print("Q2 — compress band, watch sign-agreement")
        for f_hi in (3000, 1500, 750):
            c = run_cell(8, 300, f_hi, 0.3, **K); _print(c); res["cells"].append(c); print()
        tag = f"q2_spacing_{args.readout}"
    elif args.sweep == "margin":
        print("Q3 — move off threshold")
        for m in (0.3, 1.0, 2.0, 4.0):
            c = run_cell(8, 300, 3000, m, **K); _print(c); res["cells"].append(c); print()
        tag = f"q3_margin_{args.readout}"
    elif args.sweep == "n":
        print("sign-agreement vs parameter count")
        for D in (4, 8, 16, 32):
            c = run_cell(D, 300, 3000, 0.3, **K); _print(c); res["cells"].append(c); print()
        tag = f"n_{args.readout}"

    fp = OUT / f"results_1b_{tag}.json"
    fp.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {fp.name}   [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
