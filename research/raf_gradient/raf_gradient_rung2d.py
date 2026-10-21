"""
Rung 2d — over-parameterisation as a controlled variable. Teacher/student in the
same architecture class: a hard-threshold teacher with k=4 hyperplanes that the
student can represent EXACTLY at n_hid=k. Sweep student n_hid {4,6,8,12,16} and
test whether 2c's U1/U2 failures recover at matched size (V1/V2) and degrade
monotonically with excess width (V3). U3 (sign through w2 inversions) carried; V5
adds the lr/step sweep RESULTS_2c asked for.

    python raf_gradient_rung2d.py --selftest
    python raf_gradient_rung2d.py                 # matched cell (n_hid=4), 4-method
    python raf_gradient_rung2d.py --sweep nhid    # V1/V2/V3
    python raf_gradient_rung2d.py --sweep lr      # V5

Predictions V1-V5, gates, kill condition: PREREGISTER_2d.md.
"""
import argparse
import json
import math
import sys
import time

import numpy as np

from raf_gradient_rung2 import VecBank, DT, OUT, ResonatorState, DEFAULT_RESONATOR_BASE_GAIN
from raf_gradient_rung2b import dither_amps, fd_delta, raf_delta, scale_match, sgrad
from raf_gradient_rung2c import _diagnose

K = 4


def make_teacher(seed, n=400):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    A = rng.normal(size=(K, 2)); A /= np.linalg.norm(A, axis=1, keepdims=True)
    b = rng.uniform(-1, 1, size=K)
    s = np.array([1, 1, -1, -1], float)
    c = s * rng.uniform(0.6, 1.4, size=K)
    H = (X @ A.T - b > 0).astype(float)                 # (n, K)
    y0 = H @ c
    return X, A, b, c, H, float(y0.mean())


def teacher_y(X, A, b, c, c0mean, beta_flip):
    H = (X @ A.T - b > 0).astype(float)
    cc = c.copy()
    if beta_flip:
        cc[K // 2:] *= -1.0
    return H @ cc - c0mean


class Student:
    def __init__(self, seed, n_hid):
        rng = np.random.default_rng(seed + 100)
        self.W1 = rng.normal(0, 1.0, size=(2, n_hid))
        self.b1 = rng.normal(0, 0.3, size=n_hid)
        self.w2 = 0.5 * np.tile([1.0, -1.0], n_hid)[:n_hid]
        self.b2 = 0.0
        self.n_hid = n_hid

    def forward(self, X, b1_extra=0.0):
        v = X @ self.W1 + self.b1 + b1_extra
        h = (v > 0.0).astype(float)
        return v, h, h @ self.w2 + self.b2

    def loss(self, X, y, b1_extra=0.0):
        _, _, yh = self.forward(X, b1_extra)
        return float(np.mean(0.5 * (yh - y) ** 2))

    def r2(self, X, y):
        _, _, yh = self.forward(X)
        return float(1.0 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2))

    def grads(self, X, y):
        v, h, yh = self.forward(X)
        dyh = (yh - y) / len(y)
        dv = np.outer(dyh, self.w2) * sgrad(v)
        return X.T @ dv, dv.sum(axis=0), h.T @ dyh, dyh.sum()


def train(method, n_hid, seed=0, steps=250, ticks=1500, lr=0.3, log_every=5):
    Xtr, A, b, c, _, c0 = make_teacher(seed, 400)
    Xte, *_ = make_teacher(seed + 1, 400)
    Ate, bte, cte, c0te = A, b, c, c0     # same teacher, fresh inputs -> use train teacher on test X
    net = Student(seed, n_hid)
    fr = np.logspace(math.log10(300), math.log10(3000), n_hid)
    flip = steps // 2

    H = {"step": [], "test_r2": [], "train_loss": [], "maxfd": [], "fwd": []}
    per = []
    unreach = usteps = fwd = 0

    for st in range(steps):
        fl = st >= flip
        ytr = teacher_y(Xtr, A, b, c, c0, fl)
        yte = teacher_y(Xte, A, b, c, c0, fl)
        dW1r, db1r, dw2, db2 = net.grads(Xtr, ytr)
        if method == "bptt":
            dW1, db1 = dW1r, db1r
            fwd += 1
            maxfd = np.nan
        else:
            Ad, reach = dither_amps(net, Xtr)
            usteps += n_hid; unreach += int((~reach).sum())
            fd = fd_delta(net, Xtr, ytr, Ad)
            maxfd = float(np.abs(fd).max())
            mode = {"raf_full": "full", "raf_sign": "sign", "nodepert_ideal": "ideal"}[method]
            est = raf_delta(net, Xtr, ytr, fr, Ad, ticks, mode)
            fwd += ticks + 1500 + 2 * n_hid
            db1 = scale_match(est, fd) if mode != "ideal" else est
            dW1 = np.outer(Xtr.mean(0), db1)
            wt = np.abs(fd); keep = reach & (wt > 0.1 * wt.max())
            agree = (np.sign(db1) == np.sign(fd)).astype(float)
            per.append((st, np.sign(net.w2).tolist(), (agree * keep).tolist(),
                        (wt * keep).tolist(), net.loss(Xtr, ytr)))
        net.W1 -= lr * dW1; net.b1 -= lr * db1; net.w2 -= lr * dw2; net.b2 -= lr * db2
        if st % log_every == 0 or st == steps - 1:
            H["step"].append(st); H["test_r2"].append(net.r2(Xte, yte))
            H["train_loss"].append(net.loss(Xtr, ytr)); H["maxfd"].append(maxfd); H["fwd"].append(fwd)

    d = _diagnose(per, steps, flip)
    d.update(final_test_r2=H["test_r2"][-1], fwd_total=fwd, n_hid=n_hid,
             unreachable_frac=unreach / max(usteps, 1))
    return H, d


def _wsa_at_init(seed, n_hid, ticks=6000):
    Xtr, A, b, c, _, c0 = make_teacher(seed, 400)
    y = teacher_y(Xtr, A, b, c, c0, False)
    net = Student(seed, n_hid)
    Ad, reach = dither_amps(net, Xtr)
    fd = fd_delta(net, Xtr, y, Ad)
    fr = np.logspace(math.log10(300), math.log10(3000), n_hid)
    est = scale_match(raf_delta(net, Xtr, y, fr, Ad, ticks, "full"), fd)
    wt = np.abs(fd); keep = reach & (wt > 0.1 * wt.max())
    r = np.corrcoef(est[keep], fd[keep])[0, 1] if keep.sum() >= 3 else float("nan")
    wsa = float(np.sum(wt[keep] / wt[keep].sum() * (np.sign(est[keep]) == np.sign(fd[keep]))))
    return wsa, r, int(keep.sum())


def selftest():
    print("SELFTEST 2d")
    vb = VecBank([650.0], 0.02)
    rr = ResonatorState(name="R", freq_hz=650.0, damping=0.02,
                        coupling=DEFAULT_RESONATOR_BASE_GAIN * (2 * math.pi * 650) ** 2, gate_threshold=0.0)
    md = 0.0
    for t in range(8000):
        dv = math.sin(2 * math.pi * 770 * t * DT); vb.step(dv, DT); rr.step(dv, DT)
        md = max(md, abs(vb.x[0] - rr.x), abs(vb.energy_ema[0] - rr.energy_ema))
    print(f"  vec==real max|d|={md:.1e}  {'PASS' if md < 1e-9 else 'FAIL'}")
    oks = []
    for sd in range(4):
        w, r, nk = _wsa_at_init(sd, K)
        oks.append(w >= 0.99)
        print(f"  gate W seed {sd}: sign-agree={w:.3f} r={r:+.3f} ({nk} kept)")
    okW = sum(oks) >= 3
    _, db = train("bptt", K, 0, 1500, 1, 0.3)
    print(f"  gate T  bptt R2 at n_hid={K} = {db['final_test_r2']:+.3f}  {'PASS' if db['final_test_r2'] >= 0.9 else 'FAIL'}")
    print(f"  gate W  {'PASS' if okW else 'FAIL'} ({sum(oks)}/4 seeds sign-agree>=0.99)")
    return md < 1e-9 and okW and db['final_test_r2'] >= 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", choices=["nhid", "lr"], default=None)
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    out = {"meta": vars(a), "cells": []}

    if a.sweep == "nhid":
        print("V1/V2/V3 — sweep student width against a k=4 teacher")
        base_bptt = {}
        for nh in (4, 6, 8, 12, 16):
            _, db = train("bptt", nh, a.seed, max(a.steps, 1500), 1, 0.3)
            _, df = train("raf_full", nh, a.seed, a.steps, a.ticks, 0.3)
            base_bptt[nh] = db["final_test_r2"]
            row = {"n_hid": nh, "bptt_r2": db["final_test_r2"], "raf_r2": df["final_test_r2"],
                   "sign_w2pos_all": df["sign_agree_w2pos_all"], "sign_w2neg_all": df["sign_agree_w2neg_all"],
                   "sign_w2pos_late": df["sign_agree_w2pos_late"], "sign_w2neg_late": df["sign_agree_w2neg_late"],
                   "sign_at_cross": df["sign_agree_at_crossings"], "n_cross": df["n_crossings_up"] + df["n_crossings_down"],
                   "cross_wt": df["cross_weight"], "unreach": df["unreachable_frac"]}
            out["cells"].append(row)
            print(f"  n_hid={nh:2d}  bptt R2 {row['bptt_r2']:+.3f}  raf R2 {row['raf_r2']:+.3f} "
                  f"({row['raf_r2']/max(row['bptt_r2'],1e-9):.2f}x)  "
                  f"sign +w2 {row['sign_w2pos_all']:.2f}/-w2 {row['sign_w2neg_all']:.2f} (all)  "
                  f"@cross {row['sign_at_cross']:.2f} (n={row['n_cross']},wt={row['cross_wt']:.2g})")
        m = out["cells"][0]; x4 = out["cells"][-1]
        sa_k = min(m["sign_w2pos_all"], m["sign_w2neg_all"])
        sa_4k = min(x4["sign_w2pos_all"], x4["sign_w2neg_all"])
        seq = [min(r["sign_w2pos_all"], r["sign_w2neg_all"]) for r in out["cells"]]
        mono = all(seq[i] - seq[i + 1] >= -0.08 for i in range(len(seq) - 1))
        print("\n--- verdict vs PREREGISTER_2d.md ---")
        print(f"Gate T  bptt R2 @k = {m['bptt_r2']:+.3f} >= 0.90   {'PASS' if m['bptt_r2'] >= 0.9 else 'FAIL'}")
        print(f"V1  raf R2 @k {m['raf_r2']:+.3f} >= 0.90*bptt {0.9*m['bptt_r2']:.3f}   {'PASS' if m['raf_r2'] >= 0.9*m['bptt_r2'] else 'FAIL'}")
        print(f"V2  sign @k  +w2 {m['sign_w2pos_all']:.3f}/{m['sign_w2pos_late']:.3f}  -w2 {m['sign_w2neg_all']:.3f}/{m['sign_w2neg_late']:.3f}  (all/late, >=0.85)   "
              f"{'PASS' if min(m['sign_w2pos_all'], m['sign_w2neg_all'], m['sign_w2pos_late'], m['sign_w2neg_late']) >= 0.85 else 'FAIL'}")
        print(f"V3  sign @k {sa_k:.3f} -> @4k {sa_4k:.3f}  (drop {sa_k-sa_4k:+.3f} >= 0.10, monotone {mono})   "
              f"{'PASS' if (sa_k - sa_4k) >= 0.10 and mono else 'FAIL'}")
        v4 = all((r["cross_wt"] < 1e-9 or r["sign_at_cross"] >= 0.75) for r in out["cells"])
        print(f"V4  U3 carried: @cross >= 0.75 at every n_hid   {'PASS' if v4 else 'FAIL'}")

    elif a.sweep == "lr":
        print("V5 — lr/step retry at matched n_hid=4")
        _, db = train("bptt", K, a.seed, 2000, 1, 0.3)
        best = -9
        for lr in (0.1, 0.3, 0.6):
            for stp in (200, 400):
                _, df = train("raf_full", K, a.seed, stp, a.ticks, lr)
                out["cells"].append({"lr": lr, "steps": stp, "raf_r2": df["final_test_r2"]})
                best = max(best, df["final_test_r2"])
                print(f"  lr={lr} steps={stp}: raf R2 {df['final_test_r2']:+.3f}")
        print(f"\nV5  best raf R2 {best:+.3f} >= 0.90*bptt {0.9*db['final_test_r2']:.3f}   "
              f"{'PASS' if best >= 0.9 * db['final_test_r2'] else 'FAIL'}")

    else:
        print("matched cell n_hid=4, 4-method")
        for m in ("bptt", "raf_full", "raf_sign", "nodepert_ideal"):
            ms = max(a.steps, 2000) if m == "bptt" else a.steps
            _, d = train(m, K, a.seed, ms, a.ticks, 0.3)
            out["cells"].append({"method": m, **{k: d[k] for k in ("final_test_r2", "fwd_total",
                                "sign_agree_w2pos_all", "sign_agree_w2neg_all", "sign_agree_at_crossings",
                                "n_crossings_up", "n_crossings_down", "cross_weight")}})
            print(f"  {m:15s} R2 {d['final_test_r2']:+.3f}  fwd {d['fwd_total']:>9d}  "
                  + ("" if m == "bptt" else f"sign +w2 {d['sign_agree_w2pos_all']:.2f} -w2 {d['sign_agree_w2neg_all']:.2f} "
                     f"@cross {d['sign_agree_at_crossings']:.2f}"))

    fp = OUT / f"results_2d_{a.sweep or 'matched'}.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {fp.name}   [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
