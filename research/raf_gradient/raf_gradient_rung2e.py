"""
Rung 2e — attribute the ~0.10 R^2 gap between RAF node-perturbation and BPTT.
2x2: {exact FD readout, resonator readout} x {node-pert weight rule, per-parameter}.
Plus a no-training covariance diagnostic on the dropped cov_n(delta_j, x_i) term.

    python raf_gradient_rung2e.py --selftest
    python raf_gradient_rung2e.py            # 2x2 + bptt + covariance diagnostic

Predictions W1-W4, C1 + the pre-registered alternative: PREREGISTER_2e.md.
"""
import argparse
import json
import math
import sys
import time

import numpy as np

from raf_gradient_rung2 import VecBank, DT, OUT
from raf_gradient_rung2b import dither_amps, fd_delta, raf_delta, scale_match, sgrad
from raf_gradient_rung2d import K, make_teacher, teacher_y, Student, _diagnose


# ── per-parameter (weight) readouts ──────────────────────────────────
def fd_weight_grads(net, X, y, eW, eb):
    """Central FD on every W1_ij and b1_j directly — no node-pert approximation."""
    gW = np.zeros_like(net.W1)
    for i in range(net.W1.shape[0]):
        for j in range(net.W1.shape[1]):
            W0 = net.W1[i, j]
            net.W1[i, j] = W0 + eW[i, j]; lp = net.loss(X, y)
            net.W1[i, j] = W0 - eW[i, j]; lm = net.loss(X, y)
            net.W1[i, j] = W0
            gW[i, j] = (lp - lm) / (2 * eW[i, j])
    gb = fd_delta(net, X, y, eb)
    return gW, gb


def raf_weight_grads(net, X, y, freqs, AW, ticks):
    """One resonator per W1_ij, dithered at its own frequency; sign from velocity
    phase, magnitude from energy. AW shape == W1 shape; freqs flat len n_in*n_hid."""
    nin, nh = net.W1.shape
    omega = 2 * math.pi * np.asarray(freqs).reshape(nin, nh)
    W0 = net.W1.copy()
    Ldc = net.loss(X, y)
    pk = []
    for t in range(1500):
        net.W1 = W0 + AW * np.sin(omega * (t * DT))
        L = net.loss(X, y); Ldc += 0.01 * (L - Ldc); pk.append(abs(L - Ldc))
    w = float(np.percentile(pk, 90))
    net.W1 = W0
    if w < 1e-12:
        return np.zeros_like(W0)
    g = min(1e6, 1e-2 / w)
    bank = VecBank(np.asarray(freqs).ravel(), 0.02)
    Ldc = net.loss(X, y)
    vs = np.zeros(nin * nh)
    for t in range(ticks):
        s = np.sin(omega * (t * DT))
        net.W1 = W0 + AW * s
        L = net.loss(X, y); Ldc += 0.01 * (L - Ldc)
        bank.step(g * (L - Ldc), DT)
        if t >= ticks // 2:
            vs += bank.v * s.ravel()
    net.W1 = W0
    mag = np.sqrt(np.maximum(bank.energy_ema, 0.0)).reshape(nin, nh)
    return np.sign(vs).reshape(nin, nh) * mag


# ── training ─────────────────────────────────────────────────────────
def train(method, seed=0, steps=250, ticks=1500, lr=0.3, n_hid=K, log_every=5):
    Xtr, A, b, c, _, c0 = make_teacher(seed, 400)
    Xte, *_ = make_teacher(seed + 1, 400)
    net = Student(seed, n_hid)
    fr_b = np.logspace(math.log10(300), math.log10(3000), n_hid)
    fr_w = np.logspace(math.log10(300), math.log10(3000), 2 * n_hid)
    flip = steps // 2
    H = {"step": [], "test_r2": []}
    per = []
    fwd = 0
    rmsx = np.sqrt((Xtr ** 2).mean(0))

    for st in range(steps):
        fl = st >= flip
        ytr = teacher_y(Xtr, A, b, c, c0, fl)
        yte = teacher_y(Xte, A, b, c, c0, fl)
        dW1_bp, db1_bp, dw2, db2 = net.grads(Xtr, ytr)

        if method == "bptt":
            dW1, db1 = dW1_bp, db1_bp
            fwd += 1
        else:
            Ab, reach = dither_amps(net, Xtr)
            if method in ("fd_nodepert", "raf_full"):
                fd_b = fd_delta(net, Xtr, ytr, Ab)
                if method == "fd_nodepert":
                    db1 = fd_b
                else:
                    est = raf_delta(net, Xtr, ytr, fr_b, Ab, ticks, "full")
                    db1 = scale_match(est, fd_b)
                dW1 = np.outer(Xtr.mean(0), db1)
                fwd += 1 if method == "fd_nodepert" else ticks
            elif method in ("fd_weights", "raf_weights"):
                AW = np.clip(Ab[None, :] / rmsx[:, None], 0.02, 3.0)
                if method == "fd_weights":
                    dW1, db1 = fd_weight_grads(net, Xtr, ytr, AW, Ab)
                else:
                    gW = raf_weight_grads(net, Xtr, ytr, fr_w, AW, ticks)
                    db1 = fd_delta(net, Xtr, ytr, Ab)                 # bias still via node dither
                    gref = np.concatenate([dW1_bp.ravel(), db1_bp])
                    est = np.concatenate([gW.ravel(), db1])
                    sc = scale_match(est, gref)
                    dW1 = sc[:gW.size].reshape(gW.shape); db1 = sc[gW.size:]
                fwd += (2 * dW1_bp.size if method == "fd_weights" else ticks)
            wt = np.abs(dW1_bp).sum(0) + np.abs(db1_bp)
            keep = reach & (wt > 0.1 * wt.max())
            agree = (np.sign(db1) == np.sign(db1_bp)).astype(float)
            per.append((st, np.sign(net.w2).tolist(), (agree * keep).tolist(),
                        (wt * keep).tolist(), net.loss(Xtr, ytr)))

        net.W1 -= lr * dW1; net.b1 -= lr * db1; net.w2 -= lr * dw2; net.b2 -= lr * db2
        if st % log_every == 0 or st == steps - 1:
            H["step"].append(st); H["test_r2"].append(net.r2(Xte, yte))

    d = _diagnose(per, steps, flip) if per else {}
    d.update(final_test_r2=H["test_r2"][-1], fwd_total=fwd, method=method, hist=H)
    return d


def covariance_diagnostic(seed=0, states=20):
    Xtr, A, b, c, _, c0 = make_teacher(seed, 400)
    net = Student(seed, K)
    rel, cos = [], []
    for s in range(states):
        fl = s % 2 == 1
        y = teacher_y(Xtr, A, b, c, c0, fl)
        v, h, yh = net.forward(Xtr)
        dyh = (yh - y) / len(y)
        delta = np.outer(dyh, net.w2) * sgrad(v)             # (n, n_hid) per-sample node error (1/N inside)
        g = Xtr.T @ delta                                     # true dL/dW1 = sum_n x_n (x) delta_n
        node_err = delta.sum(0)                               # dL/db1_j = sum_n delta_{n,j}
        ghat = np.outer(Xtr.mean(0), node_err)               # node-pert reconstruction: (dL/db1_j) * mean_n x_i
        rel.append(np.linalg.norm(g - ghat) / (np.linalg.norm(g) + 1e-12))
        d = g.ravel(); e = ghat.ravel()
        cos.append(float(d @ e / (np.linalg.norm(d) * np.linalg.norm(e) + 1e-12)))
        # perturb weights a little for the next state
        net.W1 += np.random.default_rng(s).normal(0, 0.15, net.W1.shape)
    return np.array(rel), np.array(cos)


def selftest():
    print("SELFTEST 2e")
    rel, cos = covariance_diagnostic(0, 6)
    print(f"  covariance rel-error first 6 states: {np.round(rel, 3).tolist()}")
    d = train("bptt", 0, 250, 1, 0.3)
    print(f"  bptt R2 = {d['final_test_r2']:+.3f}  {'PASS' if d['final_test_r2'] >= 0.9 else 'FAIL'}")
    dfw = train("fd_weights", 0, 120, 1, 0.3)
    print(f"  fd_weights R2 (120 steps) = {dfw['final_test_r2']:+.3f}  (sanity: should be near bptt)")
    return d['final_test_r2'] >= 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    rel, cos = covariance_diagnostic(a.seed, 20)
    print(f"C1 covariance diagnostic (20 states):")
    print(f"   ||g - ghat|| / ||g||   median {np.median(rel):.3f}   frac>=0.25 {np.mean(rel >= 0.25):.2f}")
    print(f"   cos(g, ghat)           median {np.median(cos):.3f}")

    R = {}
    for m in ("bptt", "fd_weights", "fd_nodepert", "raf_weights", "raf_full"):
        d = train(m, a.seed, a.steps if m != "bptt" else max(a.steps, 2000), a.ticks, 0.3)
        R[m] = d["final_test_r2"]
        print(f"  {m:13s} R2 {d['final_test_r2']:+.3f}   fwd {d['fwd_total']:>8d}")

    b = R["bptt"]
    print("\n--- verdict vs PREREGISTER_2e.md ---")
    print(f"W1  fd_weights {R['fd_weights']:+.3f} >= 0.90*bptt {0.9*b:.3f}   "
          f"{'PASS' if R['fd_weights'] >= 0.9 * b else 'FAIL'}   (hard-net FD ~= surrogate BPTT?)")
    gap_approx = R["fd_weights"] - R["fd_nodepert"]
    gap_noise = R["fd_nodepert"] - R["raf_full"]
    print(f"W2  approx cost fd_weights-fd_nodepert = {gap_approx:+.3f} (>=0.05)  "
          f"and raf_full within 0.03 of fd_nodepert (|{gap_noise:+.3f}|)   "
          f"{'PASS' if gap_approx >= 0.05 and abs(gap_noise) <= 0.03 else 'FAIL'}")
    print(f"W3  drive-noise cost: raf_full>=fd_nodepert-0.05 ({gap_noise:+.3f})  "
          f"raf_weights>=fd_weights-0.05 ({R['fd_weights']-R['raf_weights']:+.3f})   "
          f"{'PASS' if gap_noise <= 0.05 and (R['fd_weights'] - R['raf_weights']) <= 0.05 else 'FAIL'}")
    print(f"W4  raf_weights {R['raf_weights']:+.3f} >= 0.90*bptt {0.9*b:.3f}   "
          f"{'PASS' if R['raf_weights'] >= 0.9 * b else 'FAIL'}   (per-weight RAF closes the gap?)")
    alt = gap_approx < 0.05 and gap_noise > 0.05
    print(f"ALT (drive variance is the cause instead): {'YES' if alt else 'no'}")

    out = {"meta": vars(a), "R2": R, "cov_rel_median": float(np.median(rel)),
           "cov_rel_frac_ge_025": float(np.mean(rel >= 0.25)), "cov_cos_median": float(np.median(cos))}
    fp = OUT / "results_2e.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {fp.name}   [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
