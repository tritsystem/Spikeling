"""
Rung 2c — regression target + mid-training sign flip. Fixes rung 2b's two open
problems: (a) the loss no longer saturates into a staircase, so FD signal stays
alive near convergence; (b) the beta flip forces a subset of w2 through zero, so
sign-tracking THROUGH an inversion (S3/U3) is finally measured.

    python raf_gradient_rung2c.py --selftest
    python raf_gradient_rung2c.py --steps 300 --ticks 1500

Predictions U1-U5, gates, kill condition: PREREGISTER_2c.md.
Reuses the rung-2b threshold-tracking dither, FD ground truth, and resonator probe.
"""
import argparse
import json
import math
import sys
import time

import numpy as np

from raf_gradient_rung2 import VecBank, DT, OUT, ResonatorState, DEFAULT_RESONATOR_BASE_GAIN
from raf_gradient_rung2b import dither_amps, fd_delta, raf_delta, scale_match, sgrad, A_MAX


def make_reg_task(seed, n=400):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1.5, 1.5, size=(n, 2))
    g1 = np.tanh(2 * X[:, 0])
    g2 = np.tanh(2 * X[:, 1])
    return X, g1, g2                       # y = g1 + beta*g2


class RegNet:
    def __init__(self, seed, n_hid=6):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 1.0, size=(2, n_hid))
        self.b1 = rng.normal(0, 0.3, size=n_hid)
        self.w2 = 0.5 * np.array([1, -1, 1, -1, 1, -1], float)[:n_hid]
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
        return 1.0 - np.sum((yh - y) ** 2) / np.sum((y - y.mean()) ** 2)

    def grads(self, X, y):
        v, h, yh = self.forward(X)
        dyh = (yh - y) / len(y)
        dw2 = h.T @ dyh
        db2 = dyh.sum()
        dv = np.outer(dyh, self.w2) * sgrad(v)
        return X.T @ dv, dv.sum(axis=0), dw2, db2


def train(method, seed=0, steps=300, ticks=1500, lr=0.3, log_every=5):
    Xtr, g1tr, g2tr = make_reg_task(seed, 400)
    Xte, g1te, g2te = make_reg_task(seed + 1, 400)
    net = RegNet(seed)
    fr = np.logspace(math.log10(300), math.log10(3000), net.n_hid)
    flip = steps // 2

    H = {"step": [], "test_mse": [], "train_loss": [], "maxfd": [], "fwd": []}
    per = []          # (step, w2sign[], agree[], wt[], loss)
    unreach = usteps = 0
    fwd = 0

    for st in range(steps):
        beta = 1.0 if st < flip else -1.0
        ytr = g1tr + beta * g2tr
        yte = g1te + beta * g2te
        dW1r, db1r, dw2, db2 = net.grads(Xtr, ytr)

        if method == "bptt":
            dW1, db1 = dW1r, db1r
            fwd += 1
            maxfd = np.nan
        else:
            A, reach = dither_amps(net, Xtr)
            usteps += net.n_hid
            unreach += int((~reach).sum())
            fd = fd_delta(net, Xtr, ytr, A)
            maxfd = float(np.abs(fd).max())
            mode = {"raf_full": "full", "raf_sign": "sign", "nodepert_ideal": "ideal"}[method]
            est = raf_delta(net, Xtr, ytr, fr, A, ticks, mode)
            fwd += ticks + 1500 + 2 * net.n_hid
            db1 = scale_match(est, fd) if mode != "ideal" else est
            dW1 = np.outer(Xtr.mean(0), db1)

            wt = np.abs(fd)
            keep = reach & (wt > 0.1 * wt.max())
            agree = (np.sign(db1) == np.sign(fd)).astype(float)
            per.append((st, np.sign(net.w2).tolist(), (agree * keep).tolist(),
                        (wt * keep).tolist(), net.loss(Xtr, ytr)))

        net.W1 -= lr * dW1; net.b1 -= lr * db1; net.w2 -= lr * dw2; net.b2 -= lr * db2
        if st % log_every == 0 or st == steps - 1:
            H["step"].append(st); H["test_mse"].append(net.loss(Xte, yte))
            H["train_loss"].append(net.loss(Xtr, ytr)); H["maxfd"].append(maxfd); H["fwd"].append(fwd)

    diag = _diagnose(per, steps, flip)
    diag.update(final_test_mse=H["test_mse"][-1], fwd_total=fwd,
                final_r2=net.r2(Xte, g1te - g2te),
                unreachable_frac=unreach / max(usteps, 1))
    return H, diag


def _diagnose(per, steps, flip):
    if not per:
        return {}
    steps_arr = np.array([p[0] for p in per])
    w2s = np.array([p[1] for p in per])          # (T, n_hid) signs
    agree = np.array([p[2] for p in per])        # (T, n_hid) agree*keep  (0 where not kept)
    wt = np.array([p[3] for p in per])           # (T, n_hid) wt*keep     (0 where not kept)
    T, nh = w2s.shape
    late = steps_arr >= int(steps * 0.8)

    def frac(mask2d):
        m = mask2d & (wt > 0)
        w = wt[m]
        return float((agree[m] * w).sum() / w.sum()) if w.sum() > 0 else float("nan")

    ge10 = (steps_arr[:, None] >= 10) & np.ones((1, nh), bool)
    latem = late[:, None] & np.ones((1, nh), bool)

    # sign-agreement in +/-3-step windows around w2_j zero-crossings on kept units
    cross_hits = cross_wt = 0.0
    n_up = n_dn = 0
    for j in range(nh):
        s = w2s[:, j]
        for t in range(1, T):
            if s[t] != s[t - 1] and s[t] != 0 and s[t - 1] != 0:
                n_up += int(s[t] > 0); n_dn += int(s[t] < 0)
                lo, hi = max(0, t - 3), min(T, t + 4)
                cw = wt[lo:hi, j]
                cross_hits += float((agree[lo:hi, j] * cw).sum()); cross_wt += float(cw.sum())

    return {
        "sign_agree_w2pos_all": frac((w2s > 0) & ge10),
        "sign_agree_w2neg_all": frac((w2s < 0) & ge10),
        "sign_agree_w2pos_late": frac((w2s > 0) & latem),
        "sign_agree_w2neg_late": frac((w2s < 0) & latem),
        "sign_agree_at_crossings": cross_hits / cross_wt if cross_wt > 0 else float("nan"),
        "n_crossings_up": n_up, "n_crossings_down": n_dn, "cross_weight": cross_wt,
    }


def selftest():
    print("SELFTEST 2c")
    vb = VecBank([700.0], 0.02)
    r = ResonatorState(name="R", freq_hz=700.0, damping=0.02,
                       coupling=DEFAULT_RESONATOR_BASE_GAIN * (2 * math.pi * 700) ** 2, gate_threshold=0.0)
    md = 0.0
    for t in range(8000):
        d = math.sin(2 * math.pi * 810 * t * DT); vb.step(d, DT); r.step(d, DT)
        md = max(md, abs(vb.x[0] - r.x), abs(vb.energy_ema[0] - r.energy_ema))
    print(f"  vec==real max|d|={md:.1e}  {'PASS' if md < 1e-9 else 'FAIL'}")

    Xtr, g1, g2 = make_reg_task(0, 400)
    y = g1 + g2
    net = RegNet(0)
    A, reach = dither_amps(net, Xtr)
    fd = fd_delta(net, Xtr, y, A)
    est = scale_match(raf_delta(net, Xtr, y, np.logspace(math.log10(300), math.log10(3000), net.n_hid), A, 6000, "full"), fd)
    wt = np.abs(fd); keep = reach & (wt > 0.1 * wt.max())
    rr = np.corrcoef(est[keep], fd[keep])[0, 1] if keep.sum() >= 3 else float("nan")
    wsa = float(np.sum(wt[keep] / wt[keep].sum() * (np.sign(est[keep]) == np.sign(fd[keep]))))
    okW = wsa >= 0.99 and (math.isnan(rr) or rr >= 0.85)   # addendum 2026-09-07: sign-agree is the gate
    print(f"  gate W  sign-agree={wsa:+.3f}  r={rr:+.3f} ({keep.sum()} kept)  {'PASS' if okW else 'FAIL'}")
    # bptt R2
    net = RegNet(0)
    for st in range(2000):
        g = net.grads(Xtr, y)
        net.W1 -= 0.3 * g[0]; net.b1 -= 0.3 * g[1]; net.w2 -= 0.3 * g[2]; net.b2 -= 0.3 * g[3]
    Xte, h1, h2 = make_reg_task(1, 400)
    print(f"  gate T  bptt test R2 = {net.r2(Xte, h1 + h2):.3f}  {'PASS' if net.r2(Xte, h1 + h2) >= 0.85 else 'FAIL'}")
    return md < 1e-9 and okW


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    out = {"meta": vars(a), "runs": {}}
    for m in ("bptt", "raf_full", "raf_sign", "nodepert_ideal"):
        msteps = max(a.steps, 2000) if m == "bptt" else a.steps
        H, d = train(m, a.seed, msteps, a.ticks, a.lr)
        out["runs"][m] = {"hist": H, "diag": d}
        ex = "" if m == "bptt" else (
            f"  sign +w2={d['sign_agree_w2pos_all']:.2f}/{d['sign_agree_w2pos_late']:.2f}"
            f" -w2={d['sign_agree_w2neg_all']:.2f}/{d['sign_agree_w2neg_late']:.2f} (all/late)"
            f"  @cross={d['sign_agree_at_crossings']:.2f} (n={d['n_crossings_up']}+{d['n_crossings_down']}, wt={d['cross_weight']:.2g})"
            f"  unreach={d['unreachable_frac']:.2f}")
        print(f"{m:15s} test_mse {d['final_test_mse']:.4f}  R2 {d['final_r2']:+.3f}  fwd {d['fwd_total']:>9d}{ex}")

    b = out["runs"]["bptt"]["diag"]["final_test_mse"]
    rf = out["runs"]["raf_full"]["diag"]
    rs = out["runs"]["raf_sign"]["diag"]["final_test_mse"]
    hb = out["runs"]["raf_full"]["hist"]
    fd_e = np.nanmedian(hb["maxfd"][:max(1, len(hb["maxfd"]) // 5)])
    fd_l = np.nanmedian(hb["maxfd"][-max(1, len(hb["maxfd"]) // 5):])
    print("\n--- verdict vs PREREGISTER_2c.md ---")
    print(f"Gate T  bptt R2 {out['runs']['bptt']['diag']['final_r2']:+.3f} >= 0.85   "
          f"{'PASS' if out['runs']['bptt']['diag']['final_r2'] >= 0.85 else 'FAIL'}")
    print(f"Gate F  max|fd| late/early = {fd_l:.2e}/{fd_e:.2e} = {fd_l / fd_e:.2f} >= 0.25   "
          f"{'PASS' if fd_l / fd_e >= 0.25 else 'FAIL'}   <- did the signal survive")
    print(f"U1  raf_full mse {rf['final_test_mse']:.4f} <= 1.25*bptt {1.25 * b:.4f}   "
          f"{'PASS' if rf['final_test_mse'] <= 1.25 * b else 'FAIL'}")
    print(f"U2  -w2 sign  all {rf['sign_agree_w2neg_all']:.3f} / late {rf['sign_agree_w2neg_late']:.3f}  (both >= 0.85)   "
          f"{'PASS' if min(rf['sign_agree_w2neg_all'], rf['sign_agree_w2neg_late']) >= 0.85 else 'FAIL'}")
    print(f"U3  @crossings {rf['sign_agree_at_crossings']:.3f} >= 0.75  (n={rf['n_crossings_up']}+{rf['n_crossings_down']})   "
          f"{'PASS' if rf['cross_weight'] > 1e-9 and rf['sign_agree_at_crossings'] >= 0.75 else 'FAIL / N-A'}")
    print(f"U4  raf_sign mse {rs:.4f} <= 1.5*raf_full {1.5 * rf['final_test_mse']:.4f}   "
          f"{'PASS' if rs <= 1.5 * rf['final_test_mse'] else 'FAIL'}")
    print(f"U5  raf_full {rf['fwd_total']} fwd-evals vs bptt {out['runs']['bptt']['diag']['fwd_total']}")

    fp = OUT / "results_2c_compare.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {fp.name}   [{time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main()
