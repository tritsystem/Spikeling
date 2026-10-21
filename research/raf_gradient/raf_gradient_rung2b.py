"""
Rung 2b — rung 2 redone with (1) a validated task BPTT solves at ~0.98,
(2) a threshold-TRACKING dither rule that follows a unit as it drifts from the
flip, (3) per-unit finite-difference ground truth (not the biased surrogate).

    python raf_gradient_rung2b.py --selftest
    python raf_gradient_rung2b.py --steps 200 --ticks 2000

Predictions S1-S5, gates, kill condition: PREREGISTER_2b.md.
Resonator math: VecBank from raf_gradient_rung2, asserted bit-equal to the shipped
ResonatorState in rung 2's selftest and re-checked here.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from raf_gradient_rung2 import VecBank, DT, OUT, ResonatorState, DEFAULT_RESONATOR_BASE_GAIN

A_MIN, A_MAX = 0.02, 3.0
P_TRACK, P_REACH = 0.30, 0.10


def sgrad(v):
    return 1.0 / (1.0 + 10.0 * np.abs(v)) ** 2


def make_xor(seed, n=400, sigma=0.35):
    rng = np.random.default_rng(seed)
    centers = {0: [[-1, -1], [1, 1]], 1: [[-1, 1], [1, -1]]}
    X, y = [], []
    per = n // 4
    for cls, cs in centers.items():
        for c in cs:
            X.append(rng.normal(c, sigma, size=(per, 2))); y += [cls] * per
    return np.concatenate(X), np.array(y, float)


class SpikeNet:
    def __init__(self, seed, n_hid=6):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.8, size=(2, n_hid))
        self.b1 = np.zeros(n_hid)
        self.w2 = 0.5 * np.array([1, -1, 1, -1, 1, -1], float)[:n_hid]
        self.b2 = 0.0
        self.n_hid = n_hid

    def forward(self, X, b1_extra=0.0):
        v = X @ self.W1 + self.b1 + b1_extra
        h = (v > 0.0).astype(float)
        o = h @ self.w2 + self.b2
        return v, h, o, (o > 0.0).astype(float)

    def loss(self, X, y, b1_extra=0.0):
        *_, yh = self.forward(X, b1_extra)
        return float(np.mean(0.5 * (yh - y) ** 2))

    def acc(self, X, y):
        *_, yh = self.forward(X)
        return float(np.mean(yh == y))

    def bptt_grads(self, X, y):
        v, h, o, yh = self.forward(X)
        do = (yh - y) * sgrad(o) / len(y)
        dw2 = h.T @ do
        db2 = do.sum()
        dv = np.outer(do, self.w2) * sgrad(v)
        return X.T @ dv, dv.sum(axis=0), dw2, db2


def dither_amps(net, X):
    """Threshold-tracking: A_j scaled to the 30th pct of |v_j|; grows as the unit
    drifts. reachable_j := nearest 10% of samples within A_MAX of the flip."""
    v = X @ net.W1 + net.b1
    av = np.abs(v)
    A = np.clip(1.2 * np.quantile(av, P_TRACK, axis=0), A_MIN, A_MAX)
    reachable = np.quantile(av, P_REACH, axis=0) <= A_MAX
    return A, reachable


def capped_amps(net, X):
    """Rung 2's rule, for the S4 contrast only."""
    v = X @ net.W1 + net.b1
    return np.clip(1.5 * np.mean(np.abs(v), axis=0), 0.05, 0.5)


def fd_delta(net, X, y, A):
    """Per-unit central FD at each unit's own amplitude — the honest ground truth."""
    fd = np.zeros(net.n_hid)
    for j in range(net.n_hid):
        e = np.zeros(net.n_hid); e[j] = A[j]
        fd[j] = (net.loss(X, y, e) - net.loss(X, y, -e)) / (2 * A[j])
    return fd


def raf_delta(net, X, y, freqs, A, ticks, mode="full"):
    omega = 2 * math.pi * np.asarray(freqs)
    Ldc = net.loss(X, y)
    pk = []
    for t in range(1500):
        s = A * np.sin(omega * (t * DT))
        L = net.loss(X, y, s); Ldc += 0.01 * (L - Ldc); pk.append(abs(L - Ldc))
    w = float(np.percentile(pk, 90))
    if w < 1e-12:
        return np.zeros(net.n_hid)
    g = min(1e6, 1e-2 / w)
    bank = VecBank(freqs, 0.02)
    Ldc = net.loss(X, y)
    vs = np.zeros(net.n_hid); la = np.zeros(net.n_hid); n = 0
    for t in range(ticks):
        s = np.sin(omega * t * DT)
        L = net.loss(X, y, A * s); Ldc += 0.01 * (L - Ldc)
        bank.step(g * (L - Ldc), DT)
        if t >= ticks // 2:
            vs += bank.v * s; la += L * s; n += 1
    mag = np.sqrt(np.maximum(bank.energy_ema, 0.0))
    if mode == "sign":
        return np.sign(vs)
    if mode == "ideal":
        return 2.0 * la / (n * A)
    return np.sign(vs) * mag


def scale_match(est, ref):
    d = float(est @ est)
    return est * (float(est @ ref) / d) if d > 1e-30 else est


def train(method, seed=0, steps=200, ticks=2000, lr=0.5, log_every=5):
    Xtr, ytr = make_xor(seed, 400); Xte, yte = make_xor(seed + 1, 400)
    net = SpikeNet(seed)
    freqs = np.logspace(math.log10(300), math.log10(3000), net.n_hid)
    H = {"step": [], "test_acc": [], "train_loss": [], "fwd": []}
    hit = {"pos": 0.0, "tpos": 0.0, "neg": 0.0, "tneg": 0.0, "cross": 0.0, "tcross": 0.0}
    sig = {"pos": 0.0, "tpos": 0.0, "neg": 0.0, "tneg": 0.0}   # signal-present (train_loss>0.1) subset
    perstep = []                                                # (train_loss, sign_agree_kept)
    unreach_track = unreach_cap = unit_steps = 0
    prev_sign = np.sign(net.w2)
    fwd = 0

    for st in range(steps):
        dW1r, db1r, dw2, db2 = net.bptt_grads(Xtr, ytr)
        if method == "bptt":
            dW1, db1 = dW1r, db1r
            fwd += 1
        else:
            A, reach = dither_amps(net, Xtr)
            unit_steps += net.n_hid
            unreach_track += int((~reach).sum())
            unreach_cap += int((capped_amps(net, Xtr) < np.quantile(np.abs(Xtr @ net.W1 + net.b1), P_REACH, axis=0)).sum())
            fd = fd_delta(net, Xtr, ytr, A)
            mode = {"raf_full": "full", "raf_sign": "sign", "nodepert_ideal": "ideal"}[method]
            est = raf_delta(net, Xtr, ytr, freqs, A, ticks, mode)
            fwd += ticks + 1500 + 2 * net.n_hid
            db1 = scale_match(est, fd) if mode != "ideal" else est
            dW1 = np.outer(Xtr.mean(0), db1)

            wt = np.abs(fd)
            keep = reach & (wt > 0.1 * wt.max())
            agree = (np.sign(db1) == np.sign(fd)).astype(float)
            pos, neg = keep & (net.w2 > 0), keep & (net.w2 < 0)
            hit["pos"] += float((agree * wt)[pos].sum()); hit["tpos"] += float(wt[pos].sum())
            hit["neg"] += float((agree * wt)[neg].sum()); hit["tneg"] += float(wt[neg].sum())
            crossed = keep & (np.sign(net.w2) != prev_sign)
            if crossed.any():
                hit["cross"] += float((agree * wt)[crossed].sum()); hit["tcross"] += float(wt[crossed].sum())

            tl = net.loss(Xtr, ytr)
            ksa = float(np.sum((wt[keep] / wt[keep].sum()) * agree[keep])) if keep.any() else float("nan")
            perstep.append((tl, ksa))
            if tl > 0.10:                                       # signal-present subset
                sig["pos"] += float((agree * wt)[pos].sum()); sig["tpos"] += float(wt[pos].sum())
                sig["neg"] += float((agree * wt)[neg].sum()); sig["tneg"] += float(wt[neg].sum())

        prev_sign = np.sign(net.w2)
        net.W1 -= lr * dW1; net.b1 -= lr * db1; net.w2 -= lr * dw2; net.b2 -= lr * db2
        if st % log_every == 0 or st == steps - 1:
            H["step"].append(st); H["test_acc"].append(net.acc(Xte, yte))
            H["train_loss"].append(net.loss(Xtr, ytr)); H["fwd"].append(fwd)

    diag = {
        "final_test_acc": H["test_acc"][-1], "fwd_total": fwd,
        "sign_agree_w2pos": hit["pos"] / max(hit["tpos"], 1e-12),
        "sign_agree_w2neg": hit["neg"] / max(hit["tneg"], 1e-12),
        "sign_agree_w2pos_signal": sig["pos"] / max(sig["tpos"], 1e-12),
        "sign_agree_w2neg_signal": sig["neg"] / max(sig["tneg"], 1e-12),
        "sign_agree_at_crossings": hit["cross"] / max(hit["tcross"], 1e-12),
        "tcross_weight": hit["tcross"],
        "unreachable_frac_tracking": unreach_track / max(unit_steps, 1),
        "unreachable_frac_capped": unreach_cap / max(unit_steps, 1),
        "perstep_loss_signagree": perstep,
    }
    return H, diag


def selftest():
    print("SELFTEST 2b")
    vb = VecBank([500.0], 0.02)
    r = ResonatorState(name="R", freq_hz=500.0, damping=0.02,
                       coupling=DEFAULT_RESONATOR_BASE_GAIN * (2 * math.pi * 500) ** 2, gate_threshold=0.0)
    md = 0.0
    for t in range(8000):
        d = math.sin(2 * math.pi * 620 * t * DT); vb.step(d, DT); r.step(d, DT)
        md = max(md, abs(vb.x[0] - r.x), abs(vb.energy_ema[0] - r.energy_ema))
    print(f"  vec==real  max|d|={md:.1e}  {'PASS' if md < 1e-9 else 'FAIL'}")

    Xtr, ytr = make_xor(0, 400)
    net = SpikeNet(0)
    fr = np.logspace(math.log10(300), math.log10(3000), net.n_hid)
    A, reach = dither_amps(net, Xtr)
    fd = fd_delta(net, Xtr, ytr, A)
    est = scale_match(raf_delta(net, Xtr, ytr, fr, A, 6000, "full"), fd)
    wt = np.abs(fd); keep = reach & (wt > 0.1 * wt.max())
    rr = np.corrcoef(est[keep], fd[keep])[0, 1]
    wsa = float(np.sum(wt[keep] / wt[keep].sum() * (np.sign(est[keep]) == np.sign(fd[keep]))))
    print(f"  gate W  r(dhat,fd) on {keep.sum()} kept units = {rr:+.3f}  {'PASS' if rr >= 0.9 else 'FAIL'}")
    print(f"  gate W  |fd|-weighted sign-agree            = {wsa:+.3f}  {'PASS' if wsa >= 0.9 else 'FAIL'}")
    print(f"  A_j = {np.round(A,3).tolist()}   reachable = {reach.tolist()}")
    return md < 1e-9 and rr >= 0.9 and wsa >= 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--ticks", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    t0 = time.perf_counter()
    if a.selftest:
        sys.exit(0 if selftest() else 1)

    out = {"meta": vars(a), "runs": {}}
    for m in ("bptt", "raf_full", "raf_sign", "nodepert_ideal"):
        msteps = max(a.steps, 2000) if m == "bptt" else a.steps   # BPTT ceiling at convergence
        H, d = train(m, a.seed, msteps, a.ticks, a.lr)
        out["runs"][m] = {"hist": H, "diag": d}
        extra = "" if m == "bptt" else (
            f"  sign +w2={d['sign_agree_w2pos']:.3f} -w2={d['sign_agree_w2neg']:.3f} "
            f"@cross={d['sign_agree_at_crossings']:.3f}(wt={d['tcross_weight']:.2g})  "
            f"unreach track={d['unreachable_frac_tracking']:.2f} cap={d['unreachable_frac_capped']:.2f}")
        print(f"{m:15s} acc {d['final_test_acc']:.3f}  fwd {d['fwd_total']:>9d}{extra}")

    b = out["runs"]["bptt"]["diag"]["final_test_acc"]
    rf = out["runs"]["raf_full"]["diag"]
    rs = out["runs"]["raf_sign"]["diag"]["final_test_acc"]
    print("\n--- verdict vs PREREGISTER_2b.md ---")
    print(f"Gate T  bptt {b:.3f} > 0.90                      {'PASS' if b > 0.9 else 'FAIL'}")
    print(f"S1  raf_full {rf['final_test_acc']:.3f} >= 0.90*bptt {0.9*b:.3f}   {'PASS' if rf['final_test_acc'] >= 0.9*b else 'FAIL'}")
    print(f"S2  sign all-steps  +w2 {rf['sign_agree_w2pos']:.3f} / -w2 {rf['sign_agree_w2neg']:.3f}   (context)")
    print(f"S2  sign SIGNAL(loss>0.1)  +w2 {rf['sign_agree_w2pos_signal']:.3f} / -w2 {rf['sign_agree_w2neg_signal']:.3f}  (both >= 0.85)   "
          f"{'PASS' if min(rf['sign_agree_w2pos_signal'], rf['sign_agree_w2neg_signal']) >= 0.85 else 'FAIL'}")
    print(f"S3  @crossings {rf['sign_agree_at_crossings']:.3f} >= 0.75 (wt {rf['tcross_weight']:.2g})   "
          f"{'PASS' if rf['tcross_weight'] > 1e-9 and rf['sign_agree_at_crossings'] >= 0.75 else 'N/A or FAIL'}")
    print(f"S4  unreachable  tracking {rf['unreachable_frac_tracking']:.2f} < 0.10 and < capped {rf['unreachable_frac_capped']:.2f}   "
          f"{'PASS' if rf['unreachable_frac_tracking'] < 0.10 and rf['unreachable_frac_tracking'] < rf['unreachable_frac_capped'] else 'FAIL'}")
    print(f"S5  raf_full {rf['fwd_total']} fwd-evals vs bptt {out['runs']['bptt']['diag']['fwd_total']}")
    print(f"R4-carry  raf_sign {rs:.3f} >= 0.80*raf_full {0.8*rf['final_test_acc']:.3f}   {'PASS' if rs >= 0.8*rf['final_test_acc'] else 'FAIL'}")

    fp = OUT / "results_2b_compare.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {fp.name}   [{time.perf_counter()-t0:.1f}s]")


if __name__ == "__main__":
    main()
