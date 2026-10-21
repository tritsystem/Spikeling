"""
Rung 2 — can the RAF resonator readout TRAIN a 2-layer spiking net, and does it
track hidden-unit gradient-sign inversion when w2 signs flip during training?

    python raf_gradient_rung2.py --selftest        # instrument checks (vec==real, wiring)
    python raf_gradient_rung2.py                    # full comparison run + json + plot data
    python raf_gradient_rung2.py --steps 200 --ticks 2000

Predictions R1-R4, instrument gates, kill condition: PREREGISTER_2.md.
The training loops use a VECTORISED copy of ResonatorState (asserted bit-close to
the shipped class in --selftest); the real class is stepped in the selftest itself.
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "core"))
from runtime.runtime import ResonatorState, DEFAULT_RESONATOR_BASE_GAIN  # noqa: E402

DT = 1.0 / 40000.0
OUT = Path(__file__).parent
ETC = 0.0025  # ResonatorState.energy_time_constant


# ─────────────────────────────────────────────────────────────────────
#  Vectorised resonator bank — identical math to ResonatorState.step
#  with gate_threshold=0 (always compute energy). Verified in --selftest.
# ─────────────────────────────────────────────────────────────────────
class VecBank:
    def __init__(self, freqs, damping):
        self.omega = 2 * math.pi * np.asarray(freqs, float)
        self.zeta = float(damping)
        self.coupling = DEFAULT_RESONATOR_BASE_GAIN * self.omega ** 2
        self.x = np.zeros_like(self.omega)
        self.v = np.zeros_like(self.omega)
        self.energy_ema = np.zeros_like(self.omega)

    def step(self, drive, dt):
        accel = -(self.omega ** 2) * self.x - 2 * self.zeta * self.omega * self.v + self.coupling * drive
        self.v += accel * dt
        self.x += self.v * dt
        alpha = min(1.0, dt / ETC)
        self.energy_ema += alpha * (self.x * self.x - self.energy_ema)


# ─────────────────────────────────────────────────────────────────────
#  Task + network
# ─────────────────────────────────────────────────────────────────────
def make_task(seed, n_in=4, n=48):
    rng = np.random.default_rng(seed)
    # two Gaussian clusters per class -> a single linear cut can't separate; needs a hidden layer
    centers = {0: [[-1.2, -1.2], [1.2, 1.2]], 1: [[-1.2, 1.2], [1.2, -1.2]]}
    def draw(m):
        X, y = [], []
        for cls, cs in centers.items():
            for c in cs:
                pts = rng.normal(c, 0.45, size=(m, 2))
                X.append(pts); y += [cls] * m
        X = np.concatenate(X)
        pad = rng.normal(0, 0.3, size=(X.shape[0], n_in - 2))   # distractor dims
        return np.concatenate([X, pad], axis=1), np.array(y, float)
    per = n // 4
    return draw(per), draw(per)  # (Xtr,ytr), (Xte,yte)


def sgrad(v):                       # fast-sigmoid surrogate derivative
    return 1.0 / (1.0 + 10.0 * np.abs(v)) ** 2


class SpikeNet:
    def __init__(self, seed, n_in=4, n_hid=6):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, 0.8, size=(n_in, n_hid))
        self.b1 = np.zeros(n_hid)
        self.w2 = 0.5 * np.array([1, -1, 1, -1, 1, -1], float)[:n_hid]   # alternating signs
        self.b2 = 0.0
        self.n_hid = n_hid

    def forward(self, X, b1_extra=0.0):
        v = X @ self.W1 + self.b1 + b1_extra
        h = (v > 0.0).astype(float)
        o = h @ self.w2 + self.b2
        yhat = (o > 0.0).astype(float)
        return v, h, o, yhat

    def loss(self, X, y, b1_extra=0.0):
        _, _, _, yhat = self.forward(X, b1_extra)
        return float(np.mean(0.5 * (yhat - y) ** 2))

    def acc(self, X, y):
        _, _, _, yhat = self.forward(X)
        return float(np.mean(yhat == y))

    def bptt_grads(self, X, y):
        v, h, o, yhat = self.forward(X)
        do = (yhat - y) * sgrad(o) / len(y)              # dL/do  (surrogate on output step)
        dw2 = h.T @ do
        db2 = do.sum()
        dh = np.outer(do, self.w2)                        # (N, n_hid)
        dv = dh * sgrad(v)
        dW1 = X.T @ dv
        db1 = dv.sum(axis=0)                              # dL/db1_j  == delta_j reference
        return dW1, db1, dw2, db2


# ─────────────────────────────────────────────────────────────────────
#  RAF node-perturbation: one resonator per hidden unit, shared loss drive
# ─────────────────────────────────────────────────────────────────────
def raf_delta(net, X, y, freqs, A, ticks, mode="full"):
    omega = 2 * math.pi * np.asarray(freqs)
    # warm-up: calibrate a single drive gain
    Ldc = net.loss(X, y)
    peaks = []
    for t in range(1500):
        s = A * np.sin(omega * (t * DT))
        L = net.loss(X, y, b1_extra=s)
        Ldc += 0.01 * (L - Ldc)
        peaks.append(abs(L - Ldc))
    w = float(np.percentile(peaks, 90))
    if w < 1e-9:
        return np.zeros(net.n_hid), 0.0, True   # dead: dither never crossed a hidden threshold
    g = min(1e6, 1e-2 / w)

    bank = VecBank(freqs, damping=0.02)
    Ldc = net.loss(X, y)
    v_sin = np.zeros(net.n_hid)
    settle = ticks // 2
    nmeas = 0
    lossacc = np.zeros(net.n_hid)
    for t in range(ticks):
        ts = t * DT
        s = np.sin(omega * ts)
        L = net.loss(X, y, b1_extra=A * s)
        Ldc += 0.01 * (L - Ldc)
        bank.step(g * (L - Ldc), DT)
        if t >= settle:
            v_sin += bank.v * s
            lossacc += L * s
            nmeas += 1
    mag = np.sqrt(np.maximum(bank.energy_ema, 0.0))
    if mode == "sign":
        est = np.sign(v_sin)
    elif mode == "ideal":
        est = 2.0 * lossacc / (nmeas * A)                 # digital lock-in on the loss
    else:
        est = np.sign(v_sin) * mag
    return est, g, False


def scale_match(est, ref):
    """single global least-squares factor so est's magnitude matches the ideal delta"""
    denom = float(est @ est)
    return est * (float(est @ ref) / denom) if denom > 1e-30 else est


# ─────────────────────────────────────────────────────────────────────
def train(method, seed=0, steps=150, ticks=1500, lr=0.4, log_every=5):
    (Xtr, ytr), (Xte, yte) = make_task(seed)
    net = SpikeNet(seed)
    freqs = np.logspace(math.log10(300), math.log10(3000), net.n_hid)

    hist = {"step": [], "test_acc": [], "train_loss": [], "w2_sign": [], "fwd_evals": []}
    sign_hits_pos, sign_tot_pos, sign_hits_neg, sign_tot_neg = 0, 0, 0, 0
    cross_hits, cross_tot = 0, 0
    prev_w2 = net.w2.copy()
    fwd = 0

    for st in range(steps):
        dW1_ref, db1_ref, dw2, db2 = net.bptt_grads(Xtr, ytr)   # always: output layer + reference

        if method == "bptt":
            dW1, db1 = dW1_ref, db1_ref
            fwd += 1
        else:
            A = np.clip(1.5 * np.mean(np.abs(Xtr @ net.W1 + net.b1), axis=0), 0.05, 0.5)
            mode = {"raf_full": "full", "raf_sign": "sign", "nodepert_ideal": "ideal"}[method]
            est, _, dead = raf_delta(net, Xtr, ytr, freqs, A, ticks, mode=mode)
            fwd += ticks + 1500
            db1 = scale_match(est, db1_ref) if mode != "ideal" else est
            dW1 = np.outer(Xtr.mean(axis=0), db1)               # node-pert weight rule

            # ---- sign-inversion diagnostic (vs BPTT reference delta),
            #      |delta_ref|-weighted so numerically-zero units can't dominate
            #      (PREREGISTER_2.md addendum 2026-09-07) ----
            wt = np.abs(db1_ref)
            live = wt > 0.1 * wt.max()
            agree = (np.sign(db1) == np.sign(db1_ref)).astype(float)
            pos, neg = (net.w2 > 0) & live, (net.w2 < 0) & live
            sign_hits_pos += float((agree * wt)[pos].sum()); sign_tot_pos += float(wt[pos].sum())
            sign_hits_neg += float((agree * wt)[neg].sum()); sign_tot_neg += float(wt[neg].sum())
            crossed = (np.sign(net.w2) != np.sign(prev_w2)) & live
            if crossed.any():
                cross_hits += float((agree * wt)[crossed].sum()); cross_tot += float(wt[crossed].sum())

        prev_w2 = net.w2.copy()
        net.W1 -= lr * dW1; net.b1 -= lr * db1
        net.w2 -= lr * dw2; net.b2 -= lr * db2

        if st % log_every == 0 or st == steps - 1:
            hist["step"].append(st)
            hist["test_acc"].append(net.acc(Xte, yte))
            hist["train_loss"].append(net.loss(Xtr, ytr))
            hist["w2_sign"].append(net.w2.tolist())
            hist["fwd_evals"].append(fwd)

    diag = {
        "sign_agree_w2pos": sign_hits_pos / max(sign_tot_pos, 1),
        "sign_agree_w2neg": sign_hits_neg / max(sign_tot_neg, 1),
        "sign_agree_at_crossings": cross_hits / max(cross_tot, 1),
        "n_crossings": cross_tot,
        "final_test_acc": hist["test_acc"][-1],
        "fwd_evals_total": fwd,
    }
    return hist, diag


# ─────────────────────────────────────────────────────────────────────
def selftest():
    print("SELFTEST — instrument gates")
    # (1) vec bank == real ResonatorState
    freqs = [400.0, 900.0, 1700.0]
    vb = VecBank(freqs, 0.02)
    reals = [ResonatorState(name=f"R{f}", freq_hz=f, damping=0.02,
                            coupling=DEFAULT_RESONATOR_BASE_GAIN * (2 * math.pi * f) ** 2,
                            gate_threshold=0.0) for f in freqs]
    md = 0.0
    for t in range(8000):
        d = math.sin(2 * math.pi * 550 * t * DT)
        vb.step(d, DT)
        for r in reals:
            r.step(d, DT)
        md = max(md, max(abs(vb.x[i] - reals[i].x) for i in range(3)),
                 max(abs(vb.energy_ema[i] - reals[i].energy_ema) for i in range(3)))
    print(f"  vec vs real  max|d state| over 8000 ticks = {md:.2e}   {'PASS' if md < 1e-9 else 'FAIL'}")

    # (2) RAF delta wiring on a random init
    (Xtr, ytr), _ = make_task(0)
    net = SpikeNet(0)
    _, db1_ref, _, _ = net.bptt_grads(Xtr, ytr)
    fr = np.logspace(math.log10(300), math.log10(3000), net.n_hid)
    A = np.clip(1.5 * np.mean(np.abs(Xtr @ net.W1 + net.b1), axis=0), 0.05, 0.5)
    est, _, dead = raf_delta(net, Xtr, ytr, fr, A, 4000, mode="full")
    est = scale_match(est, db1_ref)
    wt = np.abs(db1_ref); big = wt > 0.1 * wt.max()
    r_all = np.corrcoef(est, db1_ref)[0, 1]
    r_big = np.corrcoef(est[big], db1_ref[big])[0, 1]
    wsa = float(np.sum(wt / wt.sum() * (np.sign(est) == np.sign(db1_ref))))
    print(f"  r(RAF dhat, d_ref) all units          = {r_all:+.3f}   (context only, not a gate)")
    print(f"  gate 3a  |d_ref|-weighted sign-agree  = {wsa:+.3f}   {'PASS' if wsa >= 0.9 else 'FAIL'}   (need >= 0.90)")
    print(f"  gate 3b  r on units > 10% max|d_ref|  = {r_big:+.3f}   {'PASS' if r_big >= 0.9 else 'FAIL'}   (need >= 0.90)")
    return md < 1e-9 and wsa >= 0.9 and r_big >= 0.9


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--ticks", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    t0 = time.perf_counter()

    if args.selftest:
        sys.exit(0 if selftest() else 1)

    out = {"meta": vars(args), "runs": {}}
    for method in ("bptt", "raf_full", "raf_sign", "nodepert_ideal"):
        h, d = train(method, seed=args.seed, steps=args.steps, ticks=args.ticks, lr=args.lr)
        out["runs"][method] = {"hist": h, "diag": d}
        print(f"{method:16s} final test acc {d['final_test_acc']:.3f}   "
              f"fwd evals {d['fwd_evals_total']:>10d}   "
              + (f"sign-agree +w2={d['sign_agree_w2pos']:.3f} -w2={d['sign_agree_w2neg']:.3f} "
                 f"@cross={d['sign_agree_at_crossings']:.3f} (n={d['n_crossings']})"
                 if method != "bptt" else ""))

    base = out["runs"]["bptt"]["diag"]["final_test_acc"]
    rf = out["runs"]["raf_full"]["diag"]
    rs = out["runs"]["raf_sign"]["diag"]["final_test_acc"]
    print("\n--- verdict vs PREREGISTER_2.md ---")
    print(f"R1 learns:  raf_full {rf['final_test_acc']:.3f}  vs  0.90*baseline {0.9*base:.3f}   "
          f"{'PASS' if rf['final_test_acc'] >= 0.9 * base else 'FAIL'}")
    print(f"R2 sign:    +w2 {rf['sign_agree_w2pos']:.3f}  -w2 {rf['sign_agree_w2neg']:.3f}  "
          f"(need >=0.85 both; crossings n={rf['n_crossings']} @ {rf['sign_agree_at_crossings']:.3f} need >=0.75)   "
          f"{'PASS' if min(rf['sign_agree_w2pos'], rf['sign_agree_w2neg']) >= 0.85 else 'FAIL'}")
    print(f"R3 cost:    raf_full {rf['fwd_evals_total'] / max(base, 1e-9):.0f}x-ish fwd evals vs bptt "
          f"({rf['fwd_evals_total']} vs {out['runs']['bptt']['diag']['fwd_evals_total']})")
    print(f"R4 sign~=:   raf_sign {rs:.3f}  vs  0.80*raf_full {0.8*rf['final_test_acc']:.3f}   "
          f"{'PASS' if rs >= 0.8 * rf['final_test_acc'] else 'FAIL'}")
    print(f"instrument: bptt baseline {base:.3f} {'PASS' if base > 0.6 else 'FAIL'}")

    fp = OUT / "results_2_compare.json"
    fp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {fp.name}   [{time.perf_counter() - t0:.1f}s]")


if __name__ == "__main__":
    main()
