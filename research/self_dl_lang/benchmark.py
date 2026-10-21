#!/usr/bin/env python3
"""Can an sdl program improve its own routing from an ENDOGENOUS signal --
"did exactly one route win the spiking race, decisively" -- with no labels?

Four arms, all on the same tiny sdl program (`sdl.ROUTING_SDL`), same
held-out eval set:

  A1_fixed        weights hand-set to the task's own route patterns, frozen.
  A2_endo         ES on the endogenous reward only. Topology frozen.
  A3_endo_restr   A2 + prune/spawn driven by the SAME endogenous reward.
  A0_oracle       ES on the TRUE reward (route == label). Ceiling for this
                  topology if it could see labels.

Primary diagnostic: Pearson corr between the per-decision endogenous reward
and per-decision true-correctness on the eval set. If that is ~0 the whole
premise is dead regardless of the arm accuracies.

    python benchmark.py            # 6 seeds
    python benchmark.py --seeds 2  # quick
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

import sdl

N_IN, K = 6, 3
ENDO_MARGIN_SCALE = 20.0


# ─────────────────────────────────────────────────────── task
def make_task(seed):
    """K fixed non-negative route patterns. A sample: pick label y uniformly,
    x = base noise + strength * pattern[y], strength drawn per-sample so a
    controllable fraction of decisions are genuinely ambiguous. Ground truth
    is y by construction."""
    r = np.random.default_rng(seed)
    patterns = r.random((K, N_IN)) ** 1.5          # peaky non-negative patterns
    patterns /= patterns.sum(1, keepdims=True)

    def gen(n, rng):
        X, Y, strong = [], [], []
        for _ in range(n):
            y = int(rng.integers(K))
            s = float(rng.choice([0.20, 0.35, 0.55, 0.80]))    # difficulty mix
            x = np.clip(0.12 + 0.06 * rng.normal(size=N_IN) + s * K * patterns[y], 0, 1)
            X.append(x); Y.append(y); strong.append(s)
        return np.array(X), np.array(Y), np.array(strong)

    return patterns, gen


def endo_reward(tr: sdl.Trace) -> float:
    win = 1.0 if tr.exactly_one else -1.0
    return win + 0.3 * math.tanh(tr.margin / ENDO_MARGIN_SCALE) - 0.5 * tr.double_fires


def true_reward(tr: sdl.Trace, y: int) -> float:
    return 1.0 if (tr.exactly_one and tr.winner == y) else -1.0


# ─────────────────────────────────────────────────── ES trainer
def es_train(prog, X, Y, reward_fn, *, epochs, batch, sigma, lr, rng, restructure=False):
    n = len(X)
    for ep in range(epochs):
        idx = rng.permutation(n)[:batch]
        w = prog.weights
        eps = rng.normal(scale=sigma, size=w.shape)
        prog.weights = w + eps
        rp = np.mean([reward_fn(prog.decide(X[i]), Y[i]) for i in idx])
        prog.weights = w - eps
        rm = np.mean([reward_fn(prog.decide(X[i]), Y[i]) for i in idx])
        prog.weights = np.clip(w + lr * (rp - rm) / (2 * sigma) * eps, 0.0, None)

        if restructure and ep % 4 == 3:
            # window stats from the last minibatch -- endogenous only
            wfc = np.zeros(prog.k)
            rs = []
            for i in idx:
                tr = prog.decide(X[i])
                wfc += tr.fire_counts
                rs.append(endo_reward(tr))
            prog.prune_dead_routes(wfc)
            if np.mean(rs) < 0.0:
                weakest = int(np.argmin([prog.decide(X[i]).potentials for i in idx[:8]],
                                        axis=0).mean() if False else np.argmin(wfc))
                prog.spawn_hidden(weakest)
    return prog


def evaluate(prog, X, Y):
    endo, corr, acc, decisive = [], [], 0, 0
    for x, y in zip(X, Y):
        tr = prog.decide(x)
        endo.append(endo_reward(tr))
        corr.append(1.0 if (tr.exactly_one and tr.winner == y) else 0.0)
        acc += int(tr.exactly_one and tr.winner == y)
        decisive += int(tr.exactly_one)
    endo, corr = np.array(endo), np.array(corr)
    c = float(np.corrcoef(endo, corr)[0, 1]) if endo.std() > 1e-9 and corr.std() > 1e-9 else 0.0
    return dict(acc=acc / len(X), decisive=decisive / len(X), endo_true_corr=c)


# ─────────────────────────────────────────────────── driver
def run_seed(seed, n_train, n_eval, epochs):
    patterns, gen = make_task(seed)
    rng = np.random.default_rng(seed + 100)
    Xtr, Ytr, _ = gen(n_train, np.random.default_rng(seed + 1))
    Xev, Yev, _ = gen(n_eval, np.random.default_rng(seed + 2))
    matched = 4.0 * (patterns.T / patterns.T.max())          # 6x3 non-negative, ~task-aligned

    out = {}

    p = sdl.compile_source(sdl.ROUTING_SDL, seed=seed)
    p.weights = matched
    out["A1_fixed"] = evaluate(p, Xev, Yev)

    p = sdl.compile_source(sdl.ROUTING_SDL, seed=seed)
    p.weights = rng.uniform(0.5, 3.0, size=(N_IN, K))
    es_train(p, Xtr, Ytr, lambda tr, y: endo_reward(tr),
             epochs=epochs, batch=24, sigma=0.6, lr=0.4, rng=rng)
    out["A2_endo"] = evaluate(p, Xev, Yev)

    p = sdl.compile_source(sdl.ROUTING_SDL, seed=seed)
    p.weights = rng.uniform(0.5, 3.0, size=(N_IN, K))
    es_train(p, Xtr, Ytr, lambda tr, y: endo_reward(tr),
             epochs=epochs, batch=24, sigma=0.6, lr=0.4, rng=rng, restructure=True)
    out["A3_endo_restr"] = evaluate(p, Xev, Yev)

    p = sdl.compile_source(sdl.ROUTING_SDL, seed=seed)
    p.weights = rng.uniform(0.5, 3.0, size=(N_IN, K))
    es_train(p, Xtr, Ytr, true_reward,
             epochs=epochs, batch=24, sigma=0.6, lr=0.4, rng=rng)
    out["A0_oracle"] = evaluate(p, Xev, Yev)

    return out


def ci95(v):
    v = np.asarray(v, float)
    m = v.mean()
    se = v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    n_train, n_eval, epochs, seeds = ((120, 120, 8, 1) if a.smoke
                                      else (400, 400, a.epochs, a.seeds))
    t0 = time.time()

    rows = {}
    for si in range(seeds):
        r = run_seed(7 + 13 * si, n_train, n_eval, epochs)
        for arm, d in r.items():
            rows.setdefault(arm, []).append(d)
        print(f"  seed {7 + 13 * si} done ({time.time() - t0:.1f}s)", flush=True)

    print(f"\n{'arm':<16} {'true acc':>20} {'decisive':>10} {'corr(endo,true)':>18}")
    agg = {}
    for arm in ("A1_fixed", "A2_endo", "A3_endo_restr", "A0_oracle"):
        accs = [d["acc"] for d in rows[arm]]
        dec = [d["decisive"] for d in rows[arm]]
        cor = [d["endo_true_corr"] for d in rows[arm]]
        m, lo, hi = ci95(accs)
        cm, clo, chi = ci95(cor)
        agg[arm] = dict(acc=ci95(accs), decisive=ci95(dec), corr=ci95(cor))
        print(f"{arm:<16} {m:.3f} [{lo:.3f},{hi:.3f}]   {np.mean(dec):.2f}     "
              f"{cm:+.3f} [{clo:+.3f},{chi:+.3f}]")

    a1, a2, a3, a0 = (agg[k]["acc"][0] for k in ("A1_fixed", "A2_endo", "A3_endo_restr", "A0_oracle"))
    corr_lo = agg["A1_fixed"]["corr"][1]
    v = {}
    v["premise_endo_tracks_truth"] = "ALIVE" if corr_lo > 0.05 else "DEAD (corr CI includes ~0)"
    v["H_A2_beats_A1"] = "CONFIRMED" if a2 > a1 + 0.01 else "REFUTED"
    v["H_A2_below_oracle"] = "CONFIRMED" if a2 < a0 - 0.02 else "REFUTED/parity"
    v["H_A3_no_better_than_A2"] = "CONFIRMED" if a3 <= a2 + 0.01 else "REFUTED (A3 helped)"
    print("\nPre-registered verdicts (see PREREGISTER.md):")
    for k, val in v.items():
        print(f"  {k:<28} {val}")

    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.json"), "w") as f:
        json.dump(dict(config=dict(seeds=seeds, epochs=epochs, n_train=n_train, n_eval=n_eval,
                                   n_in=N_IN, k=K, chance=1.0 / K),
                       agg={a_: {m_: list(t_) for m_, t_ in d.items()} for a_, d in agg.items()},
                       per_seed=rows, verdicts=v), f, indent=2)
    print(f"\nwrote RESULTS.json  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
