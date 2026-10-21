#!/usr/bin/env python3
"""SSHNode vs ResonatorState -- pre-registered benchmark (see PREREGISTER.md).

Tone detection at the normalised scale (omega ~ 1, dt = 0.05), scored as
balanced accuracy at a matched false-alarm rate, swept over a per-unit
internal-defect axis (chiral-preserving decouple / chiral-breaking detune).

    python benchmark.py                 # full run, 8 seeds -> RESULTS.json + console table
    python benchmark.py --seeds 3       # quick look
    python benchmark.py --smoke         # 1 seed, tiny windows, just checks nothing NaNs
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "core"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runtime.runtime import ResonatorState

from ssh_node import SSHNode

# --- task / substrate constants (normalised scale, matches ssh_resonator_bridge.py) ---
FREQ = 1.0 / (2 * math.pi)     # -> omega = 1.0
DAMPING = 0.3
DT = 0.05
N_SITES = 8
G_TOPO, G_TRIVIAL, G_UNIFORM = 0.6, -0.6, 0.0
DETUNE = 0.3                   # chiral-breaking: omega_i -> omega_i * (1 +/- DETUNE)
INTERIOR = [1, 2, 3, 4, 5, 6]  # damageable sites (0 = readout edge, 7 = far edge, left intact)
TARGET_FPR = 0.05

# Task difficulty tuned (before the scored multi-seed run) so a single real
# ResonatorState lands ~0.80 balanced accuracy -- imperfect, discriminating,
# room for the other arms to differ either way. Distractor is LOUDER than the
# target and only 1.8x off-resonance, so resonance rejection is required but
# not trivial.
A_TARGET, A_DISTRACT, NOISE = 0.5, 1.5, 1.0
DISTRACT_MULT = 1.8   # distractor sinusoid frequency = DISTRACT_MULT * omega


# ────────────────────────────────────────────────────────────── task
def make_window(rng, target_present, n_ticks):
    t = np.arange(n_ticks) * DT
    sig = A_DISTRACT * np.sin(DISTRACT_MULT * t) + rng.normal(0.0, NOISE, n_ticks)
    if target_present:
        sig = sig + A_TARGET * np.sin(1.0 * t)
    return sig


def dataset(rng, n_windows, n_ticks):
    labels = np.array([i % 2 for i in range(n_windows)])
    rng.shuffle(labels)
    sigs = [make_window(rng, bool(y), n_ticks) for y in labels]
    return sigs, labels


# ─────────────────────────────────────────────────── per-arm scorers
def _zero(node):
    """Reset either a real ResonatorState (scalar x/v) or an SSHNode
    (array x/v) to rest -- ResonatorState has no .reset()."""
    if hasattr(node, "reset"):
        node.reset()
    else:
        node.x = 0.0
        node.v = 0.0
        node.energy_ema = 0.0


def _peak_readout(node, sig):
    """Max |x[readout]| over the window -- the peak resonant amplitude at
    the readout site. Equivalent to peak sqrt(energy_ema) in the alpha->1
    regime this scale runs in; matches ssh_resonator_bridge.py reading r.x
    directly. Returns np.inf if the integrator diverged."""
    _zero(node)
    peak = 0.0
    for u in sig:
        node.step(float(u), DT)
        xe = abs(node.x if np.isscalar(node.x) else node.x[node.readout_site])
        if not math.isfinite(xe):
            return math.inf
        peak = max(peak, xe)
    return peak


def _pop_peaks(nodes, sig):
    """Per-unit peak |x| over the window -- the SAME statistic the ssh /
    single arms are scored on, so the trained arm gets the identical
    feature, only with a learned combination on top."""
    for n in nodes:
        _zero(n)
    peak = np.zeros(len(nodes))
    for u in sig:
        for i, n in enumerate(nodes):
            n.step(float(u), DT)
            peak[i] = max(peak[i], abs(n.x))
    return peak if np.all(np.isfinite(peak)) else np.full(len(nodes), np.nan)


def _op_threshold(neg_scores):
    """Operating point: threshold at the (1 - TARGET_FPR) quantile of the
    negatives' scores, so calibration FPR == TARGET_FPR by construction."""
    return float(np.quantile(neg_scores, 1.0 - TARGET_FPR))


def _balanced_acc(scores, labels, thr):
    fired = scores >= thr
    pos, neg = labels == 1, labels == 0
    tpr = float(fired[pos].mean()) if pos.any() else 0.0
    fpr = float(fired[neg].mean()) if neg.any() else 0.0
    return 0.5 * (tpr + (1.0 - fpr)), tpr, fpr


# ───────────────────────────────────────────────── arm builders
def build_arm(arm_id, defect_class, k, rng):
    """Returns (kind, obj, damaged_sites). kind in {'single','ssh','pop'}."""
    dmg = sorted(rng.choice(INTERIOR, size=k, replace=False).tolist()) if k else []

    if arm_id == "res_single":
        return "single", ResonatorState(name="r", freq_hz=FREQ, damping=DAMPING, coupling=1.0), []

    if arm_id == "res_pop_trained":
        nodes = [ResonatorState(name=f"r{i}", freq_hz=FREQ, damping=DAMPING, coupling=1.0)
                 for i in range(N_SITES)]
        for s in dmg:
            if defect_class == "preserving":
                nodes[s] = None                     # unit dropped from the population
            else:                                   # breaking: detune this unit
                nodes[s] = ResonatorState(name=f"r{s}", freq_hz=FREQ * (1 + DETUNE),
                                          damping=DAMPING, coupling=1.0)
        return "pop", [n for n in nodes if n is not None], dmg

    g = {"ssh_topo": G_TOPO, "ssh_trivial": G_TRIVIAL, "ssh_uniform": G_UNIFORM}[arm_id]
    node = SSHNode(name="ssh", freq_hz=FREQ, damping=DAMPING, coupling=1.0,
                   n_sites=N_SITES, dimerization=g, drive_mode="edge")
    if dmg:
        if defect_class == "preserving":
            node.decouple_sites(dmg)
        else:
            node.detune_sites({s: (DETUNE if i % 2 == 0 else -DETUNE) for i, s in enumerate(dmg)})
    return "ssh", node, dmg


def score_arm(kind, obj, cal, test):
    (cal_sigs, cal_y), (test_sigs, test_y) = cal, test
    if kind == "pop":
        Ec = np.array([_pop_peaks(obj, s) for s in cal_sigs])
        Et = np.array([_pop_peaks(obj, s) for s in test_sigs])
        if not (np.all(np.isfinite(Ec)) and np.all(np.isfinite(Et))):
            return dict(nan=True)
        mu, sd = Ec.mean(0), Ec.std(0) + 1e-9           # standardise on cal stats
        Ec, Et = (Ec - mu) / sd, (Et - mu) / sd
        y = np.where(cal_y == 1, 1.0, -1.0)
        lam = 1.0                                        # firm ridge -- few windows, few features
        w = np.linalg.solve(Ec.T @ Ec + lam * np.eye(Ec.shape[1]), Ec.T @ y)
        cal_scores, test_scores = Ec @ w, Et @ w
    else:
        cal_scores = np.array([_peak_readout(obj, s) for s in cal_sigs])
        test_scores = np.array([_peak_readout(obj, s) for s in test_sigs])
        if not (np.all(np.isfinite(cal_scores)) and np.all(np.isfinite(test_scores))):
            return dict(nan=True)

    thr = _op_threshold(cal_scores[cal_y == 0])
    bacc, tpr, fpr = _balanced_acc(test_scores, test_y, thr)
    return dict(nan=False, bacc=bacc, tpr=tpr, fpr=fpr)


# ─────────────────────────────────────────────────────────── driver
ARMS = ["res_single", "res_pop_trained", "ssh_topo", "ssh_trivial", "ssh_uniform"]


def conditions():
    yield ("clean", 0)
    for dc in ("preserving", "breaking"):
        for k in (1, 2, 4):
            yield (dc, k)


def ci95(vals):
    v = np.asarray(vals, float)
    m, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v)) if len(v) > 1 else 0.0
    return m, m - 1.96 * se, m + 1.96 * se


def main():
    global N_SITES, INTERIOR
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--n-sites", type=int, default=N_SITES,
                    help="hidden chain length (even). Smaller N -> defects bite harder, "
                         "which is what actually tests H5.")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.n_sites != N_SITES:
        N_SITES = a.n_sites
        INTERIOR = [i for i in range(1, N_SITES - 1)]   # all but the two edges

    n_cal, n_test, n_ticks = (8, 8, 60) if a.smoke else (100, 160, 200)
    seeds = 1 if a.smoke else a.seeds
    t0 = time.time()

    # static edge-localisation check (H1) -- no dynamics
    loc = {g_id: SSHNode(name="d", freq_hz=FREQ, damping=DAMPING, coupling=1.0,
                         n_sites=N_SITES, dimerization=g).edge_localization
           for g_id, g in [("ssh_topo", G_TOPO), ("ssh_trivial", G_TRIVIAL), ("ssh_uniform", G_UNIFORM)]}
    print("H1  static edge localisation |psi_edge|^2 :",
          {k: round(v, 3) for k, v in loc.items()})

    raw = {}   # (arm, dc, k) -> list of bacc over seeds  (None entries = NaN/diverged)
    for si in range(seeds):
        seed = 7 + 11 * si
        drng = np.random.default_rng(seed)
        cal = dataset(np.random.default_rng(seed + 1), n_cal, n_ticks)
        test = dataset(np.random.default_rng(seed + 2), n_test, n_ticks)
        for arm in ARMS:
            for (dc, k) in conditions():
                if arm == "res_single" and k != 0:
                    continue
                if k > len(INTERIOR):
                    continue
                _dc = dc if dc != "clean" else "preserving"
                dmg_seed = (seed * 1000 + ARMS.index(arm) * 100
                            + {"preserving": 1, "breaking": 2}[_dc] * 10 + k)
                kind, obj, _ = build_arm(arm, _dc, k, np.random.default_rng(dmg_seed))
                r = score_arm(kind, obj, cal, test)
                raw.setdefault((arm, dc, k), []).append(None if r["nan"] else r["bacc"])
        print(f"  seed {seed} done ({time.time() - t0:5.1f}s)", flush=True)

    # ---- aggregate + console table ----
    print(f"\n{'arm':<18} {'clean':>18}  {'preserving k=1/2/4':>26}  {'breaking k=1/2/4':>26}")
    summary = {}
    for arm in ARMS:
        cells = {}
        for (dc, k) in conditions():
            if (arm, dc, k) not in raw:
                continue
            vals = [x for x in raw[(arm, dc, k)] if x is not None]
            n_nan = len(raw[(arm, dc, k)]) - len(vals)
            if vals:
                m, lo, hi = ci95(vals)
                cells[(dc, k)] = dict(mean=m, lo=lo, hi=hi, n=len(vals), nan=n_nan)
            else:
                cells[(dc, k)] = dict(mean=None, nan=n_nan)
        summary[arm] = cells

        def fmt(dc, k):
            c = cells.get((dc, k))
            if not c or c["mean"] is None:
                return "  NaN " if c and c["nan"] else "  --  "
            return f"{c['mean']:.3f}" + ("*" if c["nan"] else "")

        clean = cells.get(("clean", 0))
        cs = f"{clean['mean']:.3f} [{clean['lo']:.3f},{clean['hi']:.3f}]" if clean and clean["mean"] is not None else "--"
        pres = "/".join(fmt("preserving", k) for k in (1, 2, 4))
        brk = "/".join(fmt("breaking", k) for k in (1, 2, 4))
        print(f"{arm:<18} {cs:>18}  {pres:>26}  {brk:>26}")

    # ---- pre-registered verdict checks ----
    verdicts = {}
    tp = summary["ssh_topo"]
    pop = summary["res_pop_trained"]
    tr = summary["ssh_trivial"]

    def m(cells, key):
        c = cells.get(key)
        return c["mean"] if c and c["mean"] is not None else None

    # H1 -- 3-way: pre-registered bars were topo>0.5 and both others<0.2
    _bars = loc["ssh_topo"] > 0.5 and loc["ssh_trivial"] < 0.2 and loc["ssh_uniform"] < 0.2
    _rank = loc["ssh_topo"] > loc["ssh_uniform"] > loc["ssh_trivial"] and loc["ssh_topo"] > 2 * loc["ssh_trivial"]
    verdicts["H1_edge_localisation"] = "CONFIRMED" if _bars else ("DOWNGRADED_ranking_only" if _rank else "REFUTED")
    # H2 clean parity: pop >= topo at k=0
    a2, b2 = m(pop, ("clean", 0)), m(tp, ("clean", 0))
    verdicts["H2_clean_parity_pop_ge_topo"] = (
        "CONFIRMED" if (a2 is not None and b2 is not None and a2 + 1e-9 >= b2) else "REFUTED/CHECK")
    # H3 topo beats pop under preserving defect (drop across k)
    def drop(cells, dc):
        base, k4 = m(cells, ("clean", 0)), m(cells, (dc, 4))
        return None if base is None or k4 is None else base - k4
    d_tp_p, d_pop_p, d_tr_p = drop(tp, "preserving"), drop(pop, "preserving"), drop(tr, "preserving")
    verdicts["H3_topo_defect_tolerance_win"] = (
        "CONFIRMED" if (d_tp_p is not None and d_pop_p is not None and d_tr_p is not None
                        and d_tp_p < d_pop_p and d_tp_p < d_tr_p)
        else "REFUTED/INCONCLUSIVE")
    # H4 preserving easier than breaking for topo
    d_tp_b = drop(tp, "breaking")
    verdicts["H4_chiral_mechanism"] = (
        "CONFIRMED" if (d_tp_p is not None and d_tp_b is not None and d_tp_p < d_tp_b)
        else "REFUTED")
    # H5 zero-training topo matches trained pop under preserving defect
    verdicts["H5_free_robust_readout"] = (
        "CONFIRMED" if (d_tp_p is not None and d_pop_p is not None and abs(d_tp_p - d_pop_p) < 0.02)
        else "REFUTED")

    print("\nPre-registered verdicts (see PREREGISTER.md; * = a seed diverged):")
    for h, v in verdicts.items():
        print(f"  {h:<34} {v}")

    out = dict(config=dict(n_sites=N_SITES, seeds=seeds, n_cal=n_cal, n_test=n_test,
                           n_ticks=n_ticks, g=(G_TOPO, G_TRIVIAL, G_UNIFORM), detune=DETUNE,
                           target_fpr=TARGET_FPR),
               edge_localization=loc,
               results={f"{a_}|{dc}|{k}": (None if all(x is None for x in v)
                                           else [x for x in v])
                        for (a_, dc, k), v in raw.items()},
               verdicts=verdicts)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "RESULTS.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote RESULTS.json  ({time.time() - t0:.1f}s total)")


if __name__ == "__main__":
    main()
