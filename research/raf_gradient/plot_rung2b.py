"""plot_rung2b.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).parent
d = json.loads((OUT / "results_2b_compare.json").read_text())["runs"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
col = {"bptt": "k", "raf_full": "C0", "raf_sign": "C1", "nodepert_ideal": "C2"}

for m, r in d.items():
    h = r["hist"]
    ax[0].plot(h["step"], h["test_acc"], "o-", ms=3, color=col[m], label=m)
ax[0].set_xlabel("gradient step"); ax[0].set_ylabel("test accuracy"); ax[0].set_ylim(0.4, 1.02)
ax[0].axhline(0.9 * d["bptt"]["hist"]["test_acc"][-1], ls="--", color="crimson", label="S1 bar")
ax[0].set_title("S1 PASS — RAF node-perturbation\ntrains to BPTT level"); ax[0].legend(fontsize=8)

# sign-agree vs train loss (seed-0, raf_full)
ps = np.array(d["raf_full"]["diag"]["perstep_loss_signagree"], float)
ps = ps[np.argsort(ps[:, 0])]
ax[1].scatter(ps[:, 0], ps[:, 1], s=14, alpha=0.6, color="C0")
ax[1].axvline(0.10, ls=":", color="gray"); ax[1].axhline(0.85, ls="--", color="crimson")
ax[1].set_xlabel("train loss (proxy for gradient signal)"); ax[1].set_ylabel("|fd|-weighted sign-agree, kept units")
ax[1].set_title("sign fidelity vs remaining signal\n(collapses as the hard loss saturates)")
ax[1].set_xlim(left=-0.01)

# per-seed S2 signal-present by w2 sign
seeds = [0, 1, 2, 3]
pos = [0.929, 0.972, 1.000, 0.957]
neg = [0.770, 0.823, 0.997, 0.521]
xp = np.arange(len(seeds))
ax[2].bar(xp - 0.2, pos, 0.4, color="C0", label="+w2 units")
ax[2].bar(xp + 0.2, neg, 0.4, color="C3", label="-w2 units")
ax[2].axhline(0.85, ls="--", color="crimson", label="S2 bar"); ax[2].axhline(0.5, ls=":", color="gray")
ax[2].set_xticks(xp); ax[2].set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
ax[2].set_ylabel("S2 sign-agree (signal-present)"); ax[2].set_ylim(0, 1.05)
ax[2].set_title("S2 FAIL — minority-sign (-w2) units:\nunreliable, 0.52-1.00 across seeds"); ax[2].legend(fontsize=8)

fig.suptitle("Rung 2b — validated task + threshold-tracking dither: trains fine (S1), sign credit still unreliable for -w2 (S2)")
fig.tight_layout(); fig.savefig(OUT / "rung2b_summary.png", dpi=130)
print("wrote rung2b_summary.png")
