"""plot_rung2.py — rung-2 comparison + sign-inversion diagnostic"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
OUT = Path(__file__).parent
d = json.loads((OUT / "results_2_compare.json").read_text())["runs"]

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
colors = {"bptt": "k", "raf_full": "C0", "raf_sign": "C1", "nodepert_ideal": "C2"}

for m, r in d.items():
    h = r["hist"]
    ax[0].plot(h["step"], h["test_acc"], "o-", ms=3, color=colors[m], label=m)
ax[0].axhline(0.5, ls=":", color="gray")
ax[0].set_xlabel("gradient step"); ax[0].set_ylabel("test accuracy")
ax[0].set_title("training curves (task near noise floor:\nBPTT overfits to ~0.69)"); ax[0].legend(fontsize=8); ax[0].set_ylim(0.4, 1)

meth = [m for m in d if m != "bptt"]
xp = np.arange(len(meth))
pos = [d[m]["diag"]["sign_agree_w2pos"] for m in meth]
neg = [d[m]["diag"]["sign_agree_w2neg"] for m in meth]
ax[1].bar(xp - 0.2, pos, 0.4, label="+w2 units", color="C0")
ax[1].bar(xp + 0.2, neg, 0.4, label="-w2 units", color="C3")
ax[1].axhline(0.85, ls="--", color="crimson", label="R2 bar 0.85")
ax[1].axhline(0.5, ls=":", color="gray", label="chance")
ax[1].set_xticks(xp); ax[1].set_xticklabels(meth, rotation=15, fontsize=8)
ax[1].set_ylabel("|delta|-weighted sign-agreement")
ax[1].set_title("R2: sign recovery by w2 sign\n(minority-sign units fail)"); ax[1].legend(fontsize=7); ax[1].set_ylim(0, 1.05)

fwd = [d[m]["diag"]["fwd_evals_total"] for m in d]
ax[2].bar(range(len(d)), fwd, color=[colors[m] for m in d])
ax[2].set_yscale("log"); ax[2].set_xticks(range(len(d))); ax[2].set_xticklabels(list(d), rotation=15, fontsize=8)
ax[2].set_ylabel("total forward evaluations (log)")
ax[2].set_title("R3: cost — RAF ~3500x/step,\nno wall-clock win")

fig.suptitle("Rung 2 — RAF node-perturbation training + sign-inversion stress (honest negative: R2 fails)")
fig.tight_layout(); fig.savefig(OUT / "rung2_summary.png", dpi=130)
print("wrote rung2_summary.png")
