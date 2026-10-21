"""Plot rung-1 results from the results_*.json files. python plot_rung1.py"""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent


def load(tag):
    fp = OUT / f"results_{tag}.json"
    return json.loads(fp.read_text())["cells"] if fp.exists() else []


fig, ax = plt.subplots(1, 3, figsize=(13, 4))

# P2 — cross-talk (soft)
c = load("p2_spacing_soft")
if c:
    ax[0].plot([x["spacing_ratio"] for x in c], [x["rho_res_vs_absfd"] for x in c],
               "o-", label="resonator bank")
    ax[0].plot([x["spacing_ratio"] for x in c], [x["rho_lockin_vs_absfd"] for x in c],
               "s--", color="gray", label="naive lock-in")
    ax[0].set_xlabel("band spread  f_hi / f_lo"); ax[0].set_ylabel(r"Spearman $\rho$ vs |FD|")
    ax[0].set_title("P2  cross-talk (D=8, soft)"); ax[0].set_ylim(0, 1.05); ax[0].legend()
    ax[0].invert_xaxis()

# P3 — bandwidth saturation (soft)
c = load("p3_n_soft")
if c:
    ax[1].plot([x["D"] for x in c], [x["rho_res_vs_absfd"] for x in c], "o-", label="resonator bank")
    ax[1].plot([x["D"] for x in c], [x["rho_lockin_vs_absfd"] for x in c],
               "s--", color="gray", label="naive lock-in")
    ax[1].axhline(0.8, ls=":", color="crimson", label="P1 bar 0.8")
    ax[1].set_xlabel("parameter count D"); ax[1].set_ylabel(r"Spearman $\rho$ vs |FD|")
    ax[1].set_title("P3  bandwidth saturation (band x10, soft)"); ax[1].set_ylim(0, 1.05)
    ax[1].set_xscale("log", base=2); ax[1].legend()

# P4 — hard-threshold margin cliff
c = load("p4_margin_hard")
if c:
    xs = [x["margin"] for x in c]
    ys = [x["rho_res_vs_absfd"] if x["rho_res_vs_absfd"] == x["rho_res_vs_absfd"] else 0.0 for x in c]
    dead = [x["rho_res_vs_absfd"] != x["rho_res_vs_absfd"] for x in c]
    ax[2].plot(xs, ys, "o-", color="darkorange")
    for xi, yi, d in zip(xs, ys, dead):
        if d:
            ax[2].annotate("dead\n(loss const)", (xi, 0.02), ha="center", va="bottom", fontsize=8, color="crimson")
    ax[2].set_xlabel("operating margin from threshold (logistic units)")
    ax[2].set_ylabel(r"Spearman $\rho$ vs |FD|")
    ax[2].set_title("P4  hard step: works only near the flip (D=8, A=0.08)")
    ax[2].set_ylim(0, 1.05)

fig.suptitle("Rung 1 — RAF resonator bank as a gradient probe (real Spikeling ResonatorState)")
fig.tight_layout()
fp = OUT / "rung1_summary.png"
fig.savefig(fp, dpi=130)
print("wrote", fp.name)
