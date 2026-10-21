"""plot_rung1b.py — sign-recovery figure from results_1b_*.json"""
import json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).parent
def load(t):
    fp = OUT / f"results_1b_{t}.json"
    return json.loads(fp.read_text())["cells"] if fp.exists() else []
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
c = load("q2_spacing_soft")
if c:
    xs = [x["spacing_ratio"] for x in c]
    ax[0].plot(xs, [x["sign_agree_raf_v"] for x in c], "o-", label="RAF-v sign-agree")
    ax[0].plot(xs, [x["r_raf_v_vs_fd"] for x in c], "^-", label="RAF-v signed r")
    ax[0].plot(xs, [x["rho_mag_vs_absfd"] for x in c], "s--", color="gray", label="rung-1 |grad| rho")
    ax[0].set_xlabel("band spread f_hi/f_lo"); ax[0].set_title("Q2  sign vs magnitude under cross-talk")
    ax[0].set_ylim(0, 1.05); ax[0].invert_xaxis(); ax[0].legend(fontsize=8)
c = load("n_soft")
if c:
    xs = [x["D"] for x in c]
    ax[1].plot(xs, [x["sign_agree_raf_v"] for x in c], "o-", label="RAF-v sign-agree")
    ax[1].plot(xs, [x["r_raf_v_vs_fd"] for x in c], "^-", label="RAF-v signed r")
    ax[1].plot(xs, [x["rho_mag_vs_absfd"] for x in c], "s--", color="gray", label="rung-1 |grad| rho")
    ax[1].set_xscale("log", base=2); ax[1].set_xlabel("parameter count D")
    ax[1].set_title("sign holds while magnitude decays"); ax[1].set_ylim(0, 1.05); ax[1].legend(fontsize=8)
c = load("q3_margin_hard")
if c:
    xs = [x["margin"] for x in c]
    ys = [x["sign_agree_raf_v"] if x["sign_agree_raf_v"] == x["sign_agree_raf_v"] else 0.0 for x in c]
    dead = [x["sign_agree_raf_v"] != x["sign_agree_raf_v"] for x in c]
    ax[2].plot(xs, ys, "o-", color="darkorange")
    for xi, d in zip(xs, dead):
        if d: ax[2].annotate("dead", (xi, 0.04), ha="center", color="crimson", fontsize=8)
    ax[2].set_xlabel("operating margin (logistic units)"); ax[2].set_ylim(0, 1.05)
    ax[2].set_title("Q3  hard step: sign only where signal exists")
for a in ax: a.set_ylabel("score"); a.grid(alpha=0.3)
fig.suptitle("Rung 1b — phase-preserving RAF readout (velocity-reference sign recovery)")
fig.tight_layout(); fig.savefig(OUT / "rung1b_summary.png", dpi=130)
print("wrote rung1b_summary.png")
