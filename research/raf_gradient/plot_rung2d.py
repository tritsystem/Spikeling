"""plot_rung2d.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).parent
nh = json.loads((OUT / "results_2d_nhid.json").read_text())["cells"]
lr = json.loads((OUT / "results_2d_lr.json").read_text())["cells"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))

x = [c["n_hid"] for c in nh]
ax[0].plot(x, [c["bptt_r2"] for c in nh], "o-", color="k", label="BPTT")
ax[0].plot(x, [c["raf_r2"] for c in nh], "o-", color="C0", label="RAF-full")
ax[0].axvline(4, ls=":", color="crimson"); ax[0].text(4.2, 0.7, "matched (k=4)", color="crimson", fontsize=8)
ax[0].set_xlabel("student n_hid"); ax[0].set_ylabel("test R²"); ax[0].set_ylim(0.6, 1.02)
ax[0].set_title("V1 marginal — RAF plateaus ~0.86,\nBPTT climbs with width (persistent gap)"); ax[0].legend(fontsize=8)

ax[1].plot(x, [c["sign_w2pos_all"] for c in nh], "o-", color="C0", label="+w2 units")
ax[1].plot(x, [c["sign_w2neg_all"] for c in nh], "o-", color="C3", label="-w2 units")
ax[1].axhline(0.85, ls="--", color="crimson", label="V2 bar"); ax[1].axhline(0.5, ls=":", color="gray")
ax[1].axvline(4, ls=":", color="crimson")
ax[1].annotate("2c predicted this\nDECREASING", (10, 0.62), fontsize=8, color="crimson", ha="center")
ax[1].set_xlabel("student n_hid"); ax[1].set_ylabel("bulk sign-agree(δ̂, fd)"); ax[1].set_ylim(0, 1.05)
ax[1].set_title("V3 FALSIFIED — sign fidelity RISES with width\n(over-parameterisation was not the cause)"); ax[1].legend(fontsize=8)

for stp, mk in ((200, "o-"), (400, "s--")):
    pts = [(c["lr"], c["raf_r2"]) for c in lr if c["steps"] == stp]
    ax[2].plot([p[0] for p in pts], [p[1] for p in pts], mk, color="C0", label=f"RAF, {stp} steps")
ax[2].axhline(0.954, ls="-", color="k", label="BPTT R²=0.95")
ax[2].axhline(0.9 * 0.954, ls="--", color="crimson", label="V5 bar")
ax[2].set_xlabel("learning rate"); ax[2].set_ylabel("test R²"); ax[2].set_ylim(0.6, 1.0)
ax[2].set_title("V5 — no lr/step setting closes the gap\n(best 0.860, bar 0.858: bare pass)"); ax[2].legend(fontsize=8)

fig.suptitle("Rung 2d — matched sizing FALSIFIES 2c's over-parameterisation explanation; RAF underfits BPTT by a width-independent margin")
fig.tight_layout(); fig.savefig(OUT / "rung2d_summary.png", dpi=130)
print("wrote rung2d_summary.png")
