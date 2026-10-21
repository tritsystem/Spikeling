"""plot_rung2c.py"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).parent
d = json.loads((OUT / "results_2c_compare.json").read_text())["runs"]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
col = {"bptt": "k", "raf_full": "C0", "raf_sign": "C1", "nodepert_ideal": "C2"}

for m, r in d.items():
    h = r["hist"]
    ax[0].plot(h["step"], h["test_mse"], "o-", ms=3, color=col[m], label=m)
ax[0].axvline(150, ls=":", color="crimson"); ax[0].text(152, ax[0].get_ylim()[1]*0.8, "beta flip", color="crimson", fontsize=8)
ax[0].set_xlabel("gradient step"); ax[0].set_ylabel("test MSE"); ax[0].set_yscale("log")
ax[0].set_title("U1 marginal — RAF trains (R2~0.93)\nbut misses the tight MSE bar"); ax[0].legend(fontsize=8)

# Gate F: max|fd| does not collapse
h = d["raf_full"]["hist"]
mf = np.array(h["maxfd"], float)
ax[1].plot(h["step"], mf, "o-", ms=3, color="C0")
ax[1].axvline(150, ls=":", color="crimson")
ax[1].set_xlabel("gradient step"); ax[1].set_ylabel("max_j |fd_j|  (FD signal)")
ax[1].set_title("Gate F PASS — regression signal survives\nconvergence (late/early = 0.40; 2b collapsed)")

# U3 vs U2 across seeds
seeds = [0, 1, 2, 3]
u3 = [0.987, 1.000, 1.000, 1.000]
u2neg = [0.62, 0.73, 0.67, 0.78]
xp = np.arange(len(seeds))
ax[2].bar(xp - 0.2, u3, 0.4, color="C0", label="U3: at w2 zero-crossings")
ax[2].bar(xp + 0.2, u2neg, 0.4, color="C3", label="U2: bulk -w2 steps")
ax[2].axhline(0.85, ls="--", color="crimson", label="bar"); ax[2].axhline(0.5, ls=":", color="gray")
ax[2].set_xticks(xp); ax[2].set_xticklabels([f"seed {s}" for s in seeds], fontsize=8)
ax[2].set_ylabel("sign-agree(δ̂, fd)"); ax[2].set_ylim(0, 1.05)
ax[2].set_title("U3 PASS / U2 FAIL — readout is sharp AT\nthe inversion, mushy in the redundant bulk"); ax[2].legend(fontsize=8)

fig.suptitle("Rung 2c — regression target: signal survives (Gate F) + sign-tracking through w2 inversions works (U3); bulk per-unit sign still unreliable (U1/U2)")
fig.tight_layout(); fig.savefig(OUT / "rung2c_summary.png", dpi=130)
print("wrote rung2c_summary.png")
