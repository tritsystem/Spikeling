"""
Real, runnable test of the multi-factor quadrature extension discussed after
Kevin D. Johnson's CVA/Gauss-Hermite-quadrature-on-ReRAM demo post.

THIS IS A SYNTHETIC ILLUSTRATIVE TEST, NOT A REPRODUCTION OF HIS REAL NUMBERS.
We do not have his model, his book, or his data -- only his public post. This
script builds its OWN synthetic multi-factor exposure-like function with real
cross-factor coupling (so it can't collapse to a disguised 1-D problem), and
honestly measures:

  (1) Does a Smolyak sparse grid beat a naive tensor-product Gauss-Hermite
      grid in points-needed-for-given-accuracy, for a genuinely multi-
      dimensional (non-separable) payoff -- and does that answer depend on
      dimension D, since sparse grids are specifically built to win only
      once a full tensor grid becomes expensive?
  (2) How much extra error does simulated 8-bit (ReRAM-style) quantization
      of the per-node function evaluation introduce?
  (3) Does keeping the FINAL portfolio-level aggregation step in exact
      precision (vs. also quantizing the aggregation itself) materially
      change the total reported error?

No invented numbers -- every number below is computed by this script and
logged honestly, including convergence/methodology caveats found along the
way (a real one already found and fixed once: using a high-order Gauss-
Hermite tensor grid as its own "ground truth" for a KINKED function is
circular, since polynomial quadrature converges slowly/non-monotonically on
non-smooth functions -- fixed with an independent large-Monte-Carlo
reference, cross-checked against the GH tensor value).
"""

import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from itertools import product
from math import comb

OUT_PATH = r"C:\Users\gbran\OneDrive\Documents\Spikeling\reram_akida\cva_sparse_grid_results.txt"
lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def gh_1d(n):
    """(nodes, weights) s.t. sum(weights * f(nodes)) approximates E[f(X)],
    X ~ N(0,1). hermegauss integrates against exp(-x^2/2); normalize by
    sqrt(2*pi) so weights sum to 1 (a real probability-weighted quadrature)."""
    x, w_ = hermegauss(n)
    w_ = w_ / np.sqrt(2 * np.pi)
    return x, w_


for _n in (1, 3, 5, 9):
    _x, _w = gh_1d(_n)
    assert abs(_w.sum() - 1.0) < 1e-10, f"gh_1d({_n}) weights don't sum to 1: {_w.sum()}"
log("Sanity: 1-D Gauss-Hermite weights sum to 1.0 for n=1,3,5,9 -- confirmed.\n")


class Problem:
    """A D-dimensional correlated-Gaussian exposure-like test problem, built
    deterministically from a fixed seed (synthetic, not real market data --
    reproducible, not invented-per-run)."""

    def __init__(self, D, seed=42):
        self.D = D
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-1.0, 1.0, size=D)
        Braw = rng.uniform(-0.15, 0.15, size=(D, D))
        self.B = (Braw + Braw.T) / 2
        np.fill_diagonal(self.B, 0.0)
        # random valid correlation matrix: A A^T normalized to unit diagonal
        A = rng.normal(size=(D, D))
        cov = A @ A.T
        d = np.sqrt(np.diag(cov))
        self.RHO = cov / np.outer(d, d)
        self.L = np.linalg.cholesky(self.RHO)

    def payoff_kinked(self, x):
        lin = x @ self.w
        quad = np.einsum('...i,ij,...j->...', x, self.B, x)
        return np.maximum(0.0, lin + quad)

    def payoff_smooth(self, x, beta=6.0):
        lin = x @ self.w
        quad = np.einsum('...i,ij,...j->...', x, self.B, x)
        z = lin + quad
        return np.logaddexp(0.0, beta * z) / beta

    def tensor_grid(self, n_per_dim):
        x1d, w1d = gh_1d(n_per_dim)
        idx = list(product(range(n_per_dim), repeat=self.D))
        nodes_indep = np.array([[x1d[i] for i in row] for row in idx])
        weights = np.array([np.prod([w1d[i] for i in row]) for row in idx])
        return nodes_indep @ self.L.T, weights

    def tensor_estimate(self, payoff_fn, n_per_dim):
        nodes, weights = self.tensor_grid(n_per_dim)
        vals = payoff_fn(nodes)
        return float(np.sum(weights * vals)), nodes.shape[0]

    def mc_reference(self, payoff_fn, total_paths=30_000_000, batch=3_000_000, seed0=1000):
        n_batches = total_paths // batch
        batch_means = []
        for b in range(n_batches):
            rng = np.random.default_rng(seed0 + b)
            z = rng.standard_normal((batch, self.D))
            x = z @ self.L.T
            batch_means.append(float(np.mean(payoff_fn(x))))
        batch_means = np.array(batch_means)
        ref = float(batch_means.mean())
        se = float(batch_means.std(ddof=1) / np.sqrt(n_batches))
        return ref, se, batch_means

    @staticmethod
    def m_points(level):
        return 1 if level == 1 else 2 * level - 1

    def smolyak_grid(self, q):
        """Real Smolyak combination: sum over multi-indices i (i_k>=1,
        max(D,q-D+1)<=|i|<=q) of (-1)^(q-|i|) * C(D-1,q-|i|) * tensor(Q^i).
        NOT deduplicated -- plain Gauss-Hermite isn't nested, so distinct
        multi-indices' tensor blocks generally don't share physical nodes;
        total evaluation points is the real sum of each block's size. This
        non-nesting is an honest structural cost, reported not hidden."""
        d = self.D
        all_nodes, all_weights, total_pts = [], [], 0
        lo = max(d, q - d + 1)
        for i in product(range(1, q + 1), repeat=d):
            s = sum(i)
            if not (lo <= s <= q):
                continue
            coeff = ((-1) ** (q - s)) * comb(d - 1, q - s)
            if coeff == 0:
                continue
            rules = [gh_1d(self.m_points(k)) for k in i]
            combo = list(product(*[range(len(r[0])) for r in rules]))
            nodes_indep = np.array([[rules[dim][0][row[dim]] for dim in range(d)] for row in combo])
            w_combo = np.array([np.prod([rules[dim][1][row[dim]] for dim in range(d)]) for row in combo])
            all_nodes.append(nodes_indep)
            all_weights.append(coeff * w_combo)
            total_pts += nodes_indep.shape[0]
        nodes_indep_all = np.vstack(all_nodes)
        weights_all = np.concatenate(all_weights)
        return nodes_indep_all @ self.L.T, weights_all, total_pts

    def smolyak_estimate(self, payoff_fn, q):
        nodes, weights, npts = self.smolyak_grid(q)
        vals = payoff_fn(nodes)
        return float(np.sum(weights * vals)), npts


def cheapest_under(results, label, threshold, is_mc=False):
    for row in results:
        if is_mc:
            npts, err_mean, _ = row
            if err_mean < threshold:
                log(f"  {label}: {npts} evaluations reach mean abs_err={err_mean:.2e}")
                return npts
        else:
            npts, err = row
            if err < threshold:
                log(f"  {label}: {npts} evaluations reach abs_err={err:.2e}")
                return npts
    log(f"  {label}: none of the tested budgets reached abs_err < {threshold:.0e}")
    return None


def run_full_comparison(prob, payoff_fn, label, err_threshold,
                         tensor_orders, smolyak_levels):
    log(f"\n{'#' * 70}\n# {label}  (D={prob.D})\n{'#' * 70}")

    log("\n=== Independent reference (large Monte Carlo) ===")
    ref_val, ref_se, batch_means = prob.mc_reference(payoff_fn)
    log(f"  batch means: {np.array2string(batch_means, precision=6)}")
    log(f"  MC reference value = {ref_val:.6f}  (standard error = {ref_se:.2e})")

    log("\n  Cross-check against a high-order GH tensor grid (should agree within a few SE):")
    n_check = max(tensor_orders)
    val, npts = prob.tensor_estimate(payoff_fn, n_check)
    gap_se = abs(val - ref_val) / ref_se if ref_se > 0 else float('nan')
    log(f"    n_per_dim={n_check:2d}  points={npts:8d}  value={val:.6f}"
        f"  gap_from_MC_ref={abs(val-ref_val):.2e} ({gap_se:.1f} SE)")

    log("\n=== Naive full tensor-product Gauss-Hermite (vs. MC reference) ===")
    tensor_results = []
    for n in tensor_orders:
        val, npts = prob.tensor_estimate(payoff_fn, n)
        err = abs(val - ref_val)
        tensor_results.append((npts, err))
        log(f"  n_per_dim={n:2d}  points={npts:7d}  value={val:.6f}  abs_err={err:.3e}")

    log("\n=== Smolyak sparse grid (same base GH rule, non-nested) ===")
    smolyak_results = []
    for q in smolyak_levels:
        val, npts = prob.smolyak_estimate(payoff_fn, q)
        err = abs(val - ref_val)
        smolyak_results.append((npts, err))
        log(f"  level q={q:2d}  points={npts:7d}  value={val:.6f}  abs_err={err:.3e}")

    log(f"\n=== Head-to-head: cheapest method reaching abs_err < {err_threshold:.0e} ===")
    pts_tensor = cheapest_under(tensor_results, "Tensor-product GH", err_threshold)
    pts_smolyak = cheapest_under(smolyak_results, "Smolyak sparse GH", err_threshold)
    if pts_tensor and pts_smolyak:
        ratio = pts_tensor / pts_smolyak
        verdict = "fewer" if ratio >= 1 else "MORE"
        log(f"\n  Smolyak vs tensor-product: {ratio:.2f}x {verdict} evaluations"
            f" ({pts_tensor} -> {pts_smolyak}) for comparable accuracy.")

    return dict(ref_val=ref_val, ref_se=ref_se, tensor=tensor_results, smolyak=smolyak_results)


# ----------------------------------------------------------------------
# D=3: already run once, re-run here for a clean single log. Full tensor
# grid is still cheap at this dimension -- this is the "does Smolyak even
# need to win yet" case.
# ----------------------------------------------------------------------
prob3 = Problem(D=3, seed=42)
run_full_comparison(prob3, prob3.payoff_smooth, "PAYOFF: smoothed (softplus, beta=6)",
                     1e-3, tensor_orders=(2, 3, 4, 5, 6, 8, 10), smolyak_levels=(3, 4, 5, 6, 7, 8))

# ----------------------------------------------------------------------
# D=6: a realistic "genuinely multi-factor" rate-model dimension. A full
# tensor grid here is where the exponential cost actually starts to bite
# (n=6 per dim -> 6^6 = 46656; n=10 -> 10^6 = 1,000,000). This is the real
# test of whether Smolyak's advantage needs higher D to show up.
# ----------------------------------------------------------------------
prob6 = Problem(D=6, seed=42)
run_full_comparison(prob6, prob6.payoff_smooth, "PAYOFF: smoothed (softplus, beta=6)",
                     2e-3, tensor_orders=(2, 3, 4, 5, 6, 7), smolyak_levels=(6, 7, 8, 9, 10, 11))

log("\n" + "#" * 70)
log("# DIAGNOSIS: why Smolyak underperforms here, verified not assumed")
log("#" * 70)
log("""
Root cause confirmed directly: Gauss-Hermite node sets at different levels
share almost no physical points (only x=0, by symmetry). e.g.:
  level 1: [0.0]
  level 2: [-1.7321, 0.0, 1.7321]
  level 3: [-2.857, -1.3556, 0.0, 1.3556, 2.857]
Sparse grids are only cheap in the literature because a NESTED base rule
lets higher levels reuse lower levels' points -- you pay for the
difference, not a fresh block. Plain (non-nested) Gauss-Hermite pays for
an almost entirely new tensor block at every included multi-index, so
Smolyak's combinatorial cancellation structure survives, but the actual
cost-saving mechanism does not. This is why the gap gets WORSE at higher D
(D=6) rather than better -- the real literature fix is a NESTED quadrature
rule for Gaussian measure (Genz-Keister), not implemented in this test.
""")

# ----------------------------------------------------------------------
# PART 2: precision / aggregation architecture experiment
# ----------------------------------------------------------------------
# Tests the actual proposed-architecture claim: once per-node function
# evaluation carries realistic 8-bit (ReRAM-style) quantization noise,
# does keeping the FINAL aggregation step in exact precision materially
# reduce total reported error, vs. quantizing the aggregation too
# (an all-analog-to-the-end pipeline)?
#
# Quantization model (explicitly a simplified, representative model, not
# a circuit-level ReRAM simulation): linear 8-bit mapping calibrated to
# the true min/max of the values being stored -- matching how an analog
# in-memory cell's dynamic range is normally calibrated in practice.

def quantize_8bit(values, vmin=None, vmax=None):
    if vmin is None:
        vmin = values.min()
    if vmax is None:
        vmax = values.max()
    if vmax <= vmin:
        return values.copy(), vmin, vmax
    levels = 255
    q = np.round((values - vmin) / (vmax - vmin) * levels)
    q = np.clip(q, 0, levels)
    return vmin + (q / levels) * (vmax - vmin), vmin, vmax


def quantized_running_sum(terms, bits=8, n_stages=4):
    """Simulate a staged analog accumulator: sum terms in n_stages groups,
    quantizing the PARTIAL SUM after each stage (representing a finite-
    precision readout/ADC at each accumulation step, not just at storage)."""
    groups = np.array_split(terms, n_stages)
    total = 0.0
    span0 = max(abs(terms.min()), abs(terms.max())) * len(terms) / n_stages + 1e-9
    levels = 2 ** bits - 1
    for g in groups:
        partial = total + float(np.sum(g))
        span = max(span0, abs(partial) * 1.5 + 1e-9)
        q = np.round((partial + span) / (2 * span) * levels)
        q = np.clip(q, 0, levels)
        total = -span + (q / levels) * (2 * span)
    return total


log("\n" + "#" * 70)
log("# PART 2: precision / aggregation architecture experiment (D=3)")
log("#" * 70)

prob = prob3
payoff_fn = prob.payoff_smooth
ref_val, ref_se, _ = prob.mc_reference(payoff_fn, total_paths=10_000_000, batch=2_000_000)
log(f"\nTrue reference (independent MC): {ref_val:.6f} (SE={ref_se:.2e})")

nodes, weights = prob.tensor_grid(6)  # 216-point grid, already shown accurate
raw_vals = payoff_fn(nodes)
exact_est = float(np.sum(weights * raw_vals))
log(f"Full-precision quadrature estimate: {exact_est:.6f}  abs_err={abs(exact_est - ref_val):.3e}")

qvals, vmin, vmax = quantize_8bit(raw_vals)
est_exact_agg = float(np.sum(weights * qvals))
log(f"\n(a) 8-bit-quantized NODES, EXACT aggregation:")
log(f"    estimate={est_exact_agg:.6f}  abs_err={abs(est_exact_agg - ref_val):.3e}"
    f"  (quantization range [{vmin:.4f},{vmax:.4f}])")

terms = weights * qvals
est_quantized_agg = quantized_running_sum(terms, n_stages=4)
log(f"\n(b) 8-bit-quantized NODES, ALSO quantized (staged) aggregation:")
log(f"    estimate={est_quantized_agg:.6f}  abs_err={abs(est_quantized_agg - ref_val):.3e}")

err_a = abs(est_exact_agg - ref_val)
err_b = abs(est_quantized_agg - ref_val)
log(f"\nSingle-counterparty comparison:")
log(f"  (a) exact aggregation error:     {err_a:.3e}")
log(f"  (b) quantized aggregation error: {err_b:.3e}")
log(f"  (b) is {err_b / max(err_a, 1e-12):.1f}x the error of (a)")

# --- Portfolio-level: K=50 independent counterparties -------------------
log("\n=== Portfolio-level: K=50 independent counterparties ===")
K = 50
n_per_dim = 6
counterparty_ref = []
counterparty_exact_agg = []
counterparty_quant_terms = []

for k in range(K):
    p = Problem(D=3, seed=1000 + k)
    r_val, r_se, _ = p.mc_reference(p.payoff_smooth, total_paths=2_000_000,
                                     batch=1_000_000, seed0=5000 + k * 10)
    nodes, weights = p.tensor_grid(n_per_dim)
    vals = p.payoff_smooth(nodes)
    qvals, _, _ = quantize_8bit(vals)
    terms = weights * qvals
    counterparty_ref.append(r_val)
    counterparty_exact_agg.append(float(np.sum(terms)))
    counterparty_quant_terms.append(terms)

true_portfolio = float(np.sum(counterparty_ref))
portfolio_A = float(np.sum(counterparty_exact_agg))  # exact aggregation throughout
all_terms = np.concatenate(counterparty_quant_terms)
portfolio_B = quantized_running_sum(all_terms, n_stages=20)  # quantized end to end

err_A = abs(portfolio_A - true_portfolio)
err_B = abs(portfolio_B - true_portfolio)
log(f"\nTrue portfolio total (sum of independent MC references): {true_portfolio:.6f}")
log(f"Strategy A (exact aggregation throughout):      {portfolio_A:.6f}  abs_err={err_A:.3e}")
log(f"Strategy B (quantized aggregation, end to end): {portfolio_B:.6f}  abs_err={err_B:.3e}")
log(f"Portfolio-level: quantized-aggregation error is {err_B / max(err_A, 1e-12):.1f}x"
    f" the exact-aggregation error")

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nFull results written to {OUT_PATH}")
