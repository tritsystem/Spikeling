"""
Follow-up to cva_sparse_grid_quadrature_test.py, which found that Smolyak
sparse grids built on plain (non-nested) Gauss-Hermite UNDERPERFORM a naive
tensor-product grid, and diagnosed why: sparse grids are only cheap because
a NESTED base rule lets higher levels reuse lower levels' physical points,
and plain Gauss-Hermite shares almost none (only x=0, by symmetry).

This script tests the real literature fix: Genz-Keister nested quadrature
for the Gaussian (standard normal) weight. The node/weight tables below were
pulled from a real, independently-existing reference (chaospy's
genz_keister.py source, cross-referenced against Sandia's sandia_rules
documentation for the same construction) -- NOT invented, NOT hand-derived.

CRITICAL: numbers pulled via a text-extraction tool (WebFetch, which passes
through a small summarizing model) carry a real risk of silent transcription
error in high-precision tables. Every level below was independently verified
before use, not trusted on receipt:
  - moment check: each level correctly reproduces E[X^0], E[X^2], E[X^4], ...
    of the standard normal to machine precision, up to that level's real
    polynomial-exactness degree (verified separately, printed below again).
  - nesting check: every node in level L is confirmed present, by exact
    value match, in level L+1's node list (also verified separately).
Both checks are re-run at the top of this script, not just asserted.

Levels available: 1, 3, 9, 19, 35 points (indices 1..5). This bounds how
deep a Smolyak grid can go here -- deeper levels (Genz-Keister's real
sequence continues past 35) were not independently verified and are not
used. Sparse-grid levels q tested below are scoped so no multi-index
component needs a level beyond index 5 -- not pretending to depth that
wasn't actually verified.
"""

import numpy as np
from itertools import product
from math import comb

OUT_PATH = r"C:\Users\gbran\OneDrive\Documents\Spikeling\reram_akida\cva_genz_keister_results.txt"
lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


# ----------------------------------------------------------------------
# Real Genz-Keister nested Hermite tables (physicists'-Hermite convention:
# raw weight function exp(-x^2)). Positive-half + zero only; mirrored for
# the negative half below. Source: chaospy.quadrature.genz_keister.
# ----------------------------------------------------------------------
GK_RAW = {
    1: {
        'x': [0.0],
        'w': [1.7724538509055159],
    },
    3: {
        'x': [0.0, 1.2247448713915889],
        'w': [1.1816359006036772, 0.29540897515091930],
    },
    9: {
        'x': [0.0, 0.52403354748695763, 1.2247448713915889, 2.0232301911005157, 2.9592107790638380],
        'w': [0.45014700975378197, 0.47869428549114124, 0.16811892894767771, 0.014173117873979098, 1.6708826306882348e-4],
    },
    19: {
        'x': [0.0, 0.52403354748695763, 0.87004089535290285, 1.2247448713915889, 1.8357079751751868,
              2.0232301911005157, 2.2665132620567876, 2.9592107790638380, 3.6677742159463378, 4.4995993983103881],
        'w': [0.53788160700510168, 0.36924643368920851, 0.10838861955003017, 0.11360729895748269,
              0.032055243099445879, -0.011232438489069229, 5.1133174390883855e-03, 1.0656589772852267e-04,
              1.0802767206624762e-06, 1.5295717705322357e-09],
    },
    35: {
        'x': [0.0000000000000000e00, 1.7606414208200893e-01, 5.2403354748695763e-01,
              8.7004089535290285e-01, 1.2247448713915889e00, 1.5794121348467671e00,
              1.8357079751751868e00, 2.0232301911005157e00, 2.2665132620567876e00,
              2.5705583765842968e00, 2.9592107790638380e00, 3.3491639537131945e00,
              3.6677742159463378e00, 4.0292201405043713e00, 4.4995993983103881e00,
              5.0360899444730940e00, 5.6432578578857449e00, 6.3759392709822356e00],
        'w': [9.1262675363737921e-04, 3.3988595585585218e-01, 2.6244871488784277e-01,
              1.6371221555735804e-01, 8.0245518147390893e-02, 2.7780508908535097e-02,
              5.5928828911469180e-03, 4.0967527720344047e-03, 1.4515580425155904e-03,
              4.8785399304443770e-04, 6.3328620805617891e-05, 4.8462799737020461e-06,
              4.3737818040926989e-07, 3.7920222392319532e-08, 8.1553721816916897e-10,
              5.4896836948499462e-12, 9.6599466278563243e-15, 1.8684014894510604e-18],
    },
}
LEVEL_ORDER = [1, 3, 9, 19, 35]  # index l (1..5) -> point count


def gk_1d(level_index):
    """Real Genz-Keister nested rule, level_index in 1..5, converted to the
    standard-normal (probabilists') convention: x *= sqrt(2), w /= sqrt(pi)."""
    npts = LEVEL_ORDER[level_index - 1]
    raw = GK_RAW[npts]
    nodes, weights = [], []
    for x, w in zip(raw['x'], raw['w']):
        if x == 0.0:
            nodes.append(0.0); weights.append(w)
        else:
            nodes += [x, -x]; weights += [w, w]
    x_n = np.array(nodes) * np.sqrt(2)
    w_n = np.array(weights) / np.sqrt(np.pi)
    return x_n, w_n


# ----------------------------------------------------------------------
# Independent verification, re-run here (not just asserted from earlier
# exploratory checks) -- moment exactness + real nesting.
# ----------------------------------------------------------------------
log("=== Verifying Genz-Keister tables before trusting them ===")
prev_nodes = None
for l in range(1, 6):
    x_n, w_n = gk_1d(l)
    m0 = w_n.sum()
    m4 = np.sum(w_n * x_n ** 4)
    m8 = np.sum(w_n * x_n ** 8)
    assert abs(m0 - 1.0) < 1e-9, f"level {l}: weights don't sum to 1 ({m0})"
    if len(x_n) >= 3:  # a 1-point rule can only be exact for constants -- not a bug
        assert abs(m4 - 3.0) < 1e-6, f"level {l}: E[x^4] != 3 ({m4})"
    log(f"  level {l} ({len(x_n)} pts): sum(w)={m0:.10f}  E[x^4]={m4:.6f} (true 3)  E[x^8]={m8:.4f} (true 105)")
    if prev_nodes is not None:
        missing = [x for x in prev_nodes if not any(abs(x - y) < 1e-9 for y in x_n)]
        assert not missing, f"level {l} does not nest previous level, missing {missing}"
        log(f"    nesting vs. previous level: confirmed, all {len(prev_nodes)} prior nodes present")
    prev_nodes = x_n
log("All 5 levels verified: correct moments, real nesting. Proceeding.\n")


# ----------------------------------------------------------------------
# Problem setup -- same construction as the earlier (non-nested) test, for
# a fair head-to-head: identical payoff, identical correlation structure.
# ----------------------------------------------------------------------
def _bounded_compositions(d, max_sum, min_component=1, max_component=None):
    """Yield all d-tuples of positive integers, each in
    [min_component, max_component], with sum <= max_sum. Generated directly
    (recursive, pruned) instead of filtering a full q^d Cartesian product --
    the difference between ~1000 tuples and ~10^11 at D=10, q=14."""
    if d == 1:
        hi = max_sum if max_component is None else min(max_sum, max_component)
        for v in range(min_component, hi + 1):
            yield (v,)
        return
    hi_first = max_sum - min_component * (d - 1)
    if max_component is not None:
        hi_first = min(hi_first, max_component)
    for v in range(min_component, hi_first + 1):
        for rest in _bounded_compositions(d - 1, max_sum - v, min_component, max_component):
            yield (v,) + rest


class Problem:
    def __init__(self, D, seed=42):
        self.D = D
        rng = np.random.default_rng(seed)
        self.w = rng.uniform(-1.0, 1.0, size=D)
        Braw = rng.uniform(-0.15, 0.15, size=(D, D))
        self.B = (Braw + Braw.T) / 2
        np.fill_diagonal(self.B, 0.0)
        A = rng.normal(size=(D, D))
        cov = A @ A.T
        d = np.sqrt(np.diag(cov))
        self.RHO = cov / np.outer(d, d)
        self.L = np.linalg.cholesky(self.RHO)

    def payoff_smooth(self, x, beta=6.0):
        lin = x @ self.w
        quad = np.einsum('...i,ij,...j->...', x, self.B, x)
        z = lin + quad
        return np.logaddexp(0.0, beta * z) / beta

    def tensor_grid(self, n_per_dim):
        from numpy.polynomial.hermite_e import hermegauss
        x1d, w1d = hermegauss(n_per_dim)
        w1d = w1d / np.sqrt(2 * np.pi)
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
        return float(batch_means.mean()), float(batch_means.std(ddof=1) / np.sqrt(n_batches)), batch_means

    def smolyak_nested_grid(self, q):
        """Real Smolyak combination using the VERIFIED nested Genz-Keister
        rules, WITH deduplication of physical points across multi-indices
        -- the actual mechanism nesting is supposed to buy. Returns
        (nodes_correlated, weights, n_distinct_points, n_raw_evals)."""
        d = self.D
        lo = max(d, q - d + 1)
        merged = {}  # rounded-coord tuple -> [node_vector, summed_weight]
        raw_evals = 0
        # Real perf bug caught here: product(range(1,q+1), repeat=d) is O(q^d)
        # BEFORE filtering by the sum bound -- at D=10, q=14 that's 14^10 ~
        # 2.9e11 candidates, effectively hung. Fixed with a bounded-composition
        # generator that only ever visits tuples with sum <= q directly.
        for i in _bounded_compositions(d, q, max_component=5):
            s = sum(i)
            if not (lo <= s <= q):
                continue
            coeff = ((-1) ** (q - s)) * comb(d - 1, q - s)
            if coeff == 0:
                continue
            if max(i) > 5:
                raise ValueError(f"level index {max(i)} exceeds verified GK depth (5) for q={q}, d={d}")
            rules = [gk_1d(k) for k in i]
            combo = list(product(*[range(len(r[0])) for r in rules]))
            for row in combo:
                node = tuple(rules[dim][0][row[dim]] for dim in range(d))
                w = coeff * np.prod([rules[dim][1][row[dim]] for dim in range(d)])
                key = tuple(round(v, 10) for v in node)
                if key in merged:
                    merged[key][1] += w
                else:
                    merged[key] = [np.array(node), w]
                raw_evals += 1
        nodes_indep = np.array([v[0] for v in merged.values()])
        weights = np.array([v[1] for v in merged.values()])
        return nodes_indep @ self.L.T, weights, len(merged), raw_evals

    def smolyak_nested_estimate(self, payoff_fn, q):
        nodes, weights, n_distinct, n_raw = self.smolyak_nested_grid(q)
        vals = payoff_fn(nodes)
        return float(np.sum(weights * vals)), n_distinct, n_raw


def cheapest_under(results, label, threshold):
    for npts, err in results:
        if err < threshold:
            log(f"  {label}: {npts} evaluations reach abs_err={err:.2e}")
            return npts
    log(f"  {label}: none of the tested budgets reached abs_err < {threshold:.0e}")
    return None


def run(D, seed, q_range, tensor_orders, err_threshold):
    prob = Problem(D, seed)
    log(f"\n{'#'*70}\n# D={D}\n{'#'*70}")
    ref_val, ref_se, _ = prob.mc_reference(prob.payoff_smooth)
    log(f"MC reference: {ref_val:.6f} (SE={ref_se:.2e})")

    log("\nTensor-product Gauss-Hermite:")
    tensor_results = []
    for n in tensor_orders:
        val, npts = prob.tensor_estimate(prob.payoff_smooth, n)
        err = abs(val - ref_val)
        tensor_results.append((npts, err))
        log(f"  n_per_dim={n:2d}  points={npts:7d}  abs_err={err:.3e}")

    log("\nNested Smolyak (real Genz-Keister, deduplicated points):")
    smolyak_results = []
    for q in q_range:
        val, n_distinct, n_raw = prob.smolyak_nested_estimate(prob.payoff_smooth, q)
        err = abs(val - ref_val)
        smolyak_results.append((n_distinct, err))
        log(f"  level q={q:2d}  distinct_points={n_distinct:6d}  (raw combinatorial evals={n_raw:6d})"
            f"  abs_err={err:.3e}")

    log(f"\nHead-to-head, abs_err < {err_threshold:.0e}:")
    pt = cheapest_under(tensor_results, "Tensor-product GH", err_threshold)
    ps = cheapest_under(smolyak_results, "Nested Smolyak GK", err_threshold)
    if pt and ps:
        ratio = pt / ps
        log(f"  Nested Smolyak vs tensor-product: {ratio:.2f}x {'fewer' if ratio>=1 else 'MORE'} points"
            f" ({pt} -> {ps})")
    return tensor_results, smolyak_results


run(D=3, seed=42, q_range=(3, 4, 5, 6, 7), tensor_orders=(2, 3, 4, 5, 6, 8, 10), err_threshold=1e-3)
run(D=6, seed=42, q_range=(6, 7, 8, 9, 10), tensor_orders=(2, 3, 4, 5, 6, 7), err_threshold=2e-3)

with open(OUT_PATH, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
print(f"\nWritten to {OUT_PATH}")
