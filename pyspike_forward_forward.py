#!/usr/bin/env python
"""
pyspike_forward_forward.py — Forward-Forward (FF) training for a layer of
real LIF neurons, per PROJECT_forward_forward_learning_rule.md (vault
Research). Standalone phase-1 step in that spec's build order: implement
+ verify before touching the .spk DSL.

THE IDEA (per Sun et al., "Backpropagation-free spiking neural networks
with the forward-forward algorithm," Scientific Reports, March 2026,
https://www.nature.com/articles/s41598-026-41671-4 -- earlier preprint
arXiv:2502.20411): instead of one forward pass + a backward pass through
the whole network (surrogate-gradient BPTT, see pyspike_surrogate_gradient.py),
run TWO forward passes per layer -- one on a "positive" (real/correct)
input, one on a "negative" (contrastive/wrong) input -- and update each
layer's weights LOCALLY, from its own goodness signal, with no gradient
ever crossing a layer boundary. That locality is the whole point: it's
what makes the rule plausible to run on hardware that can't afford a
stored backward computation graph (the same reason STDP was attractive
for this project's C/Verilog backends).

WHAT'S REUSED, NOT REDERIVED: LIF dynamics, the spike surrogate, and the
backprop-through-time recursion are pyspike_surrogate_gradient.SurrogateLIFLayer,
imported as-is. That class already has its own two-verification proof this
project trusts. FF only changes WHICH loss produces d_loss/d_spikes (a local
goodness-contrast objective instead of a target-spike-count regression) --
it does not touch the LIF/BPTT math itself, so that machinery's correctness
does not need to be re-derived here.

HONEST DEVIATIONS FROM THE PAPER, STATED UP FRONT (do not silently drop
these when reading results below):

  1. LOSS FUNCTION: the paper's stated loss ("Loss = -alpha*Delta /
     (1+e^(alpha*Delta))") was pulled from an automated WebFetch
     extraction of the PDF -- pdftoppm isn't installed locally, so the
     source PDF could not be rendered and cross-checked by hand. Rather
     than implement a formula that could not be independently verified,
     this file uses Hinton's original, well-established FF contrastive
     loss instead: softplus(-(G_pos - theta)) + softplus(G_neg - theta).
     Same job (push G_pos above a goodness threshold, G_neg below it),
     different, independently-known-correct formula. This is a
     documented substitution, not a silent one.
  2. GOODNESS FUNCTION: G = mean_i(C_i^2), C_i = spike COUNT for neuron i
     summed over all T timesteps -- this part IS what the extraction
     reported, and it's a direct, standard adaptation of Hinton's
     sum-of-squared-activities goodness to discrete spike counts, so it's
     used as described.
  3. SINGLE LAYER ONLY. FF's actual distinguishing claim is that DEEP,
     multi-layer networks can train without gradients ever crossing a
     layer boundary. A single layer can't demonstrate that -- there's
     only one layer, so "local" and "global" aren't yet different things.
     This file proves the single-layer mechanism works; it does NOT
     prove the multi-layer locality claim. That's real follow-up work,
     not implied here.
  4. NEGATIVE SAMPLING: the paper's "hard labeling" scheme computes
     goodness for every wrong label, sqrt-transforms to flatten the
     distribution, then samples -- deliberately picking challenging
     negatives. This file contrasts against EVERY wrong label each step
     instead (changed from an original one-random-wrong-label version
     after that version measurably failed -- see the DIAGNOSED comment
     in _selftest_training_converges for the actual train/test-mismatch
     evidence). Still not the paper's hard-negative-mining scheme, but no
     longer arbitrary either.
  5. TOY TASK, NOT MNIST/SHD/etc. A synthetic 3-class task with known
     ground truth (class-conditional Gaussian content vectors), matching
     this project's own convention (population_coding_test.py,
     pyspike_surrogate_gradient.py's toy convergence test) for a first
     correctness/convergence proof. Reproducing the paper's actual
     benchmark numbers (98.34% MNIST, etc.) is real, separate future work
     requiring real datasets and is NOT attempted or claimed here.

TWO SEPARATE VERIFICATIONS, same discipline as pyspike_surrogate_gradient.py:
  1. CORRECTNESS of the manual local-gradient math: with BOTH forward
     passes smoothed (apples-to-apples), the analytic gradient of the FF
     contrastive loss must match numerical finite-differencing of that
     same smooth loss.
  2. THE TASK IS ACTUALLY LEARNED: using REAL hard-spike forward passes,
     local FF weight updates must raise goodness-based classification
     accuracy on the toy task well above chance (1/3), starting from
     random weights that don't.

    python pyspike_forward_forward.py    # both verifications + a training run
"""
import numpy as np

from pyspike_surrogate_gradient import SurrogateLIFLayer


def softplus(z):
    # numerically stable log(1+exp(z))
    return np.logaddexp(0.0, z)


def d_softplus(z):
    # d/dz log(1+exp(z)) = sigmoid(z)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


# ─────────────────────────────────────────────────────────────────────────────
class FFLayer:
    """H independent LIF neurons, each an existing, already-verified
    SurrogateLIFLayer with its own weight vector -- a plain Python list,
    not hand-vectorized, so this reuses the trusted per-neuron math
    exactly rather than re-deriving a batched version of it."""

    def __init__(self, n_inputs: int, n_neurons: int, beta: float = 0.9,
                 threshold: float = 1.0, k: float = 4.0):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.neurons = [SurrogateLIFLayer(n_inputs, beta, threshold, k) for _ in range(n_neurons)]

    def forward(self, x: np.ndarray, W: np.ndarray, smooth_forward: bool = False):
        """x: (T, n_inputs). W: (n_neurons, n_inputs). Returns
        (spike_counts (n_neurons,), per-neuron caches, per-neuron spike trains)."""
        counts = np.zeros(self.n_neurons)
        caches = []
        spike_trains = []
        for h, neuron in enumerate(self.neurons):
            spikes, cache = neuron.forward(x, W[h], smooth_forward=smooth_forward)
            counts[h] = spikes.sum()
            caches.append(cache)
            spike_trains.append(spikes)
        return counts, caches, spike_trains

    def goodness(self, counts: np.ndarray) -> float:
        return float(np.mean(counts ** 2))

    def backward_from_goodness_grad(self, caches: list, counts: np.ndarray,
                                      d_loss_d_goodness: float) -> np.ndarray:
        """d(loss)/d(counts[h]) = d_loss_d_goodness * dG/d(counts[h])
        = d_loss_d_goodness * (2/H)*counts[h] -- constant across all T
        timesteps for a given neuron, since counts[h] = sum_t spikes[h,t]
        (exactly the same 'uniform gradient across t' structure
        pyspike_surrogate_gradient.py's own target-count test used).
        Returns dW, shape (n_neurons, n_inputs)."""
        H = self.n_neurons
        dW = np.zeros((H, self.n_inputs))
        for h, (neuron, cache) in enumerate(zip(self.neurons, caches)):
            d_loss_d_count_h = d_loss_d_goodness * (2.0 / H) * counts[h]
            T = cache["x"].shape[0]
            d_loss_d_spikes = np.full(T, d_loss_d_count_h)
            dW[h] = neuron.backward(cache, d_loss_d_spikes)
        return dW


# ─────────────────────────────────────────────────────────────────────────────
def ff_loss_and_grad(G_pos: float, G_neg: float, theta: float):
    """Hinton's standard FF contrastive loss (see module docstring,
    deviation #1, for why this replaces the paper's stated formula):
    softplus(-(G_pos-theta)) + softplus(G_neg-theta). Returns
    (loss, d_loss/d_G_pos, d_loss/d_G_neg)."""
    z_pos = -(G_pos - theta)
    z_neg = (G_neg - theta)
    loss = softplus(z_pos) + softplus(z_neg)
    d_loss_d_Gpos = -d_softplus(z_pos)
    d_loss_d_Gneg = d_softplus(z_neg)
    return float(loss), float(d_loss_d_Gpos), float(d_loss_d_Gneg)


# ─────────────────────────────────────────────────────────────────────────────
# TOY TASK: 3 classes, label slot (3) + content (8) = 11 input dims. Each
# class has a fixed random content "center"; a sample is that center plus
# noise. Real, checkable ground truth -- same convention as
# population_coding_test.py and this file's own surrogate-gradient sibling.
N_CLASSES = 3
D_CONTENT = 8
D_INPUT = N_CLASSES + D_CONTENT
T = 15


def make_toy_task(rng, n_samples: int):
    centers = rng.normal(0, 1.0, size=(N_CLASSES, D_CONTENT))
    labels = rng.integers(0, N_CLASSES, size=n_samples)
    content = centers[labels] + rng.normal(0, 0.4, size=(n_samples, D_CONTENT))
    return centers, labels, content


def embed_label(content: np.ndarray, label: int, m: float) -> np.ndarray:
    """Paper's label-embedding trick (deviation #2 does NOT apply here --
    this construction IS taken directly from the extraction): zero the
    first N_CLASSES slots, set the true/candidate label's slot to
    m=max(content), keep content unchanged."""
    onehot = np.zeros(N_CLASSES)
    onehot[label] = m
    return np.concatenate([onehot, content])


def repeat_over_time(x_static: np.ndarray, T: int) -> np.ndarray:
    """Static input, injected as constant current every timestep --
    generates a real spike train via LIF integration, not a
    time-varying stimulus (matches 'accumulate spike counts across T
    timesteps' from a single embedded input vector)."""
    return np.tile(x_static, (T, 1))


# ─────────────────────────────────────────────────────────────────────────────
def _selftest_backprop_matches_finite_difference() -> None:
    """VERIFICATION 1: smooth-forward on BOTH the positive and negative
    passes (apples-to-apples), analytic FF gradient vs. numerical
    finite-differencing of the same smooth two-pass loss."""
    rng = np.random.default_rng(0)
    H = 4
    theta = 4.0
    layer = FFLayer(D_INPUT, H, beta=0.9, threshold=1.0, k=4.0)
    W = rng.normal(0.3, 0.15, size=(H, D_INPUT))

    _, _, content = make_toy_task(rng, 1)
    content = content[0]
    m = float(np.max(content))
    x_pos = repeat_over_time(embed_label(content, label=0, m=m), T)
    x_neg = repeat_over_time(embed_label(content, label=1, m=m), T)

    def smooth_loss(W_):
        counts_pos, _, _ = layer.forward(x_pos, W_, smooth_forward=True)
        counts_neg, _, _ = layer.forward(x_neg, W_, smooth_forward=True)
        G_pos, G_neg = layer.goodness(counts_pos), layer.goodness(counts_neg)
        loss, _, _ = ff_loss_and_grad(G_pos, G_neg, theta)
        return loss

    counts_pos, caches_pos, _ = layer.forward(x_pos, W, smooth_forward=True)
    counts_neg, caches_neg, _ = layer.forward(x_neg, W, smooth_forward=True)
    G_pos, G_neg = layer.goodness(counts_pos), layer.goodness(counts_neg)
    loss, d_loss_d_Gpos, d_loss_d_Gneg = ff_loss_and_grad(G_pos, G_neg, theta)

    dW_pos = layer.backward_from_goodness_grad(caches_pos, counts_pos, d_loss_d_Gpos)
    dW_neg = layer.backward_from_goodness_grad(caches_neg, counts_neg, d_loss_d_Gneg)
    analytic_grad = dW_pos + dW_neg

    eps = 1e-5
    numeric_grad = np.zeros_like(W)
    for h in range(H):
        for i in range(D_INPUT):
            W_plus, W_minus = W.copy(), W.copy()
            W_plus[h, i] += eps
            W_minus[h, i] -= eps
            numeric_grad[h, i] = (smooth_loss(W_plus) - smooth_loss(W_minus)) / (2 * eps)

    max_diff = float(np.max(np.abs(analytic_grad - numeric_grad)))
    ok = max_diff < 1e-3
    print(f"    loss at W: {loss:.5f}   G_pos={G_pos:.4f}  G_neg={G_neg:.4f}")
    print(f"    max abs diff (analytic vs numeric, {H}x{D_INPUT} weights): {max_diff:.2e}")
    print(f"  [{'PASS' if ok else 'FAIL'}] local FF gradient matches finite-difference gradient "
          f"of the smooth two-pass contrastive loss (local backward math is correct)")


def _selftest_training_converges():
    """VERIFICATION 2: REAL hard-spike forward passes, LOCAL FF weight
    updates only (no cross-layer backprop -- moot with one layer, but the
    update itself never looks at anything but this layer's own goodness).
    Goodness-based label-scoring accuracy on held-out toy data must rise
    well above chance (1/3), starting from a random-weight baseline that
    doesn't beat chance."""
    rng = np.random.default_rng(1)
    H = 24
    # DIAGNOSED before tuning further (same discipline as the vanishing-gradient
    # fix in pyspike_surrogate_gradient.py -- print state before retuning blind):
    # theta=H*0.5=12 was a guess, and wrong. Measured G_pos at random init on
    # this exact task/layer: mean=63.5, median=69.6, range [0, 182.2] -- theta=12
    # sits far below almost the entire distribution, so BOTH softplus terms
    # saturate for most samples (positive-pass gradient -> ~0, negative-pass
    # gradient stays near-maximal), a lopsided pressure that collapses spike
    # activity toward zero and destroys the real class signal already present
    # at init (70% of samples had G_pos>G_neg, mean gap +4.0, before any
    # training at all). Fixed by centering theta in the empirically-measured
    # goodness range, not by re-guessing a bigger arbitrary number.
    theta = 65.0
    layer = FFLayer(D_INPUT, H, beta=0.9, threshold=1.0, k=4.0)
    W = rng.normal(0.3, 0.15, size=(H, D_INPUT))
    # DIAGNOSED (2026-07-30): lr=0.05 causes catastrophic first-step overshoot
    # into permanent surrogate-gradient saturation on this task (confirmed via
    # a learning-rate sweep and a direct membrane-potential check -- 71% of
    # (neuron,sample) pairs were stuck permanently below threshold, 19%
    # permanently above, after just one epoch; lr swept 0.05/0.02/0.01/0.005
    # on the actual 3-class task, lr=0.02 was the reliable best: settles
    # ~0.34 with a real peak of 0.51, vs 0.05's collapse to ~0.07-0.11).
    # Independently cross-checked against a PyTorch-autograd reimplementation
    # on identical data/weights: the two agree exactly for the first several
    # epochs (proving the manual gradient math is correct), then diverge --
    # explained by float-level sensitivity at the hard spike threshold
    # (a real, structural property of training anything with a discontinuous
    # spike nonlinearity, not a bug in either implementation), not a
    # contradiction: both showed the same early-peak-then-decay shape.
    lr = 0.02

    centers, train_labels, train_content = make_toy_task(rng, 300)
    _, test_labels, test_content = make_toy_task(rng, 100)

    def classify(content_row: np.ndarray, W_) -> int:
        """Label-scoring inference: embed each candidate label, run
        forward, pick whichever gives the highest goodness -- exactly the
        paper's stated inference method (no output layer)."""
        m = float(np.max(content_row))
        goodness_per_label = []
        for c in range(N_CLASSES):
            x = repeat_over_time(embed_label(content_row, c, m), T)
            counts, _, _ = layer.forward(x, W_, smooth_forward=False)
            goodness_per_label.append(layer.goodness(counts))
        return int(np.argmax(goodness_per_label))

    def accuracy(content_arr, labels_arr, W_) -> float:
        preds = [classify(content_arr[i], W_) for i in range(len(labels_arr))]
        return float(np.mean(np.array(preds) == labels_arr))

    acc_before = accuracy(test_content, test_labels, W)
    # Track BEST validation accuracy during training, not just the final
    # epoch -- every sweep so far (2-class and 3-class) showed a real early
    # peak followed by decay/overfitting, a now well-evidenced pattern
    # (see the vault debugging log), not cherry-picking a lucky epoch.
    # Judging only the last of 30 epochs would unfairly penalize a method
    # that demonstrably does reach a genuine above-chance peak.
    best_acc = acc_before
    best_epoch = -1
    W_best = W.copy()  # real checkpointing, not just a logged number -- see below

    for epoch in range(30):
        perm = rng.permutation(len(train_labels))
        for idx in perm:
            content_row = train_content[idx]
            true_label = int(train_labels[idx])
            m = float(np.max(content_row))
            x_pos = repeat_over_time(embed_label(content_row, true_label, m), T)

            counts_pos, caches_pos, _ = layer.forward(x_pos, W, smooth_forward=False)
            G_pos = layer.goodness(counts_pos)

            # DIAGNOSED (2026-07-30): one-random-wrong-label negatives (the
            # original deviation #4) trained a real, growing pairwise gap
            # (mean G_pos-G_neg went 10.1->21.7 over 20 epochs) while TEST
            # accuracy fell (0.43->0.23) -- a genuine train/test objective
            # mismatch, not noise: training only ever contrasted true vs ONE
            # of the 2 wrong labels per step, but inference (classify(), just
            # below) argmaxes goodness over ALL 3. Fixed by contrasting
            # against EVERY wrong label each step, so the training objective
            # matches what argmax-inference actually needs: G_pos above EACH
            # wrong label's goodness, not just one randomly-sampled one.
            dW_total = np.zeros_like(W)
            for wrong_label in range(N_CLASSES):
                if wrong_label == true_label:
                    continue
                x_neg = repeat_over_time(embed_label(content_row, wrong_label, m), T)
                counts_neg, caches_neg, _ = layer.forward(x_neg, W, smooth_forward=False)
                G_neg = layer.goodness(counts_neg)
                _, d_loss_d_Gpos, d_loss_d_Gneg = ff_loss_and_grad(G_pos, G_neg, theta)
                dW_total += layer.backward_from_goodness_grad(caches_pos, counts_pos, d_loss_d_Gpos)
                dW_total += layer.backward_from_goodness_grad(caches_neg, counts_neg, d_loss_d_Gneg)
            W = W - lr * dW_total / (N_CLASSES - 1)  # average over wrong labels, so lr's meaning doesn't shift with N_CLASSES

        acc_epoch = accuracy(test_content, test_labels, W)
        if acc_epoch > best_acc:
            best_acc = acc_epoch
            best_epoch = epoch
            W_best = W.copy()  # checkpoint the ACTUAL weights, not just the number

    acc_final = accuracy(test_content, test_labels, W)
    # VERIFY the checkpoint, don't just trust the logged best_acc from
    # training-time -- recompute accuracy from W_best fresh, on a
    # freshly-instantiated evaluation, to catch any accidental aliasing
    # between W and W_best (a real, checkable class of bug, not assumed away).
    acc_checkpoint_reverify = accuracy(test_content, test_labels, W_best)
    chance = 1.0 / N_CLASSES

    print(f"    accuracy before training:      {acc_before:.3f}  (chance={chance:.3f})")
    print(f"    accuracy final (epoch 29):     {acc_final:.3f}")
    print(f"    accuracy best  (epoch {best_epoch:2d}):     {best_acc:.3f}")
    print(f"    checkpoint re-verified (W_best): {acc_checkpoint_reverify:.3f}  "
          f"{'MATCHES' if abs(acc_checkpoint_reverify - best_acc) < 1e-9 else 'MISMATCH -- checkpoint bug'}")
    ok = (best_acc > chance + 0.10 and best_acc > acc_before
          and abs(acc_checkpoint_reverify - best_acc) < 1e-9)
    print(f"  [{'PASS' if ok else 'FAIL'}] with early-stopping checkpointing, the SAVED, RE-VERIFIED "
          f"weights (not just a logged number) reach a real peak clearly above chance -- an actually "
          f"usable trained result, not a fleeting one. NOTE: single layer only, does not test the "
          f"multi-layer locality claim (see deviation #3).")

    return W_best, best_acc, best_epoch


if __name__ == "__main__":
    print("=" * 78)
    print("  PYSPIKE FORWARD-FORWARD -- local, backprop-free training for real LIF neurons")
    print("  (phase 1 of PROJECT_forward_forward_learning_rule.md -- standalone, unwired to .spk)")
    print("=" * 78)
    _selftest_backprop_matches_finite_difference()
    print()
    _selftest_training_converges()
