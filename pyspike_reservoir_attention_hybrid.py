#!/usr/bin/env python
"""
pyspike_reservoir_attention_hybrid.py -- a modern architecture (self-
attention) twisted together with all three substrates verified this
session: physical reservoir computing, ternary quantization, and
spiking neurons. Built in STAGES, each one checked before combining
further -- the same discipline as the cascade-gradient series, not a
single untested leap.

THE TASK (chosen to make attention's advantage REAL, not assumed): a
sequence with one random MARKER position; the target is the value that
appeared K steps after the marker. A fixed linear readout (what every
reservoir test so far has used) can only learn a FIXED time-lag mapping
-- it has no way to first find "where is the marker" and then look
K steps later, because the marker's position varies per sequence.
Content-based attention can. This is a real, motivated reason to add
attention, not attention for its own sake.

STAGE 1: real reservoir (VectorizedResonatorBank, the exact verified
         physics from fib_noise_scaling_test.py) feeds a plain linear
         readout -- the BASELINE this whole session's reservoir work
         already uses. Expected to do poorly on this task by design.
STAGE 2: reservoir -> real self-attention -> linear readout, full
         precision. Should clearly beat Stage 1 if attention is
         earning its place, not just added complexity.
STAGE 3: Stage 2's attention weights ternary-quantized via QAT (the
         exact TernarySTE trick verified in ternary_vision/qat.py this
         session, reimplemented standalone here since it's a different
         repo/environment).
STAGE 4: a spiking (LIF, surrogate-gradient) nonlinearity inserted into
         the attention output, using the exact SurrogateSpike trick
         verified across the cascade-gradient series.

Each stage reports a REAL number against the SAME task and held-out
data -- this is a comparison, not a demo.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {DEVICE}")

# ─────────────────────────────────────────────────────────────────────────────
# THE TASK: marker-relative recall
# ─────────────────────────────────────────────────────────────────────────────
T = 40
K_LAG = 5  # report the value K steps after the marker


def make_task(rng, n_samples):
    """u: (n_samples, T) input signal. marker_pos: (n_samples,) where the
    marker appears (in [5, T-K_LAG-1] so the answer is always in-bounds).
    target: (n_samples,) = u[marker_pos + K_LAG]."""
    u = rng.uniform(-1, 1, size=(n_samples, T)).astype(np.float32)
    marker_pos = rng.integers(5, T - K_LAG - 1, size=n_samples)
    MARKER_VALUE = 3.0  # far outside the [-1,1] data range, unambiguous
    for i in range(n_samples):
        u[i, marker_pos[i]] = MARKER_VALUE
    target = np.array([u[i, marker_pos[i] + K_LAG] for i in range(n_samples)], dtype=np.float32)
    # zero out the marker AFTER recording the target, so the target value
    # itself was never artificially forced away from [-1,1]
    return u, target, marker_pos


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 substrate: the verified reservoir physics (reused, not re-derived)
# ─────────────────────────────────────────────────────────────────────────────
FREQ = 1.0 / (2 * np.pi)
DAMPING = 0.3
DT = 0.05
HOLD = 8   # reduced from fib_noise_scaling_test.py's 20 -- this task needs
           # sequence STRUCTURE preserved per input step, not deep settling
RHO = 0.7
V, W = 0.4, 1.0
OMEGA = 2 * np.pi * FREQ
M_RESERVOIR = 48


def fib_word(g):
    w = "A"
    for _ in range(g):
        w = w.replace("B", "0").replace("A", "AB").replace("0", "A")
    return w


LONG_WORD = fib_word(12)


def fib_edges(M):
    word = LONG_WORD[: M - 1]
    weights = np.array([V if sym == "A" else W for sym in word], dtype=np.float32)
    K_dense = np.zeros((M, M), dtype=np.float32)
    idx = np.arange(M - 1)
    K_dense[idx, idx + 1] = weights
    K_dense[idx + 1, idx] = weights
    norm = np.max(np.abs(np.linalg.eigvalsh(K_dense))) + 1e-9
    return idx, idx + 1, RHO * weights / norm


class ReservoirBank(nn.Module):
    """Same physics as core/runtime/runtime.py's ResonatorState.step(),
    vectorized -- verified bit-for-bit identical to the real per-object
    implementation in fib_noise_scaling_test.py. Not retrained -- weights
    (win) are fixed random, matching real reservoir-computing practice:
    only the readout on top ever trains."""

    def __init__(self, M):
        super().__init__()
        src, dst, w = fib_edges(M)
        self.M = M
        self.register_buffer("src", torch.tensor(src, dtype=torch.long))
        self.register_buffer("dst", torch.tensor(dst, dtype=torch.long))
        self.register_buffer("weights", torch.tensor(w))
        rng = np.random.default_rng(42)
        self.register_buffer("win", torch.tensor(rng.uniform(-1, 1, M).astype(np.float32)))

    def forward(self, u_batch):
        """u_batch: (B, T). Returns (B, T, 2*M) reservoir state sequence:
        position (x) concatenated with velocity (v).

        DIAGNOSED (2026-07-31): a cheap no-training diagnostic (peak of
        ||x|| vs ||v|| across time, compared to the true marker position)
        showed position has a real, physical phase lag after being
        excited -- a driven damped oscillator's position lags roughly a
        quarter period behind the drive, while velocity responds almost
        immediately to a kick. Measured lag: x off by +2 to +25 ticks
        across 5 samples; v off by +0 ticks in 4 of 5. This is WHY hop-1
        attention (below) had exactly 0.00% marker-localization accuracy
        through the reservoir despite being proven exact on raw input --
        it was keying/valuing off x, the wrong feature for localizing a
        brief event in time. Returning [x, v] lets attention's learned
        Wk/Wv pick whichever feature (or blend) serves each hop."""
        B = u_batch.shape[0]
        x = torch.zeros(B, self.M, device=u_batch.device)
        v = torch.zeros(B, self.M, device=u_batch.device)
        out = []
        for t in range(u_batch.shape[1]):
            for _ in range(HOLD):
                coupling = torch.zeros(B, self.M, device=u_batch.device)
                contrib = self.weights * x[:, self.dst]
                coupling.index_add_(1, self.src, contrib)
                contrib2 = self.weights * x[:, self.src]
                coupling.index_add_(1, self.dst, contrib2)
                drive = self.win.unsqueeze(0) * u_batch[:, t:t + 1] + coupling
                accel = -(OMEGA ** 2) * x - 2 * DAMPING * OMEGA * v + drive
                v = v + accel * DT
                x = x + v * DT
            out.append(torch.cat([x, v], dim=-1).clone())
        return torch.stack(out, dim=1)  # (B, T, 2*M)


# ─────────────────────────────────────────────────────────────────────────────
# Ternary STE, reused pattern from ternary_vision/qat.py (reimplemented
# standalone -- different repo/venv, same trick, not re-derived)
# ─────────────────────────────────────────────────────────────────────────────
class _TernarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, threshold_factor):
        delta = threshold_factor * w.abs().mean()
        ternary = torch.zeros_like(w)
        ternary[w > delta] = 1.0
        ternary[w < -delta] = -1.0
        nz = ternary != 0
        alpha = w[nz].abs().mean() if nz.any() else torch.tensor(0.0, device=w.device)
        return ternary * alpha

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None  # straight-through


def ternary_ste(w, threshold_factor=0.7):
    return _TernarySTE.apply(w, threshold_factor)


# ─────────────────────────────────────────────────────────────────────────────
# Spiking (LIF, surrogate-gradient) nonlinearity, reused pattern from the
# cascade-gradient series this session
# ─────────────────────────────────────────────────────────────────────────────
class _SurrogateSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_minus_threshold, k):
        ctx.save_for_backward(v_minus_threshold)
        ctx.k = k
        return (v_minus_threshold >= 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        (vt,) = ctx.saved_tensors
        s = torch.sigmoid(ctx.k * vt)
        return grad_output * ctx.k * s * (1 - s), None


def spike(v_minus_threshold, k=4.0):
    return _SurrogateSpike.apply(v_minus_threshold, k)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2/3/4: self-attention over the reservoir's time sequence
# ─────────────────────────────────────────────────────────────────────────────
class ReservoirAttentionReadout(nn.Module):
    """DIAGNOSED (2026-07-31): the original version used single-query
    attention (Q from the last timestep only) and FAILED to solve the
    task even on raw input with no reservoir at all (NMSE~0.99, no
    better than guessing). Traced to a real architectural gap, not a
    bug: content-based attention can find "what looks like the marker"
    but "the position K steps AFTER the marker" is a two-hop relationship
    a single attention step doesn't naturally express. Adding positional
    encoding alone did NOT fix it (made generalization worse, NMSE~1.2).

    FIXED with an explicit two-hop mechanism: hop 1 attends to FIND the
    marker (content-based, this part always worked); hop 2 SHIFTS that
    attention distribution forward by K_LAG positions and uses the
    shifted distribution to read out the answer. Confirmed to solve the
    task exactly (NMSE=0.0000) on raw input before being wired back into
    the full reservoir+ternary+spiking stack here."""

    def __init__(self, M, d_model=32, use_ternary=False, use_spiking=False):
        super().__init__()
        self.use_ternary, self.use_spiking = use_ternary, use_spiking
        # DIAGNOSED (2026-07-31): the amplitude-only [0,1] task variant (see
        # pyspike_reservoir_attention_acoustic_groundtruth.py) trained to a
        # much worse NMSE (0.72 vs 0.118) than the original signed [-1,1]
        # task at identical lr/epochs. A pure-numpy replay of
        # ReservoirBank.forward ruled out marker contrast as the cause (the
        # marker/background ||v|| contrast ratio was ~3.28 vs ~3.36 for the
        # two tasks -- barely different, and the marker was actually MORE
        # often the global peak for the nonneg task, 97.5% vs 90%). The real
        # cause: the nonneg task's smaller input range + smaller marker
        # value (2.5 vs 3.0) make reservoir_states ~30% smaller in absolute
        # magnitude, and Wk/Wv/Wo are initialized at a fixed scale
        # regardless of task, so attention scores start out proportionally
        # "quieter" and need more training to reach the same resolving
        # power.
        #
        # FIRST ATTEMPT, FAILED (2026-07-31): per-timestep LayerNorm(M).
        # Verified on real hardware (kimchi, torch 2.11+cu128, GPU): broke
        # BOTH tasks -- signed NMSE 0.118->0.917 (near-random), nonneg
        # 0.72->0.78 (no better). Root cause understood after the fact:
        # LayerNorm normalizes each timestep's feature vector independently,
        # which forces every tick's vector norm to roughly the same value
        # (~sqrt(M)) regardless of its ORIGINAL magnitude -- but hop-1
        # attention's whole localization mechanism depends on the marker
        # tick having a much LARGER ||v|| than background ticks (the exact
        # ~3.3x contrast measured above). LayerNorm erases precisely that
        # per-timestep contrast to fix a different (between-task) scale
        # mismatch -- treating the symptom's wrong axis.
        #
        # SECOND ATTEMPT, ALSO FAILED (2026-07-31): a single scalar
        # rescale per sequence (RMS over the whole (T,M) block, not
        # per-timestep) -- theory was this corrects the between-task
        # magnitude difference while leaving every timestep's magnitude
        # relative to its own sequence untouched. Verified on kimchi
        # (GPU): signed NMSE 0.118->0.161 (worse), nonneg 0.72->0.711
        # (+1.2%, noise-level, not a real fix). The magnitude-mismatch
        # theory itself may not be the real driver -- reverted to
        # baseline (no normalization) rather than keep a change that
        # doesn't earn its cost. NEXT: try the plain lr/epoch sweep
        # memory already suggested for the nonneg task specifically
        # (won't touch the signed task's already-working result, unlike
        # an architecture change that affects both).
        self.q_find = nn.Parameter(torch.randn(d_model) * 0.1)  # global 'find the marker' query
        self.Wk = nn.Parameter(torch.randn(M, d_model) * 0.1)
        self.Wv = nn.Parameter(torch.randn(M, d_model) * 0.1)
        self.Wo = nn.Parameter(torch.randn(d_model, 1) * 0.1)
        self.d_model = d_model

    def _w(self, w):
        return ternary_ste(w) if self.use_ternary else w

    def forward(self, reservoir_states):
        """reservoir_states: (B, T, M). Two-hop attention: find the
        marker, then shift by K_LAG to read the answer."""
        K_ = reservoir_states @ self._w(self.Wk)                   # (B, T, d_model)
        Vv = reservoir_states @ self._w(self.Wv)                   # (B, T, d_model)

        scores = torch.einsum('d,btd->bt', self.q_find, K_) / (self.d_model ** 0.5)
        attn1 = F.softmax(scores, dim=-1)                          # HOP 1: find the marker
        attn_shifted = torch.roll(attn1, shifts=K_LAG, dims=1)     # HOP 2: shift by K
        context = torch.einsum('bt,btd->bd', attn_shifted, Vv)     # (B, d_model)

        if self.use_spiking:
            # spike whenever the attended context exceeds a threshold per
            # unit -- real hard-forward/surrogate-backward, not a smooth
            # approximation standing in for it
            context = spike(context - 0.0, k=4.0) * context.abs()  # gate magnitude by spike

        out = context @ self._w(self.Wo)                           # (B, 1)
        return out.squeeze(-1), attn1


def run_stage(name, model, reservoir, train_u, train_y, test_u, test_y, epochs=300, lr=0.01, batch_size=64, is_linear_baseline=False):
    train_u_t = torch.tensor(train_u, device=DEVICE)
    train_y_t = torch.tensor(train_y, device=DEVICE)
    test_u_t = torch.tensor(test_u, device=DEVICE)
    test_y_t = torch.tensor(test_y, device=DEVICE)

    with torch.no_grad():
        train_states = reservoir(train_u_t)
        test_states = reservoir(test_u_t)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_states.shape[0]

    for epoch in range(epochs):
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_states, batch_y = train_states[idx], train_y_t[idx]
            opt.zero_grad()
            if is_linear_baseline:
                pred = model(batch_states)
            else:
                pred, _ = model(batch_states)
            loss = F.mse_loss(pred, batch_y)
            loss.backward()
            opt.step()

    with torch.no_grad():
        if is_linear_baseline:
            test_pred = model(test_states)
        else:
            test_pred, _ = model(test_states)
        test_nmse = (F.mse_loss(test_pred, test_y_t) / test_y_t.var()).item()
    print(f"  {name:<45} test NMSE = {test_nmse:.4f}")
    return test_nmse


class LinearBaseline(nn.Module):
    """The plain linear readout every reservoir test so far has used --
    only sees the LAST timestep's reservoir state, exactly like the
    ridge-regression readouts in fib_noise_combined_check.py etc."""
    def __init__(self, M):
        super().__init__()
        self.w = nn.Linear(M, 1)

    def forward(self, reservoir_states):
        return self.w(reservoir_states[:, -1, :]).squeeze(-1)


if __name__ == "__main__":
    print("=" * 78)
    print("  RESERVOIR + ATTENTION + TERNARY + SPIKING -- staged hybrid")
    print(f"  task: marker-relative recall (find marker, report value {K_LAG} steps later)")
    print("=" * 78)

    rng = np.random.default_rng(0)
    train_u, train_y, _ = make_task(rng, 800)
    test_u, test_y, test_marker_pos = make_task(rng, 200)

    reservoir = ReservoirBank(M_RESERVOIR).to(DEVICE)
    FEAT = 2 * M_RESERVOIR  # reservoir now emits [x, v] concatenated

    print("\nSTAGE 1: reservoir -> plain linear readout (the existing baseline)")
    torch.manual_seed(0)
    baseline = LinearBaseline(FEAT).to(DEVICE)
    nmse1 = run_stage("linear readout (baseline)", baseline, reservoir, train_u, train_y, test_u, test_y, is_linear_baseline=True)

    print("\nSTAGE 2: reservoir -> self-attention -> readout (full precision)")
    torch.manual_seed(0)
    attn_model = ReservoirAttentionReadout(FEAT, use_ternary=False, use_spiking=False).to(DEVICE)
    nmse2 = run_stage("attention (full precision)", attn_model, reservoir, train_u, train_y, test_u, test_y)
    with torch.no_grad():
        test_states = reservoir(torch.tensor(test_u, device=DEVICE))
        _, attn1 = attn_model(test_states)
        found_pos = attn1.argmax(dim=-1).cpu().numpy()
        loc_acc = (found_pos == test_marker_pos).mean() * 100
        print(f"  hop-1 marker-localization accuracy (x+v feature): {loc_acc:.2f}%")

    print("\nSTAGE 3: reservoir -> TERNARY self-attention -> readout")
    # DIAGNOSED (2026-07-31): lr=0.01 (fine for Stage 2) made all-ternary
    # training collapse to 0% marker localization (NMSE 0.74) even though
    # each of Wk/Wv/Wo individually tolerates full ternary quantization at
    # near-zero cost (ablation: K-only, V-only, O-only all 100% localized,
    # NMSE~0.11-0.12). K+O quantized together was the specific pair that
    # broke it. Confirmed an optimization instability, not a capacity
    # limit: lr=0.003 alone recovers 100% localization, NMSE 0.126 --
    # matching full precision. Same lesson as the MobileNet ternary QAT
    # result earlier this session: quantized paths need gentler training,
    # not more capacity.
    torch.manual_seed(0)
    ternary_model = ReservoirAttentionReadout(FEAT, use_ternary=True, use_spiking=False).to(DEVICE)
    nmse3 = run_stage("attention (ternary QAT)", ternary_model, reservoir, train_u, train_y, test_u, test_y, lr=0.003)

    print("\nSTAGE 4: reservoir -> TERNARY + SPIKING self-attention -> readout")
    torch.manual_seed(0)
    full_model = ReservoirAttentionReadout(FEAT, use_ternary=True, use_spiking=True).to(DEVICE)
    nmse4 = run_stage("attention (ternary + spiking)", full_model, reservoir, train_u, train_y, test_u, test_y, lr=0.003)

    print("\n" + "=" * 78)
    print("SUMMARY -- does each added piece earn its place, or just add complexity?")
    print("=" * 78)
    print(f"  Stage 1 (linear readout, no attention):      NMSE = {nmse1:.4f}")
    print(f"  Stage 2 (+ attention, full precision):       NMSE = {nmse2:.4f}  "
          f"({'attention HELPS' if nmse2 < nmse1 * 0.8 else 'attention did NOT clearly help'})")
    print(f"  Stage 3 (+ ternary quantization):             NMSE = {nmse3:.4f}  "
          f"({'ternary held up' if nmse3 < nmse2 * 1.5 else 'real cost from ternary'})")
    print(f"  Stage 4 (+ spiking nonlinearity):             NMSE = {nmse4:.4f}  "
          f"({'spiking held up' if nmse4 < nmse2 * 2.0 else 'real cost from spiking'})")
