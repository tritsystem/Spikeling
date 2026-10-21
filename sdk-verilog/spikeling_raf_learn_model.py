"""
Bit-accurate reference model for spikeling_raf_learn.v -- the RAF-resonator-bank
ON-CHIP LEARNING RULE.

Origin: Spikeling/research/raf_gradient/ (rungs 1 -> 2e). That study concluded a
resonator bank in the backward path is a substrate-feasible, sign-faithful learning
signal for hard SNNs -- it never matches surrogate BPTT (rung 2e: ~10-15% R^2 short,
because it measures the *true* gradient of a hard net) but on a chip with no backward
pass the comparison is "vs no learning at all". This wires that rule into the
sdk-verilog fixed-point Resonator hardware.

THE RULE (one gradient step = one measurement window of W ticks):
  every tick t in the window:
    ref_j(t)      = sin(2 pi f_j t dt)                 # per-channel dither reference
    param_out_j   = param_j + (A_j * ref_j(t))         # what the app forward-evaluates
    drive(t)      = G * (loss(t) - loss_ema(t))        # DC-removed scalar loss
    <resonator j stepped by the shared drive>          # exact spikeling_resonators.v math
    if t >= SETTLE:  vref_acc_j += v_j(t) * ref_j(t)   # rung-1b velocity-phase sign readout
  at t == W:
    SIGN mode  : param_j -= step * sign(vref_acc_j)              # signSGD, no multiplier
    SCALED mode: param_j -= step * sign(vref_acc_j) * mag(energy_ema_j)
    reset resonators, accumulators, window counter

Fixed-point matches spikeling_resonators.v exactly: Q14.18 for x/v/params/refs,
Q28.36/64-bit for energy, dt = 2^-15, alpha decay = >>ALPHA_SHIFT.
"""
import math

FRAC = 18
WIDTH = 32
DT_SHIFT = 15
ENERGY_WIDTH = 64
ENERGY_TC = 0.0025
RES_BASE_GAIN = 4.0e-4          # matches runtime.DEFAULT_RESONATOR_BASE_GAIN
DAMPING = 0.02

ONE = 1 << FRAC
IMAX = (1 << (WIDTH - 1)) - 1
IMIN = -(1 << (WIDTH - 1))


def q(x):                       # float -> Q14.18 int, saturating
    v = int(round(x * ONE))
    return max(IMIN, min(IMAX, v))


def f2(x):                      # Q14.18 int -> float
    return x / ONE


def sat(v):
    return max(IMIN, min(IMAX, v))


def alpha_shift():
    dt = 2.0 ** -DT_SHIFT
    return max(1, round(-math.log2(dt / ENERGY_TC)))


ALPHA_SHIFT = alpha_shift()
DT = 2.0 ** -DT_SHIFT


def res_consts(freq_hz):
    """(K1,K2,K3) = (omega^2*dt, 2*damping*omega*dt, coupling*dt) in Q14.18 ints.
    coupling = RES_BASE_GAIN * omega^2 (the omega^2 gain-equalisation from runtime.py)."""
    omega = 2 * math.pi * freq_hz
    coupling = RES_BASE_GAIN * omega ** 2
    return q(omega * omega * DT), q(2 * DAMPING * omega * DT), q(coupling * DT)


SINE_LUT_BITS = 8               # 256-entry full-wave sine LUT, Q14.18
SINE_LUT = [q(math.sin(2 * math.pi * i / (1 << SINE_LUT_BITS))) for i in range(1 << SINE_LUT_BITS)]
PHASE_BITS = 28                 # DDS phase accumulator width -- MUST be wide enough that
                               # round(f*dt*2^PHASE_BITS) has sub-Hz error at the lowest
                               # channel, else the dither runs off the resonator's tuned
                               # frequency and the velocity-phase sign flips.
PHASE_MASK = (1 << PHASE_BITS) - 1
PHASE_TO_IDX = PHASE_BITS - SINE_LUT_BITS


class RAFLearnCore:
    """Bit-accurate. N learnable scalar params, one resonator + dither channel each."""

    def __init__(self, freqs, amps, params0, window=4096, settle=2048,
                 loss_alpha_shift=7, drive_gain_shift=4, step_shift=14, mode="SIGN",
                 scaled_shift=16):
        self.N = len(freqs)
        self.K = [res_consts(f) for f in freqs]
        self.A = [q(a) for a in amps]
        self.param = [q(p) for p in params0]
        self.phase_inc = [int(round(f * DT * (1 << PHASE_BITS))) for f in freqs]
        self.W, self.SETTLE = window, settle
        self.LA, self.DG, self.SS = loss_alpha_shift, drive_gain_shift, step_shift
        self.mode = mode
        self.scaled_shift = scaled_shift
        self._reset_window(hard=True)

    def _reset_window(self, hard=False):
        self.x = [0] * self.N
        self.v = [0] * self.N
        self.energy = [0] * self.N
        self.vref_acc = [0] * self.N
        self.win_ctr = 0
        if hard:
            self.phase = [0] * self.N
            self.loss_ema = None

    def ref(self, j):
        return SINE_LUT[(self.phase[j] >> PHASE_TO_IDX) & ((1 << SINE_LUT_BITS) - 1)]

    def param_out(self):
        """Q14.18 dithered params the application should forward-evaluate THIS tick."""
        out = []
        for j in range(self.N):
            d = (self.A[j] * self.ref(j)) >> FRAC
            out.append(sat(self.param[j] + d))
        return out

    def tick(self, loss_q):
        """Advance one clock. loss_q = Q14.18 scalar loss of param_out() from the app.
        Returns True on the tick a weight update is applied."""
        if self.loss_ema is None:
            self.loss_ema = loss_q
        # DC removal + gain  (all shifts, no multiply)
        err = loss_q - self.loss_ema
        self.loss_ema += err >> self.LA
        drive = sat(err << self.DG)

        updated = False
        for j in range(self.N):
            k1, k2, k3 = self.K[j]
            k1x = (k1 * self.x[j]) >> FRAC
            k2v = (k2 * self.v[j]) >> FRAC
            k3d = (k3 * drive) >> FRAC
            v_new = sat(self.v[j] - k1x - k2v + k3d)
            x_new = sat(self.x[j] + (v_new >> DT_SHIFT))
            # energy: Q28.36, leaky avg of x^2, decay by shift
            x2 = x_new * x_new
            self.energy[j] += (x2 - self.energy[j]) >> ALPHA_SHIFT
            self.x[j], self.v[j] = x_new, v_new
            if self.win_ctr >= self.SETTLE:
                self.vref_acc[j] += (v_new * self.ref(j)) >> FRAC
            self.phase[j] = (self.phase[j] + self.phase_inc[j]) & PHASE_MASK

        self.win_ctr += 1
        if self.win_ctr >= self.W:
            for j in range(self.N):
                s = (self.vref_acc[j] > 0) - (self.vref_acc[j] < 0)
                if self.mode == "SIGN":
                    dstep = s << self.SS                      # signSGD: fixed step, NO multiplier
                else:  # SCALED: vref_acc itself as a signed magnitude, clamped to +/-2*step.
                    # loss-scale dependent -- needs per-app tuning of scaled_shift; SIGN is
                    # the recommended hardware default (rungs 2c R4 / 2d: raf_sign ~= raf_full).
                    cap = 2 << self.SS
                    dstep = max(-cap, min(cap, self.vref_acc[j] >> self.scaled_shift))
                self.param[j] = sat(self.param[j] - dstep)
            self._reset_window()
            updated = True
        return updated


# ── float "intent" model + a training demo ──────────────────────────────
def demo_learn(mode="SIGN", verbose=True):
    """Toy: N scalar knobs, quadratic loss L = sum_j (param_j - target_j)^2.
    Purely to show the fixed-point rule converges on-chip-realistic hardware."""
    targets = [0.35, -0.30, 0.20, -0.15]
    freqs = [311.0, 523.0, 881.0, 1483.0]        # spread, mutually non-harmonic-ish
    amps = [0.05, 0.05, 0.05, 0.05]
    core = RAFLearnCore(freqs, amps, [0.0] * 4, window=4000, settle=2000,
                        step_shift=10, mode=mode)
    hist = []
    n_windows = 200
    win = 0
    while win < n_windows:
        po = [f2(p) for p in core.param_out()]
        loss = sum((po[j] - targets[j]) ** 2 for j in range(4))
        if core.tick(q(loss)):
            cur = [f2(p) for p in core.param]
            err = math.sqrt(sum((cur[j] - targets[j]) ** 2 for j in range(4)))
            hist.append((win, err, cur))
            if verbose and win % 20 == 0:
                print(f"  win {win:3d}  |param-target| = {err:.4f}   param = {[round(c,3) for c in cur]}")
            win += 1
    return hist, targets


if __name__ == "__main__":
    print(f"Q14.18  ALPHA_SHIFT={ALPHA_SHIFT}  DT=2^-{DT_SHIFT}={DT:.2e}  SINE_LUT={1<<SINE_LUT_BITS} entries")
    for mode in ("SIGN", "SCALED"):
        print(f"\n[{mode}] fixed-point RAF learning rule, quadratic toy:")
        hist, tg = demo_learn(mode)
        e0, ef = hist[0][1], hist[-1][1]
        print(f"  target {tg}")
        print(f"  |param-target|: {e0:.4f} -> {ef:.4f}   ({'CONVERGED' if ef < 0.10 and ef < 0.5*e0 else 'did not converge'})")
