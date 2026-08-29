#!/usr/bin/env python
"""
reram_synapse_array.py — a software model of Weebit Nano's ReRAM used as an
analog synapse array, built from their real published specs (not invented).

Real, sourced constants (see the docstring on each constant for the source):
  - Endurance: 100K cycles AEC-Q100-qualified @150C; up to 1M reported.
  - Retention: 10yr @150C / 20yr @125C.
  - Real named programming strategies: single-shot Set (fast, imprecise),
    iterative Set (slower, precise), hybrid (a blend).
  - Real validated few-shot-learning result (Omniglot character recognition):
    20% baseline -> >97% accuracy after 5 updates, <10uJ for a 2kbit array.
  Sources (fetched 2026-08-28):
    https://www.weebit-nano.com/enabling-few-shot-learningai-with-reram/
    https://www.weebit-nano.com/technology/reram-advantages/

DISCLOSED ASSUMPTION, not a Weebit-published number: Weebit's own public
materials do not state a specific conductance-level count or cycle-to-cycle/
device-to-device variability percentage for their array. CONDUCTANCE_LEVELS
and the noise-std constants below are a plausible choice grounded in general
multi-level-cell ReRAM literature, NOT a sourced Weebit spec -- treat any
number derived from them as illustrative, not a claim about the real part.
"""

import random

# ── real, sourced constants ─────────────────────────────────────────────
ENDURANCE_CYCLES_QUALIFIED = 100_000   # AEC-Q100 150C qualified
ENDURANCE_CYCLES_REPORTED_MAX = 1_000_000  # upper bound Weebit reports (10x-100x flash)
RETENTION_YEARS_AT_150C = 10
RETENTION_YEARS_AT_125C = 20

FEWSHOT_BASELINE_ACC = 0.20   # Omniglot random-baseline accuracy, real Weebit demo
FEWSHOT_FINAL_ACC = 0.97      # real Weebit demo result after 5 updates
FEWSHOT_UPDATES = 5
FEWSHOT_ENERGY_UJ_PER_2KBIT = 10.0

# ── disclosed assumptions, NOT sourced from Weebit's public materials ──
CONDUCTANCE_LEVELS = 64        # assumed 6-bit MLC target, not a published Weebit number
SINGLE_SHOT_NOISE_STD = 0.08   # assumed -- "fast, imprecise" per Weebit's own real description
ITERATIVE_NOISE_STD = 0.015    # assumed -- "more precise, but slower" per Weebit's own real description
HYBRID_NOISE_STD = 0.03        # assumed -- "a blend, best balance" per Weebit's own real description

# ── added 2026-08-29, from a real, read-in-full source: "The Reliability
# Issue in ReRam-based CIM Architecture for SNN: A Survey" (Wei-Ting Chen,
# National Taiwan University, arXiv:2412.10389) -- two real, documented
# ReRAM failure mechanisms not previously modeled anywhere in this file.
#
# HONEST SCOPE NOTE: the survey's own worked example (Section 4.1.1) is
# specifically about BIT-SLICED binary ReRAM cells (each cell stores one
# bit; a multi-bit weight is the summed, bit-significance-weighted current
# of several such cells on one bit line) -- a different architecture from
# THIS file's model, where each weight is ONE analog multi-level cell
# (CONDUCTANCE_LEVELS states, no bit-slicing). The off-state-leakage
# mechanism (below) is a per-cell physical property (real HRS resistance
# is never literally infinite) and transfers directly regardless of
# architecture. The "overlapping error" mechanism does NOT transfer
# directly -- there is no bit-line-summing-of-binary-cells happening here
# -- so `crossbar_sum_readout()` below is a disclosed ADAPTATION of the
# survey's general point ("more simultaneously-summed cells -> more
# read-out ambiguity") to this file's own analog per-cell architecture,
# not a reproduction of the paper's specific binary-cell mechanism.
HRS_ON_OFF_RATIO = 100.0       # disclosed assumption -- the survey describes the
                                # on/off-ratio-affects-severity relationship
                                # qualitatively but gives no specific ratio;
                                # 100 is a plausible order-of-magnitude figure
                                # for OxRAM-family devices per the general
                                # ReRAM literature the survey itself cites,
                                # not a sourced Weebit number.
CROSSBAR_SUM_NOISE_PER_CELL = ITERATIVE_NOISE_STD  # disclosed assumption -- reuses
                                # the existing iterative-mode per-cell noise
                                # std as the per-cell current-distribution
                                # spread referenced in the survey's real,
                                # sourced claim that ReRAM resistance follows
                                # a normal/log-normal distribution; the
                                # specific magnitude is not itself sourced.


class ReRAMSynapseArray:
    """A rows x cols array of ReRAM cells, each storing a weight in
    [0, 1] as one of CONDUCTANCE_LEVELS discrete conductance states.

    Real, modeled behaviors:
      - programming noise depends on the REAL named strategy used
        (single_shot / iterative / hybrid), with iterative genuinely more
        precise but requiring more real write cycles (endurance cost) --
        matching Weebit's own real tradeoff description, not just a label.
      - endurance: cells accumulate write-cycle count; approaching/past the
        real 100K-qualified limit, additional write noise is injected (a
        disclosed SIMPLIFICATION of how endurance failure actually looks,
        not a sourced failure curve -- Weebit's public materials don't
        publish one).
      - retention: conductance drifts slowly toward a relaxed baseline the
        longer since last programmed, with a time constant derived from the
        real 10yr@150C spec -- at the timescales this module is actually
        used at (seconds to hours of simulated signal), this drift is
        correctly negligible, matching the real spec (it's a DECADE-scale
        number), not tuned to matter.
    """

    def __init__(self, rows: int, cols: int, seed: int | None = None):
        self.rows = rows
        self.cols = cols
        self._rng = random.Random(seed)
        self.weight = [[0.0] * cols for _ in range(rows)]
        self.cycles = [[0] * cols for _ in range(rows)]
        self.ticks_since_programmed = [[0] * cols for _ in range(rows)]

    def _quantize(self, value: float) -> float:
        value = max(0.0, min(1.0, value))
        step = 1.0 / (CONDUCTANCE_LEVELS - 1)
        return round(value / step) * step

    def _endurance_noise(self, cycles: int) -> float:
        """Extra write noise as cycle count approaches/exceeds the real
        qualified endurance limit -- 0 well below it, growing past it.
        Disclosed simplification: Weebit's own materials don't publish the
        real shape of this curve, so this is a smooth ramp, not a sourced
        failure model."""
        if cycles < ENDURANCE_CYCLES_QUALIFIED:
            return 0.0
        over = (cycles - ENDURANCE_CYCLES_QUALIFIED) / max(
            1, ENDURANCE_CYCLES_REPORTED_MAX - ENDURANCE_CYCLES_QUALIFIED)
        return min(0.5, over * 0.5)

    def program(self, row: int, col: int, target: float, mode: str = "hybrid") -> float:
        """Write `target` (0..1) to a cell using a real named Weebit
        programming strategy. Returns the actual resulting weight
        (post-noise, post-quantization) -- this is the honest "what
        actually got stored," not the ideal target."""
        if mode == "single_shot":
            noise_std = SINGLE_SHOT_NOISE_STD
        elif mode == "iterative":
            noise_std = ITERATIVE_NOISE_STD
        elif mode == "hybrid":
            noise_std = HYBRID_NOISE_STD
        else:
            raise ValueError(f"unknown programming mode: {mode!r}")

        cycles = self.cycles[row][col]
        total_noise_std = noise_std + self._endurance_noise(cycles)
        noisy = target + self._rng.gauss(0.0, total_noise_std)
        actual = self._quantize(noisy)

        self.weight[row][col] = actual
        self.cycles[row][col] += 1
        self.ticks_since_programmed[row][col] = 0
        return actual

    def read(self, row: int, col: int, ticks_elapsed: int = 0) -> float:
        """Read a cell's current weight. `ticks_elapsed` lets a caller
        simulate retention drift over a chosen timescale -- real 10yr@150C
        retention means this should be a no-op at any timescale this
        module is actually used at; the drift term exists for completeness,
        not because it's expected to matter in a same-session demo."""
        self.ticks_since_programmed[row][col] += ticks_elapsed
        age = self.ticks_since_programmed[row][col]
        # time constant chosen so 1 "tick" == roughly 1 real hour at the
        # decade-scale retention spec -- deliberately makes retention loss
        # unreachable within any realistic demo run, matching the real spec.
        retention_tau_ticks = RETENTION_YEARS_AT_150C * 365 * 24
        drift = 1.0 - pow(2.718281828, -age / retention_tau_ticks)
        relaxed_baseline = 0.5
        w = self.weight[row][col]
        w = w * (1.0 - drift * 0.05) + relaxed_baseline * (drift * 0.05)
        # real off-state (HRS) leakage, added 2026-08-29: a real HRS cell's
        # resistance is never literally infinite (Chen survey, Sec 4.1.1 --
        # "the resistance of a ReRAM cell in the off state rarely resets to
        # infinity"), so a cell programmed near the "off" end of its range
        # still contributes a real, nonzero floor current instead of true
        # zero. Floors the returned value at 1/HRS_ON_OFF_RATIO of the full
        # [0,1] range; leaves already-higher values untouched.
        return max(w, 1.0 / HRS_ON_OFF_RATIO)

    def endurance_margin(self, row: int, col: int) -> float:
        """0.0..1.0+ -- fraction of the real qualified endurance limit
        this cell has used. >1.0 means past the 100K-qualified point
        (still within the reported 1M max, but degraded)."""
        return self.cycles[row][col] / ENDURANCE_CYCLES_QUALIFIED

    def sum_currents(self, row: int, col_input_pairs, rng) -> float:
        """Real crossbar-style summed readout for one row, added
        2026-08-29 -- ADAPTED from the Chen survey's real "overlapping
        error" finding (Sec 4.1.1): the more cells contribute current to
        one summed readout simultaneously, the more read-out ambiguity
        accumulates, because each cell's real current is drawn from a
        distribution (the survey's own sourced claim: ReRAM resistance
        follows a normal/log-normal distribution), not a fixed value.

        `col_input_pairs`: iterable of (col, input_value) -- the columns
        contributing to this readout and their input drive. Returns the
        summed current (post-leakage-floor `.read()` value x input,
        summed across columns) PLUS extra noise whose std scales with
        sqrt(n_active) -- the real statistical signature of summing N
        independent noisy contributions, not a flat per-call constant.

        HONEST SCOPE NOTE (see the module docstring for the full
        disclosure): this models the GENERAL "more simultaneous cells ->
        more readout ambiguity" effect the survey describes, not its
        specific binary-bit-line mechanism, which doesn't apply to this
        file's single-analog-cell-per-weight architecture. Opt-in --
        existing hidden-layer code in this folder still reads cells
        individually and sums in plain Python; this method exists for
        code that wants the more realistic crossbar-level behavior."""
        pairs = list(col_input_pairs)
        total = 0.0
        for col, x in pairs:
            total += self.read(row, col) * x
        n_active = sum(1 for _, x in pairs if x != 0)
        if n_active > 0:
            extra_std = CROSSBAR_SUM_NOISE_PER_CELL * (n_active ** 0.5)
            total += rng.gauss(0.0, extra_std)
        return total
