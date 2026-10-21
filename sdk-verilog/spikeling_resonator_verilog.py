"""
Spikeling Resonator -> Verilog backend.

  tone_detector.spk (or any .spk with `type=Resonator` neurons)
      -> spikeling_resonators.v   (synthesizable fixed-point hardware)
      -> tb_resonators.v          (self-checking testbench)
      -> drive_samples.hex        (precomputed stimulus, Q14.18 fixed-point)

Each Resonator becomes REAL HARDWARE:
  - two registers (x, v)            the oscillator state
  - a 64-bit energy accumulator     mean-square amplitude (needs the
                                     extra width -- see FIXED-POINT NOTES)
  - three multiply-and-shift ops    the symplectic-Euler update
  - a comparator + edge detector    the "detected" pulse output

All resonators update in PARALLEL on the rising clock edge, same as the
existing LIF module (spikeling_verilog.py) -- driven by ONE shared
`drive` input per cycle (a single ADC sample feeding the whole bank),
matching the Python/C reference models in runtime.py and the C backend.

FIXED-POINT NOTES (read before changing FRAC_BITS or ALPHA_SHIFT):

  State (x, v) and the per-resonator constants (derived from omega^2,
  2*damping*omega, coupling -- each pre-multiplied by dt in Python, NOT
  in hardware, since omega^2 alone is far too large to represent in a
  reasonable fixed-point width) use Q14.18: 32-bit signed, 18 fractional
  bits, 13 integer bits (+/-8191). Verified against real values seen at
  1760Hz (the highest channel): K1 ~= 3732, v amplitude ~= 45 -- both
  comfortably inside range.

  The energy accumulator is a SEPARATE, wider format: Q28.36 in a 64-bit
  register. This isn't optional -- the detection threshold (0.0008
  amplitude) squared is ~6.4e-7, which rounds to ZERO in Q14.18 (whose
  smallest representable step is ~3.8e-6, *larger* than the value it
  needs to hold). Q14.18's x*x product naturally produces a Q28.36
  result before any rescale, so the energy accumulator just keeps that
  full width instead of truncating it back down -- no extra cost, just
  not throwing away precision that's already there for free.

  The leaky-average decay (`energy_ema += alpha * (x*x - energy_ema)` in
  the Python/C reference) is implemented here as a right shift by
  ALPHA_SHIFT instead of a multiply -- a real hardware design wouldn't
  spend a 64-bit multiplier on this when a power-of-two decay rate is
  close enough and free in silicon. ALPHA_SHIFT is DERIVED from DT_SHIFT
  and ENERGY_TIME_CONSTANT (not hand-picked) -- see compute_alpha_shift()
  below. This matters because of a real bug already found and fixed in
  the Python/C backends: a fixed decay rate implicitly bakes in whatever
  dt it was tuned at, and silently produces a far-too-short averaging
  window (and collapsed detection accuracy) if dt changes without
  rescaling it. Hardcoding ALPHA_SHIFT as an independent constant here
  would reintroduce that exact bug the moment someone changes DT_SHIFT
  for a different target frequency without remembering to also update
  ALPHA_SHIFT by hand -- deriving it removes that landmine entirely.

  dt is fixed at a power of two (2^-DT_SHIFT) specifically so the
  v -> x integration step is an EXACT shift, not a lossy fixed-point
  multiply by a non-power-of-two constant.
"""

import math
import os
import re
import sys

FRAC_BITS = 18          # Q14.18 for x, v, and the K1/K2/K3 per-resonator constants
WIDTH = 32               # bits for x, v, K1/K2/K3
ENERGY_FRAC_BITS = 2 * FRAC_BITS   # 36 -- exactly what x*x produces, no shift needed
ENERGY_WIDTH = 64
DT_SHIFT = 15             # dt = 2^-15 ~= 30.5us per tick (~32768 Hz)
ENERGY_TIME_CONSTANT = 0.0025  # seconds -- matches the Python/C backends' default


def compute_alpha_shift(dt_shift: int, time_constant: float) -> int:
    """alpha = dt / time_constant, rounded to the nearest power-of-two
    shift. Recomputed from dt_shift every time instead of hardcoded, so
    changing DT_SHIFT (e.g. for a different target sample rate) can't
    silently leave a stale, wrong decay rate behind -- see the module
    docstring's fixed-point notes for why that specific failure mode is
    a known, previously-hit bug, not a hypothetical one."""
    dt = 2.0 ** -dt_shift
    alpha = dt / time_constant
    return max(1, round(-math.log2(alpha)))

# Must match runtime.DEFAULT_RESONATOR_BASE_GAIN / CCodeGenerator's copy --
# kept duplicated for the same reason as the C backend: this generator is
# meant to be usable standalone (just this one file + the .spk).
DEFAULT_RESONATOR_BASE_GAIN = 4.0e-4
DEFAULT_RESONATOR_THRESHOLD = 0.0008
DEFAULT_RESONATOR_GATE_THRESHOLD = 0.00024

RESONATOR_RE = re.compile(
    r"neuron\s+(\w+)\s+type=Resonator\s+freq=([\d.]+)\s+damping=([\d.]+)(?:\s+coupling=([\d.]+))?"
)
ACTION_RE = re.compile(r"action\s+(\w+)\s*->\s*\[(\w+)\]")


def to_fixed(value: float, frac_bits: int = FRAC_BITS) -> int:
    return int(round(value * (1 << frac_bits)))


def to_hex32(value: int) -> str:
    """Two's-complement 32-bit hex, for Verilog 32'h literals."""
    return format(value & 0xFFFFFFFF, "08x")


def to_hex64(value: int) -> str:
    return format(value & 0xFFFFFFFFFFFFFFFF, "016x")


class ResonatorSpec:
    def __init__(self, name, freq_hz, damping, coupling=None):
        self.name = name
        self.freq_hz = freq_hz
        self.damping = damping
        omega = 2 * math.pi * freq_hz
        self.coupling = coupling if coupling is not None else DEFAULT_RESONATOR_BASE_GAIN * (omega ** 2)
        self.threshold = DEFAULT_RESONATOR_THRESHOLD

        dt = 2.0 ** -DT_SHIFT
        self.k1 = (omega ** 2) * dt                 # multiplies x
        self.k2 = (2 * damping * omega) * dt         # multiplies v
        self.k3 = self.coupling * dt                 # multiplies drive
        self.thresh2 = self.threshold ** 2


def parse_resonators(spk_path: str):
    with open(spk_path, encoding="utf-8") as f:
        text = f.read()
    resonators = []
    for m in RESONATOR_RE.finditer(text):
        name, freq, damping, coupling = m.groups()
        resonators.append(ResonatorSpec(
            name, float(freq), float(damping),
            float(coupling) if coupling else None,
        ))
    actions = {n: c for n, c in ACTION_RE.findall(text)}
    return resonators, actions


def generate_module(resonators, output_path: str):
    N = len(resonators)
    L = []
    w = L.append

    w("// Auto-generated by spikeling_resonator_verilog.py -- DO NOT EDIT")
    w("// Synthesizable fixed-point Resonator bank. See module docstring in")
    w("// spikeling_resonator_verilog.py for the fixed-point format notes.")
    w("")
    w("module spikeling_resonators #(")
    w(f"    parameter WIDTH = {WIDTH},")
    w(f"    parameter FRAC_BITS = {FRAC_BITS},")
    w(f"    parameter ENERGY_WIDTH = {ENERGY_WIDTH}")
    w(") (")
    w("    input  wire                   clk,")
    w("    input  wire                   rst,")
    w("    input  wire signed [WIDTH-1:0] drive,   // one shared sample feeds the whole bank")
    w(f"    output reg  [{N - 1}:0]         detected")
    w(");")
    w("")
    w(f"    localparam integer N = {N};")
    w(f"    localparam integer DT_SHIFT = {DT_SHIFT};")
    alpha_shift = compute_alpha_shift(DT_SHIFT, ENERGY_TIME_CONSTANT)
    w(f"    localparam integer ALPHA_SHIFT = {alpha_shift};  // derived: dt=2^-{DT_SHIFT}, "
      f"time_constant={ENERGY_TIME_CONSTANT*1000:.2f}ms -> alpha~=1/{2**alpha_shift}")
    w(f"    localparam signed [WIDTH-1:0] GATE_THRESH = 32'sh{to_hex32(to_fixed(DEFAULT_RESONATOR_GATE_THRESHOLD))};")
    w("")

    w("    // per-resonator constants, pre-scaled by dt at compile time in Python")
    w("    // (see module docstring for why -- omega^2 alone doesn't fit)")
    for i, r in enumerate(resonators):
        w(f"    localparam signed [WIDTH-1:0] K1_{i} = 32'sh{to_hex32(to_fixed(r.k1))}; // {r.name} omega^2*dt")
        w(f"    localparam signed [WIDTH-1:0] K2_{i} = 32'sh{to_hex32(to_fixed(r.k2))}; // {r.name} 2*damping*omega*dt")
        w(f"    localparam signed [WIDTH-1:0] K3_{i} = 32'sh{to_hex32(to_fixed(r.k3))}; // {r.name} coupling*dt")
        w(f"    localparam signed [ENERGY_WIDTH-1:0] THRESH2_{i} = 64'sh{to_hex64(to_fixed(r.thresh2, ENERGY_FRAC_BITS))}; // {r.name} threshold^2")
    w("")

    w("    function signed [WIDTH-1:0] k1_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: k1_of = K1_{i};")
    w("            default: k1_of = 0;")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    function signed [WIDTH-1:0] k2_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: k2_of = K2_{i};")
    w("            default: k2_of = 0;")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    function signed [WIDTH-1:0] k3_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: k3_of = K3_{i};")
    w("            default: k3_of = 0;")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    function signed [ENERGY_WIDTH-1:0] thresh2_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: thresh2_of = THRESH2_{i};")
    w("            default: thresh2_of = {ENERGY_WIDTH{1'b1}};")  # unreachable threshold
    w("        endcase")
    w("    end endfunction")
    w("")

    w("    // -- state registers: THESE ARE THE RESONATORS --")
    w("    reg signed [WIDTH-1:0]        x          [0:N-1];")
    w("    reg signed [WIDTH-1:0]        v          [0:N-1];")
    w("    reg signed [ENERGY_WIDTH-1:0] energy_ema [0:N-1];")
    w("")
    w("    integer i;")
    w("")
    w("    always @(posedge clk) begin")
    w("        if (rst) begin")
    w("            for (i = 0; i < N; i = i + 1) begin")
    w("                x[i]          <= 0;")
    w("                v[i]          <= 0;")
    w("                energy_ema[i] <= 0;")
    w("                detected[i]   <= 1'b0;")
    w("            end")
    w("        end else begin")
    w("            for (i = 0; i < N; i = i + 1) begin: step")
    w("                reg signed [2*WIDTH-1:0] k1x_full, k2v_full, k3drive_full;")
    w("                reg signed [WIDTH-1:0]   k1x, k2v, k3drive, v_new, x_new, gated_x;")
    w("                reg signed [ENERGY_WIDTH-1:0] x2_full, energy_new;")
    w("                reg gate_active, was_above, now_above;")
    w("")
    w("                // accel*dt = -K1*x - K2*v + K3*drive  (each term: Q14.18 * Q14.18 >>> FRAC_BITS)")
    w("                k1x_full = $signed(k1_of(i)) * $signed(x[i]);")
    w("                k2v_full = $signed(k2_of(i)) * $signed(v[i]);")
    w("                k3drive_full = $signed(k3_of(i)) * $signed(drive);")
    w("                k1x     = k1x_full     >>> FRAC_BITS;")
    w("                k2v     = k2v_full     >>> FRAC_BITS;")
    w("                k3drive = k3drive_full >>> FRAC_BITS;")
    w("")
    w("                v_new = v[i] - k1x - k2v + k3drive;")
    w("                x_new = x[i] + (v_new >>> DT_SHIFT);   // exact: dt = 2^-DT_SHIFT")
    w("")
    w("                // AMPLITUDE GATING (operand isolation): the mechanical")
    w("                // state (x, v) always has to integrate -- a real signal")
    w("                // could arrive on any cycle. But when |x_new| is below the")
    w("                // noise floor, force the energy multiplier's OPERANDS to")
    w("                // zero (gated_x) instead of just ignoring its result --")
    w("                // this is the part that actually matters in real silicon:")
    w("                // a multiplier fed unchanging zero inputs has near-zero")
    w("                // internal switching activity, which is where dynamic")
    w("                // power actually goes (NOT instruction/cycle count -- this")
    w("                // measured as a WASH in the Python/C benchmarks, see")
    w("                // benchmarks/BENCHMARKS.md entry #6; the payoff here is")
    w("                // power, a property simulation timing doesn't capture).")
    w("                gate_active = (x_new >= GATE_THRESH) || (x_new <= -GATE_THRESH);")
    w("                gated_x = gate_active ? x_new : {WIDTH{1'b0}};")
    w("")
    w("                // energy: leaky average of x^2, kept at full Q28.36 width --")
    w("                // see module docstring for why this can't be rescaled to Q14.18")
    w("                x2_full = $signed(gated_x) * $signed(gated_x);")
    w("                was_above = energy_ema[i] >= thresh2_of(i);")
    w("                if (gate_active)")
    w("                    energy_new = energy_ema[i] + ((x2_full - energy_ema[i]) >>> ALPHA_SHIFT);")
    w("                else")
    w("                    energy_new = energy_ema[i] - (energy_ema[i] >>> ALPHA_SHIFT);  // decay only, multiplier output unused")
    w("                now_above = energy_new >= thresh2_of(i);")
    w("")
    w("                x[i]          <= x_new;")
    w("                v[i]          <= v_new;")
    w("                energy_ema[i] <= energy_new;")
    w("                detected[i]   <= now_above && !was_above;")
    w("            end")
    w("        end")
    w("    end")
    w("")
    w("endmodule")
    w("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def generate_coupled_module(resonators, output_path: str, couple_v: float, couple_w: float):
    """Same Resonator bank as generate_module(), PLUS nearest-neighbor
    inter-resonator coupling wired as a chain (resonator i coupled to
    i-1 and i+1, declaration order = chain order).

    This is the feature flagged as a real, unbuilt opportunity: no
    synapse/connection weight has ever been synthesized to hardware in
    this backend before -- resonators only ever shared one drive input.
    Built the GENERATION-EFFICIENT way on purpose: instead of one
    stored weight per bond (N-1 constants for N resonators -- the same
    O(N) per-neuron pattern K1/K2/K3/THRESH2 already use below, and the
    O(N^2) trap a general connectivity matrix would fall into if this
    ever grows past a chain), bond strength is SSH-dimerized: it
    alternates between exactly TWO shared physical constants
    (COUPLE_V, COUPLE_W) by bond parity, selected with a 1-bit compare
    instead of a case-statement LUT. Same generation-efficiency idea
    measured in software earlier tonight (topological-phononics/
    fibonacci_*.py, debruijn_quasicrystal_reservoir.py): a few
    generation parameters standing in for what would otherwise be a
    stored array, at a real, bounded, and separately-measured cost --
    NOT assumed free here either, see test_chain_coupling.py for the
    fixed-point-vs-float numeric cross-check.
    """
    N = len(resonators)
    dt = 2.0 ** -DT_SHIFT
    L = []
    w = L.append

    w("// Auto-generated by spikeling_resonator_verilog.py -- DO NOT EDIT")
    w("// Synthesizable fixed-point Resonator CHAIN with SSH-dimerized nearest-")
    w("// neighbor coupling. See generate_coupled_module()'s docstring in")
    w("// spikeling_resonator_verilog.py for why this is 2 shared constants")
    w("// instead of a stored per-bond weight array.")
    w("")
    w("module spikeling_resonators #(")
    w(f"    parameter WIDTH = {WIDTH},")
    w(f"    parameter FRAC_BITS = {FRAC_BITS},")
    w(f"    parameter ENERGY_WIDTH = {ENERGY_WIDTH}")
    w(") (")
    w("    input  wire                   clk,")
    w("    input  wire                   rst,")
    w("    input  wire signed [WIDTH-1:0] drive,   // one shared sample feeds the whole bank")
    w(f"    output reg  [{N - 1}:0]         detected")
    w(");")
    w("")
    w(f"    localparam integer N = {N};")
    w(f"    localparam integer DT_SHIFT = {DT_SHIFT};")
    alpha_shift = compute_alpha_shift(DT_SHIFT, ENERGY_TIME_CONSTANT)
    w(f"    localparam integer ALPHA_SHIFT = {alpha_shift};  // derived: dt=2^-{DT_SHIFT}, "
      f"time_constant={ENERGY_TIME_CONSTANT*1000:.2f}ms -> alpha~=1/{2**alpha_shift}")
    w(f"    localparam signed [WIDTH-1:0] GATE_THRESH = 32'sh{to_hex32(to_fixed(DEFAULT_RESONATOR_GATE_THRESHOLD))};")
    w("")
    w(f"    // CHAIN COUPLING -- {N - 1} bonds specified by exactly 2 constants, not {N - 1}.")
    w(f"    // If this were stored per-bond the way K1/K2/K3 are stored per-neuron below, it")
    w(f"    // would need {N - 1} localparams + an O(N) case-statement LUT (see k1_of for what")
    w(f"    // that pattern looks like). SSH dimerization needs only these two, always, at any N.")
    w(f"    localparam signed [WIDTH-1:0] COUPLE_V = 32'sh{to_hex32(to_fixed(couple_v * dt))}; // even bonds")
    w(f"    localparam signed [WIDTH-1:0] COUPLE_W = 32'sh{to_hex32(to_fixed(couple_w * dt))}; // odd bonds")
    w("")

    w("    // per-resonator constants, pre-scaled by dt at compile time in Python")
    w("    // (see module docstring for why -- omega^2 alone doesn't fit)")
    for i, r in enumerate(resonators):
        w(f"    localparam signed [WIDTH-1:0] K1_{i} = 32'sh{to_hex32(to_fixed(r.k1))}; // {r.name} omega^2*dt")
        w(f"    localparam signed [WIDTH-1:0] K2_{i} = 32'sh{to_hex32(to_fixed(r.k2))}; // {r.name} 2*damping*omega*dt")
        w(f"    localparam signed [WIDTH-1:0] K3_{i} = 32'sh{to_hex32(to_fixed(r.k3))}; // {r.name} coupling*dt")
        w(f"    localparam signed [ENERGY_WIDTH-1:0] THRESH2_{i} = 64'sh{to_hex64(to_fixed(r.thresh2, ENERGY_FRAC_BITS))}; // {r.name} threshold^2")
    w("")

    w("    function signed [WIDTH-1:0] k1_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: k1_of = K1_{i};")
    w("            default: k1_of = 0;")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    function signed [WIDTH-1:0] k2_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: k2_of = K2_{i};")
    w("            default: k2_of = 0;")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    function signed [WIDTH-1:0] k3_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: k3_of = K3_{i};")
    w("            default: k3_of = 0;")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    function signed [ENERGY_WIDTH-1:0] thresh2_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: thresh2_of = THRESH2_{i};")
    w("            default: thresh2_of = {ENERGY_WIDTH{1'b1}};")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    // bond parity select -- 1-bit, NOT a case-statement LUT: bond j (between")
    w("    // resonator j and j+1) uses COUPLE_V if j is even, COUPLE_W if j is odd.")
    w("    function signed [WIDTH-1:0] bond_of; input integer j; begin")
    w("        bond_of = (j[0] == 1'b0) ? COUPLE_V : COUPLE_W;")
    w("    end endfunction")
    w("")

    w("    // -- state registers: THESE ARE THE RESONATORS --")
    w("    reg signed [WIDTH-1:0]        x          [0:N-1];")
    w("    reg signed [WIDTH-1:0]        v          [0:N-1];")
    w("    reg signed [ENERGY_WIDTH-1:0] energy_ema [0:N-1];")
    w("")
    w("    integer i;")
    w("")
    w("    always @(posedge clk) begin")
    w("        if (rst) begin")
    w("            for (i = 0; i < N; i = i + 1) begin")
    w("                x[i]          <= 0;")
    w("                v[i]          <= 0;")
    w("                energy_ema[i] <= 0;")
    w("                detected[i]   <= 1'b0;")
    w("            end")
    w("        end else begin")
    w("            for (i = 0; i < N; i = i + 1) begin: step")
    w("                reg signed [2*WIDTH-1:0] k1x_full, k2v_full, k3drive_full;")
    w("                reg signed [2*WIDTH-1:0] cleft_full, cright_full;")
    w("                reg signed [WIDTH-1:0]   k1x, k2v, k3drive, cleft, cright, v_new, x_new, gated_x;")
    w("                reg signed [ENERGY_WIDTH-1:0] x2_full, energy_new;")
    w("                reg gate_active, was_above, now_above;")
    w("")
    w("                k1x_full = $signed(k1_of(i)) * $signed(x[i]);")
    w("                k2v_full = $signed(k2_of(i)) * $signed(v[i]);")
    w("                k3drive_full = $signed(k3_of(i)) * $signed(drive);")
    w("                k1x     = k1x_full     >>> FRAC_BITS;")
    w("                k2v     = k2v_full     >>> FRAC_BITS;")
    w("                k3drive = k3drive_full >>> FRAC_BITS;")
    w("")
    w("                // nearest-neighbor coupling: bond (i-1) to the left, bond i to the right")
    w("                cleft_full  = (i > 0)     ? ($signed(bond_of(i-1)) * $signed(x[i-1])) : 0;")
    w("                cright_full = (i < N - 1) ? ($signed(bond_of(i))   * $signed(x[i+1])) : 0;")
    w("                cleft  = cleft_full  >>> FRAC_BITS;")
    w("                cright = cright_full >>> FRAC_BITS;")
    w("")
    w("                v_new = v[i] - k1x - k2v + k3drive + cleft + cright;")
    w("                x_new = x[i] + (v_new >>> DT_SHIFT);")
    w("")
    w("                gate_active = (x_new >= GATE_THRESH) || (x_new <= -GATE_THRESH);")
    w("                gated_x = gate_active ? x_new : {WIDTH{1'b0}};")
    w("")
    w("                x2_full = $signed(gated_x) * $signed(gated_x);")
    w("                was_above = energy_ema[i] >= thresh2_of(i);")
    w("                if (gate_active)")
    w("                    energy_new = energy_ema[i] + ((x2_full - energy_ema[i]) >>> ALPHA_SHIFT);")
    w("                else")
    w("                    energy_new = energy_ema[i] - (energy_ema[i] >>> ALPHA_SHIFT);")
    w("                now_above = energy_new >= thresh2_of(i);")
    w("")
    w("                x[i]          <= x_new;")
    w("                v[i]          <= v_new;")
    w("                energy_ema[i] <= energy_new;")
    w("                detected[i]   <= now_above && !was_above;")
    w("            end")
    w("        end")
    w("    end")
    w("")
    w("endmodule")
    w("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def generate_fibonacci_coupled_module(resonators, output_path: str, bond_values):
    """Same Resonator bank + nearest-neighbor chain coupling as
    generate_coupled_module(), but for a bond sequence that ISN'T reducible
    to 2 constants by parity -- e.g. the Fibonacci substitution word used by
    device idea #17's embedded-reservoir-sensor-processor build (see vault
    PROJECT_embedded_reservoir_sensor_processor.md), where bond i alternates
    quasi-periodically (V,W,V,V,W,V,W,... for the real word), not simply
    V,W,V,W,... like the SSH-dimerized chain generate_coupled_module()
    targets. bond_of(j) = (j even) ? V : W is WRONG for this bond sequence --
    confirmed by construction, not assumed (Fibonacci words are provably
    non-periodic).

    `bond_values` is the REAL, already-computed len(resonators)-1 array of
    per-bond coupling strengths (e.g. extracted directly from
    fib_noise_combined_check_M128.py's fib_K(M) off-diagonal, RHO-normalized
    exactly like the software keystone test that already passed at M=96/128)
    -- this function does not re-derive the Fibonacci word or its bond
    values, it only synthesizes hardware for values computed elsewhere.

    HONEST COST, stated up front: unlike the 2-constant SSH case, this
    needs one case-statement LUT entry per bond (N-1 entries) -- the same
    O(N) pattern k1_of/k2_of/k3_of already use for per-neuron constants
    below. That's still O(N) generated constants from bond_values (itself
    produced by a 3-parameter host-side rule: V, W, Fibonacci generation
    depth), NOT an O(N^2) stored connectivity matrix -- the actual
    generation-efficiency claim being carried into hardware is against the
    O(N^2) alternative, not against O(1)-per-bond, which only the strictly
    periodic SSH case can honestly claim.
    """
    N = len(resonators)
    assert len(bond_values) == N - 1, f"need {N-1} bond values for {N} resonators, got {len(bond_values)}"
    dt = 2.0 ** -DT_SHIFT
    L = []
    w = L.append

    w("// Auto-generated by spikeling_resonator_verilog.py -- DO NOT EDIT")
    w("// Synthesizable fixed-point Resonator CHAIN with FIBONACCI-WORD nearest-")
    w("// neighbor coupling (device idea #17 stage 4). See")
    w("// generate_fibonacci_coupled_module()'s docstring for why this needs a")
    w("// per-bond LUT (N-1 entries) rather than generate_coupled_module()'s")
    w("// 2-constant SSH parity select -- Fibonacci words are non-periodic.")
    w("")
    w("module spikeling_resonators #(")
    w(f"    parameter WIDTH = {WIDTH},")
    w(f"    parameter FRAC_BITS = {FRAC_BITS},")
    w(f"    parameter ENERGY_WIDTH = {ENERGY_WIDTH}")
    w(") (")
    w("    input  wire                   clk,")
    w("    input  wire                   rst,")
    w("    input  wire signed [WIDTH-1:0] drive,   // one shared sample feeds the whole bank")
    w(f"    output reg  [{N - 1}:0]         detected")
    w(");")
    w("")
    w(f"    localparam integer N = {N};")
    w(f"    localparam integer DT_SHIFT = {DT_SHIFT};")
    alpha_shift = compute_alpha_shift(DT_SHIFT, ENERGY_TIME_CONSTANT)
    w(f"    localparam integer ALPHA_SHIFT = {alpha_shift};  // derived: dt=2^-{DT_SHIFT}, "
      f"time_constant={ENERGY_TIME_CONSTANT*1000:.2f}ms -> alpha~=1/{2**alpha_shift}")
    w(f"    localparam signed [WIDTH-1:0] GATE_THRESH = 32'sh{to_hex32(to_fixed(DEFAULT_RESONATOR_GATE_THRESHOLD))};")
    w("")

    w("    // per-resonator constants, pre-scaled by dt at compile time in Python")
    for i, r in enumerate(resonators):
        w(f"    localparam signed [WIDTH-1:0] K1_{i} = 32'sh{to_hex32(to_fixed(r.k1))}; // {r.name} omega^2*dt")
        w(f"    localparam signed [WIDTH-1:0] K2_{i} = 32'sh{to_hex32(to_fixed(r.k2))}; // {r.name} 2*damping*omega*dt")
        w(f"    localparam signed [WIDTH-1:0] K3_{i} = 32'sh{to_hex32(to_fixed(r.k3))}; // {r.name} coupling*dt")
        w(f"    localparam signed [ENERGY_WIDTH-1:0] THRESH2_{i} = 64'sh{to_hex64(to_fixed(r.thresh2, ENERGY_FRAC_BITS))}; // {r.name} threshold^2")
    w("")

    w("    // FIBONACCI-WORD BOND LUT -- N-1 real values, one per-bond entry")
    w("    // (see docstring: this is O(N) generated constants, not O(N^2) stored,")
    w("    // but unlike SSH it cannot collapse to 2 constants -- the word is non-periodic).")
    for j, bv in enumerate(bond_values):
        w(f"    localparam signed [WIDTH-1:0] BOND_{j} = 32'sh{to_hex32(to_fixed(bv * dt))};")
    w("")

    for fn, prefix in (("k1_of", "K1"), ("k2_of", "K2"), ("k3_of", "K3")):
        w(f"    function signed [WIDTH-1:0] {fn}; input integer idx; begin")
        w("        case (idx)")
        for i in range(N):
            w(f"            {i}: {fn} = {prefix}_{i};")
        w("            default: " + fn + " = 0;")
        w("        endcase")
        w("    end endfunction")
        w("")

    w("    function signed [ENERGY_WIDTH-1:0] thresh2_of; input integer idx; begin")
    w("        case (idx)")
    for i in range(N):
        w(f"            {i}: thresh2_of = THRESH2_{i};")
    w("            default: thresh2_of = {ENERGY_WIDTH{1'b1}};")
    w("        endcase")
    w("    end endfunction")
    w("")
    w("    // bond parity is NOT used here -- real per-bond LUT instead (see docstring)")
    w("    function signed [WIDTH-1:0] bond_of; input integer j; begin")
    w("        case (j)")
    for j in range(N - 1):
        w(f"            {j}: bond_of = BOND_{j};")
    w("            default: bond_of = 0;")
    w("        endcase")
    w("    end endfunction")
    w("")

    w("    // -- state registers: THESE ARE THE RESONATORS --")
    w("    reg signed [WIDTH-1:0]        x          [0:N-1];")
    w("    reg signed [WIDTH-1:0]        v          [0:N-1];")
    w("    reg signed [ENERGY_WIDTH-1:0] energy_ema [0:N-1];")
    w("")
    w("    integer i;")
    w("")
    w("    always @(posedge clk) begin")
    w("        if (rst) begin")
    w("            for (i = 0; i < N; i = i + 1) begin")
    w("                x[i]          <= 0;")
    w("                v[i]          <= 0;")
    w("                energy_ema[i] <= 0;")
    w("                detected[i]   <= 1'b0;")
    w("            end")
    w("        end else begin")
    w("            for (i = 0; i < N; i = i + 1) begin: step")
    w("                reg signed [2*WIDTH-1:0] k1x_full, k2v_full, k3drive_full;")
    w("                reg signed [2*WIDTH-1:0] cleft_full, cright_full;")
    w("                reg signed [WIDTH-1:0]   k1x, k2v, k3drive, cleft, cright, v_new, x_new, gated_x;")
    w("                reg signed [ENERGY_WIDTH-1:0] x2_full, energy_new;")
    w("                reg gate_active, was_above, now_above;")
    w("")
    w("                k1x_full = $signed(k1_of(i)) * $signed(x[i]);")
    w("                k2v_full = $signed(k2_of(i)) * $signed(v[i]);")
    w("                k3drive_full = $signed(k3_of(i)) * $signed(drive);")
    w("                k1x     = k1x_full     >>> FRAC_BITS;")
    w("                k2v     = k2v_full     >>> FRAC_BITS;")
    w("                k3drive = k3drive_full >>> FRAC_BITS;")
    w("")
    w("                // nearest-neighbor coupling: bond (i-1) to the left, bond i to the right")
    w("                cleft_full  = (i > 0)     ? ($signed(bond_of(i-1)) * $signed(x[i-1])) : 0;")
    w("                cright_full = (i < N - 1) ? ($signed(bond_of(i))   * $signed(x[i+1])) : 0;")
    w("                cleft  = cleft_full  >>> FRAC_BITS;")
    w("                cright = cright_full >>> FRAC_BITS;")
    w("")
    w("                v_new = v[i] - k1x - k2v + k3drive + cleft + cright;")
    w("                x_new = x[i] + (v_new >>> DT_SHIFT);")
    w("")
    w("                gate_active = (x_new >= GATE_THRESH) || (x_new <= -GATE_THRESH);")
    w("                gated_x = gate_active ? x_new : {WIDTH{1'b0}};")
    w("")
    w("                x2_full = $signed(gated_x) * $signed(gated_x);")
    w("                was_above = energy_ema[i] >= thresh2_of(i);")
    w("                if (gate_active)")
    w("                    energy_new = energy_ema[i] + ((x2_full - energy_ema[i]) >>> ALPHA_SHIFT);")
    w("                else")
    w("                    energy_new = energy_ema[i] - (energy_ema[i] >>> ALPHA_SHIFT);")
    w("                now_above = energy_new >= thresh2_of(i);")
    w("")
    w("                x[i]          <= x_new;")
    w("                v[i]          <= v_new;")
    w("                energy_ema[i] <= energy_new;")
    w("                detected[i]   <= now_above && !was_above;")
    w("            end")
    w("        end")
    w("    end")
    w("")
    w("endmodule")
    w("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def generate_stimulus(steps: int, dt: float, components, noise_amp: float, hex_path: str, seed: int = 42):
    """Precompute the drive signal in Python (full float precision) and
    emit it as a $readmemh hex file -- real hardware gets samples from an
    ADC, it doesn't compute sin() itself, so the testbench shouldn't
    either; this also sidesteps Verilog having no standard trig functions."""
    import random
    random.seed(seed)
    lines = []
    for i in range(steps):
        t = i * dt
        val = sum(amp * math.sin(2 * math.pi * f * t) for f, amp in components)
        val += random.uniform(-noise_amp, noise_amp)
        lines.append(to_hex32(to_fixed(val)))
    with open(hex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return steps


def generate_testbench(resonators, actions, output_path: str, steps: int, hex_path: str,
                        expected_actions: set):
    N = len(resonators)
    hex_name = os.path.basename(hex_path)
    names = [r.name for r in resonators]
    action_names = [actions.get(r.name, "") for r in resonators]

    L = []
    w = L.append
    w("// Auto-generated by spikeling_resonator_verilog.py -- DO NOT EDIT")
    w("// Self-checking testbench: replays a precomputed mixed signal through")
    w("// the Resonator bank and verifies detections match expectations --")
    w("// same scenario as core/examples/test_tone_detector.py and .c.")
    w("`timescale 1ns/1ps")
    w("")
    w("module tb;")
    w(f"    localparam N = {N};")
    w(f"    localparam STEPS = {steps};")
    w(f"    localparam WIDTH = {WIDTH};")
    w("")
    w("    reg clk = 0;")
    w("    reg rst = 1;")
    w("    reg signed [WIDTH-1:0] drive_mem [0:STEPS-1];")
    w("    reg signed [WIDTH-1:0] drive;")
    w("    wire [N-1:0] detected;")
    w("")
    w("    spikeling_resonators dut (.clk(clk), .rst(rst), .drive(drive), .detected(detected));")
    w("")
    w("    always #5 clk = ~clk;  // 100 MHz")
    w("")
    w(f'    reg [127:0] names [0:N-1];')
    for i, name in enumerate(names):
        w(f'    initial names[{i}] = "{name}";')
    w("")
    w("    integer fired [0:N-1];")
    w("    integer i, t;")
    w("")
    w("    initial begin")
    w(f'        $readmemh("{hex_name}", drive_mem);')
    w("        for (i = 0; i < N; i = i + 1) fired[i] = 0;")
    w("        @(negedge clk); rst = 0;")
    w("        for (t = 0; t < STEPS; t = t + 1) begin")
    w("            drive = drive_mem[t];")
    w("            @(negedge clk);")
    w("            for (i = 0; i < N; i = i + 1) begin")
    w("                if (detected[i]) begin")
    w('                    $display("[SPIKE] %0s fired at step %0d", names[i], t);')
    w("                    fired[i] = 1;")
    w("                end")
    w("            end")
    w("        end")
    w("")
    w('        $display("");')
    w('        $display("[test] final fired state:");')
    w("        for (i = 0; i < N; i = i + 1) begin")
    w('            $display("  %0s  fired=%0d", names[i], fired[i]);')
    w("        end")
    w("")
    w("        // pass/fail check baked in at generation time from expected_actions")

    expected_idx = [i for i, a in enumerate(action_names) if a in expected_actions]
    unexpected_idx = [i for i, a in enumerate(action_names) if a and a not in expected_actions]

    # NAMED block: an unnamed `begin...end` with a local `integer` declaration is
    # SystemVerilog-only (rejected by iverilog in default/-g2005 mode with "Variable
    # declaration in unnamed block requires SystemVerilog"). Real bug, caught the
    # first time this generator's output was ever run through an actual simulator
    # rather than just read -- pre-existing, not introduced by chain coupling.
    w("        begin : check_block")
    w("            integer ok;")
    w("            ok = 1;")
    for i in expected_idx:
        w(f'            if (!fired[{i}]) begin $display("[test] FAIL -- missing expected detection: %0s", names[{i}]); ok = 0; end')
    for i in unexpected_idx:
        w(f'            if (fired[{i}]) begin $display("[test] FAIL -- unexpected false positive: %0s", names[{i}]); ok = 0; end')
    w('            if (ok) $display("\\n[test] PASS -- Resonator Verilog hardware module correctly detected the expected tones with no false positives.");')
    w("        end")
    w("        $finish;")
    w("    end")
    w("endmodule")
    w("")

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    spk_path = sys.argv[1] if len(sys.argv) > 1 else "tone_detector.spk"
    out_dir = os.path.dirname(os.path.abspath(spk_path)) or "."

    resonators, actions = parse_resonators(spk_path)
    if not resonators:
        print(f"Error: no Resonator neurons found in '{spk_path}'.")
        sys.exit(1)

    module_path = os.path.join(out_dir, "spikeling_resonators.v")
    generate_module(resonators, module_path)
    print(f"Compiled '{spk_path}' -> '{module_path}' ({len(resonators)} resonators)")

    dt = 2.0 ** -DT_SHIFT
    steps = 4000
    hex_path = os.path.join(out_dir, "drive_samples.hex")
    generate_stimulus(steps, dt, [(440, 1.0), (1760, 0.6)], noise_amp=0.1, hex_path=hex_path)
    print(f"Generated stimulus -> '{hex_path}' ({steps} samples, dt={dt:.6e}s)")

    tb_path = os.path.join(out_dir, "tb_resonators.v")
    generate_testbench(resonators, actions, tb_path, steps, hex_path,
                        expected_actions={"TONE_440HZ", "TONE_1760HZ"})
    print(f"Generated testbench -> '{tb_path}'")
