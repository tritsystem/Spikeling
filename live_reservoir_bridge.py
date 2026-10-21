"""Stage 3 of device idea #17's build plan: feed the real Arduino 5x HC-SR04
grid (arduino_spikeling_sensor_grid) into the M=128 Fibonacci-coupled
Resonator reservoir that Stage 1 (fib_noise_combined_check_M128.py, both
P1/P2 MET) and Stage 2 (reservoir_driver.c, exact NMSE parity with Python)
already validated.

HONEST STATUS, stated up front: no Arduino is connected to this machine in
this session (checked via serial.tools.list_ports -- 0 ports found). This
file is software-validated only via --replay, matching the same discipline
already used for junkyard-control-panel/junkyard-reservoir in this portfolio.
Real-hardware confirmation is the next physical step, not done here.

ARCHITECTURE NOTE, stated honestly rather than glossed over: the tested
reservoir (fib_noise_combined_check_M128.py) is a SINGLE-CHANNEL delay-recall
task -- one scalar drive u[t] feeds all M=128 units via per-unit input
weights `win`. It is NOT "5 sensors -> 5 reservoir channels" the way the
original device-idea narrative loosely implied. This bridge collapses the 5
real HC-SR04 readings into ONE scalar drive per tick (reusing the exact
`max(0, 120 - 4*d)` proximity mapping bridge.py already uses, summed across
the 5 zones) before feeding the reservoir -- a deliberate, stated design
choice, not the only possible one. Running 5 independent reservoirs (one per
sensor) is a real alternative, not attempted here.

Uses the REAL ResonatorState class and the REAL fib_K/ridge functions,
imported directly from fib_noise_combined_check_M128.py -- no re-derivation.
"""
import argparse, os, sys, time, importlib.util
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "core"))
from runtime.runtime import ResonatorState

spec = importlib.util.spec_from_file_location(
    "fib_m128", os.path.join(HERE, "research", "fib_noise_combined_check_M128.py"))
fib_m128 = importlib.util.module_from_spec(spec)
_orig_argv = sys.argv
sys.argv = [sys.argv[0]]           # suppress the module's own top-level test run's argv sensitivity
spec.loader.exec_module(fib_m128)  # NOTE: this re-runs the M=128 pre-registered test (verbose, ~17s)
sys.argv = _orig_argv

M, SEED = 128, 0
K = fib_m128.fib_K(M)
Xf, yb = fib_m128.make_XY(K, SEED)
TR = int(0.6 * len(yb))
DECODER = fib_m128.ridge(Xf[:TR], yb[:TR])   # same trained decoder Stage 2 verified in C

rng = np.random.default_rng(SEED)
_ = rng.standard_normal(fib_m128.T)          # burn the u draw to land on the same win as training
WIN = rng.uniform(-1, 1, M)

FREQ, DAMPING, DT, HOLD = fib_m128.FREQ, fib_m128.DAMPING, fib_m128.DT, fib_m128.HOLD

def build_reservoir():
    return [ResonatorState(name=f"r{i}", freq_hz=FREQ, damping=DAMPING, coupling=1.0) for i in range(M)]

def reservoir_tick(res, drive_scalar):
    """One outer tick: HOLD real substeps of the Fibonacci-coupled chain,
    identical update order to fib_noise_combined_check_M128.py's run()."""
    for _ in range(HOLD):
        x = np.array([r.x for r in res])
        drive_vec = WIN * drive_scalar + K @ x
        for i, r in enumerate(res):
            r.step(float(drive_vec[i]), DT)
    xf = np.array([r.x for r in res])
    return np.concatenate([xf, xf ** 2])

def decode(features):
    return float(features @ DECODER)

def sensors_to_scalar_drive(dists):
    """5 HC-SR04 distances (cm) -> one scalar drive, reusing bridge.py's
    proven proximity mapping (max(0, 120 - 4*d)) summed across zones."""
    total = 0.0
    for d in dists:
        if d is not None and d >= 0:
            total += max(0.0, 120.0 - 4.0 * d)
    return total / 120.0  # roughly O(1) scale, matching the synthetic u[t]~N(0,1) training regime

def run_live(port, baud=9600):
    import serial
    print(f"Opening {port}...", flush=True)
    ser = serial.Serial(port, baud, timeout=1)
    print("Connected. Streaming 5-sensor CSV -> Fibonacci reservoir -> decode.", flush=True)
    res = build_reservoir()
    tick = 0
    while True:
        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        try:
            dists = [float(x) for x in line.split(",") if x]
        except ValueError:
            continue
        if len(dists) != 5:
            continue
        drive = sensors_to_scalar_drive(dists)
        feat = reservoir_tick(res, drive)
        y_hat = decode(feat)
        tick += 1
        print(f"tick {tick:5d}  dists={dists}  drive={drive:.3f}  decode={y_hat:+.4f}", flush=True)

def run_replay(n_ticks=40, seed=1):
    """No hardware needed: synthesizes plausible 5-sensor CSV lines (a hand
    slowly approaching then leaving one zone) to validate the parse/drive/
    reservoir/decode path end-to-end without a physical Arduino. HONEST:
    this proves the software path runs, not that it says anything useful
    about real sensor data -- the synthetic distances are a smooth ramp,
    not real ultrasonic noise."""
    rng2 = np.random.default_rng(seed)
    res = build_reservoir()
    print(f"[replay] {n_ticks} synthetic ticks, no serial port opened, no hardware required", flush=True)
    for t in range(n_ticks):
        base = 5.0 + 40.0 * abs(np.sin(t / n_ticks * np.pi))  # hand approaches then leaves
        dists = [float(base + rng2.normal(0, 1.5)) for _ in range(5)]
        drive = sensors_to_scalar_drive(dists)
        feat = reservoir_tick(res, drive)
        y_hat = decode(feat)
        print(f"tick {t:3d}  dists={[f'{d:.1f}' for d in dists]}  drive={drive:.3f}  decode={y_hat:+.4f}", flush=True)
    print("[replay] completed without error -- parse/drive/reservoir/decode path is real and runs.", flush=True)
    print("[replay] NOT a claim about real sensor accuracy -- synthetic input only.", flush=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="COM3")
    ap.add_argument("--replay", action="store_true", help="run without real hardware (synthetic sensor lines)")
    ap.add_argument("--ticks", type=int, default=40)
    args = ap.parse_args()
    if args.replay:
        run_replay(n_ticks=args.ticks)
    else:
        run_live(args.port)
