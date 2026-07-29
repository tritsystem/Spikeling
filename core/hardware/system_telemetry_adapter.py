"""
core/hardware/system_telemetry_adapter.py
===========================================
SystemTelemetryAdapter — a real, multi-channel PC-stress monitor: CPU
utilization (aggregate + hottest single core), RAM usage, disk I/O rate,
and GPU temperature/utilization/power (via nvidia-smi). Every channel here
is REAL hardware telemetry already available on this machine -- no
simulated source, unlike emg_adapter.py/environmental_adapter.py, because
this machine genuinely has these sensors and nothing here needs a
substitute.

WHY THIS EXISTS: "tune EMG/environmental against real data" turned out to
be unfulfillable as asked -- no EMG or environmental hardware is connected
(see those modules' own docstrings). This is NOT a substitute for either
device. It answers a different, fully real, related question instead:
"track exactly why THIS machine is stressed, using telemetry it actually
has."

WHY A SINGLE AGGREGATE SCORE ISN'T ENOUGH (a real finding from
validate_baseline_deviation_gpu.py): BaselineDeviation.score() collapses
every channel into one RMS number -- answers "is something unusual," not
"which subsystem." A GPU-bound load and a disk-thrashing load can produce
similar aggregate scores while needing completely different responses.
diagnose() below surfaces the per-channel deviation BaselineDeviation.
deviation() was already computing (nothing new mathematically -- score()
was just discarding this), sorted by magnitude, so the dominant channel
IS the answer to "why."

HOMEOSTATIC BASELINE (self.adapt()): a PC's real "normal" legitimately
drifts over time (a new background service, a driver update). Calling
adapt() once per monitoring tick lets the baseline track that drift via
the gated closed-loop mechanism in baseline_deviation.py, instead of
needing a manual recalibration forever -- see that method's own docstring
for why it's gated (a real stress event must never get absorbed as the
new "normal").
"""

import subprocess
import time

import psutil

from .sensor_adapter import SensorAdapter
from .baseline_deviation import BaselineDeviation

CHANNELS = ["cpu_pct", "cpu_max_core_pct", "mem_pct",
            "disk_read_bps", "disk_write_bps",
            "gpu_temp_c", "gpu_util_pct", "gpu_power_w"]


def _read_gpu_telemetry():
    """Real nvidia-smi query -- the exact approach independently confirmed
    in validate_baseline_deviation_gpu.py against real ground truth (util
    jumping to 100% under a genuine CUDA load). Returns None if no NVIDIA
    GPU/driver is present rather than fabricating zeros -- "no GPU" is a
    real, different state from "GPU idle at 0%."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if out.returncode != 0:
            return None
        temp, util, power = (float(x) for x in out.stdout.strip().split(","))
        return {"gpu_temp_c": temp, "gpu_util_pct": util, "gpu_power_w": power}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return None


class SystemTelemetrySource:
    """Real telemetry from THIS machine: psutil for CPU/RAM/disk,
    nvidia-smi for GPU (if present). Stateful -- disk I/O is a RATE,
    computed from the delta between consecutive read() calls, so the
    first read() after construction reports 0.0 for both disk channels
    (no prior sample to diff against yet -- an honest "no rate known
    yet," not a guess)."""

    def __init__(self):
        self._gpu_available = _read_gpu_telemetry() is not None
        psutil.cpu_percent(percpu=True)   # primes psutil's internal counter -- its first-ever call is always 0.0
        self._prev_disk = psutil.disk_io_counters()
        self._prev_t = time.monotonic()

    @property
    def channels(self) -> list:
        return list(CHANNELS) if self._gpu_available else CHANNELS[:5]

    def read(self) -> dict:
        per_core = psutil.cpu_percent(percpu=True)
        mem_pct = psutil.virtual_memory().percent

        now = time.monotonic()
        dt = max(now - self._prev_t, 1e-6)
        disk = psutil.disk_io_counters()
        read_bps = (disk.read_bytes - self._prev_disk.read_bytes) / dt
        write_bps = (disk.write_bytes - self._prev_disk.write_bytes) / dt
        self._prev_disk = disk
        self._prev_t = now

        reading = {
            "cpu_pct": sum(per_core) / len(per_core) if per_core else 0.0,
            "cpu_max_core_pct": max(per_core) if per_core else 0.0,
            "mem_pct": mem_pct,
            "disk_read_bps": max(0.0, read_bps),
            "disk_write_bps": max(0.0, write_bps),
        }
        if self._gpu_available:
            gpu = _read_gpu_telemetry()
            if gpu is not None:
                reading.update(gpu)
        return reading


def top_processes(n: int = 3, by: str = "cpu") -> list:
    """Real, live process attribution -- the actual answer to "which
    program is doing this," which a channel-level score alone can't give.
    `by`: "cpu" (needs psutil's own per-process priming, see below) or
    "memory" (no priming needed). Returns [(name, pid, value), ...] sorted
    descending.

    NOTE: psutil's per-process cpu_percent() needs a first call to start
    its internal timer, then a real elapsed interval before a second call
    is meaningful -- the priming pass + real sleep below is deliberate,
    verified interactively against this machine's actual process list
    before this module was written (System Idle Process, node.exe,
    python.exe, etc. -- real names, real numbers)."""
    procs = list(psutil.process_iter(["pid", "name"]))
    if by == "cpu":
        for p in procs:
            try:
                p.cpu_percent(None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        time.sleep(0.3)
    scored = []
    for p in procs:
        try:
            value = p.cpu_percent(None) if by == "cpu" else p.memory_percent()
            scored.append((p.info["name"], p.info["pid"], value))
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    scored.sort(key=lambda x: -x[2])
    return scored[:n]


class SystemTelemetryAdapter(SensorAdapter):
    """See module docstring. `source` defaults to the real
    SystemTelemetrySource -- unlike emg_adapter.py/environmental_adapter.py
    there is no simulated fallback here, because this machine genuinely
    has every one of these sensors.

    `runtime` may be None: this adapter is usable purely as a diagnostic
    (calibrate/stress_score/diagnose) without ever calling tick()/encode(),
    the only methods that touch self.runtime. Pass a real SpikelingRuntime
    only if you also want this to stimulate an actual network."""

    def __init__(self, runtime=None, *args, source=None, smoothing_alpha: float = 0.3,
                 max_channel_z: float = 20.0, **kwargs):
        """max_channel_z defaults to 20.0 (not None/unclipped) -- see
        BaselineDeviation's own docstring for the real measured failure
        (a flat-during-calibration disk_write_bps channel producing a
        billions-scale aggregate score from one real write burst) that
        makes this the honest default HERE specifically: real PC telemetry
        (disk I/O especially) is genuinely bursty enough that a short
        calibration window often DOES see a channel sit at exactly 0.0 the
        whole time. 20 std-devs is still an enormous, unambiguous
        deviation for any real channel -- nothing legitimate gets
        suppressed, only the runaway-division artifact is bounded."""
        super().__init__(runtime, *args, **kwargs)
        self.source = source or SystemTelemetrySource()
        self._deviation = BaselineDeviation(smoothing_alpha=smoothing_alpha, max_channel_z=max_channel_z)
        self._active_channels = list(self.source.channels)

    @property
    def value_range(self):
        return (-4.0, 4.0)   # deviation-scored, same convention as acoustic_anomaly_detector.py

    def calibrate(self, n_samples: int = 20, delay_s: float = 0.5) -> tuple:
        """Records n_samples REAL readings, delay_s apart, as this
        specific machine's calibrated normal. Real elapsed time between
        samples matters here (unlike a single instantaneous snapshot)
        since CPU/disk readings are themselves rate-based and need real
        elapsed time to show their natural variance."""
        samples = []
        for _ in range(max(1, n_samples)):
            r = self.source.read()
            samples.append([r[c] for c in self._active_channels])
            time.sleep(delay_s)
        return self._deviation.calibrate(samples)

    @property
    def is_calibrated(self) -> bool:
        return self._deviation.is_calibrated

    def _reading_vector(self) -> list:
        r = self.source.read()
        return [r[c] for c in self._active_channels]

    def read_raw(self) -> list:
        return self._deviation.deviation(self._reading_vector())

    def stress_score(self) -> float:
        return self._deviation.score(deviation=self.read_raw())

    def smoothed_stress_score(self) -> float:
        return self._deviation.smoothed_score(deviation=self.read_raw())

    def reset_smoothing(self) -> None:
        self._deviation.reset_smoothing()

    def adapt(self, rate: float = 0.01, gate_threshold: float = 2.0) -> bool:
        """See BaselineDeviation.adapt() -- call once per real monitoring
        tick to let this machine's baseline track genuine long-run drift
        without absorbing real stress events as "normal"."""
        return self._deviation.adapt(self._reading_vector(), rate=rate, gate_threshold=gate_threshold)

    def diagnose(self, top_n: int = 3) -> list:
        """THE "why" answer: per-channel deviation, sorted by |z-score|
        descending -- the dominant channel(s) actually driving the current
        stress score, not just one opaque aggregate number. Returns
        [(channel_name, z_score), ...], most-deviant first."""
        dev = self._deviation.deviation(self._reading_vector())
        pairs = list(zip(self._active_channels, dev))
        pairs.sort(key=lambda x: -abs(x[1]))
        return pairs[:top_n]

    def sample(self, top_n: int = 3, adapt: bool = False,
               adapt_rate: float = 0.01, adapt_gate: float = 2.0) -> dict:
        """ONE real telemetry read, reused for the smoothed score,
        diagnosis, and (optionally) homeostatic adaptation -- calling
        smoothed_stress_score()/diagnose()/adapt() separately in a tight
        monitoring loop would each trigger their own real read (including
        a separate nvidia-smi subprocess spawn), reading slightly
        different, inconsistent moments and wasting real work for no
        benefit. Returns {"score": float, "diagnosis": [(name, z), ...],
        "adapted": bool | None (None if adapt=False)}."""
        vec = self._reading_vector()
        dev = self._deviation.deviation(vec)
        score = self._deviation.smoothed_score(deviation=dev)
        diagnosis = sorted(zip(self._active_channels, dev), key=lambda x: -abs(x[1]))[:top_n]
        adapted = self._deviation.adapt(vec, rate=adapt_rate, gate_threshold=adapt_gate) if adapt else None
        return {"score": score, "diagnosis": diagnosis, "adapted": adapted}

    def save_calibration(self, path: str) -> None:
        self._deviation.save(path)

    def load_calibration(self, path: str) -> tuple:
        return self._deviation.load(path, expected_channels=len(self._active_channels))
