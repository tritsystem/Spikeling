"""
core/hardware/baseline_deviation.py
=====================================
BaselineDeviation — the shared "verify a reading by its relationship to a
calibrated baseline" primitive behind every anomaly-style device in this
hardware layer.

WHY THIS EXISTS: acoustic_anomaly_detector.py built this exact pattern
first (per-band calibrated mean+std, z-score deviation, an aggregate RMS
score, EMA smoothing after a real stress test showed instantaneous scores
were too noisy for a single-reading threshold) -- then it became clear EMG
and environmental detection both need the SAME thing (EMG's baseline-
subtraction was only half of it, no deviation score at all; environmental
had no baseline-relative notion whatsoever, only a fixed-physical-range
normalization). Rather than three devices each reimplementing pieces of
"is this reading unusual FOR THIS INSTALL" slightly differently, this is
the one shared mechanism: nothing here is meaningful in isolation, only
relative to a learned baseline -- the same idea as z-scoring, but named
for what it's actually doing in this codebase's own vocabulary (verifying
X by its relationship to Y, not X alone).
"""

import json

import numpy as np


class BaselineDeviation:
    """Channel-count-agnostic: works for a single scalar channel (EMG) or
    N channels (acoustic frequency bands, environmental sensors) the same
    way -- every reading is just a list[float] of a fixed length."""

    def __init__(self, smoothing_alpha: float = 0.3, max_channel_z: float = None):
        """smoothing_alpha: weight given to each NEW reading in the
        exponential moving average smoothed_score() (0 < a <= 1; higher =
        tracks recent readings faster but smooths less).

        max_channel_z: optional cap on any single channel's contribution
        to score()/smoothed_score() (deviation() itself is never clipped
        -- see below). Defaults to None (no clipping, the original
        behavior every existing caller/test relies on). WHY THIS EXISTS --
        a REAL measured failure, not a hypothetical: system_telemetry_
        adapter.py's disk_write_bps channel calibrated with zero real
        writes in every sample (std floored to 1e-6), then a genuine real
        write of a few hundred KB/s during a live test produced a
        deviation() z-score in the billions and dragged score()'s RMS
        aggregate up with it -- mathematically correct, but useless as a
        "how stressed is this machine" number. Any channel that happens to
        be perfectly flat during a short real calibration window (common
        for coarse-precision or bursty real sensors, not just synthetic
        data) is one real reading away from this. Capping each channel's
        z BEFORE the aggregate keeps the aggregate interpretable regardless
        of which channel is degenerate, while deviation() stays unclipped
        so a caller inspecting per-channel detail (e.g. diagnose() in
        system_telemetry_adapter.py) still sees the true, uncapped
        magnitude -- that raw number IS meaningful information ("this
        channel deviated far past anything seen during calibration"),
        just not something the aggregate should be dominated by."""
        self.smoothing_alpha = smoothing_alpha
        self.max_channel_z = max_channel_z
        self._mean = None
        self._std = None
        self._smoothed_score = None

    @property
    def is_calibrated(self) -> bool:
        return self._mean is not None

    def calibrate(self, samples) -> tuple:
        """samples: an iterable of readings, each a list[float] of the same
        length (one per channel). Computes a genuine per-channel mean+std
        across all of them -- one sample can't distinguish real background
        variance from a fluke, which is why calibrate() takes many, not
        one. Returns (mean, std) as plain lists."""
        arr = np.array(list(samples))
        if arr.ndim != 2 or arr.shape[0] < 1:
            raise ValueError("calibrate() needs at least one multi-channel sample "
                              "(a 2D array of [n_samples, n_channels]).")
        self._mean = arr.mean(axis=0)
        self._std = np.maximum(arr.std(axis=0), 1e-6)   # floor avoids div-by-zero on a dead-flat channel
        self.reset_smoothing()   # old smoothed history is meaningless against a new baseline
        return self._mean.tolist(), self._std.tolist()

    def deviation(self, reading: list) -> list:
        """Per-channel z-score of `reading` against the calibrated
        baseline. Returns all-zeros (not an error) if not yet calibrated --
        "no baseline to compare against" is a real, valid state, and
        reporting zero deviation is more honest than a number computed
        against nothing."""
        if not self.is_calibrated:
            return [0.0] * len(reading)
        return [(r - m) / s for r, m, s in zip(reading, self._mean, self._std)]

    def score(self, reading: list = None, deviation: list = None) -> float:
        """A single 0+ scalar summarizing how far `reading` (or a
        precomputed `deviation`) sits from the calibrated baseline -- RMS
        of the per-channel z-score. ~0 means "matches baseline"; growing
        values mean growing deviation. There's no universal "this means
        an anomaly" cutoff -- that depends on the real install."""
        if deviation is None:
            if reading is None:
                raise ValueError("score() needs either `reading` or a precomputed `deviation`.")
            deviation = self.deviation(reading)
        if not deviation:
            return 0.0
        d = deviation
        if self.max_channel_z is not None:
            cap = self.max_channel_z
            d = [max(-cap, min(cap, x)) for x in d]
        return float(np.sqrt(np.mean(np.square(d))))

    def smoothed_score(self, reading: list = None, deviation: list = None) -> float:
        """Exponential moving average of score() -- see the module
        docstring for why this exists (a real measured need: instantaneous
        scores are too noisy for a single-reading threshold in practice).
        Each call both updates AND returns the running smoothed value --
        call once per real tick in a monitoring loop, not interchangeably
        with score() in the same loop."""
        raw = self.score(reading, deviation)
        if self._smoothed_score is None:
            self._smoothed_score = raw
        else:
            a = self.smoothing_alpha
            self._smoothed_score = a * raw + (1.0 - a) * self._smoothed_score
        return self._smoothed_score

    def reset_smoothing(self) -> None:
        self._smoothed_score = None

    def adapt(self, reading: list, rate: float = 0.01, gate_threshold: float = 2.0) -> bool:
        """Closed-loop homeostatic baseline drift-tracking: nudges the
        calibrated mean/std toward `reading`, but ONLY when the reading is
        already close to baseline (score(reading) <= gate_threshold).
        Returns True if the baseline was actually adapted, False if gated
        off (not calibrated yet, or the reading is a real deviation).

        WHY GATED, NOT A PLAIN EMA: an always-on EMA toward every new
        reading would let a genuine, ongoing anomaly slowly get absorbed
        into "normal" -- exactly the failure mode a static one-time
        calibration doesn't have, trading one bug for a worse one. Gating
        on the reading's OWN score means only readings that already match
        baseline get to move it, so real deviations stay visible as
        deviations no matter how long they persist.

        This is closed-loop feedback (measure -> compare to a target ->
        correct), the same class of mechanism that in this codebase's own
        prior research decisively beat both a static baseline and blind
        periodic ("breathing") adjustment for recovering a stuck LIF
        neuron -- see the gbranaa4-hue portfolio's breathing-modulation
        finding. That result was about recovery SPEED for one stuck
        neuron; this is a different job (long-run baseline drift for a
        continuously-running monitor), so the gate is new here, not
        carried over -- but the core principle (closed-loop beats blind
        schedule) is the same one that was actually measured to hold."""
        if not self.is_calibrated:
            return False
        arr = np.array(reading, dtype=np.float64)
        if self.score(reading=reading) > gate_threshold:
            return False
        self._mean = (1.0 - rate) * self._mean + rate * arr
        diff = arr - self._mean
        var = np.maximum(self._std, 1e-6) ** 2
        var = (1.0 - rate) * var + rate * (diff ** 2)
        self._std = np.maximum(np.sqrt(var), 1e-6)
        return True

    def save(self, path: str) -> None:
        if not self.is_calibrated:
            raise RuntimeError("save() called before calibrate() -- nothing to save.")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "num_channels": len(self._mean),
                "mean": self._mean.tolist(),
                "std": self._std.tolist(),
            }, f, indent=2)

    def load(self, path: str, expected_channels: int = None) -> tuple:
        """Loads a previously-saved baseline. Pass `expected_channels`
        (e.g. the caller's encoder.num_neurons) to refuse a file with a
        different channel count instead of silently comparing channels
        that don't correspond to the same thing."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if expected_channels is not None and data["num_channels"] != expected_channels:
            raise ValueError(
                f"calibration file has {data['num_channels']} channels, caller expects "
                f"{expected_channels} -- they must match, or the per-channel deviation "
                f"math is comparing channels that don't correspond to the same thing.")
        self._mean = np.array(data["mean"])
        self._std = np.array(data["std"])
        self.reset_smoothing()
        return self._mean.tolist(), self._std.tolist()
