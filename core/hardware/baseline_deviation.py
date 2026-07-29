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

    def __init__(self, smoothing_alpha: float = 0.3):
        """smoothing_alpha: weight given to each NEW reading in the
        exponential moving average smoothed_score() (0 < a <= 1; higher =
        tracks recent readings faster but smooths less)."""
        self.smoothing_alpha = smoothing_alpha
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
        return float(np.sqrt(np.mean(np.square(deviation))))

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
