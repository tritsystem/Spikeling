"""
core/hardware/acoustic_anomaly_detector.py
============================================
AcousticAnomalyDetector — a real anomaly-detection device built on top of
AcousticSensorAdapter (already hardware-verified against a real microphone;
see test_acoustic_adapter.py).

WHY THIS ISN'T JUST "SUBCLASS AND YOU'RE DONE" (the "ten minutes per
device" framing this whole hardware layer replaced was called out as
unrealistic, and this file is why): a raw-amplitude signal only reflects
LOUDNESS. Two very different sounds -- a healthy motor and a motor with a
bearing starting to squeal -- can have nearly identical RMS loudness while
having completely different FREQUENCY content. Rate-coding raw amplitude
the way the plain base class does would miss exactly the class of anomaly
a device like this exists to catch. So this adds a real, separate signal-
processing stage:

  1. FFT each captured chunk into a magnitude spectrum, binned into
     `num_neurons` frequency bands (see _spectrum()).
  2. Hand that spectrum to BaselineDeviation (core/hardware/
     baseline_deviation.py) -- the shared "verify a reading by its
     relationship to a calibrated baseline" primitive used identically by
     EMG and environmental detection now too, extracted from what this
     file built first. "Anomalous" only means anything relative to a
     known-normal baseline for THIS specific install (a factory floor and
     a bedroom have wildly different "normal"); the SAME mechanism (learn
     a per-channel mean+std, z-score new readings against it, RMS into one
     score, EMA-smooth for noisy real-world use) applies whether the
     channel is a frequency band, an EMG reading, or an environmental
     sensor.
  3. read_raw() returns each band's DEVIATION from that baseline, not the
     raw spectrum -- so what actually drives spikes is "how different does
     this sound from normal right now," the actual detection signal, not
     raw loudness.

Still honestly scoped: this catches devices whose failure mode changes the
SPECTRAL SHAPE of ambient sound near the sensor. It is not a general
industrial-fault classifier and has not been validated against any real
mechanical fault (no real faulty machine was available to record) --
calibrate() against the real target environment and treat initial
threshold/deviation_scale values as a starting point to tune, not a solved
problem out of the box.
"""

import numpy as np

from .acoustic_adapter import AcousticSensorAdapter
from .baseline_deviation import BaselineDeviation


class AcousticAnomalyDetector(AcousticSensorAdapter):
    def __init__(self, *args, deviation_scale: float = 4.0,
                 smoothing_alpha: float = 0.3, **kwargs):
        """deviation_scale: how many std-devs of spectral deviation maps to
        the encoder's full [-1, 1] range -- lower = more sensitive
        (smaller deviations still saturate the encoder), higher = more
        conservative.

        smoothing_alpha: passed straight through to the underlying
        BaselineDeviation -- see smoothed_anomaly_score()'s own docstring
        for the real measurement that motivated smoothing existing at
        all."""
        super().__init__(*args, **kwargs)
        self.deviation_scale = deviation_scale
        self._baseline = BaselineDeviation(smoothing_alpha=smoothing_alpha)

    @property
    def smoothing_alpha(self) -> float:
        return self._baseline.smoothing_alpha

    @smoothing_alpha.setter
    def smoothing_alpha(self, value: float) -> None:
        self._baseline.smoothing_alpha = value

    @property
    def is_calibrated(self) -> bool:
        return self._baseline.is_calibrated

    def _spectrum(self, samples: list) -> list:
        """Real FFT magnitude spectrum, Hann-windowed (reduces spectral
        leakage from the chunk boundary not being periodic) and binned into
        num_neurons linear frequency bands. Linear binning is the honest,
        simple choice for a first real implementation -- log-spaced bands
        would better match human/mechanical perceptual relevance and is a
        reasonable next improvement, not implemented here."""
        n = len(samples)
        n_bands = self.encoder.num_neurons
        if n == 0:
            return [0.0] * n_bands
        windowed = np.asarray(samples, dtype=np.float64) * np.hanning(n)
        mags = np.abs(np.fft.rfft(windowed))
        edges = np.linspace(0, len(mags), n_bands + 1, dtype=int)
        bands = []
        for i in range(n_bands):
            lo, hi = edges[i], edges[i + 1]
            bands.append(float(mags[lo:hi].mean()) if hi > lo else 0.0)
        return bands

    def calibrate(self, duration_s: float = 5.0) -> tuple:
        """Records `duration_s` seconds of REAL audio from an already-
        start()'d stream as the 'normal' baseline, computing a genuine
        per-band mean+std across many chunks (not a single-sample guess --
        one chunk can't distinguish real background variance from a fluke).
        Returns (baseline_mean, baseline_std) as plain lists."""
        if self._stream is None:
            raise RuntimeError("calibrate() requires an open stream -- call start() first.")
        chunk_s = self.chunk_samples / float(self.sample_rate)
        n_chunks = max(3, int(round(duration_s / chunk_s)))
        spectra = [self._spectrum(super(AcousticAnomalyDetector, self).read_raw())
                   for _ in range(n_chunks)]
        return self._baseline.calibrate(spectra)

    def read_raw(self) -> list:
        samples = super().read_raw()   # the real mic capture, unmodified
        bands = self._spectrum(samples)
        return self._baseline.deviation(bands)

    @property
    def value_range(self):
        return (-self.deviation_scale, self.deviation_scale)

    def save_calibration(self, path: str) -> None:
        """Persist the calibrated baseline to disk (JSON) so a real
        calibration session's result survives past the process that ran
        it -- tuning against a real machine's real ambient sound is real
        work, not something to redo every run."""
        self._baseline.save(path)

    def load_calibration(self, path: str) -> tuple:
        """Loads a previously-saved baseline. Refuses to load one computed
        with a different num_neurons -- the per-band deviation math is only
        meaningful if the band count matches this instance's encoder."""
        return self._baseline.load(path, expected_channels=self.encoder.num_neurons)

    def anomaly_score(self, deviation: list = None) -> float:
        """A single 0+ scalar summarizing how anomalous the CURRENT moment
        is (RMS of the per-band z-score deviation) -- convenience for a
        caller that wants "is something wrong" rather than per-band
        detail. ~0 means "matches baseline"; growing values mean growing
        deviation. There is no universal "this means an anomaly" cutoff --
        that depends on the real install and what counts as a real fault
        there, which nothing here can know without real fault data."""
        if deviation is None:
            deviation = self.read_raw()
        return self._baseline.score(deviation=deviation)

    def smoothed_anomaly_score(self, deviation: list = None) -> float:
        """Exponential moving average of anomaly_score() -- MOTIVATED BY A
        REAL MEASUREMENT, not a hypothetical: stress_test_fan_ramp.py's
        actual run showed instantaneous anomaly_score() swinging
        0.67-9.36 within a single sustained test window, meaning a raw
        single-reading threshold either misses real anomalies (set high)
        or false-alarms on ordinary variance (set low -- the idle
        calibration session itself had one 3.24 outlier with nothing
        happening at all). Smoothing consecutive readings is what makes a
        threshold meaningful for continuous monitoring.

        Each call both updates AND returns the running smoothed value --
        call this once per real tick in a monitoring loop, not
        interchangeably with anomaly_score() within the same loop (mixing
        them would double-count some readings into the average and skip
        others). Resets automatically on calibrate()/load_calibration()
        since old smoothed history is meaningless against a new baseline;
        call reset_smoothing() manually for any other reason to discard
        accumulated history (e.g. after a known one-off loud event you
        don't want lingering in the average)."""
        if deviation is None:
            deviation = self.read_raw()
        return self._baseline.smoothed_score(deviation=deviation)

    def reset_smoothing(self) -> None:
        """Discards the running smoothed score -- the next
        smoothed_anomaly_score() call starts fresh from that single
        reading instead of blending in stale history."""
        self._baseline.reset_smoothing()

    # kept for any external code/tests that reached into the old private
    # fields directly -- both now live on the shared BaselineDeviation
    @property
    def _baseline_mean(self):
        return self._baseline._mean

    @property
    def _baseline_std(self):
        return self._baseline._std

    @property
    def _smoothed_score(self):
        return self._baseline._smoothed_score
