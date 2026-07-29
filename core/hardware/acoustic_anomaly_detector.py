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
  2. CALIBRATE a baseline (per-band mean + std) from a real recording of
     normal operation (calibrate()) -- a genuine requirement for ANY
     anomaly detector: "anomalous" only means anything relative to a
     known-normal baseline for THIS specific install (a factory floor and
     a bedroom have wildly different "normal").
  3. read_raw() returns each band's DEVIATION from that baseline (a
     z-score), not the raw spectrum -- so what actually drives spikes is
     "how different does this sound from normal right now," which is the
     actual detection signal, not raw loudness.

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


class AcousticAnomalyDetector(AcousticSensorAdapter):
    def __init__(self, *args, deviation_scale: float = 4.0, **kwargs):
        """deviation_scale: how many std-devs of spectral deviation maps to
        the encoder's full [-1, 1] range -- lower = more sensitive
        (smaller deviations still saturate the encoder), higher = more
        conservative."""
        super().__init__(*args, **kwargs)
        self.deviation_scale = deviation_scale
        self._baseline_mean = None
        self._baseline_std = None

    @property
    def is_calibrated(self) -> bool:
        return self._baseline_mean is not None

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
        arr = np.array(spectra)
        self._baseline_mean = arr.mean(axis=0)
        self._baseline_std = np.maximum(arr.std(axis=0), 1e-6)   # floor avoids div-by-zero on a silent band
        return self._baseline_mean.tolist(), self._baseline_std.tolist()

    def read_raw(self) -> list:
        samples = super().read_raw()   # the real mic capture, unmodified
        bands = self._spectrum(samples)
        if not self.is_calibrated:
            # no baseline yet -- report "no deviation" rather than a
            # meaningless number computed against nothing
            return [0.0] * len(bands)
        return [(b - m) / s for b, m, s in
                zip(bands, self._baseline_mean, self._baseline_std)]

    @property
    def value_range(self):
        return (-self.deviation_scale, self.deviation_scale)

    def save_calibration(self, path: str) -> None:
        """Persist the calibrated baseline to disk (JSON) so a real
        calibration session's result survives past the process that ran
        it -- tuning against a real machine's real ambient sound is real
        work, not something to redo every run."""
        if not self.is_calibrated:
            raise RuntimeError("save_calibration() called before calibrate() -- nothing to save.")
        import json
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "num_neurons": self.encoder.num_neurons,
                "sample_rate": self.sample_rate,
                "chunk_ms": self.chunk_samples / float(self.sample_rate) * 1000.0,
                "baseline_mean": self._baseline_mean.tolist(),
                "baseline_std": self._baseline_std.tolist(),
            }, f, indent=2)

    def load_calibration(self, path: str) -> tuple:
        """Loads a previously-saved baseline. Refuses to load one computed
        with a different num_neurons -- the per-band deviation math is only
        meaningful if the band count matches this instance's encoder."""
        import json
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if data["num_neurons"] != self.encoder.num_neurons:
            raise ValueError(
                f"calibration file has {data['num_neurons']} bands, this detector has "
                f"{self.encoder.num_neurons} -- they must match, or the per-band deviation "
                f"math is comparing bands that don't correspond to the same frequencies.")
        self._baseline_mean = np.array(data["baseline_mean"])
        self._baseline_std = np.array(data["baseline_std"])
        return self._baseline_mean.tolist(), self._baseline_std.tolist()

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
        if not deviation:
            return 0.0
        return float(np.sqrt(np.mean(np.square(deviation))))
