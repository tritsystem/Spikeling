"""
core/hardware/acoustic_adapter.py
==================================
AcousticSensorAdapter — a REAL reference implementation of SensorAdapter,
backed by an actual microphone (via `sounddevice`), not a simulation. This
is the one of the three original reference devices that can genuinely be
tested end to end on this PC right now, no purchased hardware required --
the other two (EMG, environmental) don't have real sensors attached yet;
see their own docstrings for what's honestly simulated there.

sounddevice's float32 input is already in [-1.0, 1.0] (standard PCM
convention), so no value_range override is needed here -- the base class's
default already matches.
"""

from .sensor_adapter import SensorAdapter


class AcousticSensorAdapter(SensorAdapter):
    """Captures short chunks of real audio from an input device and feeds
    them through SensorAdapter's shared encode/stimulate pipeline. Use as
    a context manager (`with AcousticSensorAdapter(...) as mic: ...`) so the
    stream is always closed, or call start()/stop() directly."""

    def __init__(self, *args, sample_rate: int = 16000, chunk_ms: float = 100.0,
                 device=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.sample_rate = sample_rate
        self.chunk_samples = max(1, int(sample_rate * chunk_ms / 1000.0))
        self.device = device
        self._stream = None

    def start(self) -> None:
        """Opens the input stream. Raises whatever sounddevice/PortAudio
        raises if no input device is available -- not swallowed, since a
        caller needs to know their sensor genuinely isn't there."""
        import sounddevice as sd
        self._stream = sd.InputStream(samplerate=self.sample_rate, channels=1,
                                       dtype="float32", device=self.device,
                                       blocksize=self.chunk_samples)
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    def read_raw(self) -> list:
        if self._stream is None:
            raise RuntimeError(
                "AcousticSensorAdapter.read_raw() called before start() -- "
                "no audio stream is open (use as a context manager, or call start() first).")
        data, _overflowed = self._stream.read(self.chunk_samples)
        return data[:, 0].tolist()

    @staticmethod
    def list_devices():
        """Real input devices sounddevice can see on this machine -- useful
        for picking `device=` when the default input isn't the right mic."""
        import sounddevice as sd
        return [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
