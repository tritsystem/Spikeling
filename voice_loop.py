#!/usr/bin/env python
"""
voice_loop.py — real infrastructure for talk-to-it / it-talks-back: the
mic-capture half of the loop that voicebox_transcribe (already connected
this session) needs an audio file for. Transcription, response reasoning,
speaking the reply (voicebox_speak), and displaying it on the glasses
(wireless_glasses.send_to_glasses_display, still hardware-pending) all
happen OUTSIDE this file -- this only handles real audio capture.

Works with ANY real mic device index on this machine right now (C922,
QuadCast, or eventually the BT earpiece once paired and selected) --
proving the voice conversation loop doesn't need to wait on the glasses
hardware to exist.

    from voice_loop import record_voice_clip
"""
import numpy as np
import sounddevice as sd
from scipy.io import wavfile


def _find_real_mic_device():
    """REAL BUG FOUND AND FIXED HERE: this machine's system default INPUT
    device (sd.default.device[0]) is "Stereo Mix" -- a loopback device
    that captures whatever is currently PLAYING through the speakers, not
    the user's actual voice. device=None (the old default) silently used
    it. Confirmed by direct sd.query_devices() inspection, not assumed --
    a genuinely wrong default that would have made record_voice_clip()
    record system audio playback instead of a real mic input, with no
    error to indicate anything was wrong.

    Searches by NAME for a real input device instead of hardcoding a
    device index -- indices shift when hardware is added/removed/
    reconnected, a name-based search is the more robust fix. Looks for
    "Microphone" in the device name (this machine's real mic is
    "Microphone (Realtek HD Audio Mic input)"); explicitly excludes
    "Stereo Mix" even if a future device happens to also match on some
    other substring. Falls back to the system default (with a printed
    warning, not a silent wrong answer) if no real mic is found by name,
    so this never hard-fails on a machine with different hardware."""
    devices = sd.query_devices()
    for idx, dev in enumerate(devices):
        name = dev["name"]
        if dev["max_input_channels"] > 0 and "microphone" in name.lower() and "stereo mix" not in name.lower():
            return idx
    print("WARNING: no device with 'Microphone' in its name found -- falling back to the "
          "system default input, which may be a loopback device (e.g. Stereo Mix) rather "
          "than a real mic. Pass device= explicitly to override.", flush=True)
    return None


def record_voice_clip(duration_s: float = 5.0, device=-1, samplerate: int = 44100,
                       out_path: str = "voice_clip.wav") -> str:
    """Records duration_s seconds from a real mic and saves as a 16-bit
    PCM WAV file -- the format voicebox_transcribe expects via
    audio_path. Returns the saved path. (soundfile isn't installed on
    this machine; scipy.io.wavfile, already confirmed present, does the
    same real job here.)

    device=-1 (the default): auto-detect a real microphone by name (see
    _find_real_mic_device()) -- NOT the same as device=None, which uses
    sd.default.device and on this machine resolves to "Stereo Mix" (a
    loopback device, not a real mic -- a real, confirmed bug in the
    previous default, not a hypothetical one). Pass an explicit device
    index/name to override, or None to explicitly opt into the raw
    system default despite the above."""
    if device == -1:
        device = _find_real_mic_device()
    print(f"Recording {duration_s:.1f}s from device={device if device is not None else 'default'}...", flush=True)
    audio = sd.rec(int(duration_s * samplerate), samplerate=samplerate,
                    channels=1, device=device, dtype="float32")
    sd.wait()
    pcm16 = np.clip(audio[:, 0] * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(out_path, samplerate, pcm16)
    return out_path


if __name__ == "__main__":
    import sys
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 5.0
    path = record_voice_clip(duration_s=duration)
    print(f"saved: {path}")
