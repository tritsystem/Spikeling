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


def record_voice_clip(duration_s: float = 5.0, device=None, samplerate: int = 44100,
                       out_path: str = "voice_clip.wav") -> str:
    """Records duration_s seconds from a real mic (device=None uses the
    system default input) and saves as a 16-bit PCM WAV file -- the
    format voicebox_transcribe expects via audio_path. Returns the saved
    path. (soundfile isn't installed on this machine; scipy.io.wavfile,
    already confirmed present, does the same real job here.)"""
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
