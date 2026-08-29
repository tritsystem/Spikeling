---
name: wireless-glasses-voice-loop
description: Use when working on the wireless AR-glasses hardware design or the PC-side voice loop (TTS/STT).
keywords: wireless glasses, AR-glasses, voice loop, Piper, Voicebox
---

# Wireless glasses + voice loop

Full wireless AR-glasses design with a real, priced ~$166 parts list.
PC-side WiFi code is written but UNTESTED end-to-end. Voice loop TTS is
confirmed working, with two real gotchas: profile the voice by ID, not
by name, and Piper is unsupported in this setup. STT is currently
blocked on a Whisper model download inside Voicebox -- check whether
that download has completed before assuming STT is functional.
