---
name: tribe-emergent-sync
description: Use when working on Tribe's Kuramoto phase-lock system (TribeDrums, tribemember.gd sync) or debugging NPCs that seem to coordinate without direct communication.
keywords: TribeDrums, Kuramoto, phase-lock, emergent sync, order parameter
---

# Tribe emergent sync

Real, measured result: independent NPC brains phase-lock (Kuramoto
order parameter r: 0.48 -> 0.94) purely from shared drum-audio feedback
-- no direct NPC-to-NPC communication channel exists. If NPCs appear
to be coordinating, check whether they're actually synchronizing via
this audio-feedback mechanism before assuming a scripted coordination
bug.

This has a physical-hardware analog: `sync_mesh_finding.md` replicated
the same phase-lock mechanism on real microphones (2 separate devices,
6 repeated trials needed to see it reliably, ~10x smaller effect size
than the sim, gain-mismatch and single-trial-noise being the reusable
lessons for why a single trial can look like nothing happened).
