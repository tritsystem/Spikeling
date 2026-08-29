---
name: tribe-terrain-generation
description: Use when working on Tribe's terrain_gen.gd -- heightmaps, biomes, terraforming, or island mode.
keywords: terrain_gen.gd, heightmap, island mode, terraforming, biomes
---

# Tribe terrain system

`terrain_gen.gd` real feature set: heightmap + collision generation,
biomes, live terraforming, adaptive RES (resolution scales with need),
walkable floor-snap, and a built ISLAND MODE that's off by default with
spawn-wiring still pending -- don't assume island mode is live just
because the code exists; check the default flag.

Turtle-island (a SEPARATE, now-fully-removed feature) is not this --
`turtle_island.gd`, `player_island.gd`, `troll.gd` and their tests were
deleted in a surgical removal pass (commit 496d212) that specifically
preserved `water_crossing.gd` as an unrelated general archipelago
feature. If old context mentions turtle islands, it's stale.
