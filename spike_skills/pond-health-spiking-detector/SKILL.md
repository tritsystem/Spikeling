---
name: pond-health-spiking-detector
description: Use when working on pond-health's spiking-neuron anomaly detector or comparing it against the trend-based detector.
keywords: pond-health, spiking anomaly detector, trend detector
---

# Pond-health spiking detector

Spikeling's real LIF-neuron engine wired in as a SECOND anomaly
detector alongside a linear-trend detector, honestly measured against
each other, not just added for coverage's sake. Real, asymmetric
result: the spiking detector never false-alarms, but also gives no
early warning -- the trend detector does exactly the opposite trade-off.
Neither one dominates; they're complementary by design, not one being
strictly better. Don't drop either detector without losing a real
capability the other doesn't have.
