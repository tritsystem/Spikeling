---
name: ternary-vision-negative-result
description: Use when evaluating whether to ternary-quantize a vision model, or building the fridge-cam app.
keywords: MobileNetV3, ternary-quantiz, fridge-cam, vision quantization
---

# Ternary vision negative result

Real, honestly-reported negative result: ternary-quantizing
MobileNetV3-Small genuinely fails -- accuracy collapsed 78.8% -> 4.9%,
confirmed across 2 separate attempts, not a fluke. There IS a real win
(13.8x disk-size reduction) but ZERO speedup, confirmed. Given this,
the fridge-cam app's actual recommendation is to just use an off-the-
shelf Ollama vision model rather than pursuing ternary quantization
further for this use case -- don't re-attempt ternary vision
quantization for fridge-cam without a genuinely new angle, this one was
tested and closed.
