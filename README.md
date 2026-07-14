# Rotate & Quantize

A short presentation on rotation-based quantization for AI models. It covers how methods like QuIP#, QuaRot, and SpinQuant use a rotation step to make low-bit quantization (int4, int2) much less lossy, plus a hands-on demo that shows the same trick on image colors instead of model weights.

## Contents

**presentation.html** is the slide deck. It covers vectors, why AI models are huge, quantization basics, why naive quantization breaks things, the rotate-then-round fix, the math behind it, and where it's used today (LLMs, phones, edge devices).

**demo.py** is a standalone Python script that shows the core idea on a real image. Every pixel's color is a 3-number vector (R, G, B). The script reduces the color palette two ways with the same color budget:

1. Direct quantization: round each R, G, B value independently to N levels.
2. Rotate, quantize, rotate back: rotate pixel colors into their natural axes (using PCA on the color covariance matrix), quantize on that better-fitted grid, then rotate back to RGB.

It reports MSE and PSNR for both methods so you can see the quality difference directly.

## Running the demo

```bash
python demo.py path/to/image.png
```

You'll need numpy and Pillow: `pip install numpy pillow`.

## Viewing the presentation

Open presentation.html in a browser. No build step, no server needed. Navigate with the on-screen arrows or the left/right arrow keys.

## Why this matters

Modern LLMs are hundreds of gigabytes at full precision. Quantizing to 4-bit or lower shrinks them a lot, but naive rounding degrades quality because a few outlier values force a wide, imprecise grid. Rotating the data first spreads that outlier energy evenly across dimensions, which is a mathematically lossless step, so the rounding that follows has much less error to introduce. This is the core idea behind several quantization methods used in production LLMs today, including QuIP, QuIP#, QuaRot, and SpinQuant.
