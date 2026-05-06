/*
 * sharpen.h — luma-channel unsharp mask in gamma-corrected space.
 *
 * Operates on the final 8-bit RGB image after gamma correction. For
 * each pixel:
 *
 *   1. Compute perceptual luma  Y = 0.2126·R + 0.7152·G + 0.0722·B  (BT.709)
 *   2. Apply separable Gaussian blur to the Y plane → Y_blur
 *   3. Compute detail signal    d = Y - Y_blur
 *   4. Suppress |d| below `threshold` (noise-floor protection)
 *   5. Add `amount * d` to each of R, G, B equally
 *
 * The "add equally to R, G, B" step is mathematically luma-only:
 * the BT.709 Cb and Cr matrices have row sums of exactly 0, so
 * adding the same delta to all three channels leaves chroma exactly
 * unchanged. This means sharpening doesn't introduce color fringes
 * the way per-channel USM can, even at high strength.
 *
 * Why gamma space (not linear): in linear space, the same additive
 * delta produces a much larger perceptual change in shadows than in
 * highlights. The result is asymmetric halos — bright halos look
 * much stronger than dark halos at the same edge. In gamma space
 * the halo amplitude is roughly perceptually uniform, which is what
 * the eye expects from sharpening.
 *
 * Threshold mechanism: |detail| below `threshold` is set to 0
 * before the multiply. This prevents USM from amplifying noise-
 * level fluctuations while still sharpening real edges (whose
 * |detail| typically far exceeds any reasonable threshold). For
 * images that have been NR-processed, a small threshold (3-8 in
 * 8-bit units) is usually enough.
 */

#ifndef SHARPEN_H
#define SHARPEN_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * In-place unsharp mask on an interleaved RGB8 buffer.
 *
 * amount    : 0.0  → no-op (returns immediately, no allocation)
 *             0.3  → mild      (gentle pop)
 *             0.6  → moderate  (typical photographic sharpen)
 *             1.0  → strong    (very visible)
 *             2.0  → aggressive (over the top, halo-prone)
 *
 * radius    : Gaussian sigma in pixels for the lowpass blur.
 *             0.5..1.0 → fine detail (eyelashes, fabric weave, distant text)
 *             1.0..2.0 → general sharpening
 *             2.0..5.0 → coarse / haze-cutting
 *             clamped internally to [0.3, 8.0]
 *
 * threshold : |detail| (in 8-bit units) below which sharpening is
 *             suppressed. 0 = sharpen everything (amplifies noise);
 *             4-8 = standard for clean images; 10-20 = aggressive
 *             noise protection (only sharpens strong edges).
 *
 * Memory: allocates one width*height byte buffer + small scratch.
 *         For 5248x3936 that's ~21 MB peak, freed before return.
 *
 * Time at 5248x3936, NEON+threaded (D5503 measured):
 *   radius=1.0  ≈ 1.2 s   (extrapolated from desktop ratio)
 *   radius=3.0  ≈ 1.7 s   (measured directly; deploy setting)
 *
 * Position in pipeline: AFTER demosaic + gamma + any post-processing.
 * Sharpens the final perceptual image.
 */
void sharpen_rgb8(uint8_t* rgb, int width, int height,
                  float amount, float radius, int threshold);

#ifdef __cplusplus
}
#endif

#endif
