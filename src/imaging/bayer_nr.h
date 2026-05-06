/*
 * bayer_nr.h — multi-level wavelet noise reduction for Bayer data.
 *
 * Algorithm: 3-level 2D Haar wavelet decomposition of each RGGB
 * channel separately, with BayesShrink soft thresholding of detail
 * coefficients at every level, then inverse transform.
 *
 * Why multi-level (vs. single-level):
 *
 *   Single-level wavelet captures only 1-2 pixel structure in its
 *   detail bands. Noise that has spatial extent of 4-8 pixels
 *   (which low-frequency chroma noise typically does, after
 *   demosaic interpolation spreads it) ends up in the LL sub-band
 *   and survives single-level thresholding entirely.
 *
 *   Multi-level decomposition recursively applies the wavelet to
 *   the LL sub-band:
 *     - Level 1 detail bands capture 1-2 pixel structure
 *     - Level 2 detail bands capture 2-4 pixel structure
 *     - Level 3 detail bands capture 4-8 pixel structure
 *   Thresholding at every level removes noise across the full
 *   spatial-frequency range. Real signal at any frequency still
 *   concentrates in a few large coefficients per scale, so it
 *   survives soft thresholding.
 *
 * Why wavelet thresholding (vs. bilateral or other local filters):
 *
 *   Bilateral filtering compares pixel values to decide which
 *   neighbors to average. It cannot tell low-amplitude texture
 *   from noise — both look like "small variations from the local
 *   mean" — so any setting strong enough to denoise also smooths
 *   subtle texture.
 *
 *   Wavelet thresholding works on a different principle: real
 *   signal concentrates in a few large coefficients (because real
 *   signal has structure), while noise spreads uniformly across
 *   all coefficients (because noise has none). Soft-thresholding
 *   removes most noise while keeping nearly all signal, regardless
 *   of the signal's amplitude, as long as it has spatial structure.
 *
 *   Sharp edges produce coefficients far above any reasonable
 *   threshold, so they pass through unmodified. Periodic textures
 *   (fabric, foliage) concentrate energy at their characteristic
 *   spatial frequencies, so they survive even when individual
 *   pixels are at noise-level amplitude.
 *
 * Threshold selection — BayesShrink:
 *
 *   For each detail sub-band at every level,
 *     T = sigma_noise^2 / sigma_signal
 *   where sigma_signal = sqrt(max(0, sigma_band^2 - sigma_noise^2)).
 *
 *   sigma_noise is estimated ONCE from the level-1 HH sub-band
 *   via MAD (Donoho-Johnstone convention). The orthonormal Haar
 *   transform preserves noise variance across scales, so this
 *   single estimate applies to all levels without rescaling.
 *
 *   Per-level estimation would over-threshold coarser scales
 *   because their HH bands contain aliased real signal that
 *   level-1 didn't smooth away.
 *
 *   The user's `strength` parameter is just a linear multiplier
 *   on the auto-derived per-band threshold.
 *
 * Per-channel processing means there is no cross-color contamination.
 * Filtering R uses only R neighbors, G1 only G1, etc. Color textures
 * and chromatic edges are preserved exactly.
 */

#ifndef BAYER_NR_H
#define BAYER_NR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * In-place wavelet noise reduction on a 10-bit-valued Bayer plane.
 *
 * strength : 0.0 -> no-op (returns immediately)
 *            0.5 -> mild     (half the BayesShrink default)
 *            1.0 -> moderate (BayesShrink auto-threshold)   [default]
 *            2.0 -> aggressive
 *
 * Memory: ~99 MB peak for 5248x3936 (four float channel buffers held
 *         simultaneously for the inverse Hadamard, plus ~20 MB of
 *         per-thread wavelet scratch across 4 worker threads). All
 *         scratch is freed before return.
 *
 * Time at 5248x3936:
 *   D5503 (Krait 400, 4 cores, post-threading):    ~1.7 s at strength=1.0
 *   D5503 pre-threading (single-threaded scalar):  ~5-7 s
 */
void bayer_nr_uint16(uint16_t* bayer, int width, int height, float strength);

#ifdef __cplusplus
}
#endif

#endif
