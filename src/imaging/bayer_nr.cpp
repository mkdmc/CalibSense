/*
 * bayer_nr.cpp — wavelet-based per-channel noise reduction.
 *
 * Algorithm: 2D Haar wavelet transform on each of the 4 RGGB Bayer
 * channels separately, with BayesShrink soft thresholding of the
 * detail coefficients.
 *
 * Why this preserves detail better than bilateral filtering:
 *
 *   A bilateral filter compares single-pixel values to decide which
 *   neighbors to average. It cannot distinguish noise from
 *   low-amplitude texture — both look like "small variations from
 *   the local mean." Set the range threshold high enough to actually
 *   denoise and you also smooth real low-amplitude detail.
 *
 *   Wavelet thresholding works on a structurally different principle:
 *
 *     - Real signal has structure, so when transformed it concentrates
 *       in a few large coefficients. An edge produces 1-2 large
 *       coefficients per scale at its location; a periodic texture
 *       produces large coefficients at the texture's spatial frequency.
 *
 *     - White noise has no structure, so its coefficients are
 *       distributed uniformly across all positions and scales, with
 *       roughly Gaussian magnitude.
 *
 *     - Soft-thresholding the small coefficients removes most of the
 *       noise while keeping nearly all the signal — including signal
 *       whose AMPLITUDE is at or below the noise level, as long as
 *       it has any spatial structure.
 *
 * Threshold selection — BayesShrink:
 *
 *   For each detail sub-band, the threshold is auto-derived as
 *
 *     T = σ_noise² / σ_signal
 *
 *   where σ_noise is the global noise estimate (from MAD on the
 *   finest HH band) and σ_signal = sqrt(max(0, σ_band² - σ_noise²)).
 *
 *   This adapts: bands with mostly noise (σ_band ≈ σ_noise) get a
 *   large threshold (everything zeroed), bands with strong signal
 *   (σ_band >> σ_noise) get a small threshold (light filtering).
 *   The parameter-free default works well; the user-supplied
 *   `strength` is just a linear multiplier on this auto-derived T.
 */

#include "bayer_nr.h"
#include "threading.h"

#include <algorithm>     // std::nth_element
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


namespace {

constexpr float INV_SQRT2 = 0.7071067811865476f;


// ============================================================================
// 1D Haar transform on a contiguous row
// ============================================================================

// Forward transform. Input: a[0..len-1]. Output replaces it with
// approximations in a[0..len/2-1] followed by details in a[len/2..len-1].
// `tmp` is a scratch buffer of >= len floats.
static void haar_1d_forward(float* a, int len, float* tmp) {
    const int half = len / 2;
    for (int k = 0; k < half; ++k) {
        const float x0 = a[2 * k];
        const float x1 = a[2 * k + 1];
        tmp[k]        = (x0 + x1) * INV_SQRT2;
        tmp[half + k] = (x0 - x1) * INV_SQRT2;
    }
    memcpy(a, tmp, (size_t)len * sizeof(float));
}

// Inverse transform: a[0..len/2-1] = approximations, a[len/2..len-1] = details.
// Output replaces a[] with the reconstructed signal.
static void haar_1d_inverse(float* a, int len, float* tmp) {
    const int half = len / 2;
    for (int k = 0; k < half; ++k) {
        const float s = a[k];
        const float d = a[half + k];
        tmp[2 * k]     = (s + d) * INV_SQRT2;
        tmp[2 * k + 1] = (s - d) * INV_SQRT2;
    }
    memcpy(a, tmp, (size_t)len * sizeof(float));
}


// ============================================================================
// 2D Haar transform on a W×H buffer (one level, in-place)
// ============================================================================
//
// Layout after forward transform:
//
//   +----+----+
//   | LL | HL |    each quadrant W/2 × H/2
//   +----+----+
//   | LH | HH |
//   +----+----+
//
//   LL = low-pass both directions (smooth approximation; left alone)
//   HL = high horizontal, low vertical (vertical edges)
//   LH = low horizontal, high vertical (horizontal edges)
//   HH = high both directions (diagonal detail / noise)

// Stride-aware 2D Haar forward transform.
//
// Operates on the top-left W × H sub-region of a buffer whose row stride is
// `stride`. This is needed for multi-level wavelet decomposition: at level 2
// we transform the LL1 quadrant (size cw/2 × ch/2) while it lives inside the
// larger cw × ch buffer at row stride `cw`.
//
// When stride == W this is identical to a packed 2D transform.
static void haar_2d_forward(float* buf, int W, int H, int stride,
                            float* tmp_row, float* tmp_col)
{
    // Horizontal pass on each row
    for (int y = 0; y < H; ++y) {
        haar_1d_forward(buf + (size_t)y * stride, W, tmp_row);
    }
    // Vertical pass: extract column, transform, write back
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) tmp_col[y] = buf[(size_t)y * stride + x];
        haar_1d_forward(tmp_col, H, tmp_row);
        for (int y = 0; y < H; ++y) buf[(size_t)y * stride + x] = tmp_col[y];
    }
}

static void haar_2d_inverse(float* buf, int W, int H, int stride,
                            float* tmp_row, float* tmp_col)
{
    // Vertical pass first (reverse of forward)
    for (int x = 0; x < W; ++x) {
        for (int y = 0; y < H; ++y) tmp_col[y] = buf[(size_t)y * stride + x];
        haar_1d_inverse(tmp_col, H, tmp_row);
        for (int y = 0; y < H; ++y) buf[(size_t)y * stride + x] = tmp_col[y];
    }
    // Horizontal pass on each row
    for (int y = 0; y < H; ++y) {
        haar_1d_inverse(buf + (size_t)y * stride, W, tmp_row);
    }
}


// ============================================================================
// Noise estimation via MAD on the HH sub-band
// ============================================================================
//
// Median Absolute Deviation is robust to a small number of outliers
// (i.e. genuine sharp diagonal edges in the data). The factor 0.6745
// ≈ Φ⁻¹(0.75) converts MAD to standard deviation under a Gaussian
// noise assumption.
//
// HH is the bottom-right quadrant after a 2D Haar transform.
//
// `abs_scratch` must be at least (W/2) * (H/2) floats. Passing the
// scratch in (rather than malloc'ing internally) lets the caller reuse
// it across multiple invocations and across threads — the original
// internal malloc was a 5 MB allocation in the hot path.
static float estimate_sigma_mad(const float* buf, int W, int H,
                                float* abs_scratch) {
    const int hw = W / 2, hh = H / 2;
    const int n  = hw * hh;
    if (n < 100) return 0.0f;
    if (!abs_scratch) return 0.0f;

    int idx = 0;
    for (int y = hh; y < H; ++y) {
        for (int x = hw; x < W; ++x) {
            abs_scratch[idx++] = fabsf(buf[(size_t)y * W + x]);
        }
    }

    // Median via std::nth_element — O(n) average, no full sort needed.
    std::nth_element(abs_scratch, abs_scratch + n / 2, abs_scratch + n);
    const float median = abs_scratch[n / 2];

    return median / 0.6745f;
}


// ============================================================================
// BayesShrink soft thresholding on an axis-aligned sub-band
// ============================================================================

static inline float soft_thresh(float x, float T) {
    if (x >  T) return x - T;
    if (x < -T) return x + T;
    return 0.0f;
}

// Standard deviation of coefficients in a rectangular sub-band.
static float band_stddev(const float* buf, int W,
                         int x0, int y0, int bw, int bh)
{
    const int n = bw * bh;
    if (n < 1) return 0.0f;
    double s = 0.0, s2 = 0.0;
    for (int y = y0; y < y0 + bh; ++y) {
        for (int x = x0; x < x0 + bw; ++x) {
            const double v = (double)buf[(size_t)y * W + x];
            s  += v;
            s2 += v * v;
        }
    }
    const double mean = s / n;
    const double var  = s2 / n - mean * mean;
    return var > 0.0 ? (float)sqrt(var) : 0.0f;
}

// Apply BayesShrink soft threshold to one sub-band, scaled by strength.
static void bayes_threshold_band(float* buf, int W,
                                 int x0, int y0, int bw, int bh,
                                 float sigma_noise, float strength)
{
    if (strength <= 0.0f || sigma_noise <= 0.0f) return;

    const float sigma_band = band_stddev(buf, W, x0, y0, bw, bh);
    const float diff = sigma_band * sigma_band - sigma_noise * sigma_noise;

    if (diff <= 0.0f) {
        // σ_signal² ≤ 0 means no detectable signal above the noise floor —
        // zero the entire band. (This is the BayesShrink limit case.)
        for (int y = y0; y < y0 + bh; ++y) {
            float* row = buf + (size_t)y * W + x0;
            memset(row, 0, (size_t)bw * sizeof(float));
        }
        return;
    }

    const float sigma_signal = sqrtf(diff);
    const float T = (sigma_noise * sigma_noise / sigma_signal) * strength;

    for (int y = y0; y < y0 + bh; ++y) {
        for (int x = x0; x < x0 + bw; ++x) {
            buf[(size_t)y * W + x] = soft_thresh(buf[(size_t)y * W + x], T);
        }
    }
}


// ============================================================================
// Per-channel processing — 3-level wavelet decomposition
// ============================================================================
//
// Level 1: forward Haar on entire cw × ch sub-image. Produces LL1 in the
//          top-left cw/2 × ch/2 quadrant; details HL1, LH1, HH1 fill the
//          other three quadrants.
//
// Level 2: forward Haar on the LL1 quadrant only (cw/2 × ch/2). Produces
//          LL2 in the top-left cw/4 × ch/4 of the buffer; details HL2,
//          LH2, HH2 fill the rest of the LL1 region (still inside the
//          original cw × ch buffer, but with stride still = cw).
//
// Level 3: forward Haar on the LL2 quadrant (cw/4 × ch/4). Produces LL3
//          and details HL3, LH3, HH3.
//
// Frequency bands captured:
//   Level 1 details: 1-2 px structure (highest frequency)
//   Level 2 details: 2-4 px structure
//   Level 3 details: 4-8 px structure
//
// Sigma is estimated ONCE on level-1's HH band (the standard approach).
// Why not per-level: at coarser scales the HH band contains both noise
// AND aliased real signal that didn't get smoothed by the level-1
// decomposition. Estimating sigma there contaminates the noise estimate
// with signal energy and over-thresholds the band. Donoho-Johnstone
// recommends always estimating noise from the finest-scale HH.
//
// The orthonormal Haar transform preserves noise variance across scales,
// so the level-1 sigma applies directly to all coarser levels without
// rescaling.
//
// Note on stride vs W: `bayer_stride` is the row stride of the caller's
// uint16_t buffer (= the original image width). `W` and `H` are the
// largest 4-aligned dimensions we actually process. They differ when
// the caller's `width` is not a multiple of 4 — the trailing 1-3
// columns are skipped. Indexing into the caller's buffer ALWAYS uses
// `bayer_stride`; using `W` would cause a slow diagonal skew across
// the image when width is not 4-aligned.

// Apply 3-level wavelet denoise (BayesShrink) in place on a packed
// W × H float buffer. Used as the inner kernel for both per-channel
// and Hadamard-domain processing.
//
// `tmp_row_scratch` must hold max(W, H) floats; `tmp_col_scratch` holds H
// floats; `mad_scratch` holds (W/2) * (H/2) floats (used by the sigma
// estimator). All three are caller-allocated so multiple threads can
// process channels in parallel without allocator contention.
//
// Returns silently on OOM (caller's buffer left as-is).
static void wavelet_denoise_inplace(float* buf, int W, int H, float strength,
                                    float* tmp_row_scratch,
                                    float* tmp_col_scratch,
                                    float* mad_scratch)
{
    if (W < 4 || H < 4) return;

    // Determine max level count that fits.
    int levels = 1;
    if ((W % 4) == 0 && (H % 4) == 0 && (W / 4) >= 8 && (H / 4) >= 8) levels = 2;
    if ((W % 8) == 0 && (H % 8) == 0 && (W / 8) >= 8 && (H / 8) >= 8) levels = 3;

    // Forward: cascade L levels of Haar.
    int Lw = W, Lh = H;
    for (int lvl = 0; lvl < levels; ++lvl) {
        haar_2d_forward(buf, Lw, Lh, /*stride=*/W,
                        tmp_row_scratch, tmp_col_scratch);
        Lw /= 2;
        Lh /= 2;
    }

    // Estimate noise sigma from level-1's HH band only.
    const float sigma = estimate_sigma_mad(buf, W, H, mad_scratch);

    // Threshold detail bands at every level.
    Lw = W; Lh = H;
    for (int lvl = 0; lvl < levels; ++lvl) {
        const int hw = Lw / 2, hh = Lh / 2;
        bayes_threshold_band(buf, /*stride=*/W, hw,  0, hw, hh, sigma, strength);
        bayes_threshold_band(buf, /*stride=*/W,  0, hh, hw, hh, sigma, strength);
        bayes_threshold_band(buf, /*stride=*/W, hw, hh, hw, hh, sigma, strength);
        Lw = hw;
        Lh = hh;
    }

    // Inverse: reverse order.
    for (int lvl = 0; lvl < levels; ++lvl) {
        Lw *= 2;
        Lh *= 2;
        haar_2d_inverse(buf, Lw, Lh, /*stride=*/W,
                        tmp_row_scratch, tmp_col_scratch);
    }
}

} // namespace


// ============================================================================
// Public entry point — Hadamard-domain wavelet noise reduction
// ============================================================================
//
// For each 2x2 RGGB block in the Bayer plane, apply a 4-channel
// orthonormal Hadamard transform:
//
//   Y  = (R + G1 + G2 + B) / 2     -- pseudo-luma  (DC, where signal lives)
//   CH = (R - G1 + G2 - B) / 2     -- horizontal contrast
//   CV = (R + G1 - G2 - B) / 2     -- vertical contrast
//   CD = (R - G1 - G2 + B) / 2     -- diagonal contrast (mostly noise)
//
// This decomposition exploits the inter-channel correlation of real
// scene content: a real edge or texture causes correlated changes in
// R, G1, G2, B in the same 2x2 block, so its energy concentrates into
// Y. Per-pixel sensor noise is uncorrelated across channels, so it
// spreads roughly equally into all four bands.
//
// Concretely, in a flat region with no real signal:
//   Y noise variance  = sigma^2  (signal-floor in the SAME flat region)
//   CH variance       = sigma^2  (essentially pure noise — no signal)
//   CV variance       = sigma^2  (same)
//   CD variance       = sigma^2  (same)
// But Y *also* contains all the real low-frequency scene content
// (which can be huge), so its signal-to-noise ratio is much higher
// than the C channels'. BayesShrink's per-band auto-threshold then:
//   - thresholds Y lightly (high signal-to-noise; preserve signal)
//   - thresholds CH/CV/CD aggressively (low signal-to-noise; mostly noise)
//
// The CD band in particular contains almost no real signal for most
// scenes — diagonal correlations of opposite sign across a 2x2 block
// only arise from chroma transitions, not from luma edges. So CD is
// nearly pure chroma-noise channel and gets near-total suppression.
//
// Orthonormality of the Hadamard matrix means: noise variance is
// preserved exactly (no amplification, no rebalancing needed), and
// the inverse transform is the same matrix applied again — exact
// round-trip.
//
// PRECONDITION: caller has already applied black-level subtract and
// per-channel WB to the Bayer plane. Without that, R/G1/G2/B have
// systematic gain offsets and the Hadamard decomposition becomes
// biased ("Y" becomes G-dominated and biased toward the illuminant,
// real edges leak into CH/CV/CD). See demosaic.cpp for the upstream
// black/WB pass.

extern "C" void bayer_nr_uint16(uint16_t* bayer, int width, int height, float strength) {
    if (!bayer || strength <= 0.0f) return;
    if (width < 16 || height < 16) return;

    const int W = width  & ~3;
    const int H = height & ~3;
    if (W < 16 || H < 16) return;

    // 2x2 block grid: cw × ch blocks.
    const int cw = W / 2, ch = H / 2;
    const size_t N = (size_t)cw * (size_t)ch;

    // Allocate the four Hadamard channels (kept simultaneously since
    // the inverse needs all four). 4 × cw × ch × sizeof(float).
    float* Y_buf  = (float*)malloc(N * sizeof(float));
    float* CH_buf = (float*)malloc(N * sizeof(float));
    float* CV_buf = (float*)malloc(N * sizeof(float));
    float* CD_buf = (float*)malloc(N * sizeof(float));
    if (!Y_buf || !CH_buf || !CV_buf || !CD_buf) {
        free(Y_buf); free(CH_buf); free(CV_buf); free(CD_buf);
        return;
    }

    // ---- Forward Hadamard (parallel by row bands) ----
    //
    // Each 2x2 block at (i, j) reads only from its own four Bayer
    // pixels and writes only its own slot in Y/CH/CV/CD. Different
    // rows of blocks (different j values) are completely independent.
    {
        struct fwd_ctx_t {
            const uint16_t *bayer;
            int             width;
            int             cw;
            float          *Y, *CH, *CV, *CD;
        };
        fwd_ctx_t fwd{ bayer, width, cw, Y_buf, CH_buf, CV_buf, CD_buf };
        auto fwd_worker = [](int j0, int j1, void *cv) {
            const fwd_ctx_t *c = (const fwd_ctx_t*)cv;
            const int width = c->width;
            const int cw    = c->cw;
            for (int j = j0; j < j1; ++j) {
                for (int i = 0; i < cw; ++i) {
                    const float R  = (float)c->bayer[(size_t)(2 * j    ) * width + (2 * i    )];
                    const float G1 = (float)c->bayer[(size_t)(2 * j    ) * width + (2 * i + 1)];
                    const float G2 = (float)c->bayer[(size_t)(2 * j + 1) * width + (2 * i    )];
                    const float B  = (float)c->bayer[(size_t)(2 * j + 1) * width + (2 * i + 1)];
                    const size_t idx = (size_t)j * cw + i;
                    c->Y [idx] = (R + G1 + G2 + B) * 0.5f;
                    c->CH[idx] = (R - G1 + G2 - B) * 0.5f;
                    c->CV[idx] = (R + G1 - G2 - B) * 0.5f;
                    c->CD[idx] = (R - G1 - G2 + B) * 0.5f;
                }
            }
        };
        parallel_rows(0, ch, /*auto*/ 0, +fwd_worker, &fwd);
    }

    // ---- Wavelet denoise each Hadamard channel — PARALLEL across the 4 channels ----
    //
    // The four channels are completely independent (each has its own
    // float buffer; thresholds are derived from the channel's own HH
    // band). On a 4-core Krait 400 this gives a near-4× speedup on what
    // was previously the dominant single-threaded cost in the entire
    // pipeline (~5–7 s on device pre-threading).
    //
    // Per-thread scratch is allocated inside the worker rather than
    // pre-built and passed in; the alternative would be stuffing four
    // sets of buffers into the context struct. This keeps the lifetime
    // tied to the worker's stack frame (auto-freed on return) and the
    // allocations are 4 × ~5 MB which is fine.
    //
    // Channel-specific strength multipliers exploit the very different
    // signal-to-noise profiles of the four Hadamard channels:
    //
    //   Y  carries real luma — edges, textures, gradients. Real signal
    //      energy here is huge. Light denoising preserves detail.
    //   CH carries within-block horizontal contrast. Some real signal
    //      at horizontally-aligned chromatic edges, but mostly noise.
    //   CV same as CH but vertical. Same statistics.
    //   CD carries within-block diagonal "checkerboard" contrast.
    //      Real scenes essentially never have natural color content
    //      at this 2x2 frequency, so it's ~99% pure demosaic/sensor
    //      noise. Aggressive thresholding removes it without touching
    //      real detail.
    //
    // The multipliers tell BayesShrink "be more aggressive in this
    // channel" — they inject prior knowledge about per-channel signal
    // concentration that BayesShrink (which only sees one channel at
    // a time) can't discover on its own.
    constexpr float Y_STRENGTH_MULT  = 1.0f;
    constexpr float CH_STRENGTH_MULT = 2.5f;
    constexpr float CV_STRENGTH_MULT = 2.5f;
    constexpr float CD_STRENGTH_MULT = 5.0f;

    {
        struct den_ctx_t {
            float *bufs[4];
            float  strengths[4];
            int    cw, ch;
        };
        den_ctx_t den{
            { Y_buf,  CH_buf,  CV_buf,  CD_buf },
            { strength * Y_STRENGTH_MULT,
              strength * CH_STRENGTH_MULT,
              strength * CV_STRENGTH_MULT,
              strength * CD_STRENGTH_MULT },
            cw, ch
        };
        auto den_worker = [](int c0, int c1, void *cv) {
            const den_ctx_t *c = (const den_ctx_t*)cv;
            const int cw      = c->cw;
            const int ch      = c->ch;
            const int max_dim = (cw > ch ? cw : ch);
            const size_t mad_n = (size_t)(cw / 2) * (size_t)(ch / 2);
            // Per-thread scratch.
            float *tmp_row = (float*)malloc((size_t)max_dim * sizeof(float));
            float *tmp_col = (float*)malloc((size_t)ch * sizeof(float));
            float *mad     = (float*)malloc(mad_n * sizeof(float));
            if (!tmp_row || !tmp_col || !mad) {
                free(tmp_row); free(tmp_col); free(mad);
                return;
            }
            for (int k = c0; k < c1; ++k) {
                wavelet_denoise_inplace(c->bufs[k], cw, ch, c->strengths[k],
                                        tmp_row, tmp_col, mad);
            }
            free(mad); free(tmp_col); free(tmp_row);
        };
        // Dispatch one thread per channel (up to 4).  parallel_rows
        // splits [0, 4) across however many threads it chooses, so on
        // a 4-core box each channel runs concurrently.
        parallel_rows(0, 4, /*auto*/ 0, +den_worker, &den);
    }

    // ---- Inverse Hadamard (parallel by row bands) ----
    //
    // The inverse of an orthonormal Hadamard matrix is itself (it's its
    // own inverse up to a transpose, but the matrix is symmetric):
    //   R  = (Y + CH + CV + CD) / 2
    //   G1 = (Y - CH + CV - CD) / 2
    //   G2 = (Y + CH - CV - CD) / 2
    //   B  = (Y - CH - CV + CD) / 2
    //
    // Clamp to [0, 65535] (uint16_t range). Post-WB Bayer values can
    // legitimately be much larger than 1023 (since gB ~2 pushes
    // saturated B up to ~2000 or higher in 10-bit raw scaled to
    // 16-bit storage), so we don't clamp at 1023.
    {
        struct inv_ctx_t {
            uint16_t *bayer;
            int       width;
            int       cw;
            const float *Y, *CH, *CV, *CD;
        };
        inv_ctx_t inv{ bayer, width, cw, Y_buf, CH_buf, CV_buf, CD_buf };
        auto inv_worker = [](int j0, int j1, void *cv) {
            const inv_ctx_t *c = (const inv_ctx_t*)cv;
            const int width = c->width;
            const int cw    = c->cw;
            for (int j = j0; j < j1; ++j) {
                for (int i = 0; i < cw; ++i) {
                    const size_t idx = (size_t)j * cw + i;
                    const float y = c->Y [idx];
                    const float h = c->CH[idx];
                    const float v = c->CV[idx];
                    const float d = c->CD[idx];
                    float R  = (y + h + v + d) * 0.5f;
                    float G1 = (y - h + v - d) * 0.5f;
                    float G2 = (y + h - v - d) * 0.5f;
                    float B  = (y - h - v + d) * 0.5f;
                    if (R  < 0.0f) R  = 0.0f; else if (R  > 65535.0f) R  = 65535.0f;
                    if (G1 < 0.0f) G1 = 0.0f; else if (G1 > 65535.0f) G1 = 65535.0f;
                    if (G2 < 0.0f) G2 = 0.0f; else if (G2 > 65535.0f) G2 = 65535.0f;
                    if (B  < 0.0f) B  = 0.0f; else if (B  > 65535.0f) B  = 65535.0f;
                    c->bayer[(size_t)(2 * j    ) * width + (2 * i    )] = (uint16_t)(R  + 0.5f);
                    c->bayer[(size_t)(2 * j    ) * width + (2 * i + 1)] = (uint16_t)(G1 + 0.5f);
                    c->bayer[(size_t)(2 * j + 1) * width + (2 * i    )] = (uint16_t)(G2 + 0.5f);
                    c->bayer[(size_t)(2 * j + 1) * width + (2 * i + 1)] = (uint16_t)(B  + 0.5f);
                }
            }
        };
        parallel_rows(0, ch, /*auto*/ 0, +inv_worker, &inv);
    }

    free(CD_buf); free(CV_buf); free(CH_buf); free(Y_buf);
}
