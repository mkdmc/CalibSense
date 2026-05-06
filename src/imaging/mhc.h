/*
 * mhc.h — Malvar–He–Cutler 5×5 demosaic kernels, with directional
 *         (Hamilton-Adams) per-pixel adaptation.
 *
 * HEADER-ONLY MODULE — do NOT include from more than one TU.
 *
 * Currently included only by demosaic.cpp. Everything below has
 * internal linkage (`static`), so each TU that includes this header
 * gets its own private copy of the kernel data and code. The
 * `mhc_at` function is the per-pixel inner-loop entry point and
 * MUST inline at the call site for acceptable performance on the
 * Krait 400 — that's the reason the implementation lives here in
 * the header rather than in a separate translation unit.
 *
 * Public entry points used by demosaic.cpp:
 *   bilinear_at — bilinear demosaic for border pixels (used in the
 *                 2-pixel halo around the image where mhc_at can't
 *                 reach all of its 5x5 neighbors).
 *   mhc_at      — directional MHC demosaic for interior pixels.
 *
 * Internal helpers (kernel_5x5, the K_* kernel arrays) are not
 * intended for outside use; they're exposed at file scope only
 * because mhc_at is `static inline` and needs to see them.
 */

#ifndef CALIBSENSE_IMAGING_MHC_H
#define CALIBSENSE_IMAGING_MHC_H

#include <stdint.h>

// ---------- Malvar-He-Cutler 5×5 kernels (denominator 16) ----------

static const int8_t K_G_AT_RB[25] = {
     0,  0, -2,  0,  0,
     0,  0,  4,  0,  0,
    -2,  4,  8,  4, -2,
     0,  0,  4,  0,  0,
     0,  0, -2,  0,  0,
};

// Color at G with same-row neighbors (e.g. R at G in R-row, B at G in B-row).
static const int8_t K_HORIZ[25] = {
     0,  0,  1,  0,  0,
     0, -2,  0, -2,  0,
    -2,  8, 10,  8, -2,
     0, -2,  0, -2,  0,
     0,  0,  1,  0,  0,
};

// Color at G with same-column neighbors (e.g. R at G in B-row, B at G in R-row).
static const int8_t K_VERT[25] = {
     0,  0, -2,  0,  0,
     0, -2,  8, -2,  0,
     1,  0, 10,  0,  1,
     0, -2,  8, -2,  0,
     0,  0, -2,  0,  0,
};

// Color at the opposite corner (R at B, B at R).
static const int8_t K_DIAG[25] = {
     0,  0, -3,  0,  0,
     0,  4,  0,  4,  0,
    -3,  0, 12,  0, -3,
     0,  4,  0,  4,  0,
     0,  0, -3,  0,  0,
};

// ---------- Directional MHC kernels (Hamilton-Adams) ----------
//
// For each parity case, we have a horizontal-only and a vertical-only
// kernel. These are derived from the standard Hamilton-Adams formula:
//
//   At R/B pixel, G_horiz = (G_l + G_r)/2 + (2*C_c - C_l2 - C_r2)/4
//   At R/B pixel, G_vert  = (G_u + G_d)/2 + (2*C_c - C_u2 - C_d2)/4
//
//   At G pixel, C_horiz = (C_l + C_r)/2 + (2*G_c - G_l2 - G_r2)/4
//   At G pixel, C_vert  = (C_u + C_d)/2 + (2*G_c - G_u2 - G_d2)/4
//
// In denominator-16 form, the coefficients become:
//   center same-color : 8
//   axial same-color  : 8 each (only along chosen axis)
//   distance-2 same   : -4 each (only along chosen axis, correction term)
//
// All sum to 16, giving unity DC gain.
//
// Note: the symmetric K_G_AT_RB is essentially (K_G_AT_RB_HORIZ + K_G_AT_RB_VERT) / 2
// up to coefficient rounding, which is why the symmetric MHC is a reasonable
// fallback for the "no clear direction" case.

static const int8_t K_G_AT_RB_HORIZ[25] = {
     0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,
    -4,  8,  8,  8, -4,
     0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,
};

static const int8_t K_G_AT_RB_VERT[25] = {
     0,  0, -4,  0,  0,
     0,  0,  8,  0,  0,
     0,  0,  8,  0,  0,
     0,  0,  8,  0,  0,
     0,  0, -4,  0,  0,
};

// For R-at-G in R-row (parity 1) and B-at-G in B-row (parity 2),
// the natural neighbors lie along the row (horizontal). The
// "directional" choice here is whether to use the existing horizontal
// neighbors (always available) or fall back if a vertical edge crosses
// through. Symmetric MHC's K_HORIZ already uses horizontal R neighbors;
// the directional version drops the perpendicular correction terms.
static const int8_t K_RB_AT_G_HORIZ[25] = {
     0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,
    -4,  8,  8,  8, -4,
     0,  0,  0,  0,  0,
     0,  0,  0,  0,  0,
};

static const int8_t K_RB_AT_G_VERT[25] = {
     0,  0, -4,  0,  0,
     0,  0,  8,  0,  0,
     0,  0,  8,  0,  0,
     0,  0,  8,  0,  0,
     0,  0, -4,  0,  0,
};

// Note: K_RB_AT_G_HORIZ and K_G_AT_RB_HORIZ end up structurally identical —
// both are "5-tap horizontal with distance-2 correction." The same is true
// for the vertical variants. We keep them as separate names for clarity
// at the call site.

// Apply 5×5 kernel centered on `c` with row stride `stride` (in uint16_t).
// Returns the unscaled sum; caller divides by 16.
static inline int32_t kernel_5x5(const uint16_t *c, int stride, const int8_t *k) {
    int32_t s = 0;
    for (int ky = -2; ky <= 2; ++ky) {
        const uint16_t *row = c + ky * stride;
        const int8_t   *kr  = &k[(ky + 2) * 5];
        s += (int32_t)row[-2] * kr[0]
           + (int32_t)row[-1] * kr[1]
           + (int32_t)row[ 0] * kr[2]
           + (int32_t)row[ 1] * kr[3]
           + (int32_t)row[ 2] * kr[4];
    }
    return s;
}

// Bilinear demosaic for a single border pixel. `src` is the full Bayer plane.
// `stride` is row stride in uint16_t.
static void bilinear_at(const uint16_t *src, int width, int height, int stride,
                        int x, int y,
                        int32_t *outR, int32_t *outG, int32_t *outB)
{
    auto get = [&](int xx, int yy) -> int32_t {
        if (xx < 0) xx = 0; else if (xx >= width)  xx = width  - 1;
        if (yy < 0) yy = 0; else if (yy >= height) yy = height - 1;
        return src[yy * stride + xx];
    };

    int parity = ((y & 1) << 1) | (x & 1);
    int32_t v  = get(x, y);
    int32_t up = get(x,   y-1), dn = get(x,   y+1);
    int32_t lf = get(x-1, y  ), rt = get(x+1, y  );
    int32_t ul = get(x-1, y-1), ur = get(x+1, y-1);
    int32_t dl = get(x-1, y+1), dr = get(x+1, y+1);

    int32_t r = 0, g = 0, b = 0;
    switch (parity) {
        case 0: r = v;
                g = (up + dn + lf + rt) >> 2;
                b = (ul + ur + dl + dr) >> 2;
                break;
        case 1: r = (lf + rt) >> 1; g = v; b = (up + dn) >> 1; break;
        case 2: r = (up + dn) >> 1; g = v; b = (lf + rt) >> 1; break;
        case 3: r = (ul + ur + dl + dr) >> 2;
                g = (up + dn + lf + rt) >> 2;
                b = v;
                break;
    }
    *outR = r; *outG = g; *outB = b;
}

// MHC demosaic for an interior pixel (2 ≤ x < W-2, 2 ≤ y < H-2).
//
// Directional Hamilton-Adams variant: at each pixel we estimate the
// local edge orientation and pick a horizontal-only or vertical-only
// kernel for the G interpolation (at R/B pixels) and the off-axis
// color reconstruction (at G pixels). This avoids the zipper /
// color-fringe artifacts that occur when the symmetric MHC kernel
// averages neighbors from across an edge.
//
// Direction selection:
//   At each pixel we compute horizontal and vertical gradients using
//   the immediately-adjacent G samples (sharp edge detector) plus the
//   distance-2 same-color samples (noise-robust signal). If either
//   direction has a substantially smaller gradient, that's the edge-
//   parallel direction and we use the corresponding directional kernel.
//   If the gradients are comparable (no clear edge), we fall back to
//   the symmetric MHC kernel — the safer choice in flat or texture-
//   dominated regions where a directional choice would amplify noise.
//
// The diagonal kernel (K_DIAG, used for R-at-B and B-at-R) is left
// unchanged. Diagonal-edge zippering is less visually problematic
// than horizontal/vertical zippering, and making it adaptive
// requires a more complex multi-pass approach.
static inline void mhc_at(const uint16_t *src, int stride,
                          int x, int y,
                          int32_t *outR, int32_t *outG, int32_t *outB)
{
    const uint16_t *c = &src[y * stride + x];
    int parity = ((y & 1) << 1) | (x & 1);
    int32_t v = (int32_t)*c;
    int32_t r, g, b;

    // Compute horizontal and vertical gradients. Same form regardless
    // of parity: the immediate cardinal neighbors are always G samples
    // for parity 0 and 3 (R/B center), and same-color non-G samples for
    // parity 1 and 2 (G center). The distance-2 neighbors are always
    // same-color as center. Both forms produce a valid edge estimator,
    // because a real edge changes any color channel along its gradient.
    const int32_t up   = (int32_t)c[-stride];
    const int32_t dn   = (int32_t)c[+stride];
    const int32_t lf   = (int32_t)c[-1];
    const int32_t rt   = (int32_t)c[+1];
    const int32_t up2  = (int32_t)c[-2*stride];
    const int32_t dn2  = (int32_t)c[+2*stride];
    const int32_t lf2  = (int32_t)c[-2];
    const int32_t rt2  = (int32_t)c[+2];

    // Horizontal gradient: variation along the row
    //   |G_l - G_r| at distance 1 + |C_l2 - C_r2| at distance 2
    // Vertical: same form, perpendicular axis
    const int32_t hgrad = (lf  > rt  ? lf  - rt  : rt  - lf )
                        + (lf2 > rt2 ? lf2 - rt2 : rt2 - lf2);
    const int32_t vgrad = (up  > dn  ? up  - dn  : dn  - up )
                        + (up2 > dn2 ? up2 - dn2 : dn2 - up2);

    // Direction-decision threshold. If |hgrad - vgrad| < threshold the
    // edge orientation isn't clear enough to commit to a direction;
    // we use the symmetric MHC kernel. The threshold is in the same
    // units as the Bayer sample sum used for the gradient (4 samples
    // contribute to each gradient, so values in [0, 4*max_sample]).
    //
    // 64 in post-WB Bayer space is roughly 3% of the typical highlight
    // value — comfortably above sensor noise variation but well below
    // any real edge. Tuning lower (e.g. 32) makes the algorithm more
    // aggressive about picking a direction; higher (128) is more
    // conservative. 64 is a middle-of-the-road default.
    constexpr int32_t DIR_THRESHOLD = 64;
    const int32_t grad_diff = hgrad > vgrad ? hgrad - vgrad : vgrad - hgrad;
    const bool use_directional = grad_diff > DIR_THRESHOLD;
    const bool prefer_horiz = hgrad < vgrad;  // smaller gradient = edge-parallel

    switch (parity) {
        case 0: { // R pixel: known R, interpolate G and B
            r = v;
            // G: pick directional kernel based on local edge orientation
            const int8_t* kg = use_directional
                ? (prefer_horiz ? K_G_AT_RB_HORIZ : K_G_AT_RB_VERT)
                : K_G_AT_RB;
            g = kernel_5x5(c, stride, kg) >> 4;
            // B: diagonal neighbors only — keep symmetric K_DIAG
            b = kernel_5x5(c, stride, K_DIAG) >> 4;
            break;
        }
        case 1: { // G in R-row: R is in same row, B is in same column
            // R interpolation is naturally horizontal; the symmetric
            // K_HORIZ already does horizontal R interpolation but with
            // perpendicular correction terms. Drop those when we have
            // confidence in a horizontal edge.
            const int8_t* kr = use_directional && prefer_horiz
                ? K_RB_AT_G_HORIZ : K_HORIZ;
            // B is vertical: similar logic but for the perpendicular axis.
            const int8_t* kb = use_directional && !prefer_horiz
                ? K_RB_AT_G_VERT : K_VERT;
            r = kernel_5x5(c, stride, kr) >> 4;
            g = v;
            b = kernel_5x5(c, stride, kb) >> 4;
            break;
        }
        case 2: { // G in B-row: R is vertical, B is horizontal
            const int8_t* kr = use_directional && !prefer_horiz
                ? K_RB_AT_G_VERT : K_VERT;
            const int8_t* kb = use_directional && prefer_horiz
                ? K_RB_AT_G_HORIZ : K_HORIZ;
            r = kernel_5x5(c, stride, kr) >> 4;
            g = v;
            b = kernel_5x5(c, stride, kb) >> 4;
            break;
        }
        default: { // case 3: B pixel: known B, interpolate G and R
            const int8_t* kg = use_directional
                ? (prefer_horiz ? K_G_AT_RB_HORIZ : K_G_AT_RB_VERT)
                : K_G_AT_RB;
            r = kernel_5x5(c, stride, K_DIAG) >> 4;
            g = kernel_5x5(c, stride, kg)     >> 4;
            b = v;
            break;
        }
    }

    if (r < 0) r = 0;
    if (g < 0) g = 0;
    if (b < 0) b = 0;
    *outR = r; *outG = g; *outB = b;
}

#endif  // CALIBSENSE_IMAGING_MHC_H
