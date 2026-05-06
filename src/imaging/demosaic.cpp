/*
 * demosaic.cpp — Malvar-He-Cutler RGGB demosaic + linear sRGB color pipeline.
 *
 * Layout of the input Bayer plane (RGGB, row 0 col 0 = R):
 *
 *   R G R G R G ...
 *   G B G B G B ...
 *   R G R G R G ...
 *   G B G B G B ...
 *
 * Per-pixel parity table:
 *   (y%2, x%2) = (0,0): R location (need G, B)
 *   (y%2, x%2) = (0,1): G in R-row (need R via horizontal kernel, B via vertical)
 *   (y%2, x%2) = (1,0): G in B-row (need R via vertical kernel, B via horizontal)
 *   (y%2, x%2) = (1,1): B location (need G, R)
 *
 * Malvar-He-Cutler kernels (denominator 16, all integer):
 *
 *   K_G_AT_RB  — green at red or blue location
 *   K_HORIZ    — color whose same-row pattern has neighbors at columns ±1
 *   K_VERT     — color whose same-column pattern has neighbors at rows ±1
 *   K_DIAG     — color at the "opposite" corner (R at B, B at R)
 */

#include "demosaic.h"
#include "bayer_nr.h"
#include "lsc.h"
#include "mhc.h"
#include "threading.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define HAVE_NEON 1
#else
#  define HAVE_NEON 0
#endif
#include <vector>


namespace {

// Malvar-He-Cutler kernels + directional (Hamilton-Adams) demosaic
// helpers live in mhc.h. That header is included OUTSIDE this anonymous
// namespace block — see the #include at the top of this file.


// ---------- packed-Bayer unpack (matches pipeline.cpp's format) ----------
//
// 6 × 10-bit pixels per 64-bit LE word, words_per_row = (width+5)/6,
// trailing pixels in the last word are padding.

static int unpack_bayer(uint16_t *dst, const uint8_t *packed,
                        int width, int height, size_t words_per_row)
{
    size_t bytes_per_row = words_per_row * 8;
    for (int r = 0; r < height; ++r) {
        const uint8_t *src_row = packed + (size_t)r * bytes_per_row;
        uint16_t      *out_row = dst + (size_t)r * width;
        size_t o = 0;
        for (size_t w = 0; w < words_per_row; ++w) {
            uint64_t word =
                  (uint64_t)src_row[w*8 + 0]        | ((uint64_t)src_row[w*8 + 1] <<  8)
                | ((uint64_t)src_row[w*8 + 2] << 16) | ((uint64_t)src_row[w*8 + 3] << 24)
                | ((uint64_t)src_row[w*8 + 4] << 32) | ((uint64_t)src_row[w*8 + 5] << 40)
                | ((uint64_t)src_row[w*8 + 6] << 48) | ((uint64_t)src_row[w*8 + 7] << 56);
            uint16_t pix[6] = {
                (uint16_t)((word >>  0) & 0x3FF),
                (uint16_t)((word >> 10) & 0x3FF),
                (uint16_t)((word >> 20) & 0x3FF),
                (uint16_t)((word >> 30) & 0x3FF),
                (uint16_t)((word >> 40) & 0x3FF),
                (uint16_t)((word >> 50) & 0x3FF),
            };
            for (int i = 0; i < 6 && o < (size_t)width; ++i)
                out_row[o++] = pix[i];
        }
    }
    return 0;
}

// ---------- gamma 2.2 + tone-curve LUT (14-bit linear → 8-bit gamma) ----
//
// In addition to gamma 2.2, this LUT bakes in an S-curve in linear space
// to approximate photographic contrast. Without this curve the output
// looks flat and "milky" compared to typical camera JPEGs.
//
// The curve is a smoothstep blend at strength `tone_contrast`:
//
//   smoothstep(x) = 3x² - 2x³
//   toned(x)      = x * (1 - k) + smoothstep(x) * k
//
// k = 0.0 → pure linear (no extra contrast)
// k = 0.6 → moderate (default — clearly more contrasty than linear)
// k = 1.0 → full smoothstep (very strong S-curve, can crush midtones)
//
// The LUT is keyed on tone_contrast and rebuilt when that value changes.
// Build is ~16k powf calls — microseconds, not worth caching across
// runs but trivially cached within a run.

static uint8_t  g_gamma_lut[16384];
static float    g_lut_tone_contrast = -1.0f;  // sentinel forces first build

static void build_gamma_lut(float tone_contrast) {
    if (g_lut_tone_contrast == tone_contrast) return;

    for (int i = 0; i < 16384; ++i) {
        float lin = (float)i / 16383.0f;

        // Linear-space S-curve (smoothstep blend).
        float ss    = lin * lin * (3.0f - 2.0f * lin);
        float toned = lin * (1.0f - tone_contrast) + ss * tone_contrast;
        if (toned < 0.0f) toned = 0.0f;
        if (toned > 1.0f) toned = 1.0f;

        // Gamma 2.2 encode.
        float v = powf(toned, 1.0f / 2.2f) * 255.0f + 0.5f;
        if (v < 0.0f)   v = 0.0f;
        if (v > 255.0f) v = 255.0f;
        g_gamma_lut[i] = (uint8_t)v;
    }
    g_lut_tone_contrast = tone_contrast;
}

} // namespace

extern "C" int demosaic_to_srgb_ex(
    const void  *raw_packed,
    size_t       raw_size,
    int          width,
    int          height,
    int          black_level,
    int          white_level,
    uint32_t     r_num, uint32_t r_den,
    uint32_t     g_num, uint32_t g_den,
    uint32_t     b_num, uint32_t b_den,
    const float *cam_to_srgb,
    float        bayer_nr_strength,
    float        black_point_lift,
    float        saturation,
    float        tone_contrast,
    float        exposure,
    const void  *lsc_table,
    float        lsc_strength,
    uint8_t     *rgb_out)
{
    if (!raw_packed || !rgb_out || !cam_to_srgb) return 1;
    if (width  <= 4 || height <= 4)              return 1;
    if (white_level <= 0)                        return 1;

    size_t words_per_row = ((size_t)width + 5) / 6;
    size_t expected_raw  = (size_t)height * words_per_row * 8;
    if (raw_size < expected_raw)                 return 1;


    // Unpack Bayer plane to a contiguous 16-bit buffer.
    // 5248 × 3936 × 2 = 41.3 MB. Same order as calibsense_process_frame allocates.
    size_t bayer_pixels = (size_t)width * (size_t)height;
    uint16_t *bayer = (uint16_t*)malloc(bayer_pixels * sizeof(uint16_t));
    if (!bayer) {
        return 2;
    }

    if (unpack_bayer(bayer, (const uint8_t*)raw_packed, width, height, words_per_row) != 0) {
        free(bayer);
        return 3;
    }

    // WB gains relative to G. Compute as float once.
    // gain_R = (r_num/r_den) / (g_num/g_den) = (r_num * g_den) / (r_den * g_num)
    double gR = ((double)r_num * (double)g_den) / ((double)r_den * (double)g_num);
    double gG = 1.0;
    double gB = ((double)b_num * (double)g_den) / ((double)b_den * (double)g_num);
    if (!(gR > 0.0)) gR = 1.0;
    if (!(gB > 0.0)) gB = 1.0;

    // ============================================================
    // LSC (Lens Shading Correction) — pre-blend the per-channel gain
    // grid for use in the BL+WB pass.
    //
    // The Sony RAW header carries TWO illuminant-specific LSC tables
    // (Daylight + Tungsten), each with 4 channels. We blend
    // them based on a "warmth" estimate derived from the WB gains
    // (gR/gB ratio) and apply the result inline with the BL+WB pass
    // to avoid an extra 41 MB walk over the Bayer plane.
    //
    // The caller passes a pre-parsed lsc_table_t (or NULL to skip).
    // If the table is invalid (e.g., high-ISO mode where the camera
    // writes a different value pattern), the inner loop skips LSC.
    // ============================================================
    lsc_blended_t lsc_blended;
    lsc_blended.valid = 0;
    if (lsc_table && lsc_strength > 0.0f) {
        const lsc_table_t *parsed = (const lsc_table_t*)lsc_table;
        if (parsed->valid) {
            float warmth = lsc_warmth_from_wb((float)gR, (float)gB);
            lsc_blend(parsed, warmth, lsc_strength, &lsc_blended);
            // Diagnostic: log the warmth blend factor that was chosen.
            // 0.00 = pure set A (cool/daylight assumption)
            // 1.00 = pure set B (warm/tungsten assumption)
            // intermediate = linear blend of the two factory tables.
            const char* mode =
                  (warmth <= 0.05f) ? "set A only (cool)"
                : (warmth >= 0.95f) ? "set B only (warm)"
                :                     "blended A+B";
            fprintf(stderr, "[lsc] gR=%.3f gB=%.3f ratio=%.3f → "
                            "warmth=%.2f strength=%.2f (%s)\n",
                    (double)gR, (double)gB, (double)(gR/gB),
                    (double)warmth, (double)lsc_strength, mode);
        }
    }

    // Pre-compute LSC sampling scales (used inside the BL+WB loop only
    // when lsc_blended.valid). Width/height are guaranteed > 4 by the
    // earlier validation.
    const float lsc_scale_x = (float)(LSC_GRID_W - 1) / (float)(width  - 1);
    const float lsc_scale_y = (float)(LSC_GRID_H - 1) / (float)(height - 1);

    // ============================================================
    // Precompute a SPATIAL fade map at LSC grid resolution (7×9).
    //
    // Background: the previous "per-2x2 block adaptive fade" worked
    // for synthetic uniform tests but produced cartoonish/posterized
    // artifacts on real images with mixed content (e.g., dark leaves
    // against bright sky). The reason is that the fade factor was
    // computed independently for every 2×2 block — so a dim leaf
    // pixel got full LSC (3× gain) while the adjacent bright sky pixel
    // got no LSC. The leaf was lifted to nearly sky brightness,
    // collapsing local contrast.
    //
    // Fix: compute the fade factor over a SPATIALLY SMOOTH map. We
    // first scan the Bayer plane to find the maximum input value
    // within each of the 7×9 LSC grid cells. From that, we compute a
    // single fade value per cell (does the cell contain a pixel that
    // would clip after LSC × WB?). Within the inner loop we sample
    // this fade map bilinearly, so neighboring pixels get nearly
    // identical fade factors. Local content (dim leaf vs bright sky)
    // gets the SAME fade → relative brightness ratios are preserved.
    //
    // Trade-off: bright highlights in cells whose neighborhood is
    // mostly mid-tone may not get faded enough and will clip. That's
    // accepted because clipping a few bright specular highlights
    // looks much more natural than collapsing local contrast.
    // ============================================================
    constexpr float SAFE_FRAC = 0.85f;  // start fading at 85% of white_level
    float lsc_fade_grid[LSC_GRID_H][LSC_GRID_W];
    if (lsc_blended.valid) {
        // Reset fade grid: 0 = no fade (full LSC).
        for (int gy = 0; gy < LSC_GRID_H; ++gy)
            for (int gx = 0; gx < LSC_GRID_W; ++gx)
                lsc_fade_grid[gy][gx] = 0.0f;

        // Fast pre-pass: find max raw value in each of the 7×9 cells.
        // We don't need to be exact — strided sampling at every 8th
        // pixel is plenty since cells span hundreds of pixels and
        // the max is dominated by truly bright regions, not single-
        // pixel noise.
        uint16_t cell_max[LSC_GRID_H][LSC_GRID_W];
        for (int gy = 0; gy < LSC_GRID_H; ++gy)
            for (int gx = 0; gx < LSC_GRID_W; ++gx)
                cell_max[gy][gx] = 0;

        const int STRIDE = 8;  // sample every 8th pixel
        for (int y = 0; y < height; y += STRIDE) {
            const uint16_t* row = bayer + (size_t)y * (size_t)width;
            // Map y → grid row (with bounds check).
            float gy = (float)y * lsc_scale_y;
            int   gyi = (int)(gy + 0.5f);
            if (gyi < 0) gyi = 0;
            if (gyi >= LSC_GRID_H) gyi = LSC_GRID_H - 1;

            for (int x = 0; x < width; x += STRIDE) {
                float gxf = (float)x * lsc_scale_x;
                int   gxi = (int)(gxf + 0.5f);
                if (gxi < 0) gxi = 0;
                if (gxi >= LSC_GRID_W) gxi = LSC_GRID_W - 1;
                if (row[x] > cell_max[gyi][gxi]) {
                    cell_max[gyi][gxi] = row[x];
                }
            }
        }

        // Convert cell max → safe LSC fraction in [0, 1].
        //
        // Goal: for each cell, find the largest fraction f of the full
        // LSC correction that, when combined with WB, will not push the
        // cell's brightest pixel past white_level.
        //
        // Per-channel applied gain at fraction f:
        //   applied_gain(f) = 1 + (full_gain - 1) * f
        //
        // We need:
        //   cell_max * applied_gain(f) * wb_gain ≤ white_level
        //
        // Solving for f, channel by channel:
        //   applied_gain ≤ white_level / (cell_max * wb_gain)
        //   1 + (full_gain - 1) * f ≤ headroom_ratio
        //   f ≤ (headroom_ratio - 1) / (full_gain - 1)
        //
        // The cell's safe fraction is the minimum across all 4 channels
        // (the most-constrained channel sets the limit), clamped to [0, 1].
        //
        // FADE convention in the rest of the code: fade = 1 - f, so
        //   fade=0  → full LSC (cell has plenty of headroom)
        //   fade=1  → no LSC   (cell would clip even at full strength)
        //
        // Compared to the previous "linear fade between 85% and 100% of
        // white_level", this is mathematically exact: cells get the
        // largest LSC fraction that mathematically fits, not the heuristic
        // amount. That makes a big difference for mid-bright regions like
        // sky in a landscape: with the old heuristic, a 60%-saturated sky
        // got fade=1 (no LSC at all) because pred = 0.6 * 3.3 * 1.85 ≈
        // 3.7×WL was way past the linear ramp end. With exact headroom,
        // it gets a partial LSC factor that brings it to exactly white_level.
        const float fgR = (float)gR;
        const float fgB = (float)gB;
        const float fgG = 1.0f;

        for (int gy = 0; gy < LSC_GRID_H; ++gy) {
            for (int gx = 0; gx < LSC_GRID_W; ++gx) {
                int32_t mx_post_bl = (int32_t)cell_max[gy][gx] - black_level;
                if (mx_post_bl <= 0) {
                    lsc_fade_grid[gy][gx] = 0.0f;  // no signal → no clipping risk → full LSC
                    continue;
                }

                // Per-channel full LSC gain at this grid cell.
                const float gain_R  = 1.0f / lsc_blended.grid[0][gy][gx];
                const float gain_Gr = 1.0f / lsc_blended.grid[1][gy][gx];
                const float gain_Gb = 1.0f / lsc_blended.grid[2][gy][gx];
                const float gain_B  = 1.0f / lsc_blended.grid[3][gy][gx];

                // Per-channel "headroom ratio": the max applied_gain that
                // doesn't push cell_max past white_level after WB.
                //
                // SAFE_FRAC scales the "safe" line down from white_level
                // (e.g. 0.85 means we aim for 85% of white_level as the
                // post-WB ceiling). This leaves headroom for the
                // CAM_TO_SRGB matrix's per-row amplification (~1.91x for R)
                // which can push linear-sRGB past 1.0 even when post-WB
                // is just at white_level.
                const float ceiling = SAFE_FRAC * (float)white_level;
                const float denom_R  = (float)mx_post_bl * fgR;
                const float denom_Gr = (float)mx_post_bl * fgG;
                const float denom_Gb = (float)mx_post_bl * fgG;
                const float denom_B  = (float)mx_post_bl * fgB;

                // Per-channel max safe gain. Guard against denom==0
                // (mx_post_bl > 0 above ensures denom > 0).
                const float h_R  = ceiling / denom_R;
                const float h_Gr = ceiling / denom_Gr;
                const float h_Gb = ceiling / denom_Gb;
                const float h_B  = ceiling / denom_B;

                // Per-channel safe fraction f in [0, 1].
                //   f = (h - 1) / (full_gain - 1)
                // If h ≥ full_gain: full LSC fits → f=1
                // If h ≤ 1: even no-LSC (gain=1) would clip → f=0 (no LSC)
                // Otherwise: partial.
                auto safe_frac_for = [](float h, float full_gain) -> float {
                    if (h >= full_gain) return 1.0f;
                    if (h <= 1.0f)      return 0.0f;
                    return (h - 1.0f) / (full_gain - 1.0f);
                };
                float f_R  = safe_frac_for(h_R,  gain_R );
                float f_Gr = safe_frac_for(h_Gr, gain_Gr);
                float f_Gb = safe_frac_for(h_Gb, gain_Gb);
                float f_B  = safe_frac_for(h_B,  gain_B );

                // Cell's safe fraction = min of per-channel fractions
                // (the most-constrained channel limits the cell).
                float f_min = f_R;
                if (f_Gr < f_min) f_min = f_Gr;
                if (f_Gb < f_min) f_min = f_Gb;
                if (f_B  < f_min) f_min = f_B;

                // Convert to fade (= 1 - safe_fraction).
                lsc_fade_grid[gy][gx] = 1.0f - f_min;
            }
        }

        // OPTIONAL: smooth the fade grid with a small box blur so cell
        // boundaries don't create visible discontinuities. The 7×9 grid
        // is already coarse, but blurring once with [1 2 1] / 4 reduces
        // any per-cell flicker.
        float tmp[LSC_GRID_H][LSC_GRID_W];
        for (int gy = 0; gy < LSC_GRID_H; ++gy) {
            for (int gx = 0; gx < LSC_GRID_W; ++gx) {
                int x0 = (gx > 0)              ? gx - 1 : gx;
                int x2 = (gx < LSC_GRID_W - 1) ? gx + 1 : gx;
                tmp[gy][gx] = (lsc_fade_grid[gy][x0]
                             + lsc_fade_grid[gy][gx] * 2.0f
                             + lsc_fade_grid[gy][x2]) * 0.25f;
            }
        }
        for (int gx = 0; gx < LSC_GRID_W; ++gx) {
            for (int gy = 0; gy < LSC_GRID_H; ++gy) {
                int y0 = (gy > 0)              ? gy - 1 : gy;
                int y2 = (gy < LSC_GRID_H - 1) ? gy + 1 : gy;
                lsc_fade_grid[gy][gx] = (tmp[y0][gx]
                                       + tmp[gy][gx] * 2.0f
                                       + tmp[y2][gx]) * 0.25f;
            }
        }
    }

    // ============================================================
    // Black-level subtract + LSC + WB applied to the BAYER PLANE,
    // BEFORE both Bayer NR and demosaic.
    //
    // Why before NR: the wavelet NR uses a 2x2 Hadamard transform
    // to separate luma from chroma in each Bayer block. That
    // separation is only meaningful when R, G1, G2, B share a
    // common scale (i.e. a uniform white patch reads as R=G1=G2=B).
    // Pre-WB the channels have systematic gain offsets — under
    // daylight, raw G is ~1.5-2x raw R and ~2x raw B — so the
    // "Y = (R+G1+G2+B)/2" channel becomes G-dominated and biased
    // toward the illuminant, leaking real edges into the chroma
    // channels and reducing the algorithm's ability to concentrate
    // signal into Y vs noise into the C channels.
    //
    // Why before demosaic: the MHC kernels mix samples from
    // different colors (e.g. K_G_AT_RB applied at an R pixel uses
    // the center R value with coefficient 8, four G neighbors with
    // coefficient 4 each, and four R neighbors at distance 2 with
    // coefficient -2). The kernels were derived assuming all colors
    // share a common scale. Linear filtering does NOT commute with
    // per-channel scaling when the filter mixes channels.
    //
    // Working pedestal: when black-level subtract puts a noise
    // sample at -X, the natural choice would be to clamp at 0
    // because uint16_t is unsigned. But that destroys the symmetric
    // negative tail of the noise distribution, which means BayesShrink
    // downstream sees an artificially-reduced noise σ (clipped noise
    // has lower variance than its source) and under-thresholds
    // shadow regions. The wavelet's detail bands, computed from
    // differences, would see the same underestimated σ and miss
    // genuine noise.
    //
    // Fix: add a constant pedestal of WORKING_PEDESTAL (= 1024)
    // after black/WB so all values stay non-negative without
    // clipping. The pedestal lives entirely in the Haar LL band
    // and the Hadamard Y component (both DC), so it doesn't affect
    // detail-band statistics or noise estimation. After demosaic
    // we subtract the pedestal back out.
    //
    // Headroom check: maximum negative residual for a deeply-black
    // pixel with read noise σ~30 and WB gain ~2.1 is around -200.
    // 1024 leaves comfortable margin for the entire negative noise
    // tail to remain non-negative.
    //
    // Implementation: scale each Bayer sample by its parity-color's
    // gain, then add the pedestal, in-place. Q16 fixed-point keeps
    // the math integer; results fit comfortably in uint16_t.
    // ============================================================
    constexpr int32_t WORKING_PEDESTAL = 1024;
    {
        const int32_t qR = (int32_t)(gR * 65536.0 + 0.5);
        const int32_t qG = (int32_t)(gG * 65536.0 + 0.5);
        const int32_t qB = (int32_t)(gB * 65536.0 + 0.5);

        // Pre-compute LSC per-column grid coordinates if LSC is active.
        // Each pixel's LSC contribution depends only on (gy, gx) and the
        // channel parity. Computing gx and the column-interpolation
        // weights once per column saves enormous work in the inner loop.
        std::vector<int32_t> lsc_gx0;     // grid column index (left)
        std::vector<float>   lsc_fx;       // fractional weight in [0, 1]
        if (lsc_blended.valid) {
            lsc_gx0.resize(width);
            lsc_fx .resize(width);
            for (int x = 0; x < width; ++x) {
                float gx = (float)x * lsc_scale_x;
                int   x0 = (int)gx;
                if (x0 < 0) x0 = 0;
                if (x0 >= LSC_GRID_W - 1) x0 = LSC_GRID_W - 2;
                lsc_gx0[x] = x0;
                float fx = gx - (float)x0;
                if (fx < 0) fx = 0; else if (fx > 1) fx = 1;
                lsc_fx[x] = fx;
            }
        }

        // Process the image in 2x2 Bayer blocks. This lets us share a
        // single LSC gain-cap across the four channels (R, Gr, Gb, B) at
        // each block, which preserves hue when the LSC's per-channel gains
        // would push some pixels past max_signal.
        //
        // Why 2x2 awareness matters: at corners, LSC asks for ~3x gain on
        // each channel. A blue sky pixel might have raw values like
        // R=316, Gr=536, Gb=536, B=456 (post-BL). Applying 3x gain
        // independently makes Gr and Gb saturate first (1672 → clip), but
        // R only goes to 1036 (just clips), B to 1537 (clips). After
        // per-channel capping, all four sit at max_signal — the sky goes
        // white. With 2x2 awareness we instead compute one shared scaling
        // factor that lets the brightest channel just touch max_signal,
        // and applies the same factor to the other three. R:G:B ratio is
        // preserved → sky stays blue.
        //
        // For odd width or height we fall back to the simple per-pixel
        // path on the trailing row/column. In practice raw sensors are
        // even-dimensioned so this branch rarely fires.
        const int H_even = height & ~1;
        const int W_even = width  & ~1;

        // ============================================================
        // Threaded BL+WB+LSC pass
        //
        // Each iteration of the outer y loop processes a 2x2 BLOCK ROW
        // (image rows y and y+1) and only touches bayer[y..y+1] and
        // read-only metadata (lsc grids, qR/qG/qB, etc). Block rows are
        // entirely independent, so we parallelize across them.
        //
        // parallel_rows splits a row range across worker threads. We
        // give it the BLOCK-ROW index range [0, H_even/2) — each worker
        // call gets a band of block-row indices, multiplies by 2 to get
        // the actual image y, and runs the original 2x2 logic. With 4
        // cores this is the same near-4x speedup pattern we used for
        // bayer_nr's 4-channel parallelism and the demosaic main loop.
        //
        // Trailing odd column (when width is odd) stays inside the
        // worker — each worker handles its own block-row's odd column.
        // Trailing odd row (when height is odd) stays serial after the
        // parallel pass; it's a single row so the cost is negligible.
        // ============================================================
        struct lsc_ctx_t {
            uint16_t *bayer;
            int       width;
            int       black_level;
            int32_t   qR, qG, qB;
            int       W_even;
            // LSC state — all read-only across workers.
            bool                  lsc_valid;
            const lsc_blended_t     *lsc_blended;
            const int32_t        *lsc_gx0_data;
            const float          *lsc_fx_data;
            float                 lsc_scale_y;
            const float (*lsc_fade_grid)[LSC_GRID_W];
        };

        lsc_ctx_t lc;
        lc.bayer         = bayer;
        lc.width         = width;
        lc.black_level   = black_level;
        lc.qR            = qR;
        lc.qG            = qG;
        lc.qB            = qB;
        lc.W_even        = W_even;
        lc.lsc_valid     = lsc_blended.valid;
        lc.lsc_blended   = &lsc_blended;
        lc.lsc_gx0_data  = lsc_blended.valid ? lsc_gx0.data() : nullptr;
        lc.lsc_fx_data   = lsc_blended.valid ? lsc_fx .data() : nullptr;
        lc.lsc_scale_y   = lsc_scale_y;
        lc.lsc_fade_grid = lsc_fade_grid;

        auto lsc_worker = [](int by0, int by1, void *cv) {
            const lsc_ctx_t *c = (const lsc_ctx_t*)cv;
            const int       width       = c->width;
            const int       black_level = c->black_level;
            const int32_t   qR          = c->qR;
            const int32_t   qG          = c->qG;
            const int32_t   qB          = c->qB;
            const int       W_even      = c->W_even;
            const bool      lsc_valid   = c->lsc_valid;
            const int32_t  *lsc_gx0     = c->lsc_gx0_data;
            const float    *lsc_fx      = c->lsc_fx_data;
            const float     lsc_scale_y = c->lsc_scale_y;
            const float    (*lsc_fade_grid)[LSC_GRID_W] = c->lsc_fade_grid;
            const lsc_blended_t *lsc_blended = c->lsc_blended;
            uint16_t       *bayer       = c->bayer;

            // Per-pixel: BL subtract, optional LSC gain, WB Q16 multiply,
            // pedestal, clamp.
            auto apply_pixel = [black_level](uint16_t* p, int32_t q_wb, float lsc_gain) {
                int32_t v = (int32_t)*p - black_level;
                if (lsc_gain != 1.0f) {
                    float scaled_f = (float)v * lsc_gain;
                    v = (int32_t)(scaled_f >= 0.0f ? scaled_f + 0.5f : scaled_f - 0.5f);
                }
                int32_t scaled = (v * q_wb + 32768) >> 16;
                scaled += WORKING_PEDESTAL;
                if (scaled < 0)     scaled = 0;
                if (scaled > 65535) scaled = 65535;
                *p = (uint16_t)scaled;
            };

            for (int by = by0; by < by1; ++by) {
                const int y = by * 2;
                uint16_t* row0 = bayer + (size_t)(y    ) * (size_t)width;
                uint16_t* row1 = bayer + (size_t)(y + 1) * (size_t)width;

                // Per-row LSC state for the TOP row of the 2x2 block. The
                // bottom row is interpolated separately because the 2x2 block
                // straddles two image rows. For most images these two rows are
                // very close in fy (since LSC has only 7 grid rows over the
                // image height), but we compute both for correctness.
                float row_grid_top[LSC_NUM_CHANNELS][LSC_GRID_W];
                float row_grid_bot[LSC_NUM_CHANNELS][LSC_GRID_W];
                if (lsc_valid) {
                    auto fill_row_grid = [lsc_blended, lsc_scale_y]
                                         (int img_y,
                                          float dst[LSC_NUM_CHANNELS][LSC_GRID_W]) {
                        float gy = (float)img_y * lsc_scale_y;
                        int   y0 = (int)gy;
                        if (y0 < 0) y0 = 0;
                        if (y0 >= LSC_GRID_H - 1) y0 = LSC_GRID_H - 2;
                        int y1 = y0 + 1;
                        float fy = gy - (float)y0;
                        if (fy < 0) fy = 0; else if (fy > 1) fy = 1;
                        for (int ch = 0; ch < LSC_NUM_CHANNELS; ++ch) {
                            for (int gx = 0; gx < LSC_GRID_W; ++gx) {
                                dst[ch][gx] =
                                      lsc_blended->grid[ch][y0][gx] * (1.0f - fy)
                                    + lsc_blended->grid[ch][y1][gx] *        fy;
                            }
                        }
                    };
                    fill_row_grid(y    , row_grid_top);
                    fill_row_grid(y + 1, row_grid_bot);
                }

                for (int x = 0; x < W_even; x += 2) {
                    // RGGB layout: R=top-left, Gr=top-right, Gb=bot-left, B=bot-right
                    uint16_t* pR  = &row0[x    ];
                    uint16_t* pGr = &row0[x + 1];
                    uint16_t* pGb = &row1[x    ];
                    uint16_t* pB  = &row1[x + 1];

                    float gain_R = 1.0f, gain_Gr = 1.0f, gain_Gb = 1.0f, gain_B = 1.0f;

                    if (lsc_valid) {
                        const int   x0  = lsc_gx0[x];
                        const float fx0 = lsc_fx [x];
                        const int   x1  = lsc_gx0[x + 1];
                        const float fx1 = lsc_fx [x + 1];

                        // Each Bayer position has its own (row, column) for LSC
                        // sampling. Top-left R uses row_grid_top[0], top-right
                        // Gr uses row_grid_top[1], bottom-left Gb uses
                        // row_grid_bot[2], bottom-right B uses row_grid_bot[3].
                        auto sample = [](const float (*rg)[LSC_GRID_W], int ch,
                                         int gx0, float fx) -> float {
                            return rg[ch][gx0] * (1.0f - fx) + rg[ch][gx0 + 1] * fx;
                        };
                        float sh_R  = sample(row_grid_top, 0, x0, fx0);
                        float sh_Gr = sample(row_grid_top, 1, x1, fx1);
                        float sh_Gb = sample(row_grid_bot, 2, x0, fx0);
                        float sh_B  = sample(row_grid_bot, 3, x1, fx1);

                        float full_gain_R  = 1.0f / sh_R;
                        float full_gain_Gr = 1.0f / sh_Gr;
                        float full_gain_Gb = 1.0f / sh_Gb;
                        float full_gain_B  = 1.0f / sh_B;

                        // Sample the spatially smooth fade map computed in the
                        // pre-pass. This fade was derived from the maximum
                        // input value within each LSC grid cell, so it varies
                        // smoothly across the image. Within a small region
                        // (e.g. a leaf cluster on sky), nearby pixels see
                        // essentially the same fade value, which preserves
                        // local contrast.
                        //
                        // Fade interpretation:
                        //   t = 0 → full LSC applied (cell has no clipping risk)
                        //   t = 1 → no LSC applied  (cell would clip)
                        //
                        // Per-channel actual gain blended toward 1.0 by t.
                        // We use the SAME fade for all 4 channels of this 2x2
                        // block (and effectively the same fade for nearby
                        // blocks too, thanks to bilinear sampling).
                        auto fade_at = [lsc_gx0, lsc_fx, lsc_scale_y, lsc_fade_grid]
                                       (int gx, int gy_top) -> float {
                            // Bilinear sample of lsc_fade_grid at (gy_top, gx).
                            // We use the top row of the 2x2 for simplicity;
                            // both rows of the block are within one cell anyway.
                            int   gxi = lsc_gx0[gx];
                            float fxi = lsc_fx [gx];
                            // gy_top is the absolute image y; convert to grid.
                            float gyf = (float)gy_top * lsc_scale_y;
                            int   gyi = (int)gyf;
                            if (gyi < 0) gyi = 0;
                            if (gyi >= LSC_GRID_H - 1) gyi = LSC_GRID_H - 2;
                            float fyi = gyf - (float)gyi;
                            if (fyi < 0) fyi = 0; else if (fyi > 1) fyi = 1;

                            float v00 = lsc_fade_grid[gyi    ][gxi    ];
                            float v01 = lsc_fade_grid[gyi    ][gxi + 1];
                            float v10 = lsc_fade_grid[gyi + 1][gxi    ];
                            float v11 = lsc_fade_grid[gyi + 1][gxi + 1];
                            float v0  = v00 * (1.0f - fxi) + v01 * fxi;
                            float v1  = v10 * (1.0f - fxi) + v11 * fxi;
                            return v0  * (1.0f - fyi) + v1  * fyi;
                        };

                        float t = fade_at(x, y);
                        if (t > 1.0f) t = 1.0f;
                        if (t < 0.0f) t = 0.0f;
                        const float keep = 1.0f - t;
                        gain_R  = 1.0f + (full_gain_R  - 1.0f) * keep;
                        gain_Gr = 1.0f + (full_gain_Gr - 1.0f) * keep;
                        gain_Gb = 1.0f + (full_gain_Gb - 1.0f) * keep;
                        gain_B  = 1.0f + (full_gain_B  - 1.0f) * keep;
                    }

                    apply_pixel(pR,  qR, gain_R );
                    apply_pixel(pGr, qG, gain_Gr);
                    apply_pixel(pGb, qG, gain_Gb);
                    apply_pixel(pB,  qB, gain_B );
                }

                // Trailing odd column (only if width is odd; rare).
                if (W_even != width) {
                    int x = W_even;
                    int32_t q0 = ((x & 1) == 0) ? qR : qG;
                    int32_t q1 = ((x & 1) == 0) ? qG : qB;
                    apply_pixel(&row0[x], q0, 1.0f);
                    apply_pixel(&row1[x], q1, 1.0f);
                }
            }
        };

        // Dispatch over block-row indices.
        parallel_rows(0, H_even / 2, /*num_threads=auto*/ 0, +lsc_worker, &lc);

        // Trailing odd row (only if height is odd; rare). Single row,
        // serial — cost is negligible.
        if (H_even != height) {
            auto apply_pixel = [black_level](uint16_t* p, int32_t q_wb, float lsc_gain) {
                int32_t v = (int32_t)*p - black_level;
                if (lsc_gain != 1.0f) {
                    float scaled_f = (float)v * lsc_gain;
                    v = (int32_t)(scaled_f >= 0.0f ? scaled_f + 0.5f : scaled_f - 0.5f);
                }
                int32_t scaled = (v * q_wb + 32768) >> 16;
                scaled += WORKING_PEDESTAL;
                if (scaled < 0)     scaled = 0;
                if (scaled > 65535) scaled = 65535;
                *p = (uint16_t)scaled;
            };
            int y = H_even;
            uint16_t* row = bayer + (size_t)y * (size_t)width;
            int32_t q_even = qR; // y%2=0, x%2=0 → R parity
            int32_t q_odd  = qG;
            for (int x = 0; x < width; ++x) {
                int32_t q = ((x & 1) == 0) ? q_even : q_odd;
                apply_pixel(&row[x], q, 1.0f);
            }
        }
    }

    // Bayer-domain noise reduction. Runs AFTER WB so that the
    // Hadamard transform inside bayer_nr produces a meaningful
    // luma/chroma separation (see comment above).
    if (bayer_nr_strength > 0.0f) {
        bayer_nr_uint16(bayer, width, height, bayer_nr_strength);
    }

    // Pre-matrix normalization scale: the demosaic now produces values
    // already in the WB-applied space, in 0..white_level (with possibly
    // some pixels above white_level for saturated R/B that got pushed
    // up by WB gain). Normalize to [0, 1+] linear by dividing by the
    // (post-black-subtract) white level.
    const float pre_matrix_scale = 1.0f / (float)white_level;

    const float *M = cam_to_srgb;

    build_gamma_lut(tone_contrast);

    int stride = width;

    // ---- Threaded main pixel loop ----
    //
    // Each output row depends only on read-only inputs (Bayer plane,
    // matrix, gamma LUT, exposure curve LUT, etc.) and writes to a
    // disjoint output row, so we split rows into bands and run each
    // band on its own thread. On the D5503's quad Krait 400, this
    // gives ~3x speedup on this loop alone.
    //
    // EXPOSURE CURVE LUT
    //
    // The Reinhard luma-driven filmic curve
    //
    //     ratio(Y) = exposure / (1 + (exposure - 1) * Y)
    //
    // requires a float divide per pixel. We pre-tabulate it here in 1024
    // entries spanning Y in [0, 2.0] (linear sRGB Y rarely exceeds 1.5
    // even at saturated highlights post-WB). The LUT is rebuilt only when
    // exposure changes; at exposure==1.0 the curve is identity and the
    // entire step is skipped per-pixel below, so the LUT is unused in
    // that case.
    //
    // Quantization: 1024 entries over [0, 2] gives a step of 0.00195 in Y,
    // which after multiplication by exposure produces an output error of
    // <0.001 in ratio — sub-LSB at 14-bit precision, so output is
    // visually indistinguishable from the divide-based formula.
    constexpr int EXPOSURE_LUT_SIZE = 1024;
    constexpr float EXPOSURE_LUT_DOMAIN = 2.0f;  // Y_in range [0, 2.0]
    static float g_exposure_lut[EXPOSURE_LUT_SIZE];
    static float g_exposure_lut_built_for = 1e30f;  // sentinel
    if (g_exposure_lut_built_for != exposure) {
        const float e_minus_1 = exposure - 1.0f;
        for (int i = 0; i < EXPOSURE_LUT_SIZE; ++i) {
            float Y = (float)i * (EXPOSURE_LUT_DOMAIN / (float)EXPOSURE_LUT_SIZE);
            // ratio = (e * Y) / (Y * (1 + (e-1)*Y))  = e / (1 + (e-1)*Y)
            float ratio = exposure / (1.0f + e_minus_1 * Y);
            g_exposure_lut[i] = ratio;
        }
        g_exposure_lut_built_for = exposure;
    }

    struct demosaic_ctx_t {
        const uint16_t *bayer;
        uint8_t        *rgb_out;
        int             width;
        int             height;
        int             stride;
        float           pre_matrix_scale;
        const float    *M;
        float           exposure;
        float           saturation;
        float           black_point_lift;
        const float    *exposure_lut;
    };
    demosaic_ctx_t ctx;
    ctx.bayer            = bayer;
    ctx.rgb_out          = rgb_out;
    ctx.width            = width;
    ctx.height           = height;
    ctx.stride           = stride;
    ctx.pre_matrix_scale = pre_matrix_scale;
    ctx.M                = M;
    ctx.exposure         = exposure;
    ctx.saturation       = saturation;
    ctx.black_point_lift = black_point_lift;
    ctx.exposure_lut     = g_exposure_lut;

    auto worker = [](int y0, int y1, void *ctxv) {
        const demosaic_ctx_t *c = (const demosaic_ctx_t*)ctxv;
        const uint16_t *bayer = c->bayer;
        uint8_t        *rgb_out = c->rgb_out;
        const int       width  = c->width;
        const int       height = c->height;
        const int       stride = c->stride;
        const float     pre_matrix_scale = c->pre_matrix_scale;
        const float    *M = c->M;
        const float     exposure = c->exposure;
        const float     saturation = c->saturation;
        const float     black_point_lift = c->black_point_lift;
        const float    *exposure_lut = c->exposure_lut;

    for (int y = y0; y < y1; ++y) {
        bool y_border = (y < 2 || y >= height - 2);
        uint8_t *out_row = rgb_out + (size_t)y * (size_t)width * 3;

        int x = 0;

#if HAVE_NEON
        // NEON-only setup: precompute the constant float vectors used
        // inside the 4-pixel batch.  These are loop-invariant within
        // a worker call and the compiler hoists them out of the inner
        // loops at -O2.
        const float bplift = black_point_lift;
        const float sat    = saturation;
        const bool  do_exp = (exposure != 1.0f);
        const float em1    = exposure - 1.0f;
        const float32x4_t v_one    = vdupq_n_f32(1.0f);
        const float32x4_t v_M0     = vdupq_n_f32(M[0]);
        const float32x4_t v_M1     = vdupq_n_f32(M[1]);
        const float32x4_t v_M2     = vdupq_n_f32(M[2]);
        const float32x4_t v_M3     = vdupq_n_f32(M[3]);
        const float32x4_t v_M4     = vdupq_n_f32(M[4]);
        const float32x4_t v_M5     = vdupq_n_f32(M[5]);
        const float32x4_t v_M6     = vdupq_n_f32(M[6]);
        const float32x4_t v_M7     = vdupq_n_f32(M[7]);
        const float32x4_t v_M8     = vdupq_n_f32(M[8]);
        const float32x4_t v_w_R    = vdupq_n_f32(0.2126f);
        const float32x4_t v_w_G    = vdupq_n_f32(0.7152f);
        const float32x4_t v_w_B    = vdupq_n_f32(0.0722f);
        const float32x4_t v_bplift = vdupq_n_f32(bplift);
        const float32x4_t v_sat    = vdupq_n_f32(sat);
        const float32x4_t v_em1    = vdupq_n_f32(em1);
        const float32x4_t v_exp    = vdupq_n_f32(exposure);
        const float32x4_t v_16383  = vdupq_n_f32(16383.0f);
        const int32x4_t   v_i16383 = vdupq_n_s32(16383);
        const int32x4_t   v_i0     = vdupq_n_s32(0);
        // Suppress "unused" warnings on builds where we never enter
        // the NEON branch (e.g. tiny images with width <= 5).
        (void)bplift; (void)sat; (void)em1; (void)v_one;
#endif

        // The pixel loop.  The NEON branch (when available + applicable)
        // processes 4 interior pixels per iteration; otherwise we fall
        // through to the scalar body.  Using a `while` loop with manual
        // x-advancement lets the SAME scalar body handle both the left
        // border (x = 0, 1), the right border (x ≥ width - 2), and any
        // tail of 1–3 interior pixels left over after NEON — no
        // duplicated code.
        while (x < width) {
#if HAVE_NEON
            // Conditions for the NEON 4-batch:
            //   - Not on the y-direction Bayer border (mhc_at requires it).
            //   - Past the left x-direction border (x ≥ 2).
            //   - At least 4 interior pixels remain (x + 4 ≤ width - 2).
            if (!y_border && x >= 2 && x + 4 <= width - 2) {
                // ---- Scalar: 4 × MHC + pedestal + normalize ----
                float Rfs[4], Gfs[4], Bfs[4];
                for (int dx = 0; dx < 4; ++dx) {
                    int32_t Rr, Gr, Br;
                    mhc_at(bayer, stride, x + dx, y, &Rr, &Gr, &Br);
                    Rr -= WORKING_PEDESTAL; Gr -= WORKING_PEDESTAL; Br -= WORKING_PEDESTAL;
                    if (Rr < 0) Rr = 0;
                    if (Gr < 0) Gr = 0;
                    if (Br < 0) Br = 0;
                    Rfs[dx] = (float)Rr * pre_matrix_scale;
                    Gfs[dx] = (float)Gr * pre_matrix_scale;
                    Bfs[dx] = (float)Br * pre_matrix_scale;
                }

                // ---- NEON: pre-matrix highlight clip to ≤ 1 ----
                float32x4_t Rf4 = vminq_f32(vld1q_f32(Rfs), v_one);
                float32x4_t Gf4 = vminq_f32(vld1q_f32(Gfs), v_one);
                float32x4_t Bf4 = vminq_f32(vld1q_f32(Bfs), v_one);

                // ---- NEON: matrix multiply (cam → linear sRGB) ----
                float32x4_t lr4 = vmulq_f32(Rf4, v_M0);
                lr4 = vmlaq_f32(lr4, Gf4, v_M1);
                lr4 = vmlaq_f32(lr4, Bf4, v_M2);
                float32x4_t lg4 = vmulq_f32(Rf4, v_M3);
                lg4 = vmlaq_f32(lg4, Gf4, v_M4);
                lg4 = vmlaq_f32(lg4, Bf4, v_M5);
                float32x4_t lb4 = vmulq_f32(Rf4, v_M6);
                lb4 = vmlaq_f32(lb4, Gf4, v_M7);
                lb4 = vmlaq_f32(lb4, Bf4, v_M8);

                // ---- NEON: exposure curve via vrecpe + Newton refine ----
                //
                //   ratio = exposure / (1 + (e-1) * Y_in)
                //
                // Replaces the scalar exposure_lut path on this branch.
                // vrecpe gives ~12-bit reciprocal estimate; one round of
                // vrecps Newton refinement brings it to ~22-bit accuracy
                // — strictly better than the 1024-entry LUT used in the
                // scalar path.  This intentionally drops Y_in's "≤ 1e-6"
                // black-pixel guard: the reciprocal of (1 + (e-1)·0) = 1
                // is exact, so for Y_in = 0 the formula yields exactly
                // `exposure`, matching the scalar fallback's behaviour.
                if (do_exp) {
                    float32x4_t Yin = vmulq_f32(lr4, v_w_R);
                    Yin = vmlaq_f32(Yin, lg4, v_w_G);
                    Yin = vmlaq_f32(Yin, lb4, v_w_B);
                    float32x4_t denom = vmlaq_f32(v_one, Yin, v_em1);
                    float32x4_t est = vrecpeq_f32(denom);
                    est = vmulq_f32(est, vrecpsq_f32(denom, est));
                    float32x4_t ratio = vmulq_f32(v_exp, est);
                    lr4 = vmulq_f32(lr4, ratio);
                    lg4 = vmulq_f32(lg4, ratio);
                    lb4 = vmulq_f32(lb4, ratio);
                }

                // ---- NEON: black-point lift ----
                lr4 = vsubq_f32(lr4, v_bplift);
                lg4 = vsubq_f32(lg4, v_bplift);
                lb4 = vsubq_f32(lb4, v_bplift);

                // ---- NEON: saturation ----
                if (sat != 1.0f) {
                    float32x4_t Y4 = vmulq_f32(lr4, v_w_R);
                    Y4 = vmlaq_f32(Y4, lg4, v_w_G);
                    Y4 = vmlaq_f32(Y4, lb4, v_w_B);
                    lr4 = vmlaq_f32(Y4, vsubq_f32(lr4, Y4), v_sat);
                    lg4 = vmlaq_f32(Y4, vsubq_f32(lg4, Y4), v_sat);
                    lb4 = vmlaq_f32(Y4, vsubq_f32(lb4, Y4), v_sat);
                }

                // ---- NEON: scale to 14-bit, clip to [0, 16383] ----
                int32x4_t ir4 = vcvtq_s32_f32(vmulq_f32(lr4, v_16383));
                int32x4_t ig4 = vcvtq_s32_f32(vmulq_f32(lg4, v_16383));
                int32x4_t ib4 = vcvtq_s32_f32(vmulq_f32(lb4, v_16383));
                ir4 = vmaxq_s32(ir4, v_i0); ir4 = vminq_s32(ir4, v_i16383);
                ig4 = vmaxq_s32(ig4, v_i0); ig4 = vminq_s32(ig4, v_i16383);
                ib4 = vmaxq_s32(ib4, v_i0); ib4 = vminq_s32(ib4, v_i16383);

                // ---- Scalar: 4 × gamma LUT lookup + interleaved RGB ----
                int32_t ir_arr[4], ig_arr[4], ib_arr[4];
                vst1q_s32(ir_arr, ir4);
                vst1q_s32(ig_arr, ig4);
                vst1q_s32(ib_arr, ib4);
                for (int dx = 0; dx < 4; ++dx) {
                    out_row[3*(x+dx) + 0] = g_gamma_lut[ir_arr[dx]];
                    out_row[3*(x+dx) + 1] = g_gamma_lut[ig_arr[dx]];
                    out_row[3*(x+dx) + 2] = g_gamma_lut[ib_arr[dx]];
                }
                x += 4;
                continue;
            }
#endif
            // ---- Scalar body for one pixel (border or tail) ----
            int32_t Rr, Gr, Br;
            if (y_border || x < 2 || x >= width - 2) {
                bilinear_at(bayer, width, height, stride, x, y, &Rr, &Gr, &Br);
            } else {
                mhc_at(bayer, stride, x, y, &Rr, &Gr, &Br);
            }

            // Black-level + WB are already done on the Bayer plane.
            // The demosaiced values share a common WB-applied scale,
            // plus a constant pedestal of WORKING_PEDESTAL added during
            // the WB pass to preserve the negative noise tail through
            // Bayer NR.
            //
            // Subtract the pedestal here. The MHC and bilinear kernels
            // both have unity DC gain (kernel sums = 16/16, 4/4, etc.),
            // so the pedestal passes through unchanged: the demosaicer
            // outputs (true_signal + pedestal). After subtraction the
            // values represent post-WB linear scene RGB starting from 0.
            //
            // Negative residuals are now meaningful — they're shadow
            // noise that dipped below the sensor's effective black,
            // surviving NR as honest noise. Clamp at 0 here because
            // the rest of the pipeline (matrix, gamma) needs
            // non-negative values.
            Rr -= WORKING_PEDESTAL;
            Gr -= WORKING_PEDESTAL;
            Br -= WORKING_PEDESTAL;
            if (Rr < 0) Rr = 0;
            if (Gr < 0) Gr = 0;
            if (Br < 0) Br = 0;

            // Normalize to linear [0, ~2] using the post-black-subtract
            // white level. WB has already been applied per sample.
            float Rf = (float)Rr * pre_matrix_scale;
            float Gf = (float)Gr * pre_matrix_scale;
            float Bf = (float)Br * pre_matrix_scale;

            // HIGHLIGHT CLIPPING — clamp WB-corrected camera RGB to [0,1]
            // BEFORE the color matrix.
            //
            // The color matrix is calibrated to map valid camera RGB
            // in [0,1] to linear sRGB. WB gains for non-green channels
            // are typically > 1 (here gR ≈ 1.65, gB ≈ 2.08), so a fully
            // saturated sensor pixel (raw R=G=B=1023) becomes
            // (Rf, Gf, Bf) ≈ (1.65, 1.0, 2.08) after WB — outside the
            // matrix's calibrated range.
            //
            // Feeding such values into a matrix with negative
            // off-diagonals (which is normal for cam→sRGB matrices)
            // produces hue-shifted highlights — typically pink/magenta,
            // because the green channel ends up at e.g. 0.64 while
            // R and B clip to 1.0.
            //
            // Clamping per-channel before the matrix gives saturated
            // sensor pixels neutral (1, 1, 1) input, which the matrix
            // (whose row sums ≈ 1) maps to neutral white. Loses some
            // hue information about strongly-coloured highlights, but
            // that information is unrecoverable when the sensor was
            // saturated anyway.
            if (Rf > 1.0f) Rf = 1.0f;
            if (Gf > 1.0f) Gf = 1.0f;
            if (Bf > 1.0f) Bf = 1.0f;

            // Color matrix: cam_RGB_wb → linear sRGB.
            float lr = M[0]*Rf + M[1]*Gf + M[2]*Bf;
            float lg = M[3]*Rf + M[4]*Gf + M[5]*Bf;
            float lb = M[6]*Rf + M[7]*Gf + M[8]*Bf;

            // EXPOSURE (luminance-driven filmic rolloff)
            //
            // Brightens shadows and midtones with a smooth shoulder
            // that protects highlights. Shadows get the full multiplier;
            // values approaching 1.0 in linear sRGB get progressively
            // less lift; values at 1.0 stay at 1.0 (already-clipped
            // highlights don't get pushed further).
            //
            // The curve is the Reinhard tone-mapping form:
            //
            //   Y' = e * Y / (1 + (e - 1) * Y)        where e = exposure
            //
            // Properties (anchored at 0 and 1):
            //   Y = 0.0  → Y' = 0.0          blacks stay black
            //   Y = 0.05 → Y' ≈ 0.072        shadow boost ~1.45x for e=1.5
            //   Y = 0.50 → Y' = 0.600        midtone boost ~1.20x
            //   Y = 0.85 → Y' ≈ 0.919        highlight boost ~1.08x
            //   Y = 1.0  → Y' = 1.0          saturated highlights pinned
            //
            // Crucially, the curve is computed on LUMINANCE only, then
            // each color channel is scaled by the same ratio Y'/Y. This
            // preserves hue exactly through the rolloff. Without this
            // step, a blue sky (R=0.7, G=0.78, B=0.94) would have its
            // B channel compressed more than R and G — shifting toward
            // yellow at exactly the bright pixels we're trying to
            // protect. Per-channel filmic curves are a common bug;
            // luma-driven is the correct form for "exposure that
            // doesn't blow out the sky."
            //
            // For exposure < 1.0 (darkening), the same formula works
            // unchanged: it darkens linearly with no curve, since the
            // "shoulder" only comes from the +(e-1)*Y term being
            // positive when e > 1.
            //
            // Skip when exposure == 1.0 (no-op early-out).
            if (exposure != 1.0f) {
                // BT.709 luminance (matches sRGB primaries)
                const float Y_in = 0.2126f * lr + 0.7152f * lg + 0.0722f * lb;
                if (Y_in > 1e-6f) {
                    // LUT lookup with linear interpolation between entries.
                    // The LUT was built over Y in [0, 2.0] with 1024 entries
                    // (step = 0.00195). Out-of-domain Y >= 2.0 clamps to the
                    // last entry, which represents the smallest "ratio" the
                    // curve produces — physically correct for highlight pixels
                    // outside the calibrated range.
                    //
                    // At exposure=1.5 the LUT-vs-divide error is bounded by
                    // ~0.5 LSB after gamma encoding — sub-perceptual.
                    const float pos = Y_in * (1024.0f / 2.0f);  // = Y_in * 512
                    int idx = (int)pos;
                    float frac = pos - (float)idx;
                    if (idx < 0)        { idx = 0;    frac = 0.0f; }
                    else if (idx > 1022){ idx = 1022; frac = 1.0f; }
                    const float r0 = exposure_lut[idx];
                    const float r1 = exposure_lut[idx + 1];
                    const float ratio = r0 + (r1 - r0) * frac;
                    lr *= ratio;
                    lg *= ratio;
                    lb *= ratio;
                } else {
                    // Black pixel: ratio is exposure (the curve's slope at 0).
                    // Identity for exposure=1.0; just scales for darkening.
                    lr *= exposure;
                    lg *= exposure;
                    lb *= exposure;
                }
            }

            // BLACK POINT LIFT ("dehaze")
            //
            // Subtract a small constant from linear sRGB before clipping
            // and gamma. Pixels at or below the threshold become clean
            // black; everything else gets stretched proportionally by
            // the subsequent gamma / S-curve.
            //
            // This is the single biggest fix for the "smoky" look in
            // raw-rendered images. Causes of residual lift in shadows:
            //   - sensor read noise floor (not all dark pixels read 0
            //     after black-level subtract)
            //   - small calibration error in the color matrix
            //   - veiling glare in the optical path
            // All three add a small, roughly uniform offset that the
            // matrix can't correct. This subtraction is the post-matrix
            // counterpart to the per-channel sensor-black subtraction
            // that happened earlier — same mechanism, different stage.
            //
            // Sub-zero values clip to 0 in the cast-to-int below.
            lr -= black_point_lift;
            lg -= black_point_lift;
            lb -= black_point_lift;

            // SATURATION BOOST
            //
            // Pull each pixel's RGB triple away from its luma value.
            // Luma weights are Rec.709 (matches sRGB primaries).
            //
            //   Y     = 0.2126·R + 0.7152·G + 0.0722·B
            //   out_c = Y + (in_c - Y) * S
            //
            // S = 1.0  → no change
            // S = 1.2  → typical mild boost (default)
            // S > 1.5  → over the top
            // S = 0.0  → monochrome (luma)
            //
            // Hue is preserved exactly; only the magnitude of the
            // chroma vector changes. Negative results clip to 0 below;
            // values >1 clip to white. Skip if saturation==1 to avoid
            // arithmetic in the common no-op case.
            if (saturation != 1.0f) {
                float Y = 0.2126f * lr + 0.7152f * lg + 0.0722f * lb;
                lr = Y + (lr - Y) * saturation;
                lg = Y + (lg - Y) * saturation;
                lb = Y + (lb - Y) * saturation;
            }

            // Clip [0,1] then gamma via 14-bit LUT.
            int ir = (int)(lr * 16383.0f);
            int ig = (int)(lg * 16383.0f);
            int ib = (int)(lb * 16383.0f);
            if (ir < 0) ir = 0; else if (ir > 16383) ir = 16383;
            if (ig < 0) ig = 0; else if (ig > 16383) ig = 16383;
            if (ib < 0) ib = 0; else if (ib > 16383) ib = 16383;

            out_row[3*x + 0] = g_gamma_lut[ir];
            out_row[3*x + 1] = g_gamma_lut[ig];
            out_row[3*x + 2] = g_gamma_lut[ib];
            ++x;
        }
    }
    };  // end lambda

    // Captureless lambda → C function pointer conversion is part of C++14.
    parallel_rows(0, height, /*num_threads=auto*/ 0, +worker, &ctx);

    free(bayer);
    return 0;
}

// Default-parameters wrapper for callers that want a "good enough" look
// without specifying tone parameters. Used by the on-device daemon path
// in pipeline.cpp.
extern "C" int demosaic_to_srgb(
    const void  *raw_packed,
    size_t       raw_size,
    int          width,
    int          height,
    int          black_level,
    int          white_level,
    uint32_t     r_num, uint32_t r_den,
    uint32_t     g_num, uint32_t g_den,
    uint32_t     b_num, uint32_t b_den,
    const float *cam_to_srgb,
    uint8_t     *rgb_out)
{
    constexpr float DEFAULT_BAYER_NR_STRENGTH = 0.0f;
    constexpr float DEFAULT_BLACK_POINT_LIFT  = 0.025f;
    constexpr float DEFAULT_SATURATION        = 1.20f;
    constexpr float DEFAULT_TONE_CONTRAST     = 0.60f;
    constexpr float DEFAULT_EXPOSURE          = 1.0f;  // no-op
    constexpr float DEFAULT_LSC_STRENGTH      = 1.0f;  // ignored when table is NULL
    return demosaic_to_srgb_ex(
        raw_packed, raw_size, width, height,
        black_level, white_level,
        r_num, r_den, g_num, g_den, b_num, b_den,
        cam_to_srgb,
        DEFAULT_BAYER_NR_STRENGTH,
        DEFAULT_BLACK_POINT_LIFT,
        DEFAULT_SATURATION,
        DEFAULT_TONE_CONTRAST,
        DEFAULT_EXPOSURE,
        /*lsc_table=*/nullptr, DEFAULT_LSC_STRENGTH,
        rgb_out);
}
