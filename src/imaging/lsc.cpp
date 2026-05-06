/*
 * lsc.cpp — Lens Shading Correction parser and applicator.
 *
 * See lsc.h for the file format and pipeline placement.
 *
 * Algorithm:
 *   1. parse: read 8 grids of 7x9 uint8 from the header, normalize so center
 *      becomes 1.0 (divide by 128). Validate that each grid has its peak
 *      value at the center (3,4) — this is the integrity check that lets us
 *      detect malformed or high-ISO tables.
 *
 *   2. apply: for each Bayer position (R, Gr, Gb, B), bilinearly upsample the
 *      blended-by-warmth grid to the per-channel sub-resolution (width/2 ×
 *      height/2), and divide each pixel by the upsampled value. Equivalent
 *      to multiplying by gain = 1 / shading.
 *
 * Performance: we don't materialize a full per-pixel float gain map. Instead
 * we compute the bilinear interpolation on-the-fly per pixel — cheap because
 * the 7x9 source grid is tiny. The rolling state per row is a few floats.
 */

#include "lsc.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

/* ============================================================================
 * Parsing
 * ============================================================================ */

static int parse_one_slot(const uint8_t *p, float out[LSC_GRID_H][LSC_GRID_W])
{
    /* Sanity-check the center value. Normal-mode LSC tables always have
     * value 128 at position (3, 4). Anything else means we're either
     * looking at the wrong bytes or the camera is in a special mode
     * (high-ISO, mfg test, etc.) where we should skip LSC. */
    int center = p[3 * LSC_GRID_W + 4];
    if (center != LSC_PEAK_VALUE) {
        return 1;
    }

    /* Also reject any value of 0, which would cause a division by zero.
     * Real LSC values are bounded below by ~30. A zero strongly suggests
     * we've fallen off the end of the table. */
    for (int i = 0; i < LSC_GRID_BYTES; ++i) {
        if (p[i] == 0) return 2;
    }

    /* Normalize: stored value / 128 gives a "shading factor" in (0, 1].
     * 1.0 at center, ~0.30 at corners. The applicator divides signal
     * by this factor to recover uniform brightness. */
    for (int y = 0; y < LSC_GRID_H; ++y) {
        for (int x = 0; x < LSC_GRID_W; ++x) {
            out[y][x] = (float)p[y * LSC_GRID_W + x] / (float)LSC_PEAK_VALUE;
        }
    }
    return 0;
}

extern "C" int lsc_parse_from_table(const uint8_t *lsc_bytes, lsc_table_t *out)
{
    if (!lsc_bytes || !out) return -1;

    memset(out, 0, sizeof(*out));

    /* Parse slots 0..3 (set A) and 4..7 (set B). The two sets sit at
     * adjacent offsets in slot-stride units. */
    for (int ch = 0; ch < LSC_NUM_CHANNELS; ++ch) {
        const uint8_t *pa = lsc_bytes +  ch                       * LSC_SLOT_STRIDE;
        const uint8_t *pb = lsc_bytes + (ch + LSC_NUM_CHANNELS)   * LSC_SLOT_STRIDE;
        if (parse_one_slot(pa, out->set_a[ch]) != 0) return -3;
        if (parse_one_slot(pb, out->set_b[ch]) != 0) return -4;
    }

    out->valid = 1;
    return 0;
}

extern "C" int lsc_parse(const uint8_t *header_bytes, size_t header_size,
                          lsc_table_t *out)
{
    if (!header_bytes || !out) return -1;

    /* Need at least 8 slots' worth of data after the header offset. */
    size_t needed = LSC_HEADER_OFFSET + 8 * LSC_SLOT_STRIDE;
    if (header_size < needed) {
        if (out) memset(out, 0, sizeof(*out));
        return -2;  /* header too short */
    }

    return lsc_parse_from_table(header_bytes + LSC_HEADER_OFFSET, out);
}

/* ============================================================================
 * Warmth heuristic
 * ============================================================================ */

extern "C" float lsc_warmth_from_wb(float gR, float gB)
{
    /* The ratio gR/gB is a clean color-temperature indicator:
     *   cool (daylight)  → ratio ≈ 1.0 - 1.4
     *   warm (tungsten)  → ratio ≈ 0.4 - 0.7
     *
     * We map ratio ≥ 1.1 to warmth=0 (pure set A) and ratio ≤ 0.6 to
     * warmth=1 (pure set B), with linear blend in between. The thresholds
     * leave a "neutral daylight" zone where set A is fully used.
     *
     * If gB is non-positive (sanity), default to fully cool. */
    if (gB <= 0.0f) return 0.0f;
    float ratio = gR / gB;

    const float COOL_RATIO = 1.1f;  /* warmth = 0 above this */
    const float WARM_RATIO = 0.6f;  /* warmth = 1 below this */

    if (ratio >= COOL_RATIO) return 0.0f;
    if (ratio <= WARM_RATIO) return 1.0f;
    return (COOL_RATIO - ratio) / (COOL_RATIO - WARM_RATIO);
}

/* ============================================================================
 * Application
 * ============================================================================ */

/* Bilinear interpolate a 7x9 grid at fractional position (gy, gx) in
 * [0..H-1] × [0..W-1]. */
static inline float bilinear_grid(const float grid[LSC_GRID_H][LSC_GRID_W],
                                    float gy, float gx)
{
    int y0 = (int)gy;  if (y0 < 0) y0 = 0;
    int x0 = (int)gx;  if (x0 < 0) x0 = 0;
    int y1 = y0 + 1;   if (y1 >= LSC_GRID_H) { y1 = LSC_GRID_H - 1; y0 = y1 - 1; }
    int x1 = x0 + 1;   if (x1 >= LSC_GRID_W) { x1 = LSC_GRID_W - 1; x0 = x1 - 1; }
    float fy = gy - (float)y0;
    float fx = gx - (float)x0;
    if (fy < 0) fy = 0; else if (fy > 1) fy = 1;
    if (fx < 0) fx = 0; else if (fx > 1) fx = 1;

    float v00 = grid[y0][x0];
    float v01 = grid[y0][x1];
    float v10 = grid[y1][x0];
    float v11 = grid[y1][x1];
    float v0  = v00 * (1.0f - fx) + v01 * fx;
    float v1  = v10 * (1.0f - fx) + v11 * fx;
    return v0  * (1.0f - fy) + v1  * fy;
}

extern "C" int lsc_blend(const lsc_table_t *table, float warmth, float strength,
                          lsc_blended_t *out)
{
    if (!table || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (!table->valid) return 0;  /* out->valid stays 0 */

    /* Blend per-channel sets A and B by warmth, then apply strength.
     *
     * blended_per_ch = (1-warmth) * set_a + warmth * set_b
     * attenuated     = 1 + strength * (blended_per_ch - 1)
     *
     * At strength=0 attenuated == 1 (no correction). At strength=1 attenuated
     * equals the full blended grid. The caller divides signal by these values,
     * so attenuated=1 means "leave signal alone."
     *
     * BALANCING CHROMATIC BIAS (preserves shape, removes global tint):
     *
     * The factory tables encode wavelength-dependent vignetting: R, Gr, Gb, B
     * have slightly different shading profiles. Applied per-channel as-is,
     * this shifts R:G:B ratios everywhere except dead-center, producing a
     * mild pink/magenta tint over the whole image (~2-3% per channel).
     *
     * The tint cause: AsShotNeutral is calibrated to the lens-attenuated
     * signal at center (where LSC=identity). Off-center, LSC introduces a
     * chromatic adjustment that WB doesn't compensate.
     *
     * Fix: rebalance the per-channel gains so that
     *   (a) Center gain stays exactly 1.0 for every channel (no center shift)
     *   (b) Each channel has the same MEAN gain across the grid (no global tint)
     *   (c) The relative SHAPE differences between channels are preserved
     *       — i.e. corners still get slight R/B vs G chromatic correction,
     *       just less aggressively than the raw table dictates.
     *
     * Implementation: per-channel scale_ch in
     *   gain'_ch(pos) = 1 + (gain_ch(pos) - 1) * scale_ch
     * chosen so that mean(gain'_ch) equals a common target across channels.
     * Solving:
     *   target = 1 + (mean_gain_ch - 1) * scale_ch
     *   scale_ch = (target - 1) / (mean_gain_ch - 1)
     *
     * We pick target = mean of Gr's full gain (an arbitrary common reference;
     * any of the four channel means would work and produces equivalent
     * relative balance).
     *
     * Effect:
     *   At center: gain'_ch(center) = 1 + (1 - 1) * scale_ch = 1.0  ✓
     *   At corners: per-channel differences are preserved but ~75% of the
     *               global chromatic offset is removed.
     *   On average across grid: all channels have equal mean gain → no tint.
     *
     * Compared to the previous luma-only fix, this keeps about 25% of the
     * lens's wavelength-dependent vignetting correction at corners.
     */
    const float wa = 1.0f - warmth;
    const float wb = warmth;

    /* Step 1: blend sets A/B and apply strength, per channel. */
    float per_channel[LSC_NUM_CHANNELS][LSC_GRID_H][LSC_GRID_W];
    for (int ch = 0; ch < LSC_NUM_CHANNELS; ++ch) {
        for (int y = 0; y < LSC_GRID_H; ++y) {
            for (int x = 0; x < LSC_GRID_W; ++x) {
                float v = wa * table->set_a[ch][y][x]
                        + wb * table->set_b[ch][y][x];
                v = 1.0f + strength * (v - 1.0f);
                if (v < 1e-3f) v = 1e-3f;
                per_channel[ch][y][x] = v;
            }
        }
    }

    /* Step 2: compute per-channel mean gain.
     *
     * mean_gain_ch = mean over grid of (1 / per_channel_value)
     */
    float mean_gain[LSC_NUM_CHANNELS];
    for (int ch = 0; ch < LSC_NUM_CHANNELS; ++ch) {
        float sum = 0.0f;
        for (int y = 0; y < LSC_GRID_H; ++y) {
            for (int x = 0; x < LSC_GRID_W; ++x) {
                sum += 1.0f / per_channel[ch][y][x];
            }
        }
        mean_gain[ch] = sum / (LSC_GRID_H * LSC_GRID_W);
    }

    /* Step 3: pick a target mean gain. We use the average across the four
     * channel means — this minimizes the total scale-factor magnitude
     * (each channel scales toward the common average rather than toward
     * one specific channel's mean).
     */
    float target_mean = 0.0f;
    for (int ch = 0; ch < LSC_NUM_CHANNELS; ++ch) {
        target_mean += mean_gain[ch];
    }
    target_mean /= (float)LSC_NUM_CHANNELS;

    /* Step 4: per-channel scale.
     *
     * For channel ch:
     *   scale_ch = (target_mean - 1) / (mean_gain_ch - 1)
     *
     * If mean_gain_ch is essentially 1.0 (no correction at all anywhere —
     * i.e. all values are 128, e.g. a degenerate table), scale becomes
     * undefined. Fall back to scale = 1 in that edge case.
     */
    float scale[LSC_NUM_CHANNELS];
    for (int ch = 0; ch < LSC_NUM_CHANNELS; ++ch) {
        float denom = mean_gain[ch] - 1.0f;
        if (denom < 1e-4f && denom > -1e-4f) {
            scale[ch] = 1.0f;
        } else {
            scale[ch] = (target_mean - 1.0f) / denom;
        }
    }

    /* Step 5: write the rebalanced shading values.
     *
     * The applied gain is: gain'_ch(pos) = 1 + (gain_ch(pos) - 1) * scale_ch
     * Equivalently in shading-value form (which is what we store):
     *   shading' = 1 / gain'_ch
     * The downstream code divides signal by shading'.
     *
     * Compute via gain → invert. Defensive clamp away from zero.
     */
    for (int ch = 0; ch < LSC_NUM_CHANNELS; ++ch) {
        for (int y = 0; y < LSC_GRID_H; ++y) {
            for (int x = 0; x < LSC_GRID_W; ++x) {
                float orig_gain   = 1.0f / per_channel[ch][y][x];
                float adj_gain    = 1.0f + (orig_gain - 1.0f) * scale[ch];
                /* adj_gain >= 1.0 always (since orig_gain >= 1.0 and
                 * scale_ch >= 0 in any non-pathological table), but clamp
                 * defensively. */
                if (adj_gain < 1e-3f) adj_gain = 1e-3f;
                out->grid[ch][y][x] = 1.0f / adj_gain;
            }
        }
    }
    out->valid = 1;
    return 0;
}

extern "C" float lsc_sample(const lsc_blended_t *blended, int channel,
                              float gy, float gx)
{
    if (!blended || !blended->valid) return 1.0f;
    if (channel < 0 || channel >= LSC_NUM_CHANNELS) return 1.0f;
    return bilinear_grid(blended->grid[channel], gy, gx);
}

extern "C" void lsc_apply_bayer(uint16_t *bayer, int width, int height,
                                  const lsc_table_t *table,
                                  float warmth, float strength)
{
    if (!table || !table->valid) return;
    if (!bayer || width <= 0 || height <= 0) return;
    if (strength <= 0.0f) return;

    lsc_blended_t blended;
    if (lsc_blend(table, warmth, strength, &blended) != 0) return;
    if (!blended.valid) return;

    const float scale_x = (float)(LSC_GRID_W - 1) / (float)(width  - 1);
    const float scale_y = (float)(LSC_GRID_H - 1) / (float)(height - 1);

    /* Bayer parity for RGGB:
     *   (0,0) = R     channel 0
     *   (0,1) = Gr    channel 1
     *   (1,0) = Gb    channel 2
     *   (1,1) = B     channel 3 */
    for (int y = 0; y < height; ++y) {
        const float gy = (float)y * scale_y;
        const int   yp = y & 1;
        uint16_t   *row = bayer + (size_t)y * (size_t)width;

        for (int x = 0; x < width; ++x) {
            const int xp = x & 1;
            const int ch = (yp << 1) | xp;
            float gx = (float)x * scale_x;

            float shading = bilinear_grid(blended.grid[ch], gy, gx);
            float gain = 1.0f / shading;

            int v = (int)((float)row[x] * gain + 0.5f);
            if (v < 0) v = 0;
            else if (v > 65535) v = 65535;
            row[x] = (uint16_t)v;
        }
    }
}
