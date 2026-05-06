/*
 * lsc.h — Lens Shading Correction parser and applicator for Sony D5503.
 *
 * The Sony RAW header carries factory-calibrated LSC tables at offset 0x5c8.
 *
 *   Slot  0 @ 0x5c8 : R-channel,  illuminant A (daylight)
 *   Slot  1 @ 0x608 : Gr-channel, illuminant A
 *   Slot  2 @ 0x648 : Gb-channel, illuminant A
 *   Slot  3 @ 0x688 : B-channel,  illuminant A
 *   Slot  4 @ 0x6c8 : R-channel,  illuminant B (tungsten)
 *   Slot  5 @ 0x708 : Gr-channel, illuminant B
 *   Slot  6 @ 0x748 : Gb-channel, illuminant B
 *   Slot  7 @ 0x788 : B-channel,  illuminant B
 *
 * Each slot is a 7×9 grid of uint8 values stored at 64-byte stride
 * (63 bytes payload + 1 padding). Values represent the MEASURED relative
 * brightness profile of the lens, peaking at 128 in the center (3,4)
 * and falling off to ~38-42 at corners. To compute gain: gain = 128 / value.
 *
 * The high-ISO mode (e.g., ISO 12800) writes signed near-zero values instead
 * of the normal 38-128 range. We detect this case by checking the center
 * value (must be 128 for normal-mode tables) and skip LSC if absent.
 */
#ifndef LSC_H
#define LSC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Geometry constants — match the on-disk layout. */
#define LSC_GRID_W        9          /* horizontal samples */
#define LSC_GRID_H        7          /* vertical samples */
#define LSC_GRID_BYTES    (LSC_GRID_W * LSC_GRID_H)   /* 63 */
#define LSC_SLOT_STRIDE   64         /* 63 + 1 byte padding */
#define LSC_HEADER_OFFSET 0x5c8      /* start of slot 0 in Sony RAW header */
#define LSC_NUM_CHANNELS  4          /* R, Gr, Gb, B */
#define LSC_PEAK_VALUE    128        /* center value of every normal-mode grid */

/*
 * Parsed LSC table, ready for application.
 *
 * `valid` is false if parsing failed or if the table is in a non-normal mode
 * (e.g., the high-ISO 0xffffeec8 case). When invalid, callers should skip LSC.
 *
 * `set_a` and `set_b` are the two illuminant-specific table sets, each holding
 * four channel grids in RGGB order. In float form, normalized so center == 1.0
 * (i.e., raw_value / 128.0). Apply as `corrected = signal / set[ch][y][x]`,
 * which is equivalent to multiplying by `1 / lsc` = the gain map.
 */
typedef struct {
    int   valid;
    float set_a[LSC_NUM_CHANNELS][LSC_GRID_H][LSC_GRID_W]; /* illuminant A */
    float set_b[LSC_NUM_CHANNELS][LSC_GRID_H][LSC_GRID_W]; /* illuminant B */
} lsc_table_t;

/*
 * Parse a Sony RAW header and populate `out`. Returns 0 on success.
 * On failure, sets out->valid = 0 and returns nonzero.
 *
 * `header_bytes` must contain at least LSC_HEADER_OFFSET + 8*LSC_SLOT_STRIDE
 * (= 0x7c8) bytes.
 */
int lsc_parse(const uint8_t *header_bytes, size_t header_size, lsc_table_t *out);

/*
 * Parse only the LSC payload region (8 slots × 64 bytes = 512 bytes),
 * starting at slot 0. Equivalent to lsc_parse on a header where the LSC
 * region is at offset 0. Useful for callers that have already extracted
 * the LSC bytes (e.g., a hardcoded copy for desktop testing).
 *
 * `lsc_bytes` must point to at least 8 * LSC_SLOT_STRIDE = 512 bytes.
 */
int lsc_parse_from_table(const uint8_t *lsc_bytes, lsc_table_t *out);

/*
 * Pre-blended LSC grids ready for per-pixel sampling. Computed once per image
 * by lsc_blend; then sampled per pixel during the BL+WB pass in demosaic.cpp
 * to avoid an extra 41 MB pass over the Bayer plane.
 *
 * Each entry is a "shading factor" in (0, 1] — peak 1.0 at center, ~0.30 at
 * corners. Apply as: corrected = signal / shading_factor.
 */
typedef struct {
    int   valid;
    float grid[LSC_NUM_CHANNELS][LSC_GRID_H][LSC_GRID_W];
} lsc_blended_t;

/*
 * Compute the per-channel grid blended between set A and set B by `warmth`.
 * `strength` in [0,1] scales the deviation from 1.0 (no-correction):
 *   shading = 1 + strength * (raw_shading - 1)
 * so strength=0 → grid is all 1.0 (LSC disabled), strength=1 → full correction.
 *
 * Returns 0 on success. If !table->valid, sets out->valid = 0 and returns 0
 * (caller's loop should check valid before applying).
 */
int lsc_blend(const lsc_table_t *table, float warmth, float strength,
              lsc_blended_t *out);

/*
 * Bilinearly sample the blended grid at fractional grid coordinates.
 * Caller computes (gy, gx) from pixel coordinates using:
 *   gy = pixel_y * (LSC_GRID_H - 1) / (image_height - 1)
 *   gx = pixel_x * (LSC_GRID_W - 1) / (image_width  - 1)
 */
float lsc_sample(const lsc_blended_t *blended, int channel, float gy, float gx);

/*
 * Apply LSC to a Bayer plane (RGGB layout).
 *
 *   bayer       : uint16_t plane of `width` × `height`, post-black-level-subtract
 *                 but pre-WB. Modified in place.
 *   width,height: dimensions; both must be even.
 *   table       : parsed LSC table. If !table->valid, this function is a no-op.
 *   warmth      : color-temperature signal in [0, 1]. 0 = use set A only
 *                 (default; suitable for daylight). 1 = use set B only
 *                 (suitable for tungsten). Intermediate values blend linearly.
 *                 The caller computes this from WB ratios — see lsc_warmth_from_wb.
 *   strength    : 0.0..1.0, scales the correction. 1.0 = full correction,
 *                 0.0 = no correction (multiply gain by 1.0). Useful for
 *                 dialing back at high ISO where corner gain amplifies noise.
 *
 * Note: this is a convenience wrapper that performs an extra pass over the
 * Bayer plane. For best performance, callers that already iterate the plane
 * (like demosaic_to_srgb_ex) should use lsc_blend + lsc_sample inline instead.
 */
void lsc_apply_bayer(uint16_t *bayer, int width, int height,
                     const lsc_table_t *table,
                     float warmth, float strength);

/*
 * Estimate "warmth" in [0, 1] from white-balance gains.
 *
 * gR, gB are WB gains relative to G=1.0. Under daylight gR ≈ gB; under
 * tungsten gR << gB (much more blue gain needed because the scene is red-rich).
 *
 * Returns:
 *   warmth = 0.0  for cool (gR/gB ≥ 1.1)   → use set A only
 *   warmth = 1.0  for warm (gR/gB ≤ 0.6)   → use set B only
 *   linear blend in between
 */
float lsc_warmth_from_wb(float gR, float gB);

#ifdef __cplusplus
}
#endif

#endif /* LSC_H */
