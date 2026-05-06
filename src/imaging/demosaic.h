/*
 * demosaic.h — Bayer → sRGB demosaic + color pipeline.
 *
 * One-shot conversion of a packed 10-bit RGGB Bayer buffer (the same
 * format calibsense_process_frame receives) into an interleaved 8-bit sRGB image
 * suitable for JPEG encoding.
 *
 * Pipeline:
 *   1. unpack 10-bit Bayer
 *   2. black-level subtract (64)
 *   3. white-balance multiply (per-channel, relative to G)
 *   4. Malvar–He–Cutler 5×5 demosaic
 *   5. cam_RGB_wb → linear sRGB via 3×3 matrix
 *   6. clip [0,1]
 *   7. gamma 2.2 via 14-bit LUT
 */

#ifndef DEMOSAIC_H
#define DEMOSAIC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * raw_packed   : 6×10-bit-per-uint64 packed Bayer plane as supplied by
 *                the caller. words_per_row = (width+5)/6.
 * raw_size     : packed buffer length; must cover height × words_per_row × 8
 * width/height : sensor active dimensions (5248 × 3936 on D5503)
 * black_level  : value to subtract from every sample (64 for D5503)
 * white_level  : sample range upper bound after black subtract (959)
 * r_num/r_den  : Q16.16 R-channel WB gain (numerator/denominator from sidecar)
 * g_num/g_den  : Q16.16 G-channel WB gain
 * b_num/b_den  : Q16.16 B-channel WB gain
 * cam_to_srgb  : 9-element row-major float matrix (cam_RGB_wb → linear sRGB)
 * rgb_out      : caller-owned output buffer, width*height*3 bytes
 *
 * Returns 0 on success, non-zero on failure.
 *
 * This wrapper uses default look parameters tuned for normal scenes.
 * Use demosaic_to_srgb_ex below to override.
 */
int demosaic_to_srgb(
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
    uint8_t     *rgb_out);

/*
 * Same as demosaic_to_srgb, with explicit look-tuning parameters.
 *
 * bayer_nr_strength: 0.0..2.0, default 0.0
 *                    per-channel bilateral filter on the Bayer plane,
 *                    applied AFTER unpacking but BEFORE the demosaic.
 *                    Each channel filtered using only same-color
 *                    neighbors, so chromatic edges and color textures
 *                    are preserved exactly. 0=off, 0.5=mild ISO 100,
 *                    1.0=moderate (~ISO 400-800), 2.0=aggressive (high
 *                    ISO). D5503 measured: ~1.7 s at strength=1.0,
 *                    full res, with 4-channel parallel wavelet denoise.
 *
 * black_point_lift : 0.0..0.1, default 0.025
 *                    subtracted from linear sRGB before clip; deepens shadows
 *                    and removes the "smoky" residual lift from sensor noise +
 *                    matrix error + veiling glare. Higher = more contrast in
 *                    shadows but loses dark-tone detail past ~0.05.
 *
 * saturation       : 0.5..2.0, default 1.20
 *                    multiplier applied to chroma deviation from luma in
 *                    linear sRGB. 1.0 = neutral, >1.0 = punchier colors,
 *                    <1.0 = more pastel. Hue is preserved.
 *
 * tone_contrast    : 0.0..1.0, default 0.60
 *                    strength of an S-curve baked into the gamma LUT.
 *                    0 = pure linear→2.2-gamma, 1 = full smoothstep curve.
 *                    Mid-values give photographic contrast; high values can
 *                    crush midtones.
 *
 * exposure         : 0.25..4.0, default 1.0
 *                    linear-space exposure multiplier applied AFTER the
 *                    color matrix and BEFORE black-point lift, saturation,
 *                    and gamma. 1.0 = no change; 1.5 ≈ +0.6 stops;
 *                    2.0 = +1 stop; 0.5 = -1 stop.
 *                    Linear-space placement is correct for "exposure":
 *                    a true +1 stop is multiplication by 2 in linear,
 *                    not in gamma. Doing this in gamma space would
 *                    over-lift shadows and under-lift highlights.
 *                    Highlights past 1.0 in linear sRGB clip to white
 *                    at the gamma stage — that's the photographic
 *                    cost of brightening.
 *
 * lsc_table        : pointer to a parsed LSC table (see lsc.h), or NULL
 *                    to disable LSC. Caller is responsible for parsing
 *                    via lsc_parse / lsc_parse_from_table. This is a
 *                    void* so the public header avoids depending on
 *                    lsc.h; cast to (const lsc_table_t*) internally.
 * lsc_strength     : 0.0..1.0, default 1.0. Scales the LSC correction.
 *                    1.0 = full per-channel/per-illuminant correction.
 *                    0.0 = no correction (LSC effectively disabled).
 *                    Useful for high-ISO shots where corner gain amplifies
 *                    noise; try 0.5 at ISO > 1600.
 */
int demosaic_to_srgb_ex(
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
    uint8_t     *rgb_out);

#ifdef __cplusplus
}
#endif

#endif
