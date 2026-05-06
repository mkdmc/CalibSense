/*
 * pipeline.cpp — CalibSense top-level frame pipeline (Sony D5503).
 *
 * Public entry point calibsense_process_frame(): receives a packed 10-bit RGGB Bayer
 * frame from the caller, builds a DNG (hand-rolled TIFF writer below),
 * and optionally also runs demosaic → sharpen → JPEG-encode and emits
 * both frames over the @calibsense_socket delivery channel.
 *
 * No external deps beyond libc / libstdc++.
 *
 * Pixel format: 6 × 10-bit Bayer pixels packed into one 64-bit LE word,
 *               875 words per row (= 7000 bytes), 5250 unpacked pixels
 *               per row of which the trailing 2 are padding (active = 5248).
 * DNG layout:    16-bit-per-pixel uncompressed Bayer (RGGB), DNG 1.4.
 */

#include "calibsense/calibsense.h"
#include "delivery.h"
#include "tiff_writer.h"
#include "demosaic.h"
#include "jpeg_enc.h"
#include "sharpen.h"
#include "lsc.h"

#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/stat.h>


namespace {

// All TIFF/DNG type and tag constants, plus the IfdBuilder class, live in
// tiff_writer.{h,cpp}. Pull them into this TU's lookup scope so calibsense_process_frame
// can write `IfdBuilder ifd; ifd.add_long_inline(TAG_NewSubfileType, 0);`
// without qualifying every reference.
using namespace calibsense::tiff;  // NOLINT(google-build-using-namespace)

// ---------- Sony D5503 / IMX2000A specifics -------------------------------
constexpr uint32_t SONY_BLACK_LEVEL  = 64;
constexpr uint32_t SONY_WHITE_LEVEL  = 1023;

// CFA pattern: RGGB
//   row 0: R G
//   row 1: G B
constexpr uint8_t CFA_PATTERN[4]      = { 0, 1, 1, 2 };
constexpr uint8_t CFA_PLANE_COLOR[3]  = { 0, 1, 2 };

// RAW header layout
constexpr size_t RAW_HEADER_BYTES         = 3272;       // 0xCC8
constexpr size_t RAW_HDR_OFF_MAGIC        = 0x000;      // "RAW\0"
constexpr size_t RAW_HDR_OFF_WIDTH        = 0x00C;
constexpr size_t RAW_HDR_OFF_HEIGHT       = 0x010;
constexpr size_t RAW_HDR_OFF_EXPOSURE     = 0x020;
constexpr size_t RAW_HDR_OFF_ANALOG_GAIN  = 0x024;
constexpr size_t RAW_HDR_OFF_GGAIN        = 0x030;
constexpr size_t RAW_HDR_OFF_RGAIN        = 0x038;
constexpr size_t RAW_HDR_OFF_BGAIN        = 0x054;
constexpr size_t RAW_HDR_OFF_AUTOMODE     = 0x060;

// Pixel-pack layout — derived per-call from `width`.
//   words_per_row    = ceil(width / 6)
//   bytes_per_row    = words_per_row * 8     (actual stride on disk)
//   pixels_per_row   = words_per_row * 6     (unpacked, includes padding)
// For the production D5503 full-res capture: width=5248,
//   words_per_row=875, bytes_per_row=7000, pixels_per_row=5250.

/*
 * Color matrices, stored as integer numerator/denominator with denominator
 * 128 (every value in the source data is exactly representable that way).
 *
 *   ColorMatrix1   — XYZ(D50) → camera RGB under CalibrationIlluminant1 (D65)
 *   ColorMatrix2   — XYZ(D50) → camera RGB under CalibrationIlluminant2 (Standard A)
 *   ForwardMatrix1 — camera RGB → XYZ(D50) at illuminant 1 (with white-balance)
 *   ForwardMatrix2 — camera RGB → XYZ(D50) at illuminant 2
 *   CameraCalibration{1,2} — per-unit calibration; identity for stock units.
 *
 * Calibration illuminants (DNG light source codes):
 *   21 = D65               (~6500 K, daylight)   — illuminant 1
 *   17 = Standard light A  (~2856 K, tungsten)   — illuminant 2
 */
constexpr int32_t COLOR_MATRIX_1[18] = {
     90, 128,    -9, 128,   -12, 128,
    -80, 128,   183, 128,    19, 128,
    -31, 128,    49, 128,    70, 128,
};
constexpr int32_t COLOR_MATRIX_2[18] = {
    127, 128,   -43, 128,    43, 128,
   -100, 128,   238, 128,    -1, 128,
    -15, 128,    38, 128,    80, 128,
};
constexpr int32_t FORWARD_MATRIX_1[18] = {
    102, 128,     6, 128,    15, 128,
     43, 128,    97, 128,   -12, 128,
     11, 128,   -49, 128,   144, 128,
};
constexpr int32_t FORWARD_MATRIX_2[18] = {
    111, 128,    19, 128,    -6, 128,
     46, 128,    86, 128,    -4, 128,
     -3, 128,   -85, 128,   194, 128,
};
constexpr int32_t CAMERA_CALIBRATION[18] = {
    1, 1,    0, 1,    0, 1,
    0, 1,    1, 1,    0, 1,
    0, 1,    0, 1,    1, 1,
};
constexpr uint16_t CALIB_ILLUM_1 = 21;   // D65
constexpr uint16_t CALIB_ILLUM_2 = 17;   // Standard A

// ---------- Little-endian helpers (alignment-safe) ------------------------
inline uint32_t read_u32(const uint8_t *p) {
    return (uint32_t)p[0]
         | ((uint32_t)p[1] << 8)
         | ((uint32_t)p[2] << 16)
         | ((uint32_t)p[3] << 24);
}
inline uint64_t read_u64_unaligned(const uint8_t *p) {
    uint64_t v;
    memcpy(&v, p, 8);
    return v;   // host is LE on ARMv7-A and x86
}

void format_datetime(char out[20]) {
    time_t t = time(nullptr);
    struct tm tmv;
    localtime_r(&t, &tmv);
    strftime(out, 20, "%Y:%m:%d %H:%M:%S", &tmv);
}

/*
 * Stream-unpack the packed Bayer data into a destination memory buffer.
 *
 * For each row we read `words_per_row` × 64-bit words from packed[],
 * extract 6 × 10-bit pixels from each word, and write the first `width`
 * 16-bit pixels into dst. The trailing pixels in the last word (e.g. 2
 * padding pixels at width=5248) are dropped.
 *
 * Returns 0 on success.
 */
int unpack_into_buffer(uint8_t *dst, const uint8_t *packed,
                       int width, int height, size_t words_per_row) {
    size_t bytes_per_row = words_per_row * 8;
    size_t out_stride    = (size_t)width * 2;

    for (int r = 0; r < height; ++r) {
        const uint8_t *src_row = packed + (size_t)r * bytes_per_row;
        uint16_t      *out_row = (uint16_t*)(dst + (size_t)r * out_stride);
        size_t         o = 0;

        for (size_t w = 0; w < words_per_row; ++w) {
            uint64_t word = read_u64_unaligned(src_row + w * 8);
            uint16_t pix[6] = {
                (uint16_t)((word >>  0) & 0x3FF),
                (uint16_t)((word >> 10) & 0x3FF),
                (uint16_t)((word >> 20) & 0x3FF),
                (uint16_t)((word >> 30) & 0x3FF),
                (uint16_t)((word >> 40) & 0x3FF),
                (uint16_t)((word >> 50) & 0x3FF),
            };
            for (int i = 0; i < 6 && o < (size_t)width; ++i) {
                out_row[o++] = pix[i];
            }
        }
    }
    return 0;
}

} // anonymous namespace


// ---------- Public API ---------------------------------------------------
extern "C" int calibsense_process_frame(
    const void *raw_header,
    size_t      header_size,
    const void *raw_packed,
    size_t      raw_size,
    int         width,
    int         height,
    const char *out_path,
    const void *wb_data)
{
    if (!raw_packed || !out_path) return 1;
    if (width <= 0 || height <= 0) return 1;


    // Compute packed-row stride from width.
    size_t words_per_row  = ((size_t)width + 5) / 6;
    size_t bytes_per_row  = words_per_row * 8;
    size_t pixels_per_row = words_per_row * 6;
    if ((size_t)width > pixels_per_row) return 1;
    size_t expected_raw = (size_t)height * bytes_per_row;
    if (raw_size < expected_raw) return 1;
    // raw_size may exceed expected_raw due to OS buffer padding; trailing slack ignored.

    // ---- Defaults: 1:1:1 (no WB applied)
    uint32_t r_gain = 1, g_gain = 1, b_gain = 1;
    bool wb_from_sidecar = false;

    // ---- Prefer WB sidecar gains (real Q16.16 AWB output)
    // Format: magic 'WB\0\0' at +0x00, RGain at +0x08, GGain at +0x0C, BGain at +0x10
    if (wb_data) {
        const uint8_t *w = (const uint8_t*)wb_data;
        uint32_t magic = read_u32(w + 0x00);
        if (magic == 0x00004257u) {  // 'WB\0\0' little-endian
            uint32_t rg = read_u32(w + 0x08);
            uint32_t gg = read_u32(w + 0x0C);
            uint32_t bg = read_u32(w + 0x10);
            if (gg && rg && bg) {
                g_gain = gg; r_gain = rg; b_gain = bg;
                wb_from_sidecar = true;
            }
        }
    }

    // ---- Fallback: RAW header AE/AWB metering values (less accurate)
    if (!wb_from_sidecar && raw_header && header_size >= RAW_HEADER_BYTES) {
        const uint8_t *h = (const uint8_t*)raw_header;
        if (memcmp(h, "RAW\0", 4) == 0) {
            uint32_t gg = read_u32(h + RAW_HDR_OFF_GGAIN);
            uint32_t rg = read_u32(h + RAW_HDR_OFF_RGAIN);
            uint32_t bg = read_u32(h + RAW_HDR_OFF_BGAIN);
            if (gg && rg && bg) { g_gain = gg; r_gain = rg; b_gain = bg; }
        }
    }

    // AsShotNeutral = (g/r, 1, g/b) — what the camera saw as neutral.
    // Stored as RATIONAL pairs (numerator, denominator) so we don't lose
    // precision to float conversion.
    uint32_t as_shot_pairs[6] = {
        g_gain, r_gain,
        1,      1,
        g_gain, b_gain,
    };

    // ---- Build the IFD
    IfdBuilder ifd;

    ifd.add_long_inline (TAG_NewSubfileType,  0);
    ifd.add_long_inline (TAG_ImageWidth,      (uint32_t)width);
    ifd.add_long_inline (TAG_ImageLength,     (uint32_t)height);
    ifd.add_short_inline(TAG_BitsPerSample,   16);          // 10-bit data, 16-bit container
    ifd.add_short_inline(TAG_Compression,     1);
    ifd.add_short_inline(TAG_Photometric,     32803);       // CFA
    ifd.add_ascii       (TAG_Make,            "Sony");
    ifd.add_ascii       (TAG_Model,           "D5503");
    ifd.add_long_inline (TAG_StripOffsets,    0);           // patched below
    ifd.add_short_inline(TAG_Orientation,     1);
    ifd.add_short_inline(TAG_SamplesPerPixel, 1);
    ifd.add_long_inline (TAG_RowsPerStrip,    (uint32_t)height);
    ifd.add_long_inline (TAG_StripByteCounts, (uint32_t)width * (uint32_t)height * 2u);
    ifd.add_short_inline(TAG_PlanarConfig,    1);
    ifd.add_ascii       (TAG_Software,        "libcalibsense 0.1");

    char dt[20];
    format_datetime(dt);
    ifd.add_ascii(TAG_DateTime, dt);

    // Display resolution — informational. Pixel-density metadata, not actual DPI.
    ifd.add_rational_one(TAG_XResolution, 72, 1);
    ifd.add_rational_one(TAG_YResolution, 72, 1);
    ifd.add_short_inline(TAG_ResolutionUnit, 2);    // 2 = inches

    // CFA pattern
    ifd.add_two_shorts        (TAG_CFARepeatPatternDim, 2, 2);
    ifd.add_byte_array_inline (TAG_CFAPattern, CFA_PATTERN, 4);

    // DNG-specific
    static const uint8_t dng_version[4]     = { 1, 4, 0, 0 };
    static const uint8_t dng_back_version[4]= { 1, 0, 0, 0 };
    ifd.add_byte_array_inline(TAG_DNGVersion,         dng_version, 4);
    ifd.add_byte_array_inline(TAG_DNGBackwardVersion, dng_back_version, 4);
    ifd.add_ascii            (TAG_UniqueCameraModel,  "Sony D5503 (IMX2000A)");
    ifd.add_byte_array_inline(TAG_CFAPlaneColor,      CFA_PLANE_COLOR, 3);
    ifd.add_short_inline     (TAG_CFALayout,          1);

    // Black/white levels (10-bit data => 64..1023)
    // Per-CFA-cell black levels. RepeatDim 2x2 means each CFA cell of the
    // 2x2 RGGB block has its own value (here all the same — Sony's master
    // dark frame already calibrated them out).
    {
        uint16_t bl4[4] = { SONY_BLACK_LEVEL, SONY_BLACK_LEVEL,
                            SONY_BLACK_LEVEL, SONY_BLACK_LEVEL };
        ifd.add_short_array(TAG_BlackLevelRepeatDim, (const uint16_t[]){2, 2}, 2);
        ifd.add_short_array(TAG_BlackLevel, bl4, 4);
    }
    ifd.add_long_inline (TAG_WhiteLevel,  SONY_WHITE_LEVEL);

    // Default scale, crop origin, crop size
    {
        uint32_t scale[4] = { 1,1, 1,1 };
        ifd.add_rationals(TAG_DefaultScale, scale, 2);
    }
    {
        uint32_t origin[4] = { 0,1, 0,1 };
        ifd.add_rationals(TAG_DefaultCropOrigin, origin, 2);
    }
    {
        uint32_t crop[4] = {
            (uint32_t)width,  1,
            (uint32_t)height, 1,
        };
        ifd.add_rationals(TAG_DefaultCropSize, crop, 2);
    }

    // Two-illuminant colour calibration
    ifd.add_short_inline(TAG_CalibrationIlluminant1, CALIB_ILLUM_1);
    ifd.add_short_inline(TAG_CalibrationIlluminant2, CALIB_ILLUM_2);
    ifd.add_srationals  (TAG_ColorMatrix1,        COLOR_MATRIX_1, 9);
    ifd.add_srationals  (TAG_ColorMatrix2,        COLOR_MATRIX_2, 9);
    ifd.add_srationals  (TAG_ForwardMatrix1,      FORWARD_MATRIX_1, 9);
    ifd.add_srationals  (TAG_ForwardMatrix2,      FORWARD_MATRIX_2, 9);
    ifd.add_srationals  (TAG_CameraCalibration1,  CAMERA_CALIBRATION, 9);
    ifd.add_srationals  (TAG_CameraCalibration2,  CAMERA_CALIBRATION, 9);

    // Per-frame WB
    ifd.add_rationals(TAG_AsShotNeutral, as_shot_pairs, 3);

    // Baseline exposure compensation (zero)
    {
        int32_t pair[2] = { 0, 100 };
        ifd.add_srationals(TAG_BaselineExposure, pair, 1);
    }

    // ---- Finalise: now we know the prefix size and can patch StripOffsets.
    // ----- EXIF IFD: per-shot exposure / aperture / iso / focal length
    {
        uint32_t exposure_us = 0;
        uint16_t iso         = 0;
        if (raw_header && header_size >= RAW_HEADER_BYTES) {
            const uint8_t *h = (const uint8_t*)raw_header;
            if (memcmp(h, "RAW\0", 4) == 0) {
                exposure_us = read_u32(h + RAW_HDR_OFF_EXPOSURE);
                // offset offset 0x24 stores ISO directly as a uint32 — no Q-scaling.
                uint32_t v = read_u32(h + RAW_HDR_OFF_ANALOG_GAIN);
                if (v) {
                    if (v > 25600) v = 25600;  // sanity clamp
                    iso = (uint16_t)v;
                }
            }
        }
        // Sony D5503: fixed f/2.0, 4.8mm focal length.
        ifd.add_exif_ifd(exposure_us, iso,
                         /*FNumber*/   2u, 1u,
                         /*FocalLen*/  49u, 10u,
                         /*DT*/        dt);
    }

    size_t   prefix_size  = ifd.finalize_prefix();
    uint32_t strip_offset = (uint32_t)prefix_size;
    ifd.patch_long_value(TAG_StripOffsets, strip_offset);

    // ---- Decide on a delivery sink before we touch any heap.
    //
    // The host process is typically 32-bit and address-space-constrained;
    // a 41 MB malloc per shot under memory pressure causes downstream
    // allocations to fail. So we only materialise the DNG when we have
    // somewhere to send it:
    //
    //   1. Socket listener connected → build DNG, stream it.
    //   2. No listener, but explicit out_path → build DNG, write file
    //      (offline / host-side testing path).
    //   3. Else → no-op. The caller's photo capture proceeds undisturbed.
    //
    bool have_listener = calibsense_delivery_has_peer();
    bool have_outpath  = (out_path && out_path[0]);

    if (!have_listener && !have_outpath) {
        return 0;
    }

    size_t pixel_bytes = (size_t)width * (size_t)height * 2;
    size_t total       = prefix_size + pixel_bytes;

    uint8_t *buf = (uint8_t*)malloc(total);
    if (!buf) {
        return 3;
    }

    memcpy(buf, ifd.data(), prefix_size);

    int rc = unpack_into_buffer(buf + prefix_size, (const uint8_t*)raw_packed,
                                width, height, words_per_row);
    if (rc != 0) {
        free(buf);
        return rc;
    }

    int sink_rc = 0;
    if (have_listener) {
        if (calibsense_delivery_send(FRAME_MAGIC_DNG, buf, total) != 0) {
            sink_rc = 4;
        }
    } else {
        // have_outpath, no listener (offline / test path)
        int fd = open(out_path, O_CREAT | O_WRONLY | O_TRUNC, 0644);
        if (fd < 0) {
            sink_rc = 2;
        } else {
            ssize_t got = write(fd, buf, total);
            close(fd);
            if (got != (ssize_t)total) sink_rc = 3;
        }
    }

    free(buf);

    // ====================================================================
    // Sibling JPEG: demosaic the same Bayer data into a 4:2:0 baseline JPEG
    // and ship it via the same delivery sink (socket or file). This runs
    // only if the DNG path itself succeeded — failure here is logged but
    // doesn't change the function's return code, since the DNG is the
    // primary deliverable.
    //
    // Memory peak shifts from "DNG buf alive" (~41 MB) to "RGB buf alive"
    // (~62 MB at full res) plus the JPEG output (~3-6 MB). The DNG buf is
    // already freed before we enter this block, so the two peaks don't
    // overlap.
    //
    // Time cost on the calling thread (D5503 measured, post-NEON, post-threading):
    //   demosaic_to_srgb_ex   ~3.0 s     (LSC + Bayer NR + MHC + tone block)
    //   sharpen_rgb8          ~1.7 s     (radius=3 unsharp mask, NEON+threaded)
    //   jpeg_encode_rgb       ~1.0-1.4 s (4:2:0, threaded YCbCr+DCT)
    //   ─────────────────────
    //   total                 ~5.7-6.1 s at 20 MP
    // The caller is already blocked through the DNG build at this
    // point; this extends that wait but introduces no new
    // synchronisation hazards.
    // ====================================================================

    if (sink_rc == 0) {
        // cam_RGB_wb → linear sRGB matrix, derived from FM1 (D65) above
        // composed with Bradford D50→D65 and standard XYZ_D65→linear sRGB.
        // Row sums ≈ 1.0 (neutral grey preserved).
        static const float CAM_TO_SRGB[9] = {
            +1.912046f, -0.890579f, -0.033183f,
            -0.133173f, +1.393261f, -0.256702f,
            +0.101192f, -0.707967f, +1.610468f,
        };

        constexpr int BLACK_LEVEL = 64;
        constexpr int WHITE_LEVEL = 1023 - 64;  // 959

        size_t rgb_bytes = (size_t)width * (size_t)height * 3u;
        uint8_t *rgb = (uint8_t*)malloc(rgb_bytes);
        if (rgb) {
            // The WB-gain numerator/denominator pairs the DNG path stores in
            // as_shot_pairs are (g/r, 1, g/b) — i.e. AsShotNeutral. For the
            // demosaic API we want raw R/G/B gains as (num, den) ratios; the
            // values we extracted above (r_gain, g_gain, b_gain) are exactly
            // those numerators against a common denominator (1 if scratched
            // back to integers, or 65536 if Q16.16 — either way the ratio
            // r_gain/g_gain is what matters and demosaic_to_srgb just needs
            // each as a num/den pair to compute that ratio.
            uint32_t r_num = r_gain, r_den = 1u;
            uint32_t g_num = g_gain, g_den = 1u;
            uint32_t b_num = b_gain, b_den = 1u;

            // ---- LSC table from RAW header ----
            //
            // The RAW header is 3272 bytes (RAW_HEADER_BYTES); LSC sits at
            // offset 0x5c8..0x7c8 (8 slots × 64 bytes = 512 bytes). We
            // parse it from the LIVE header rather than using the desktop's
            // embedded copy, so each unit gets its own factory calibration.
            // (The bytes happen to be byte-identical across captures from
            // a single body, but using the live header is more correct.)
            lsc_table_t lsc_parsed;
            const bool have_lsc = (raw_header
                                   && header_size >= 0x7c8 + 64
                                   && lsc_parse_from_table(
                                          (const uint8_t*)raw_header + 0x5c8,
                                          &lsc_parsed) == 0
                                   && lsc_parsed.valid);
            if (!have_lsc) {
                lsc_parsed.valid = 0;
            }

            // ---- Demosaic with on-device tuning ----
            //
            // Pipeline tuning constants. These values were chosen by
            // extensive A/B testing on representative shots:
            //
            //   bayer_nr_strength = 1.0    (Hadamard wavelet NR)
            //   black_point_lift  = 0.02   (mild dehaze)
            //   saturation        = 1.4    (punchy colors)
            //   tone_contrast     = 0.4    (moderate S-curve)
            //   exposure          = 1.5    (+0.58 stops, filmic curve)
            //   lsc_strength      = 1.0    (full lens-shading correction)
            //
            // Sharpening runs as a separate pass after demosaic (see below).
            constexpr float BAYER_NR_STRENGTH  = 1.0f;
            constexpr float BLACK_POINT_LIFT   = 0.02f;
            constexpr float SATURATION         = 1.4f;
            constexpr float TONE_CONTRAST      = 0.4f;
            constexpr float EXPOSURE           = 1.5f;
            constexpr float LSC_STRENGTH       = 1.0f;

            int dm = demosaic_to_srgb_ex(
                raw_packed, raw_size, width, height,
                BLACK_LEVEL, WHITE_LEVEL,
                r_num, r_den, g_num, g_den, b_num, b_den,
                CAM_TO_SRGB,
                BAYER_NR_STRENGTH,
                BLACK_POINT_LIFT,
                SATURATION,
                TONE_CONTRAST,
                EXPOSURE,
                /*lsc_table=*/ have_lsc ? &lsc_parsed : nullptr,
                LSC_STRENGTH,
                rgb);
            if (dm != 0) {
                free(rgb);
            } else {
                // ---- Sharpen (luma USM in gamma sRGB space) ----
                //
                //   amount=0.8, radius=3.0, threshold=3
                //
                // The radius-3 Gaussian gives a wider halo profile than
                // the more conservative radius=1.0 default, intended to
                // recover the perceptual sharpness lost to Sony's optical
                // softness on this fixed-lens sensor. amount=0.8 keeps the
                // pop modest enough to avoid over-sharpening artefacts on
                // edges; threshold=3 skips amplifying sub-3-luma noise.
                //
                // Soft-thresholded shrinkage + headroom-aware additive
                // delta (see sharpen.cpp) keeps colored edges and bright
                // walls from clipping or speckling.
                constexpr float SHARPEN_AMOUNT    = 0.8f;
                constexpr float SHARPEN_RADIUS    = 3.0f;
                constexpr int   SHARPEN_THRESHOLD = 3;
                sharpen_rgb8(rgb, width, height,
                             SHARPEN_AMOUNT, SHARPEN_RADIUS, SHARPEN_THRESHOLD);

                // Build EXIF from the RAW header (same fields as the DNG).
                exif_info ex = {};
                if (raw_header && header_size >= RAW_HEADER_BYTES) {
                    const uint8_t *h = (const uint8_t*)raw_header;
                    if (memcmp(h, "RAW\0", 4) == 0) {
                        ex.exposure_us = read_u32(h + RAW_HDR_OFF_EXPOSURE);
                        uint32_t v = read_u32(h + RAW_HDR_OFF_ANALOG_GAIN);
                        if (v) {
                            if (v > 25600) v = 25600;
                            ex.iso = (uint16_t)v;
                        }
                    }
                }
                ex.fnum_n  = 2;  ex.fnum_d  = 1;     // f/2.0
                ex.focal_n = 49; ex.focal_d = 10;    // 4.9 mm
                ex.datetime    = dt;
                ex.make        = "Sony";
                ex.model       = "D5503";
                ex.orientation = 1;

                uint8_t *jpg      = nullptr;
                size_t   jpg_size = 0;
                // Quality 96 to match deployed desktop settings.
                int je = jpeg_encode_rgb(rgb, width, height, /*quality*/ 96,
                                         &ex, &jpg, &jpg_size);
                free(rgb);

                if (je == 0 && jpg) {
                    if (have_listener) {
                        (void)calibsense_delivery_send(FRAME_MAGIC_JPG, jpg, jpg_size);
                    } else {
                        // Offline / file fallback: derive a sibling path by
                        // swapping the DNG extension to .jpg (or appending .jpg
                        // if there's no dot in the basename).
                        const char *slash = strrchr(out_path, '/');
                        const char *base  = slash ? slash + 1 : out_path;
                        const char *dot   = strrchr(base, '.');
                        size_t stem_len = dot ? (size_t)(dot - out_path)
                                              : strlen(out_path);
                        size_t jp_len = stem_len + 4 + 1;  // ".jpg\0"
                        char *jp = (char*)malloc(jp_len);
                        if (jp) {
                            memcpy(jp, out_path, stem_len);
                            memcpy(jp + stem_len, ".jpg", 5);
                            int fd = open(jp, O_CREAT | O_WRONLY | O_TRUNC, 0644);
                            if (fd >= 0) {
                                // Best-effort: the DNG (the primary
                                // deliverable) has already been written
                                // by this point, so a failed JPEG write
                                // is silently ignored.
                                ssize_t got = write(fd, jpg, jpg_size);
                                (void)got;
                                close(fd);
                            }
                            free(jp);
                        }
                    }

                    free(jpg);
                }
            }
        }
    }

    return sink_rc;
}

extern "C" int calibsense_process_file(const char *raw_path, const char *wb_path, const char *out_path) {
    if (!raw_path || !out_path) return 1;

    int fd = open(raw_path, O_RDONLY);
    if (fd < 0) return 2;

    struct stat st;
    if (fstat(fd, &st) < 0) { close(fd); return 2; }

    size_t file_size = (size_t)st.st_size;
    if (file_size < RAW_HEADER_BYTES) { close(fd); return 1; }

    uint8_t *buf = (uint8_t*)malloc(file_size);
    if (!buf) { close(fd); return 3; }

    if (read(fd, buf, file_size) != (ssize_t)file_size) {
        free(buf); close(fd); return 3;
    }
    close(fd);

    int width  = (int)read_u32(buf + RAW_HDR_OFF_WIDTH);
    int height = (int)read_u32(buf + RAW_HDR_OFF_HEIGHT);

    size_t words_per_row = ((size_t)width + 5) / 6;
    size_t bytes_per_row = words_per_row * 8;

    if (width <= 0 || height <= 0 ||
        file_size < RAW_HEADER_BYTES + (size_t)height * bytes_per_row) {
        free(buf);
        return 1;
    }

    // Optionally load WB sidecar
    uint8_t wb_buf[256] = {0};
    const void *wb_ptr = nullptr;
    if (wb_path) {
        int wfd = open(wb_path, O_RDONLY);
        if (wfd >= 0) {
            ssize_t n = read(wfd, wb_buf, sizeof(wb_buf));
            close(wfd);
            if (n >= 0x14) wb_ptr = wb_buf;
        }
    }

    int rc = calibsense_process_frame(
        buf, RAW_HEADER_BYTES,
        buf + RAW_HEADER_BYTES, (size_t)height * bytes_per_row,
        width, height,
        out_path, wb_ptr);

    free(buf);
    return rc;
}
