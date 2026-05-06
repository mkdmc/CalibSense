/*
 * jpeg_enc.h — minimal baseline JPEG encoder with EXIF support.
 *
 * Encodes interleaved 8-bit RGB into a baseline JFIF JPEG with 4:2:0
 * chroma subsampling. Optional EXIF APP1 segment is inserted right
 * after the JFIF APP0 segment.
 *
 * Self-contained: no external libraries beyond libc/libm. Output goes
 * to a caller-owned growable buffer via simple realloc.
 *
 * Quality range 1..100. Output buffer is malloc'd; caller free()s.
 */

#ifndef JPEG_ENC_H
#define JPEG_ENC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * EXIF metadata to embed. Any zero numerator means the field is omitted.
 */
typedef struct {
    uint32_t exposure_us;     // ExposureTime numerator (denominator 1000000)
    uint32_t fnum_n, fnum_d;  // FNumber as rational
    uint16_t iso;             // ISOSpeedRatings
    uint32_t focal_n, focal_d;// FocalLength as rational
    const char *datetime;     // "YYYY:MM:DD HH:MM:SS" or NULL
    const char *make;         // "Sony" or NULL
    const char *model;        // "D5503" or NULL
    uint16_t orientation;     // 1 = normal; 0 = omit
} exif_info;

/*
 * Encode RGB → JPEG. Returns 0 on success.
 *
 * On success, *out_buf is set to a malloc'd buffer the caller owns,
 * and *out_size to its length.
 *
 * exif may be NULL to skip the APP1 segment.
 */
int jpeg_encode_rgb(
    const uint8_t *rgb,
    int            width,
    int            height,
    int            quality,
    const exif_info *exif,
    uint8_t      **out_buf,
    size_t        *out_size);

#ifdef __cplusplus
}
#endif

#endif
