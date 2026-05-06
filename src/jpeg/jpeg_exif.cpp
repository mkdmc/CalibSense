/*
 * jpeg_exif.cpp — APP1 EXIF segment builder. See jpeg_exif.h.
 *
 * Hand-rolled to avoid pulling in libexif. Builds two IFDs (IFD0 with
 * camera-identification fields, EXIF sub-IFD with shot parameters) plus
 * an overflow area for strings/rationals that don't fit in the 4-byte
 * inline value slot.
 */

#include "jpeg_exif.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

namespace {

// TIFF/EXIF type codes.
constexpr uint16_t T_BYTE      = 1;
constexpr uint16_t T_ASCII     = 2;
constexpr uint16_t T_SHORT     = 3;
constexpr uint16_t T_LONG      = 4;
constexpr uint16_t T_RATIONAL  = 5;
constexpr uint16_t T_UNDEFINED = 7;

// Tag numbers we emit.
constexpr uint16_t T_TAG_MAKE    = 271;
constexpr uint16_t T_TAG_MODEL   = 272;
constexpr uint16_t T_TAG_ORI     = 274;
constexpr uint16_t T_TAG_DT      = 306;
constexpr uint16_t T_TAG_EXIFIFD = 34665;
constexpr uint16_t T_TAG_EXPTIME = 33434;
constexpr uint16_t T_TAG_FNUM    = 33437;
constexpr uint16_t T_TAG_ISO     = 34855;
constexpr uint16_t T_TAG_EXIFVER = 36864;
constexpr uint16_t T_TAG_DTORIG  = 36867;
constexpr uint16_t T_TAG_FOCAL   = 37386;

inline void le16w(uint8_t *p, uint16_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
}
inline void le32w(uint8_t *p, uint32_t v) {
    p[0] = (uint8_t)(v & 0xff);
    p[1] = (uint8_t)((v >> 8) & 0xff);
    p[2] = (uint8_t)((v >> 16) & 0xff);
    p[3] = (uint8_t)((v >> 24) & 0xff);
}

}  // namespace

void emit_app1_exif(OutBuf *out, const exif_info *ex) {
    if (!ex) return;

    // Decide which fields are present.
    bool h_make    = ex->make     && ex->make[0];
    bool h_model   = ex->model    && ex->model[0];
    bool h_dt      = ex->datetime && ex->datetime[0];
    bool h_ori     = ex->orientation != 0;

    bool h_exptime = ex->exposure_us != 0;
    bool h_fnum    = ex->fnum_n != 0 && ex->fnum_d != 0;
    bool h_iso     = ex->iso != 0;
    bool h_focal   = ex->focal_n != 0 && ex->focal_d != 0;
    bool h_dt_orig = h_dt;

    int n_ifd0 = (h_make?1:0) + (h_model?1:0) + (h_ori?1:0) + (h_dt?1:0) + 1;
    int n_exif = (h_exptime?1:0) + (h_fnum?1:0) + (h_iso?1:0)
               + 1 /* ExifVersion */ + (h_dt_orig?1:0) + (h_focal?1:0);

    if (n_ifd0 == 1 && n_exif == 1) {
        // Nothing meaningful to emit.
        return;
    }

    // Compute layout.
    //   [0..7]                       TIFF header (II*\0 + offset_to_IFD0=8)
    //   [8..]                        IFD0: 2(count) + n_ifd0*12 + 4(next=0)
    //   [exif_off..]                 EXIF SubIFD
    //   [overflow_off..]             Overflow data (strings, rationals)
    uint32_t ifd0_off    = 8;
    uint32_t ifd0_size   = 2 + (uint32_t)n_ifd0 * 12 + 4;
    uint32_t exif_off    = ifd0_off + ifd0_size;
    uint32_t exif_size   = 2 + (uint32_t)n_exif * 12 + 4;
    uint32_t overflow_off= exif_off + exif_size;

    // Build TIFF body in a scratch buffer first.
    OutBuf tiff;     out_init(&tiff, 256);
    OutBuf overflow; out_init(&overflow, 256);

    auto put_overflow = [&](const void *p, size_t n) -> uint32_t {
        uint32_t off = (uint32_t)overflow.size;
        out_bytes(&overflow, p, n);
        if (overflow.size & 1) out_byte(&overflow, 0);  // word-align
        return off;
    };

    auto write_entry = [&](uint16_t tag, uint16_t type, uint32_t count, uint32_t value) {
        uint8_t e[12];
        le16w(e+0, tag);
        le16w(e+2, type);
        le32w(e+4, count);
        le32w(e+8, value);
        out_bytes(&tiff, e, 12);
    };

    auto put_ascii = [&](uint16_t tag, const char *s) {
        size_t len = strlen(s) + 1;
        if (len <= 4) {
            uint8_t v[4] = {0};
            memcpy(v, s, len);
            uint32_t value = (uint32_t)v[0] | ((uint32_t)v[1]<<8)
                            | ((uint32_t)v[2]<<16) | ((uint32_t)v[3]<<24);
            write_entry(tag, T_ASCII, (uint32_t)len, value);
        } else {
            uint32_t off = put_overflow(s, len);
            write_entry(tag, T_ASCII, (uint32_t)len, overflow_off + off);
        }
    };
    auto put_rational = [&](uint16_t tag, uint32_t num, uint32_t den) {
        uint8_t r[8];
        le32w(r+0, num);
        le32w(r+4, den);
        uint32_t off = put_overflow(r, 8);
        write_entry(tag, T_RATIONAL, 1, overflow_off + off);
    };

    // TIFF header.
    out_byte(&tiff, 'I'); out_byte(&tiff, 'I');
    out_byte(&tiff, 0x2A); out_byte(&tiff, 0x00);
    out_byte(&tiff, 0x08); out_byte(&tiff, 0x00); out_byte(&tiff, 0x00); out_byte(&tiff, 0x00);

    // IFD0: count, then entries (ascending tag), then next-IFD = 0.
    out_byte(&tiff, (uint8_t)(n_ifd0 & 0xff));
    out_byte(&tiff, (uint8_t)((n_ifd0 >> 8) & 0xff));

    // 271 Make, 272 Model, 274 Orientation, 306 DateTime, 34665 ExifIFD
    if (h_make)  put_ascii(T_TAG_MAKE,    ex->make);
    if (h_model) put_ascii(T_TAG_MODEL,   ex->model);
    if (h_ori)   write_entry(T_TAG_ORI,   T_SHORT, 1, (uint32_t)ex->orientation);
    if (h_dt)    put_ascii(T_TAG_DT,      ex->datetime);
    write_entry(T_TAG_EXIFIFD, T_LONG, 1, exif_off);

    // IFD0 next-IFD = 0.
    out_byte(&tiff, 0); out_byte(&tiff, 0); out_byte(&tiff, 0); out_byte(&tiff, 0);

    // EXIF Sub-IFD.
    out_byte(&tiff, (uint8_t)(n_exif & 0xff));
    out_byte(&tiff, (uint8_t)((n_exif >> 8) & 0xff));

    // Ascending: 33434 ExposureTime, 33437 FNumber, 34855 ISO,
    //            36864 ExifVersion, 36867 DateTimeOriginal, 37386 FocalLength
    if (h_exptime) put_rational(T_TAG_EXPTIME, ex->exposure_us, 1000000u);
    if (h_fnum)    put_rational(T_TAG_FNUM,    ex->fnum_n, ex->fnum_d);
    if (h_iso)     write_entry (T_TAG_ISO,     T_SHORT, 1, (uint32_t)ex->iso);
    {
        uint8_t v[4] = {'0','2','3','0'};
        uint32_t value = (uint32_t)v[0] | ((uint32_t)v[1]<<8)
                        | ((uint32_t)v[2]<<16) | ((uint32_t)v[3]<<24);
        write_entry(T_TAG_EXIFVER, T_UNDEFINED, 4, value);
    }
    if (h_dt_orig) put_ascii   (T_TAG_DTORIG,  ex->datetime);
    if (h_focal)   put_rational(T_TAG_FOCAL,   ex->focal_n, ex->focal_d);

    // EXIF next-IFD = 0.
    out_byte(&tiff, 0); out_byte(&tiff, 0); out_byte(&tiff, 0); out_byte(&tiff, 0);

    // Append overflow.
    out_bytes(&tiff, overflow.data, overflow.size);
    free(overflow.data);

    if (tiff.err) { free(tiff.data); return; }

    // Now the APP1 segment in `out`:
    //   FF E1
    //   length-2bytes (covers length field + payload, NOT marker bytes)
    //   "Exif\0\0"  (6 bytes identifier)
    //   <tiff body>
    size_t payload = 6 + tiff.size;
    if (payload + 2 > 0xffff) {
        // EXIF too big (>~64KB). Skip entirely rather than truncate.
        free(tiff.data);
        return;
    }
    out_byte(out, 0xff); out_byte(out, 0xe1);
    out_be16(out, (uint16_t)(payload + 2));
    static const char EXIF_HDR[6] = { 'E','x','i','f',0,0 };
    out_bytes(out, EXIF_HDR, 6);
    out_bytes(out, tiff.data, tiff.size);
    free(tiff.data);
}
