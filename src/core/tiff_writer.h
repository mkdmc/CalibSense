/*
 * tiff_writer.h — TIFF/DNG type and tag constants + IfdBuilder.
 *
 * Internal-only header (single TU includes it: pipeline.cpp). Provides
 * a small builder that constructs a TIFF/DNG IFD0 in a fixed-size buffer.
 *
 * Layout produced:
 *   [TIFF header 8B] [IFD0 entries 12B each] [next-IFD u32=0]
 *                                            [variable-size payload area]
 *
 * The class deliberately uses a stack-sized fixed buffer (8 KiB) — the
 * IFD has a hard cap of MAX_ENTRIES (64) entries and the payloads we
 * emit (color matrices, EXIF strings, AsShotNeutral) easily fit.
 *
 * For STRIP DATA we don't use the IfdBuilder: pipeline.cpp emits the
 * Bayer plane separately and patches the StripOffsets entry afterwards
 * via patch_long_value().
 */

#ifndef CALIBSENSE_TIFF_WRITER_H
#define CALIBSENSE_TIFF_WRITER_H

#include <stddef.h>
#include <stdint.h>

namespace calibsense {
namespace tiff {

// ---------- TIFF/DNG type constants ---------------------------------------
constexpr uint16_t T_BYTE      = 1;
constexpr uint16_t T_ASCII     = 2;
constexpr uint16_t T_SHORT     = 3;
constexpr uint16_t T_LONG      = 4;
constexpr uint16_t T_RATIONAL  = 5;
constexpr uint16_t T_UNDEFINED = 7;
constexpr uint16_t T_SRATIONAL = 10;

// ---------- TIFF / DNG tags -----------------------------------------------
constexpr uint16_t TAG_NewSubfileType         = 254;
constexpr uint16_t TAG_ImageWidth             = 256;
constexpr uint16_t TAG_ImageLength            = 257;
constexpr uint16_t TAG_BitsPerSample          = 258;
constexpr uint16_t TAG_Compression            = 259;
constexpr uint16_t TAG_Photometric            = 262;
constexpr uint16_t TAG_Make                   = 271;
constexpr uint16_t TAG_Model                  = 272;
constexpr uint16_t TAG_StripOffsets           = 273;
constexpr uint16_t TAG_Orientation            = 274;
constexpr uint16_t TAG_SamplesPerPixel        = 277;
constexpr uint16_t TAG_RowsPerStrip           = 278;
constexpr uint16_t TAG_StripByteCounts        = 279;
constexpr uint16_t TAG_PlanarConfig           = 284;
constexpr uint16_t TAG_Software               = 305;
constexpr uint16_t TAG_XResolution            = 282;
constexpr uint16_t TAG_YResolution            = 283;
constexpr uint16_t TAG_ResolutionUnit         = 296;
constexpr uint16_t TAG_DateTime               = 306;
constexpr uint16_t TAG_CFARepeatPatternDim    = 33421;
constexpr uint16_t TAG_CFAPattern             = 33422;
constexpr uint16_t TAG_ExifIFD                = 34665;
// EXIF (used inside the EXIF sub-IFD)
constexpr uint16_t TAG_ExposureTime           = 33434;
constexpr uint16_t TAG_FNumber                = 33437;
constexpr uint16_t TAG_ISOSpeedRatings        = 34855;
constexpr uint16_t TAG_ExifVersion            = 36864;
constexpr uint16_t TAG_DateTimeOriginal       = 36867;
constexpr uint16_t TAG_FocalLength            = 37386;
constexpr uint16_t TAG_DNGVersion             = 50706;
constexpr uint16_t TAG_DNGBackwardVersion     = 50707;
constexpr uint16_t TAG_UniqueCameraModel      = 50708;
constexpr uint16_t TAG_CFAPlaneColor          = 50710;
constexpr uint16_t TAG_CFALayout              = 50711;
constexpr uint16_t TAG_BlackLevelRepeatDim    = 50713;
constexpr uint16_t TAG_BlackLevel             = 50714;
constexpr uint16_t TAG_WhiteLevel             = 50717;
constexpr uint16_t TAG_DefaultScale           = 50718;
constexpr uint16_t TAG_DefaultCropOrigin      = 50719;
constexpr uint16_t TAG_DefaultCropSize        = 50720;
constexpr uint16_t TAG_ColorMatrix1           = 50721;
constexpr uint16_t TAG_ColorMatrix2           = 50722;
constexpr uint16_t TAG_CameraCalibration1     = 50723;
constexpr uint16_t TAG_CameraCalibration2     = 50724;
constexpr uint16_t TAG_AsShotNeutral          = 50728;
constexpr uint16_t TAG_BaselineExposure       = 50730;
constexpr uint16_t TAG_CalibrationIlluminant1 = 50778;
constexpr uint16_t TAG_CalibrationIlluminant2 = 50779;
constexpr uint16_t TAG_ForwardMatrix1         = 50964;
constexpr uint16_t TAG_ForwardMatrix2         = 50965;

// ---------- IfdBuilder -----------------------------------------------------
//
// Builds IFD0 (and an embedded EXIF sub-IFD) into an 8 KiB fixed buffer.
// All add_* methods append one entry to IFD0; values that don't fit in
// the entry's 4-byte inline slot are stashed in the payload area and
// referenced by offset.
class IfdBuilder {
public:
    IfdBuilder();

    void add_short_inline(uint16_t tag, uint16_t v);
    void add_long_inline (uint16_t tag, uint32_t v);
    void add_two_shorts  (uint16_t tag, uint16_t a, uint16_t b);

    void add_byte_array_inline(uint16_t tag, const uint8_t *bytes, uint32_t n);

    // Multi-SHORT: inline if count <= 2, otherwise stored in payload.
    void add_short_array(uint16_t tag, const uint16_t *vals, uint32_t count);

    // Single-RATIONAL convenience wrapper around add_rationals.
    void add_rational_one(uint16_t tag, uint32_t num, uint32_t den);

    void add_ascii      (uint16_t tag, const char *s);
    void add_rationals  (uint16_t tag, const uint32_t *pairs, uint32_t count);
    void add_srationals (uint16_t tag, const int32_t  *pairs, uint32_t count);

    /*
     * Embed an EXIF sub-IFD inside the payload area and add an ExifIFDPointer
     * entry to IFD0. Layout matches IFD0:
     *   [u16 entry_count][entries][u32 next_ifd=0][offset values...]
     *
     * A zero in any of {exposure_us, iso, fnum_*, focal_*, date_time}
     * causes that field to be omitted.
     */
    void add_exif_ifd(uint32_t exposure_us, uint16_t iso,
                      uint32_t fnum_n, uint32_t fnum_d,
                      uint32_t focal_n, uint32_t focal_d,
                      const char *date_time);

    // Write the entry count back into the IFD header and zero-terminate
    // the chain. Returns the buffer size in bytes (= payload end).
    size_t finalize_prefix();

    // Patch the inline value of an already-added LONG entry. Used to
    // rewrite TAG_StripOffsets after the strip is appended on disk.
    void patch_long_value(uint16_t tag, uint32_t value);

    const uint8_t *data() const { return buf_; }
    size_t         size() const { return payload_pos_; }

private:
    static constexpr size_t BUF_SIZE    = 8192;
    static constexpr size_t MAX_ENTRIES = 64;

    uint8_t  buf_[BUF_SIZE];
    size_t   header_pos_;
    size_t   payload_base_;
    size_t   payload_pos_;
    uint16_t entry_count_;

    void     write_u16_at(size_t off, uint16_t v);
    void     write_u32_at(size_t off, uint32_t v);
    uint16_t read_u16_at (size_t off) const;

    void write_entry_inline(uint16_t tag, uint16_t type, uint32_t count, uint32_t value);
    void write_entry_offset(uint16_t tag, uint16_t type, uint32_t count, uint32_t offset);
    uint32_t append_payload(const void *data, size_t n);
};

}  // namespace tiff
}  // namespace calibsense

#endif  // CALIBSENSE_TIFF_WRITER_H
