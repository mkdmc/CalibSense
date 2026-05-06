/*
 * tiff_writer.cpp — IfdBuilder implementation. See tiff_writer.h.
 */

#include "tiff_writer.h"

#include <string.h>

namespace calibsense {
namespace tiff {

// ---------- raw byte LE writers (alignment-safe) --------------------------

void IfdBuilder::write_u16_at(size_t off, uint16_t v) {
    buf_[off]   =  v        & 0xFF;
    buf_[off+1] = (v >> 8)  & 0xFF;
}

void IfdBuilder::write_u32_at(size_t off, uint32_t v) {
    buf_[off]   =  v        & 0xFF;
    buf_[off+1] = (v >> 8)  & 0xFF;
    buf_[off+2] = (v >> 16) & 0xFF;
    buf_[off+3] = (v >> 24) & 0xFF;
}

uint16_t IfdBuilder::read_u16_at(size_t off) const {
    return (uint16_t)buf_[off] | ((uint16_t)buf_[off+1] << 8);
}

void IfdBuilder::write_entry_inline(uint16_t tag, uint16_t type,
                                    uint32_t count, uint32_t value) {
    write_u16_at(header_pos_ + 0, tag);
    write_u16_at(header_pos_ + 2, type);
    write_u32_at(header_pos_ + 4, count);
    write_u32_at(header_pos_ + 8, value);
    header_pos_ += 12;
    ++entry_count_;
}

void IfdBuilder::write_entry_offset(uint16_t tag, uint16_t type,
                                    uint32_t count, uint32_t offset) {
    write_u16_at(header_pos_ + 0, tag);
    write_u16_at(header_pos_ + 2, type);
    write_u32_at(header_pos_ + 4, count);
    write_u32_at(header_pos_ + 8, offset);
    header_pos_ += 12;
    ++entry_count_;
}

uint32_t IfdBuilder::append_payload(const void *data, size_t n) {
    if (payload_pos_ & 1) ++payload_pos_;
    uint32_t off = (uint32_t)payload_pos_;
    memcpy(buf_ + payload_pos_, data, n);
    payload_pos_ += n;
    return off;
}

// ---------- ctor + simple add_* methods -----------------------------------

IfdBuilder::IfdBuilder() : header_pos_(0), payload_pos_(0), entry_count_(0) {
    memset(buf_, 0, sizeof(buf_));
    // TIFF header: 'II' (little-endian), magic 42, IFD0 offset = 8.
    buf_[0] = 'I'; buf_[1] = 'I';
    write_u16_at(2, 42);
    write_u32_at(4, 8);
    header_pos_   = 8 + 2;
    payload_base_ = 8 + 2 + MAX_ENTRIES * 12 + 4;
    payload_pos_  = payload_base_;
}

void IfdBuilder::add_short_inline(uint16_t tag, uint16_t v) {
    write_entry_inline(tag, T_SHORT, 1, v);
}

void IfdBuilder::add_long_inline(uint16_t tag, uint32_t v) {
    write_entry_inline(tag, T_LONG, 1, v);
}

void IfdBuilder::add_two_shorts(uint16_t tag, uint16_t a, uint16_t b) {
    write_entry_inline(tag, T_SHORT, 2, (uint32_t)a | ((uint32_t)b << 16));
}

void IfdBuilder::add_byte_array_inline(uint16_t tag, const uint8_t *bytes, uint32_t n) {
    uint32_t v = 0;
    for (uint32_t i = 0; i < n && i < 4; ++i) v |= (uint32_t)bytes[i] << (8 * i);
    write_entry_inline(tag, T_BYTE, n, v);
}

void IfdBuilder::add_short_array(uint16_t tag, const uint16_t *vals, uint32_t count) {
    if (count <= 2) {
        uint32_t v = 0;
        for (uint32_t i = 0; i < count; ++i) v |= ((uint32_t)vals[i] << (i * 16));
        write_entry_inline(tag, T_SHORT, count, v);
    } else {
        uint32_t off = append_payload(vals, count * 2);
        write_entry_offset(tag, T_SHORT, count, off);
    }
}

void IfdBuilder::add_rational_one(uint16_t tag, uint32_t num, uint32_t den) {
    uint32_t pair[2] = {num, den};
    add_rationals(tag, pair, 1);
}

void IfdBuilder::add_ascii(uint16_t tag, const char *s) {
    uint32_t n = (uint32_t)strlen(s) + 1;
    if (n <= 4) {
        uint32_t v = 0;
        memcpy(&v, s, n);
        write_entry_inline(tag, T_ASCII, n, v);
    } else {
        uint32_t off = append_payload(s, n);
        write_entry_offset(tag, T_ASCII, n, off);
    }
}

void IfdBuilder::add_rationals(uint16_t tag, const uint32_t *pairs, uint32_t count) {
    uint32_t off = append_payload(pairs, count * 8);
    write_entry_offset(tag, T_RATIONAL, count, off);
}

void IfdBuilder::add_srationals(uint16_t tag, const int32_t *pairs, uint32_t count) {
    uint32_t off = append_payload(pairs, count * 8);
    write_entry_offset(tag, T_SRATIONAL, count, off);
}

size_t IfdBuilder::finalize_prefix() {
    write_u16_at(8, entry_count_);
    size_t next_ifd_offset = 8 + 2 + entry_count_ * 12;
    write_u32_at(next_ifd_offset, 0);
    return payload_pos_;
}

void IfdBuilder::patch_long_value(uint16_t tag, uint32_t value) {
    size_t entries_start = 8 + 2;
    for (uint16_t i = 0; i < entry_count_; ++i) {
        size_t entry_off = entries_start + i * 12;
        if (read_u16_at(entry_off) == tag) {
            write_u32_at(entry_off + 8, value);
            return;
        }
    }
}

// ---------- add_exif_ifd: build the EXIF sub-IFD --------------------------

void IfdBuilder::add_exif_ifd(uint32_t exposure_us, uint16_t iso,
                              uint32_t fnum_n, uint32_t fnum_d,
                              uint32_t focal_n, uint32_t focal_d,
                              const char *date_time)
{
    if (payload_pos_ & 1) ++payload_pos_;
    const uint32_t exif_offset = (uint32_t)payload_pos_;

    // Count entries we'll emit.
    uint16_t n = 0;
    if (true)            ++n;  // ExifVersion (always)
    if (date_time)       ++n;  // DateTimeOriginal
    if (exposure_us)     ++n;  // ExposureTime
    if (fnum_n && fnum_d) ++n; // FNumber
    if (iso)             ++n;  // ISOSpeedRatings
    if (focal_n && focal_d) ++n; // FocalLength

    // Layout:  [count u16][entries n*12][next_ifd u32=0][value payload]
    const size_t hdr_pos     = payload_pos_;
    const size_t entries_pos = hdr_pos + 2;
    const size_t next_pos    = entries_pos + n * 12;
    const size_t value_pos   = next_pos + 4;

    write_u16_at(hdr_pos, n);
    write_u32_at(next_pos, 0);

    // Advance payload_pos_ past entries+next, then we'll lay values after.
    payload_pos_ = value_pos;

    // Helper to emit one entry into the current entries slot.
    size_t entry_w = entries_pos;
    auto emit_inline_u32 = [&](uint16_t tag, uint16_t type,
                               uint32_t count, uint32_t v) {
        write_u16_at(entry_w + 0, tag);
        write_u16_at(entry_w + 2, type);
        write_u32_at(entry_w + 4, count);
        write_u32_at(entry_w + 8, v);
        entry_w += 12;
    };
    auto emit_inline_bytes = [&](uint16_t tag, uint16_t type,
                                 uint32_t count, const void *p) {
        write_u16_at(entry_w + 0, tag);
        write_u16_at(entry_w + 2, type);
        write_u32_at(entry_w + 4, count);
        memcpy(buf_ + entry_w + 8, p, count <= 4 ? count : 4);
        entry_w += 12;
    };
    auto emit_offset = [&](uint16_t tag, uint16_t type,
                           uint32_t count, const void *p, size_t nbytes) {
        uint32_t off = (uint32_t)payload_pos_;
        memcpy(buf_ + payload_pos_, p, nbytes);
        payload_pos_ += nbytes;
        write_u16_at(entry_w + 0, tag);
        write_u16_at(entry_w + 2, type);
        write_u32_at(entry_w + 4, count);
        write_u32_at(entry_w + 8, off);
        entry_w += 12;
    };

    // ExifVersion: 4 bytes "0230"
    static const uint8_t EXIF_VERSION[4] = { '0','2','3','0' };
    emit_inline_bytes(TAG_ExifVersion, T_UNDEFINED, 4, EXIF_VERSION);

    // DateTimeOriginal: 20-byte ASCII (incl NUL)
    if (date_time) {
        emit_offset(TAG_DateTimeOriginal, T_ASCII, 20, date_time, 20);
    }

    // ExposureTime: RATIONAL (us / 1000000)
    if (exposure_us) {
        uint32_t pair[2] = { exposure_us, 1000000u };
        emit_offset(TAG_ExposureTime, T_RATIONAL, 1, pair, 8);
    }

    // FNumber: RATIONAL
    if (fnum_n && fnum_d) {
        uint32_t pair[2] = { fnum_n, fnum_d };
        emit_offset(TAG_FNumber, T_RATIONAL, 1, pair, 8);
    }

    // ISOSpeedRatings: SHORT (inline)
    if (iso) {
        emit_inline_u32(TAG_ISOSpeedRatings, T_SHORT, 1, iso);
    }

    // FocalLength: RATIONAL
    if (focal_n && focal_d) {
        uint32_t pair[2] = { focal_n, focal_d };
        emit_offset(TAG_FocalLength, T_RATIONAL, 1, pair, 8);
    }

    // Add the IFD0 → EXIF pointer entry.
    add_long_inline(TAG_ExifIFD, exif_offset);
}

}  // namespace tiff
}  // namespace calibsense
