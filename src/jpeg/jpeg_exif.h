/*
 * jpeg_exif.h — APP1 EXIF segment builder for the JPEG encoder.
 *
 * Internal-only header (single TU includes it: jpeg_enc.cpp). Builds a
 * complete APP1 EXIF segment from an exif_info struct and appends it to
 * the encoder's output buffer.
 *
 * The segment layout produced is:
 *
 *   FF E1                          (APP1 marker)
 *   <length: 16-bit BE>            (covers length field + payload)
 *   "Exif\0\0"                     (6-byte identifier)
 *   <TIFF body>                    (II header, IFD0, EXIF sub-IFD, overflow)
 *
 * If the source struct is NULL, has only empty/zero fields, or would
 * produce more than 64 KB of payload, the function silently emits nothing.
 */

#ifndef CALIBSENSE_JPEG_EXIF_H
#define CALIBSENSE_JPEG_EXIF_H

#include "jpeg_enc.h"      // exif_info
#include "jpeg_outbuf.h"   // OutBuf

void emit_app1_exif(OutBuf *out, const exif_info *ex);

#endif  // CALIBSENSE_JPEG_EXIF_H
