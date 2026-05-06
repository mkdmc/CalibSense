/*
 * calibsense.h — CalibSense camera pipeline.
 *
 * Public C API for the imaging pipeline:
 *   raw 10-bit RGGB Bayer + RAW header → DNG (always)
 *                                     → optional demosaic → 8-bit sRGB → JPEG.
 *
 * The DNG output is a standard uncompressed file with two-illuminant
 * calibration (D65 + StdA), proper black/white levels, RGGB CFA pattern,
 * and AsShotNeutral derived from real WB gains.
 *
 * Output delivery: prefers the abstract Unix socket @calibsense_socket
 * (brought up by an explicit call to calibsense_delivery_start() — see delivery.h);
 * falls back to writing the file at out_path if no socket peer is connected
 * and out_path is non-empty; if neither is usable the conversion is a
 * no-op and returns 0.
 */

#ifndef CALIBSENSE_H
#define CALIBSENSE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Convert a packed Bayer frame plus its RAW header into a DNG.
 *
 *   raw_header  : pointer to the 3272-byte header. Pass NULL to produce a
 *                 DNG without exposure metadata.
 *   header_size : header buffer length (must be >= 3272 if non-NULL).
 *   raw_packed  : packed pixel data, 7000 bytes per row, height rows.
 *   raw_size    : packed data length in bytes; must be at least
 *                 height * 7000 (any trailing slack is ignored).
 *   width       : active pixel count per row (5248 for full-res Sony D5503).
 *   height      : row count (3936 for full-res Sony D5503).
 *   out_path    : optional file sink. If a socket peer is connected the
 *                 bytes are streamed there and out_path is ignored.
 *                 If no peer, and out_path is non-NULL/non-empty, the DNG
 *                 is written to that path. Pass NULL to disable file
 *                 output entirely (silent skip when no socket peer).
 *   wb_data     : optional pointer to a WB sidecar buffer (>= 0x14 bytes
 *                 readable). Format:
 *                   +0x00  uint32  magic ('WB\0\0' = 0x00004257)
 *                   +0x08  uint32  RGain (Q16.16)
 *                   +0x0C  uint32  GGain (Q16.16)
 *                   +0x10  uint32  BGain (Q16.16)
 *                 Pass NULL or a buffer with bad magic to fall back to
 *                 the metering values in the RAW header.
 *
 * Returns 0 on success or successful no-op, non-zero on error.
 */
int calibsense_process_frame(
    const void *raw_header,
    size_t      header_size,
    const void *raw_packed,
    size_t      raw_size,
    int         width,
    int         height,
    const char *out_path,
    const void *wb_data
);

/*
 * Convenience wrapper: read a complete frame file from disk (header
 * concatenated with packed Bayer in one blob) and convert it.
 * If wb_path is non-NULL, the WB sidecar is also read and forwarded.
 * Intended for host-side testing — bypasses the socket and writes a file.
 */
int calibsense_process_file(
    const char *raw_path,
    const char *wb_path,
    const char *out_path
);

#ifdef __cplusplus
}
#endif

#endif /* CALIBSENSE_H */
