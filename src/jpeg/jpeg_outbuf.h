/*
 * jpeg_outbuf.h — growable output buffer used by the JPEG encoder.
 *
 * Internal-only header, shared between jpeg_enc.cpp and jpeg_exif.cpp.
 * Defines a tiny realloc-backed byte buffer and a handful of write
 * helpers. All functions are `inline`; the linker collapses duplicate
 * definitions across TUs.
 *
 * Error model: every function checks `o->err` first and is a no-op if
 * an earlier allocation failed. The caller checks `o->err` once at the
 * end rather than on every byte write.
 */

#ifndef CALIBSENSE_JPEG_OUTBUF_H
#define CALIBSENSE_JPEG_OUTBUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct OutBuf {
    uint8_t *data;
    size_t   size;
    size_t   cap;
    int      err;
};

inline void out_init(OutBuf *o, size_t initial) {
    o->data = (uint8_t*)malloc(initial);
    o->size = 0;
    o->cap  = o->data ? initial : 0;
    o->err  = o->data ? 0 : 1;
}

inline void out_grow(OutBuf *o, size_t need) {
    if (o->err) return;
    if (o->size + need <= o->cap) return;
    size_t new_cap = o->cap;
    if (new_cap < 1024) new_cap = 1024;
    while (new_cap < o->size + need) new_cap = new_cap + new_cap / 2 + 64;
    uint8_t *p = (uint8_t*)realloc(o->data, new_cap);
    if (!p) { o->err = 1; return; }
    o->data = p;
    o->cap  = new_cap;
}

inline void out_byte(OutBuf *o, uint8_t b) {
    out_grow(o, 1); if (!o->err) o->data[o->size++] = b;
}

inline void out_be16(OutBuf *o, uint16_t v) {
    out_byte(o, (uint8_t)(v >> 8));
    out_byte(o, (uint8_t)(v & 0xff));
}

inline void out_bytes(OutBuf *o, const void *p, size_t n) {
    out_grow(o, n);
    if (!o->err) { memcpy(o->data + o->size, p, n); o->size += n; }
}

#endif  // CALIBSENSE_JPEG_OUTBUF_H
