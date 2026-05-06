/*
 * jpeg_tables.h — standard JPEG quantization, zig-zag and Huffman tables.
 *
 * Internal-only header. Declares the Annex K reference tables (luma and
 * chroma quantization, both DC/AC Huffman code-length specs, zig-zag
 * scan order) and the small HuffEnc struct that represents the encoder
 * lookup once the (BITS, VALUES) form has been expanded.
 *
 * The tables themselves are baseline ITU-T T.81 Annex K values. The
 * encoder applies a quality-scaling multiplier on top of the quant
 * tables — that's done in jpeg_enc.cpp, not here.
 */

#ifndef CALIBSENSE_JPEG_TABLES_H
#define CALIBSENSE_JPEG_TABLES_H

#include <stdint.h>

// ---------- Standard quantization tables (Annex K, K.1 / K.2) -------------
extern const uint8_t QTABLE_LUMA_BASE[64];
extern const uint8_t QTABLE_CHROMA_BASE[64];

// ---------- Zigzag scan order (Figure A.6) --------------------------------
extern const uint8_t ZIGZAG[64];

// ---------- Standard Huffman tables (Annex K, K.3..K.6) -------------------
//
// Each pair is the "BITS" / "VALUES" form from T.81: BITS[i] gives the
// number of codes of length (i+1), and VALUES lists the symbols in code
// order. build_huff_enc() expands these into a code/size lookup.
extern const uint8_t HUFF_DC_LUMA_BITS[16];
extern const uint8_t HUFF_DC_LUMA_VAL [12];

extern const uint8_t HUFF_AC_LUMA_BITS[16];
extern const uint8_t HUFF_AC_LUMA_VAL [162];

extern const uint8_t HUFF_DC_CHROMA_BITS[16];
extern const uint8_t HUFF_DC_CHROMA_VAL [12];

extern const uint8_t HUFF_AC_CHROMA_BITS[16];
extern const uint8_t HUFF_AC_CHROMA_VAL [162];

// ---------- Encoder Huffman lookup ---------------------------------------
//
// `code` and `size` are indexed by the symbol byte (0..255). A `size` of
// zero indicates an unused symbol — should never appear in a valid
// stream, but the encoder asserts (or in our case, just produces an
// unusable code) if it does.
struct HuffEnc {
    uint16_t code[256];
    uint8_t  size[256];   // 0 = symbol unused
};

// Expand the (BITS, VALUES) form into a HuffEnc lookup table.
void build_huff_enc(HuffEnc *h, const uint8_t bits[16], const uint8_t *vals);

#endif  // CALIBSENSE_JPEG_TABLES_H
