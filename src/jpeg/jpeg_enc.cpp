/*
 * jpeg_enc.cpp — minimal baseline JPEG encoder (4:2:0) with EXIF APP1.
 *
 * Self-contained: no external dependencies beyond libc/libm.
 *
 * Quality scaling follows IJG convention. Quantization tables are the
 * standard ones from JPEG Annex K. Huffman tables are also the standard
 * Annex K tables (no Huffman optimization pass — saves a full pre-pass
 * over the data at the cost of ~5% larger files).
 *
 * Forward DCT is the integer-accurate "jfdctint" algorithm
 * (Loeffler-Ligtenberg-Moschytz 11-mul, refined by IJG).
 *
 * Output: a malloc'd buffer containing a full JPEG file (SOI to EOI),
 *         caller-owned, freed via free().
 */

#include "jpeg_enc.h"
#include "jpeg_outbuf.h"
#include "jpeg_tables.h"
#include "jpeg_exif.h"
#include "threading.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define HAVE_NEON 1
#else
#  define HAVE_NEON 0
#endif


namespace {

// === Bit writer with JPEG byte-stuffing ===

struct BitWriter {
    OutBuf  *out;
    uint32_t buf;
    int      n;     // bits in `buf`, MSB-aligned
};

static void bw_init(BitWriter *bw, OutBuf *out) {
    bw->out = out; bw->buf = 0; bw->n = 0;
}

static void bw_put(BitWriter *bw, uint32_t code, int nbits) {
    if (nbits == 0) return;
    code &= (1u << nbits) - 1u;
    bw->buf |= code << (24 - bw->n - nbits);
    bw->n   += nbits;
    while (bw->n >= 8) {
        uint8_t b = (uint8_t)((bw->buf >> 16) & 0xff);
        out_byte(bw->out, b);
        if (b == 0xff) out_byte(bw->out, 0x00);
        bw->buf <<= 8;
        bw->n   -= 8;
    }
}

static void bw_flush(BitWriter *bw) {
    if (bw->n > 0) {
        // Pad with 1-bits to byte boundary.
        bw_put(bw, (1u << (8 - bw->n)) - 1u, 8 - bw->n);
    }
}

// === Forward DCT (jfdctint, integer-accurate; CONST_BITS=13, PASS1_BITS=2) ===

#define CONST_BITS 13
#define PASS1_BITS 2

#define FIX_0_298631336  ((int32_t) 2446)
#define FIX_0_390180644  ((int32_t) 3196)
#define FIX_0_541196100  ((int32_t) 4433)
#define FIX_0_765366865  ((int32_t) 6270)
#define FIX_0_899976223  ((int32_t) 7373)
#define FIX_1_175875602  ((int32_t) 9633)
#define FIX_1_501321110  ((int32_t)12299)
#define FIX_1_847759065  ((int32_t)15137)
#define FIX_1_961570560  ((int32_t)16069)
#define FIX_2_053119869  ((int32_t)16819)
#define FIX_2_562915447  ((int32_t)20995)
#define FIX_3_072711026  ((int32_t)25172)

#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

static void fdct(int32_t *data) {
    // Pass 1: rows.
    int32_t *p = data;
    for (int ctr = 0; ctr < 8; ++ctr) {
        int32_t tmp0 = p[0] + p[7];
        int32_t tmp7 = p[0] - p[7];
        int32_t tmp1 = p[1] + p[6];
        int32_t tmp6 = p[1] - p[6];
        int32_t tmp2 = p[2] + p[5];
        int32_t tmp5 = p[2] - p[5];
        int32_t tmp3 = p[3] + p[4];
        int32_t tmp4 = p[3] - p[4];

        int32_t tmp10 = tmp0 + tmp3;
        int32_t tmp13 = tmp0 - tmp3;
        int32_t tmp11 = tmp1 + tmp2;
        int32_t tmp12 = tmp1 - tmp2;

        p[0] = (tmp10 + tmp11) << PASS1_BITS;
        p[4] = (tmp10 - tmp11) << PASS1_BITS;

        int32_t z1 = (tmp12 + tmp13) * FIX_0_541196100;
        p[2] = DESCALE(z1 + tmp13 * FIX_0_765366865,  CONST_BITS - PASS1_BITS);
        p[6] = DESCALE(z1 - tmp12 * FIX_1_847759065,  CONST_BITS - PASS1_BITS);

        z1 = tmp4 + tmp7;
        int32_t z2 = tmp5 + tmp6;
        int32_t z3 = tmp4 + tmp6;
        int32_t z4 = tmp5 + tmp7;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;

        tmp4 *= FIX_0_298631336;
        tmp5 *= FIX_2_053119869;
        tmp6 *= FIX_3_072711026;
        tmp7 *= FIX_1_501321110;
        z1   *= -FIX_0_899976223;
        z2   *= -FIX_2_562915447;
        z3   *= -FIX_1_961570560;
        z4   *= -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        p[7] = DESCALE(tmp4 + z1 + z3, CONST_BITS - PASS1_BITS);
        p[5] = DESCALE(tmp5 + z2 + z4, CONST_BITS - PASS1_BITS);
        p[3] = DESCALE(tmp6 + z2 + z3, CONST_BITS - PASS1_BITS);
        p[1] = DESCALE(tmp7 + z1 + z4, CONST_BITS - PASS1_BITS);

        p += 8;
    }

    // Pass 2: columns.
    p = data;
    for (int ctr = 0; ctr < 8; ++ctr) {
        int32_t tmp0 = p[8*0] + p[8*7];
        int32_t tmp7 = p[8*0] - p[8*7];
        int32_t tmp1 = p[8*1] + p[8*6];
        int32_t tmp6 = p[8*1] - p[8*6];
        int32_t tmp2 = p[8*2] + p[8*5];
        int32_t tmp5 = p[8*2] - p[8*5];
        int32_t tmp3 = p[8*3] + p[8*4];
        int32_t tmp4 = p[8*3] - p[8*4];

        int32_t tmp10 = tmp0 + tmp3;
        int32_t tmp13 = tmp0 - tmp3;
        int32_t tmp11 = tmp1 + tmp2;
        int32_t tmp12 = tmp1 - tmp2;

        p[8*0] = DESCALE(tmp10 + tmp11, PASS1_BITS);
        p[8*4] = DESCALE(tmp10 - tmp11, PASS1_BITS);

        int32_t z1 = (tmp12 + tmp13) * FIX_0_541196100;
        p[8*2] = DESCALE(z1 + tmp13 * FIX_0_765366865, CONST_BITS + PASS1_BITS);
        p[8*6] = DESCALE(z1 - tmp12 * FIX_1_847759065, CONST_BITS + PASS1_BITS);

        z1 = tmp4 + tmp7;
        int32_t z2 = tmp5 + tmp6;
        int32_t z3 = tmp4 + tmp6;
        int32_t z4 = tmp5 + tmp7;
        int32_t z5 = (z3 + z4) * FIX_1_175875602;

        tmp4 *= FIX_0_298631336;
        tmp5 *= FIX_2_053119869;
        tmp6 *= FIX_3_072711026;
        tmp7 *= FIX_1_501321110;
        z1   *= -FIX_0_899976223;
        z2   *= -FIX_2_562915447;
        z3   *= -FIX_1_961570560;
        z4   *= -FIX_0_390180644;

        z3 += z5;
        z4 += z5;

        p[8*7] = DESCALE(tmp4 + z1 + z3, CONST_BITS + PASS1_BITS);
        p[8*5] = DESCALE(tmp5 + z2 + z4, CONST_BITS + PASS1_BITS);
        p[8*3] = DESCALE(tmp6 + z2 + z3, CONST_BITS + PASS1_BITS);
        p[8*1] = DESCALE(tmp7 + z1 + z4, CONST_BITS + PASS1_BITS);

        p++;
    }
}

// === Block encode: DCT, quantize, zigzag, Huffman ===

static inline int bit_size(int v) {
    if (v < 0) v = -v;
    int n = 0;
    while (v) { v >>= 1; n++; }
    return n;
}

// === Block encoding: split into two phases for threading ===
//
// Phase 1 (parallelizable): DCT + quantize + zigzag.  Pure pixel-local
// work; each MCU's 6 blocks are independent.
//
// Phase 2 (serial): DC-delta accumulation + Huffman bit emission to
// the bitstream.  Carries `prev_dc` state across all blocks of a given
// component (Y, Cb, Cr) and writes serially to the BitWriter, so this
// must be done in MCU order on a single thread.
//
// Splitting the work this way lets us run phase 1 across multiple
// worker threads (one per MCU row) while the main thread does phase 2
// over the completed rows in order.

// Phase 1: 8×8 int32 block → 64 zigzag-ordered quantized int16 coeffs.
static inline void encode_block_phase1(int32_t block[64],
                                       const uint16_t divisor[64],
                                       int16_t zz[64])
{
    fdct(block);
    for (int i = 0; i < 64; ++i) {
        int32_t v = block[ZIGZAG[i]];
        int     q = divisor[ZIGZAG[i]];
        if (v >= 0) v = (v + q/2) / q;
        else        v = -((-v + q/2) / q);
        if (v >  32767) v =  32767;
        if (v < -32768) v = -32768;
        zz[i] = (int16_t)v;
    }
}

// Phase 2: emit one block's Huffman bits into the BitWriter.  Updates
// `*prev_dc` to this block's DC value.  Must run serially in MCU order.
static int encode_block_phase2(BitWriter *bw, const int16_t zz[64],
                               const HuffEnc *dc_huff, const HuffEnc *ac_huff,
                               int *prev_dc) {
    // DC delta.
    int dc_diff = (int)zz[0] - *prev_dc;
    *prev_dc    = zz[0];
    {
        int s = bit_size(dc_diff);
        if (dc_huff->size[s] == 0) {
            return 1;
        }
        bw_put(bw, dc_huff->code[s], dc_huff->size[s]);
        if (s > 0) {
            int v = dc_diff;
            if (v < 0) v = (1 << s) + v - 1;
            bw_put(bw, (uint32_t)v, s);
        }
    }

    // AC.
    int run = 0;
    for (int i = 1; i < 64; ++i) {
        int v = zz[i];
        if (v == 0) { run++; continue; }
        while (run > 15) {
            if (ac_huff->size[0xF0] == 0) return 1;
            bw_put(bw, ac_huff->code[0xF0], ac_huff->size[0xF0]);
            run -= 16;
        }
        int s   = bit_size(v);
        int sym = (run << 4) | s;
        if (ac_huff->size[sym] == 0) {
            return 1;
        }
        bw_put(bw, ac_huff->code[sym], ac_huff->size[sym]);
        int vv = v;
        if (vv < 0) vv = (1 << s) + vv - 1;
        bw_put(bw, (uint32_t)vv, s);
        run = 0;
    }
    if (run > 0) {
        // EOB.
        if (ac_huff->size[0x00] == 0) return 1;
        bw_put(bw, ac_huff->code[0x00], ac_huff->size[0x00]);
    }
    return 0;
}

// === RGB → YCbCr (JFIF full-range, Q16 fixed-point) ===

static inline uint8_t clamp_u8(int v) {
    if (v < 0)   return 0;
    if (v > 255) return 255;
    return (uint8_t)v;
}

// === Effective qtable from base + quality ===

static void build_qtable(uint16_t out[64], const uint8_t base[64], int quality) {
    if (quality < 1)   quality = 1;
    if (quality > 100) quality = 100;
    int scale;
    if (quality < 50) scale = 5000 / quality;
    else              scale = 200 - 2 * quality;
    for (int i = 0; i < 64; ++i) {
        int q = (base[i] * scale + 50) / 100;
        if (q < 1)   q = 1;
        if (q > 255) q = 255;
        out[i] = (uint16_t)q;
    }
}

// =============================================================================
// Phase-1 worker: per-MCU RGB→YCbCr→DCT→quantize→zigzag.
// =============================================================================
//
// Produces 6 zigzag-ordered 8×8 quantized coefficient blocks for one
// MCU.  Output layout: zz_out[6][64] = { Y_TL, Y_TR, Y_BL, Y_BR, Cb, Cr }.
//
// This contains all the parallelizable work — no shared mutable state
// across MCUs.  The only external reads are from the source RGB buffer
// (read-only) and the two quant divisors (read-only).
static void process_mcu_phase1(const uint8_t *rgb,
                               int img_w, int img_h,
                               int mx, int my,
                               const uint16_t qy_div[64],
                               const uint16_t qc_div[64],
                               int16_t zz_out[6][64])
{
    const int base_x = mx * 16;
    const int base_y = my * 16;
    const bool full_w = (base_x + 16 <= img_w);
    const bool full_h = (base_y + 16 <= img_h);

    // 16×16 RGB scratch (768 bytes).
    uint8_t rgb_block[16 * 16 * 3];
    int32_t Y_TL[64], Y_TR[64], Y_BL[64], Y_BR[64], Cb_blk[64], Cr_blk[64];

    // ---- Step 1: gather 16×16 RGB ----
    if (full_w && full_h) {
        for (int yy = 0; yy < 16; ++yy) {
            const uint8_t *src =
                rgb + ((size_t)(base_y + yy) * (size_t)img_w + (size_t)base_x) * 3;
            memcpy(&rgb_block[yy * 16 * 3], src, 48);
        }
    } else {
        for (int yy = 0; yy < 16; ++yy) {
            int sy = base_y + yy;
            if (sy >= img_h) sy = img_h - 1;
            const uint8_t *src_row =
                rgb + (size_t)sy * (size_t)img_w * 3;
            for (int xx = 0; xx < 16; ++xx) {
                int sx = base_x + xx;
                if (sx >= img_w) sx = img_w - 1;
                const uint8_t *p = src_row + (size_t)sx * 3;
                uint8_t *dst = &rgb_block[(yy * 16 + xx) * 3];
                dst[0] = p[0]; dst[1] = p[1]; dst[2] = p[2];
            }
        }
    }

    // ---- Step 2: build 4 Y blocks ----
    for (int sub = 0; sub < 4; ++sub) {
        const int sx = (sub & 1) * 8;
        const int sy = (sub & 2) * 4;
        int32_t *Yblk = (sub == 0) ? Y_TL : (sub == 1) ? Y_TR
                      : (sub == 2) ? Y_BL : Y_BR;
        for (int yy = 0; yy < 8; ++yy) {
            const uint8_t *row = &rgb_block[((sy + yy) * 16 + sx) * 3];
#if HAVE_NEON
            const uint8x8x3_t rgb8 = vld3_u8(row);
            const uint16x8_t R16 = vmovl_u8(rgb8.val[0]);
            const uint16x8_t G16 = vmovl_u8(rgb8.val[1]);
            const uint16x8_t B16 = vmovl_u8(rgb8.val[2]);
            const uint32x4_t bias = vdupq_n_u32(32768);
            uint32x4_t Yl = vmlal_n_u16(bias, vget_low_u16(R16), 19595);
            Yl = vmlal_n_u16(Yl, vget_low_u16(G16), 38470);
            Yl = vmlal_n_u16(Yl, vget_low_u16(B16),  7471);
            uint32x4_t Yh = vmlal_n_u16(bias, vget_high_u16(R16), 19595);
            Yh = vmlal_n_u16(Yh, vget_high_u16(G16), 38470);
            Yh = vmlal_n_u16(Yh, vget_high_u16(B16),  7471);
            const uint16x4_t Yl_u16 = vshrn_n_u32(Yl, 16);
            const uint16x4_t Yh_u16 = vshrn_n_u32(Yh, 16);
            const int16x8_t Y8     = vreinterpretq_s16_u16(
                                        vcombine_u16(Yl_u16, Yh_u16));
            const int16x8_t Y8m128 = vsubq_s16(Y8, vdupq_n_s16(128));
            vst1q_s32(&Yblk[yy*8 + 0], vmovl_s16(vget_low_s16 (Y8m128)));
            vst1q_s32(&Yblk[yy*8 + 4], vmovl_s16(vget_high_s16(Y8m128)));
#else
            for (int xx = 0; xx < 8; ++xx) {
                const int r = row[3*xx + 0];
                const int g = row[3*xx + 1];
                const int b = row[3*xx + 2];
                const int Y = (19595*r + 38470*g + 7471*b + 32768) >> 16;
                Yblk[yy*8 + xx] = Y - 128;
            }
#endif
        }
    }

    // ---- Step 3: build Cb/Cr blocks ----
    for (int yy = 0; yy < 8; ++yy) {
        const int py = yy * 2;
        const uint8_t *row0 = &rgb_block[(py     * 16) * 3];
        const uint8_t *row1 = &rgb_block[((py+1) * 16) * 3];
#if HAVE_NEON
        const uint8x16x3_t rgb_r0 = vld3q_u8(row0);
        const uint8x16x3_t rgb_r1 = vld3q_u8(row1);
        const uint16x8_t Rsum = vaddq_u16(vpaddlq_u8(rgb_r0.val[0]),
                                          vpaddlq_u8(rgb_r1.val[0]));
        const uint16x8_t Gsum = vaddq_u16(vpaddlq_u8(rgb_r0.val[1]),
                                          vpaddlq_u8(rgb_r1.val[1]));
        const uint16x8_t Bsum = vaddq_u16(vpaddlq_u8(rgb_r0.val[2]),
                                          vpaddlq_u8(rgb_r1.val[2]));
        const uint16x8_t Rb = vrshrq_n_u16(Rsum, 2);
        const uint16x8_t Gb = vrshrq_n_u16(Gsum, 2);
        const uint16x8_t Bb = vrshrq_n_u16(Bsum, 2);
        const int16x8_t Rs = vreinterpretq_s16_u16(Rb);
        const int16x8_t Gs = vreinterpretq_s16_u16(Gb);
        const int16x8_t Bs = vreinterpretq_s16_u16(Bb);
        const int32x4_t bias_cbcr = vdupq_n_s32(32768 + (128 << 16));

        int32x4_t acc_lo = vshll_n_s16(vget_low_s16(Bs),  15);
        acc_lo = vmlal_n_s16(acc_lo, vget_low_s16(Rs), -11059);
        acc_lo = vmlal_n_s16(acc_lo, vget_low_s16(Gs), -21709);
        acc_lo = vaddq_s32(acc_lo, bias_cbcr);
        int32x4_t Cb_lo = vshrq_n_s32(acc_lo, 16);
        int32x4_t acc_hi = vshll_n_s16(vget_high_s16(Bs), 15);
        acc_hi = vmlal_n_s16(acc_hi, vget_high_s16(Rs), -11059);
        acc_hi = vmlal_n_s16(acc_hi, vget_high_s16(Gs), -21709);
        acc_hi = vaddq_s32(acc_hi, bias_cbcr);
        int32x4_t Cb_hi = vshrq_n_s32(acc_hi, 16);
        Cb_lo = vmaxq_s32(Cb_lo, vdupq_n_s32(0));
        Cb_lo = vminq_s32(Cb_lo, vdupq_n_s32(255));
        Cb_lo = vsubq_s32(Cb_lo, vdupq_n_s32(128));
        Cb_hi = vmaxq_s32(Cb_hi, vdupq_n_s32(0));
        Cb_hi = vminq_s32(Cb_hi, vdupq_n_s32(255));
        Cb_hi = vsubq_s32(Cb_hi, vdupq_n_s32(128));
        vst1q_s32(&Cb_blk[yy*8 + 0], Cb_lo);
        vst1q_s32(&Cb_blk[yy*8 + 4], Cb_hi);

        int32x4_t Cr_acc_lo = vshll_n_s16(vget_low_s16(Rs),  15);
        Cr_acc_lo = vmlal_n_s16(Cr_acc_lo, vget_low_s16(Gs), -27439);
        Cr_acc_lo = vmlal_n_s16(Cr_acc_lo, vget_low_s16(Bs),  -5329);
        Cr_acc_lo = vaddq_s32(Cr_acc_lo, bias_cbcr);
        int32x4_t Cr_lo = vshrq_n_s32(Cr_acc_lo, 16);
        int32x4_t Cr_acc_hi = vshll_n_s16(vget_high_s16(Rs), 15);
        Cr_acc_hi = vmlal_n_s16(Cr_acc_hi, vget_high_s16(Gs), -27439);
        Cr_acc_hi = vmlal_n_s16(Cr_acc_hi, vget_high_s16(Bs),  -5329);
        Cr_acc_hi = vaddq_s32(Cr_acc_hi, bias_cbcr);
        int32x4_t Cr_hi = vshrq_n_s32(Cr_acc_hi, 16);
        Cr_lo = vmaxq_s32(Cr_lo, vdupq_n_s32(0));
        Cr_lo = vminq_s32(Cr_lo, vdupq_n_s32(255));
        Cr_lo = vsubq_s32(Cr_lo, vdupq_n_s32(128));
        Cr_hi = vmaxq_s32(Cr_hi, vdupq_n_s32(0));
        Cr_hi = vminq_s32(Cr_hi, vdupq_n_s32(255));
        Cr_hi = vsubq_s32(Cr_hi, vdupq_n_s32(128));
        vst1q_s32(&Cr_blk[yy*8 + 0], Cr_lo);
        vst1q_s32(&Cr_blk[yy*8 + 4], Cr_hi);
#else
        for (int xx = 0; xx < 8; ++xx) {
            const int px = xx * 2;
            const uint8_t *p00 = row0 + (px    ) * 3;
            const uint8_t *p01 = row0 + (px + 1) * 3;
            const uint8_t *p10 = row1 + (px    ) * 3;
            const uint8_t *p11 = row1 + (px + 1) * 3;
            const int R = (p00[0] + p01[0] + p10[0] + p11[0] + 2) >> 2;
            const int G = (p00[1] + p01[1] + p10[1] + p11[1] + 2) >> 2;
            const int B = (p00[2] + p01[2] + p10[2] + p11[2] + 2) >> 2;
            int Cb = (-11059*R - 21709*G + 32768*B + 32768 + (128<<16)) >> 16;
            int Cr = ( 32768*R - 27439*G - 5329*B  + 32768 + (128<<16)) >> 16;
            if (Cb < 0) Cb = 0; else if (Cb > 255) Cb = 255;
            if (Cr < 0) Cr = 0; else if (Cr > 255) Cr = 255;
            Cb_blk[yy*8 + xx] = Cb - 128;
            Cr_blk[yy*8 + xx] = Cr - 128;
        }
#endif
    }

    // ---- Step 4: DCT + quantize + zigzag for each of the 6 blocks ----
    encode_block_phase1(Y_TL,   qy_div, zz_out[0]);
    encode_block_phase1(Y_TR,   qy_div, zz_out[1]);
    encode_block_phase1(Y_BL,   qy_div, zz_out[2]);
    encode_block_phase1(Y_BR,   qy_div, zz_out[3]);
    encode_block_phase1(Cb_blk, qc_div, zz_out[4]);
    encode_block_phase1(Cr_blk, qc_div, zz_out[5]);
}

// Worker context for parallel phase-1 dispatch over a band of MCU rows.
struct phase1_band_ctx {
    const uint8_t  *rgb;
    int             img_w;
    int             img_h;
    int             mcu_w;
    int             my_band_start;       // first MCU row of this band
    const uint16_t *qy_div;
    const uint16_t *qc_div;
    // Output buffer, flat layout:
    //   zz_buf[((row_rel * mcu_w + mx) * 6 + block) * 64 + coeff]
    int16_t        *zz_buf;
};

// parallel_rows worker: each thread owns MCU rows [my_low, my_high).
static void phase1_band_worker(int my_low, int my_high, void *cv) {
    const phase1_band_ctx *c = (const phase1_band_ctx*)cv;
    for (int my = my_low; my < my_high; ++my) {
        const int row_rel = my - c->my_band_start;
        for (int mx = 0; mx < c->mcu_w; ++mx) {
            int16_t (*zz_out)[64] = (int16_t(*)[64])
                (c->zz_buf + (size_t)((row_rel * c->mcu_w + mx) * 6) * 64);
            process_mcu_phase1(c->rgb, c->img_w, c->img_h,
                               mx, my, c->qy_div, c->qc_div, zz_out);
        }
    }
}

} // namespace

extern "C" int jpeg_encode_rgb(
    const uint8_t *rgb,
    int            width,
    int            height,
    int            quality,
    const exif_info *exif,
    uint8_t      **out_buf,
    size_t        *out_size)
{
    if (!rgb || !out_buf || !out_size) return 1;
    if (width <= 0 || height <= 0)     return 1;


    // Even-dimension requirement for 4:2:0. Round up by 1 row/col if odd
    // by replicating the last sample (handled per-MCU below).
    int img_w = width, img_h = height;

    // Build qtables (the values stored in the DQT segment).
    uint16_t qy[64], qc[64];
    build_qtable(qy, QTABLE_LUMA_BASE,   quality);
    build_qtable(qc, QTABLE_CHROMA_BASE, quality);

    // jfdctint outputs DCT scaled by 8. To get a properly-quantized result
    // we divide by (qtable[i] * 8). We keep two arrays: the stored qtable
    // for the DQT segment, and the divisor (× 8) for the actual division.
    uint16_t qy_div[64], qc_div[64];
    for (int i = 0; i < 64; ++i) {
        qy_div[i] = (uint16_t)((uint32_t)qy[i] << 3);
        qc_div[i] = (uint16_t)((uint32_t)qc[i] << 3);
    }

    // Build Huffman encoding tables.
    HuffEnc dc_y, ac_y, dc_c, ac_c;
    build_huff_enc(&dc_y, HUFF_DC_LUMA_BITS,   HUFF_DC_LUMA_VAL);
    build_huff_enc(&ac_y, HUFF_AC_LUMA_BITS,   HUFF_AC_LUMA_VAL);
    build_huff_enc(&dc_c, HUFF_DC_CHROMA_BITS, HUFF_DC_CHROMA_VAL);
    build_huff_enc(&ac_c, HUFF_AC_CHROMA_BITS, HUFF_AC_CHROMA_VAL);

    OutBuf out;
    out_init(&out, (size_t)img_w * (size_t)img_h / 8 + 65536);
    if (out.err) return 2;

    // === Header ===
    // SOI
    out_byte(&out, 0xff); out_byte(&out, 0xd8);

    // APP0 JFIF
    out_byte(&out, 0xff); out_byte(&out, 0xe0);
    out_be16(&out, 16);
    out_byte(&out, 'J'); out_byte(&out, 'F'); out_byte(&out, 'I'); out_byte(&out, 'F'); out_byte(&out, 0);
    out_byte(&out, 1); out_byte(&out, 1);  // version 1.1
    out_byte(&out, 1);                     // density unit = inches
    out_be16(&out, 72); out_be16(&out, 72);
    out_byte(&out, 0); out_byte(&out, 0);  // no thumbnail

    // APP1 EXIF (optional)
    emit_app1_exif(&out, exif);

    // DQT (luma)
    out_byte(&out, 0xff); out_byte(&out, 0xdb);
    out_be16(&out, 2 + 1 + 64);
    out_byte(&out, 0x00);  // Pq=0 (8-bit), Tq=0
    for (int i = 0; i < 64; ++i) out_byte(&out, (uint8_t)qy[ZIGZAG[i]]);

    // DQT (chroma)
    out_byte(&out, 0xff); out_byte(&out, 0xdb);
    out_be16(&out, 2 + 1 + 64);
    out_byte(&out, 0x01);  // Tq=1
    for (int i = 0; i < 64; ++i) out_byte(&out, (uint8_t)qc[ZIGZAG[i]]);

    // SOF0 (baseline)
    out_byte(&out, 0xff); out_byte(&out, 0xc0);
    out_be16(&out, 2 + 1 + 2 + 2 + 1 + 3*3);
    out_byte(&out, 8);                   // P
    out_be16(&out, (uint16_t)img_h);
    out_be16(&out, (uint16_t)img_w);
    out_byte(&out, 3);                   // Nf
    // Component 1 (Y): id=1, H=2, V=2, qtable=0
    out_byte(&out, 1); out_byte(&out, 0x22); out_byte(&out, 0);
    // Component 2 (Cb): id=2, H=1, V=1, qtable=1
    out_byte(&out, 2); out_byte(&out, 0x11); out_byte(&out, 1);
    // Component 3 (Cr): id=3, H=1, V=1, qtable=1
    out_byte(&out, 3); out_byte(&out, 0x11); out_byte(&out, 1);

    // DHT — write all 4 tables in one segment.
    {
        size_t total = 0;
        total += 1 + 16 + sizeof(HUFF_DC_LUMA_VAL);
        total += 1 + 16 + sizeof(HUFF_AC_LUMA_VAL);
        total += 1 + 16 + sizeof(HUFF_DC_CHROMA_VAL);
        total += 1 + 16 + sizeof(HUFF_AC_CHROMA_VAL);
        out_byte(&out, 0xff); out_byte(&out, 0xc4);
        out_be16(&out, (uint16_t)(2 + total));
        // DC luma:    Tc=0, Th=0
        out_byte(&out, 0x00);
        out_bytes(&out, HUFF_DC_LUMA_BITS, 16);
        out_bytes(&out, HUFF_DC_LUMA_VAL,  sizeof(HUFF_DC_LUMA_VAL));
        // AC luma:    Tc=1, Th=0
        out_byte(&out, 0x10);
        out_bytes(&out, HUFF_AC_LUMA_BITS, 16);
        out_bytes(&out, HUFF_AC_LUMA_VAL,  sizeof(HUFF_AC_LUMA_VAL));
        // DC chroma:  Tc=0, Th=1
        out_byte(&out, 0x01);
        out_bytes(&out, HUFF_DC_CHROMA_BITS, 16);
        out_bytes(&out, HUFF_DC_CHROMA_VAL,  sizeof(HUFF_DC_CHROMA_VAL));
        // AC chroma:  Tc=1, Th=1
        out_byte(&out, 0x11);
        out_bytes(&out, HUFF_AC_CHROMA_BITS, 16);
        out_bytes(&out, HUFF_AC_CHROMA_VAL,  sizeof(HUFF_AC_CHROMA_VAL));
    }

    // SOS
    out_byte(&out, 0xff); out_byte(&out, 0xda);
    out_be16(&out, 2 + 1 + 2*3 + 3);
    out_byte(&out, 3);                   // Ns
    out_byte(&out, 1); out_byte(&out, 0x00);  // Y:  Td=0, Ta=0
    out_byte(&out, 2); out_byte(&out, 0x11);  // Cb: Td=1, Ta=1
    out_byte(&out, 3); out_byte(&out, 0x11);  // Cr: Td=1, Ta=1
    out_byte(&out, 0); out_byte(&out, 63);    // Ss, Se
    out_byte(&out, 0);                        // Ah/Al = 0

    // === Scan data ===
    BitWriter bw;
    bw_init(&bw, &out);

    int prev_dc_y = 0, prev_dc_cb = 0, prev_dc_cr = 0;

    // 4:2:0: MCU is 16×16 pixels = 4 Y blocks (TL, TR, BL, BR) + 1 Cb + 1 Cr.
    int mcu_w = (img_w + 15) / 16;
    int mcu_h = (img_h + 15) / 16;

    // === Threaded phase 1, serial phase 2 ===
    //
    // The JPEG bitstream has serial state: DC coefficients are stored as
    // deltas from the previous block of the same component, and the bit
    // buffer fills sequentially with 0xff stuffing.  But the per-MCU work
    // BEFORE Huffman emission — RGB→YCbCr conversion, DCT, quantization,
    // zigzag — is purely pixel-local and can run on multiple cores.
    //
    // We split the encode into bands of MCU rows.  Each band:
    //   1. allocates a temporary buffer for the band's quantized
    //      zigzag coefficients (6 blocks × 64 int16 per MCU);
    //   2. dispatches phase-1 work across worker threads with
    //      parallel_rows();
    //   3. once all workers finish, the main thread walks the band's
    //      MCUs in order and emits each block's Huffman bits.
    //
    // Memory: with band_rows = 8 and a 5248×3936 image (mcu_w = 328),
    //         the band buffer is 8 × 328 × 6 × 64 × 2 = 2 MB.  Comfortable
    //         on the D5503's 2 GB RAM alongside the 41 MB Bayer plane and
    //         the 60 MB RGB plane the demosaicer hands us.
    //
    // Speedup ceiling: phase 2 (Huffman) becomes the bottleneck since it
    // can't parallelize.  Roughly ~30% of original JPEG time is phase 2,
    // so the absolute floor for the JPEG portion is ~30% of the original
    // serial time, regardless of core count.  In practice we get ~2.5×
    // on a 4-core Krait 400.
    const int band_rows = 8;
    const size_t band_buf_count =
        (size_t)band_rows * (size_t)mcu_w * 6 * 64;
    int16_t *zz_buf = (int16_t*)malloc(band_buf_count * sizeof(int16_t));
    if (!zz_buf) {
        free(out.data);
        return 5;
    }

    int rc_phase2 = 0;
    for (int band_start = 0; band_start < mcu_h; band_start += band_rows) {
        int band_end = band_start + band_rows;
        if (band_end > mcu_h) band_end = mcu_h;

        // Phase 1: parallel.
        phase1_band_ctx pctx;
        pctx.rgb            = rgb;
        pctx.img_w          = img_w;
        pctx.img_h          = img_h;
        pctx.mcu_w          = mcu_w;
        pctx.my_band_start  = band_start;
        pctx.qy_div         = qy_div;
        pctx.qc_div         = qc_div;
        pctx.zz_buf         = zz_buf;
        parallel_rows(band_start, band_end, /*num_threads=auto*/ 0,
                      +phase1_band_worker, &pctx);

        // Phase 2: serial Huffman emit, in MCU order.  Walks the band
        // buffer that's now fully populated by the workers.
        for (int my = band_start; my < band_end; ++my) {
            const int row_rel = my - band_start;
            for (int mx = 0; mx < mcu_w; ++mx) {
                const int16_t *zz_mcu =
                    zz_buf + (size_t)((row_rel * mcu_w + mx) * 6) * 64;
                if (encode_block_phase2(&bw, zz_mcu + 0*64,
                                        &dc_y, &ac_y, &prev_dc_y) != 0) { rc_phase2 = 1; goto huff_err; }
                if (encode_block_phase2(&bw, zz_mcu + 1*64,
                                        &dc_y, &ac_y, &prev_dc_y) != 0) { rc_phase2 = 1; goto huff_err; }
                if (encode_block_phase2(&bw, zz_mcu + 2*64,
                                        &dc_y, &ac_y, &prev_dc_y) != 0) { rc_phase2 = 1; goto huff_err; }
                if (encode_block_phase2(&bw, zz_mcu + 3*64,
                                        &dc_y, &ac_y, &prev_dc_y) != 0) { rc_phase2 = 1; goto huff_err; }
                if (encode_block_phase2(&bw, zz_mcu + 4*64,
                                        &dc_c, &ac_c, &prev_dc_cb) != 0) { rc_phase2 = 1; goto huff_err; }
                if (encode_block_phase2(&bw, zz_mcu + 5*64,
                                        &dc_c, &ac_c, &prev_dc_cr) != 0) { rc_phase2 = 1; goto huff_err; }
            }
        }
    }

huff_err:
    free(zz_buf);
    if (rc_phase2 != 0) goto err;

    bw_flush(&bw);

    // EOI.
    out_byte(&out, 0xff); out_byte(&out, 0xd9);

    if (out.err) {
        free(out.data);
        return 3;
    }

    *out_buf  = out.data;
    *out_size = out.size;
    return 0;

err:
    free(out.data);
    return 4;
}
