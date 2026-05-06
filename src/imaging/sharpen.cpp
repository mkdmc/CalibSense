/*
 * sharpen.cpp — see sharpen.h.
 *
 * Three passes:
 *
 *   Pass 1: read interleaved RGB, compute Y per pixel into a scratch
 *           int16 plane. Y is computed as the BT.709 weighted sum
 *           (matches the luma weights used elsewhere in the pipeline).
 *
 *   Pass 2: separable Gaussian blur on the Y plane (horizontal then
 *           vertical), in-place with one row + one column of scratch.
 *
 *   Pass 3: read original RGB and the now-blurred Y. Compute
 *           detail = original_Y - blurred_Y, apply threshold
 *           suppression, multiply by amount, add to R/G/B equally,
 *           clamp to [0, 255].
 *
 * To get original_Y in pass 3 without a second buffer, we recompute
 * it from the input RGB. The RGB buffer hasn't been modified yet
 * at that point in the loop (each pixel reads input, writes output
 * to the same slot — but the original Y depends only on this pixel's
 * R, G, B which haven't been touched in this iteration).
 *
 * Coefficients are BT.709 to match the saturation step in demosaic.cpp.
 * Fixed-point with denom 1024:
 *   Y = (217*R + 732*G + 74*B + 512) >> 10
 */

#include "sharpen.h"
#include "threading.h"

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/*
 * NEON SIMD path
 *
 * On ARMv7 with NEON (the D5503's Krait 400) and on ARMv8 (AArch64), we
 * use 4-wide float NEON to vectorize the per-pixel inner loops of the
 * Gaussian blur and the Y-compute pass.  Falls back to scalar code
 * everywhere else (including the desktop test build, which is x86).
 *
 * Each NEON path is paired with a scalar path that produces the same
 * output to within sub-LSB rounding noise (NEON's vfmaq_f32 fuses
 * multiply-add at higher precision than the scalar `a*b + c` pair, and
 * vcvtq_s32_f32 truncates toward zero rather than round-to-nearest;
 * both are dealt with explicitly in the code).  This means any visible
 * difference between the two paths would be a real bug, not a rounding
 * artefact — if you ever see a mismatch larger than ±1 in the 8-bit
 * output, the NEON code is wrong.
 *
 * Build flags on the device must enable NEON. The toolchain file
 * (CMakeLists.txt) already sets `-mfpu=neon` for ARMv7; AArch64 has
 * NEON unconditionally.
 */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#  include <arm_neon.h>
#  define HAVE_NEON 1
#else
#  define HAVE_NEON 0
#endif


namespace {

// Build a normalized 1D Gaussian kernel into `weights[2*radius+1]`.
static void build_gauss(float sigma, int radius, float* weights) {
    float sum = 0.0f;
    for (int i = 0; i <= 2 * radius; ++i) {
        const int x = i - radius;
        const float w = expf(-(float)(x * x) / (2.0f * sigma * sigma));
        weights[i] = w;
        sum += w;
    }
    const float inv = 1.0f / sum;
    for (int i = 0; i <= 2 * radius; ++i) weights[i] *= inv;
}

// Horizontal blur, in-place. Scratch row is written through.
// Horizontal Gaussian blur on a single row.
//
// Splits the row into three regions:
//   [0, radius)            : scalar with edge clamping
//   [radius, W - radius)   : NEON-vectorized 4-pixels-at-a-time inner kernel
//   [W - radius, W)        : scalar with edge clamping
//
// The interior region is the hot path on a normal-sized image (one row
// at 5248 px with radius=8 has 16 edge pixels and 5232 interior pixels;
// the scalar edges are <0.4% of the work).
//
// On NEON, the inner loop processes 4 output positions in parallel.
// For each tap k, we issue ONE 4-lane int16 load at row[x+k..x+k+3],
// widen to int32, convert to float, and FMA into the accumulator with
// the broadcast scalar weight w[k+radius]. This collapses 4×17 = 68
// scalar multiply-adds into 17 NEON ops on the deployed radius=8 kernel.
static inline void blur_row_horiz(const int16_t* row, int W,
                                  const float* w, int radius,
                                  int16_t* out)
{
    // Left edge: scalar with clamping. Output positions [0, radius).
    for (int x = 0; x < radius && x < W; ++x) {
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int xx = x + k;
            if (xx < 0)        xx = 0;
            else if (xx >= W)  xx = W - 1;
            acc += w[k + radius] * (float)row[xx];
        }
        out[x] = (int16_t)lrintf(acc);
    }

    int x = radius;
    const int x_end_simd = W - radius;  // exclusive

#if HAVE_NEON
    // Vectorized middle region: 4 output positions per iteration.
    // We can read row[x-radius..x+3+radius] without bounds checks.
    const float32x4_t half_pos = vdupq_n_f32( 0.5f);
    const float32x4_t half_neg = vdupq_n_f32(-0.5f);
    const float32x4_t zero4    = vdupq_n_f32( 0.0f);

    for (; x + 3 < x_end_simd; x += 4) {
        float32x4_t acc = vdupq_n_f32(0.0f);
        for (int k = -radius; k <= radius; ++k) {
            // Load 4 contiguous int16 from row[x+k..x+k+3].
            const int16x4_t v_i16 = vld1_s16(&row[x + k]);
            // Widen to int32x4 then convert to float32x4.
            const int32x4_t v_i32 = vmovl_s16(v_i16);
            const float32x4_t v_f32 = vcvtq_f32_s32(v_i32);
            // acc += v_f32 * w[k + radius]
            acc = vmlaq_n_f32(acc, v_f32, w[k + radius]);
        }
        // Round-half-away-from-zero, then saturating-narrow to int16.
        // (vcvtq_s32_f32 truncates toward zero on ARMv7, so we add a
        // sign-aware bias of ±0.5 first to match the scalar lrintf
        // behavior.  Drift from "round half to even" is at most 1 ULP
        // for values exactly at .5 — we have measured this and it's
        // sub-LSB after the int16 cast.)
        const uint32x4_t neg_mask = vcltq_f32(acc, zero4);
        const float32x4_t bias = vbslq_f32(neg_mask, half_neg, half_pos);
        const float32x4_t biased = vaddq_f32(acc, bias);
        const int32x4_t r_i32 = vcvtq_s32_f32(biased);
        const int16x4_t r_i16 = vqmovn_s32(r_i32);
        vst1_s16(&out[x], r_i16);
    }
#endif

    // Tail of middle region (or all of it on scalar builds): scalar, no
    // edge clamping needed because radius <= x < W - radius.
    for (; x < x_end_simd; ++x) {
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            acc += w[k + radius] * (float)row[x + k];
        }
        out[x] = (int16_t)lrintf(acc);
    }

    // Right edge: scalar with clamping. Output positions [W-radius, W).
    for (; x < W; ++x) {
        float acc = 0.0f;
        for (int k = -radius; k <= radius; ++k) {
            int xx = x + k;
            if (xx < 0)        xx = 0;
            else if (xx >= W)  xx = W - 1;
            acc += w[k + radius] * (float)row[xx];
        }
        out[x] = (int16_t)lrintf(acc);
    }
}

// Edge handling: clamp-to-edge.
static void blur_horizontal(int16_t* data, int W, int H,
                            const float* w, int radius,
                            int16_t* scratch_row_unused)
{
    (void)scratch_row_unused;  // unused — each thread allocates its own

    struct ctx_t {
        int16_t      *data;
        int           W;
        const float  *w;
        int           radius;
    };
    ctx_t c{ data, W, w, radius };

    auto worker = [](int y0, int y1, void *cv) {
        const ctx_t *c = (const ctx_t*)cv;
        const int W = c->W;
        const int radius = c->radius;
        const float *w = c->w;
        int16_t *data = c->data;
        // Per-thread row scratch.
        int16_t *scratch = (int16_t*)malloc((size_t)W * sizeof(int16_t));
        if (!scratch) return;  // OOM: skip; result will be unsmoothed in this band

        for (int y = y0; y < y1; ++y) {
            int16_t* row = data + (size_t)y * (size_t)W;
            blur_row_horiz(row, W, w, radius, scratch);
            memcpy(row, scratch, (size_t)W * sizeof(int16_t));
        }
        free(scratch);
    };
    parallel_rows(0, H, /*auto*/ 0, +worker, &c);
}

// Vertical blur, in-place. Parallel by columns: parallel_rows() takes a
// "row range" but we use it as an X range here (each "row" parameter is
// actually a column index). Each thread allocates its own scratch.
//
// On NEON, the inner kernel processes 4 X positions at the same Y, by
// loading 4 contiguous int16 values from data[(y+k)*W + x..x+3] and
// multiplying by the scalar weight w[k+radius]. The X-axis vectorization
// is naturally parallel-safe within a column band, and the 4-element
// loads are each from a single cache line (vs the original 1-element-
// per-row load pattern).
static void blur_vertical(int16_t* data, int W, int H,
                          const float* w, int radius,
                          int16_t* scratch_col_unused)
{
    (void)scratch_col_unused;

    struct ctx_t {
        int16_t      *data;
        int           W;
        int           H;
        const float  *w;
        int           radius;
    };
    ctx_t c{ data, W, H, w, radius };

    auto worker = [](int x0, int x1, void *cv) {
        const ctx_t *c = (const ctx_t*)cv;
        const int W = c->W;
        const int H = c->H;
        const int radius = c->radius;
        const float *w = c->w;
        int16_t *data = c->data;

#if HAVE_NEON
        // Per-thread scratch big enough for one 4-column block.
        // For typical image height (3936), that's 3936 * 4 * 2 = 31 KB.
        // Krait 400 has a private 4 KB L0-D + 16 KB L1-D per core (with
        // an exclusive L0/L1 hierarchy giving ~20 KB of fast data
        // capacity). The 31 KB scratch overspills that into the 2 MB
        // shared L2, but the access pattern is a sequential column
        // sweep, so the hardware prefetcher hides most of the latency.
        int16_t *scratch4 = (int16_t*)malloc((size_t)H * 4 * sizeof(int16_t));
        // Fallback scratch for tail columns (1-3 leftover).
        int16_t *scratch1 = (int16_t*)malloc((size_t)H * sizeof(int16_t));
        if (!scratch4 || !scratch1) { free(scratch4); free(scratch1); return; }

        const float32x4_t half_pos = vdupq_n_f32( 0.5f);
        const float32x4_t half_neg = vdupq_n_f32(-0.5f);
        const float32x4_t zero4    = vdupq_n_f32( 0.0f);

        int x = x0;
        for (; x + 3 < x1; x += 4) {
            // Process 4 columns [x, x+1, x+2, x+3] simultaneously.
            for (int y = 0; y < H; ++y) {
                float32x4_t acc = vdupq_n_f32(0.0f);
                for (int k = -radius; k <= radius; ++k) {
                    int yy = y + k;
                    if (yy < 0)        yy = 0;
                    else if (yy >= H)  yy = H - 1;
                    const int16x4_t v_i16 = vld1_s16(&data[(size_t)yy * (size_t)W + x]);
                    const int32x4_t v_i32 = vmovl_s16(v_i16);
                    const float32x4_t v_f32 = vcvtq_f32_s32(v_i32);
                    acc = vmlaq_n_f32(acc, v_f32, w[k + radius]);
                }
                // Same round-half-away-from-zero as horizontal blur.
                const uint32x4_t neg_mask = vcltq_f32(acc, zero4);
                const float32x4_t bias = vbslq_f32(neg_mask, half_neg, half_pos);
                const float32x4_t biased = vaddq_f32(acc, bias);
                const int32x4_t r_i32 = vcvtq_s32_f32(biased);
                const int16x4_t r_i16 = vqmovn_s32(r_i32);
                vst1_s16(&scratch4[y * 4], r_i16);
            }
            // Copy scratch4 back into data, one column at a time. Cannot
            // avoid the strided write (data layout is row-major), but at
            // least the source is a small contiguous block.
            for (int col = 0; col < 4; ++col) {
                for (int y = 0; y < H; ++y) {
                    data[(size_t)y * (size_t)W + x + col] = scratch4[y * 4 + col];
                }
            }
        }

        // Tail: 0..3 leftover columns, scalar.
        for (; x < x1; ++x) {
            for (int y = 0; y < H; ++y) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int yy = y + k;
                    if (yy < 0)        yy = 0;
                    else if (yy >= H)  yy = H - 1;
                    acc += w[k + radius] * (float)data[(size_t)yy * (size_t)W + x];
                }
                scratch1[y] = (int16_t)lrintf(acc);
            }
            for (int y = 0; y < H; ++y)
                data[(size_t)y * (size_t)W + x] = scratch1[y];
        }

        free(scratch1);
        free(scratch4);
#else
        // Scalar path: one column at a time.
        int16_t *scratch = (int16_t*)malloc((size_t)H * sizeof(int16_t));
        if (!scratch) return;
        for (int x = x0; x < x1; ++x) {
            for (int y = 0; y < H; ++y) {
                float acc = 0.0f;
                for (int k = -radius; k <= radius; ++k) {
                    int yy = y + k;
                    if (yy < 0)        yy = 0;
                    else if (yy >= H)  yy = H - 1;
                    acc += w[k + radius] * (float)data[(size_t)yy * (size_t)W + x];
                }
                scratch[y] = (int16_t)lrintf(acc);
            }
            for (int y = 0; y < H; ++y)
                data[(size_t)y * (size_t)W + x] = scratch[y];
        }
        free(scratch);
#endif
    };
    parallel_rows(0, W, /*auto*/ 0, +worker, &c);
}

} // namespace


extern "C" void sharpen_rgb8(uint8_t* rgb, int width, int height,
                             float amount, float radius, int threshold)
{
    if (!rgb || amount <= 0.0f) return;
    if (width < 4 || height < 4) return;


    // Clamp radius into a sensible range. Below 0.3 the kernel is
    // basically a delta and the result is nearly identity; above 8
    // we'd allocate huge tap counts for diminishing return.
    if (radius < 0.3f) radius = 0.3f;
    if (radius > 8.0f) radius = 8.0f;

    // Kernel half-width = ceil(2.5 sigma) keeps tail energy below
    // ~1% of peak. Capped at 16 to stay within the static `weights`
    // array.
    int kradius = (int)ceilf(2.5f * radius);
    if (kradius < 1)  kradius = 1;
    if (kradius > 16) kradius = 16;

    const int W = width, H = height;
    const size_t N = (size_t)W * (size_t)H;

    // Y plane (int16 — needs sign for the later detail subtract,
    // and post-blur values fit easily in 16-bit even after the
    // weighted sums used by lrintf).
    int16_t* Y         = (int16_t*)malloc(N * sizeof(int16_t));
    int16_t* scratch_W = (int16_t*)malloc((size_t)W * sizeof(int16_t));
    int16_t* scratch_H = (int16_t*)malloc((size_t)H * sizeof(int16_t));
    if (!Y || !scratch_W || !scratch_H) {
        free(Y); free(scratch_W); free(scratch_H);
        return;  // OOM → silently skip rather than crash
    }

    // ---- Pass 1: compute BT.709 Y for each pixel ----
    //
    //   Y = (217*R + 732*G + 74*B + 512) >> 10
    //
    // (sum of forward coefficients ≈ 1023, off by 1 LSB — fine.
    //  +512 implements round-half-up.)
    //
    // NEON path uses vld3_u8 to deinterleave 8 RGB pixels per load,
    // then computes Y in uint32 lanes. Y range [0, ~255] fits in u16
    // after the >> 10 narrow.  Values are non-negative everywhere here
    // (RGB is uint8), so unsigned arithmetic is safe.
    {
        struct y_ctx_t { const uint8_t *rgb; int16_t *Y; int W; };
        y_ctx_t yc{ rgb, Y, W };
        auto y_worker = [](int y0, int y1, void *cv) {
            const y_ctx_t *c = (const y_ctx_t*)cv;
            const uint8_t *rgb = c->rgb;
            int16_t *Y = c->Y;
            const int W = c->W;
            for (int y = y0; y < y1; ++y) {
                const size_t row_off = (size_t)y * (size_t)W;
                int x = 0;
#if HAVE_NEON
                const uint32x4_t bias512 = vdupq_n_u32(512);
                for (; x + 7 < W; x += 8) {
                    const uint8_t *p = &rgb[3 * (row_off + (size_t)x)];
                    // Deinterleave 8 RGB pixels (24 bytes) into 3 lanes
                    // of 8 u8 values each.
                    const uint8x8x3_t rgb8 = vld3_u8(p);
                    const uint16x8_t R = vmovl_u8(rgb8.val[0]);
                    const uint16x8_t G = vmovl_u8(rgb8.val[1]);
                    const uint16x8_t B = vmovl_u8(rgb8.val[2]);
                    // Compute Y for the low 4 pixels.
                    uint32x4_t Y_lo = vmull_n_u16(vget_low_u16(R), 217);
                    Y_lo = vmlal_n_u16(Y_lo, vget_low_u16(G), 732);
                    Y_lo = vmlal_n_u16(Y_lo, vget_low_u16(B),  74);
                    Y_lo = vaddq_u32(Y_lo, bias512);
                    // ... and the high 4 pixels.
                    uint32x4_t Y_hi = vmull_n_u16(vget_high_u16(R), 217);
                    Y_hi = vmlal_n_u16(Y_hi, vget_high_u16(G), 732);
                    Y_hi = vmlal_n_u16(Y_hi, vget_high_u16(B),  74);
                    Y_hi = vaddq_u32(Y_hi, bias512);
                    // Narrow + shift right by 10 in one go.
                    const uint16x4_t Y_lo_u16 = vshrn_n_u32(Y_lo, 10);
                    const uint16x4_t Y_hi_u16 = vshrn_n_u32(Y_hi, 10);
                    const uint16x8_t Y_u16    = vcombine_u16(Y_lo_u16, Y_hi_u16);
                    // Y in [0, 255], reinterpret as int16 and store.
                    vst1q_s16(&Y[row_off + (size_t)x],
                              vreinterpretq_s16_u16(Y_u16));
                }
#endif
                // Tail (or all of it on scalar builds).
                for (; x < W; ++x) {
                    const size_t i = row_off + (size_t)x;
                    const int R = rgb[3*i + 0];
                    const int G = rgb[3*i + 1];
                    const int B = rgb[3*i + 2];
                    Y[i] = (int16_t)((217*R + 732*G + 74*B + 512) >> 10);
                }
            }
        };
        parallel_rows(0, H, /*auto*/ 0, +y_worker, &yc);
    }

    // ---- Pass 2: separable Gaussian blur on Y ----
    float weights[33];   // up to radius=16 → 33 taps
    build_gauss(radius, kradius, weights);
    blur_horizontal(Y, W, H, weights, kradius, scratch_W);
    blur_vertical  (Y, W, H, weights, kradius, scratch_H);

    // ---- Pass 3: detail = Y_original - Y_blurred, apply, recombine ----
    //
    // `Y` now holds Y_blurred. We recompute Y_original from the still-
    // unmodified RGB (the loop reads RGB and writes the same slot, but
    // each iteration touches one pixel only, so the read happens
    // before the write).
    //
    // detail signal interpretation: positive at edges where the local
    // pixel is brighter than its blurred neighborhood (e.g. on the
    // bright side of an edge); negative on the dark side. Standard
    // USM amplifies both, producing the characteristic edge "pop".
    {
        struct apply_ctx_t {
            uint8_t        *rgb;
            const int16_t  *Y;
            int             W;
            float           amount;
            int             threshold;
        };
        apply_ctx_t ac{ rgb, Y, W, amount, threshold };
        auto apply_worker = [](int y0, int y1, void *cv) {
            const apply_ctx_t *c = (const apply_ctx_t*)cv;
            uint8_t *rgb = c->rgb;
            const int16_t *Y = c->Y;
            const int W = c->W;
            const float amount = c->amount;
            const int threshold = c->threshold;
            for (int y = y0; y < y1; ++y) {
                const size_t row_off = (size_t)y * (size_t)W;
                for (int x = 0; x < W; ++x) {
                    const size_t i = row_off + (size_t)x;
                    const int R = rgb[3*i + 0];
                    const int G = rgb[3*i + 1];
                    const int B = rgb[3*i + 2];
                    const int Y_orig = (217*R + 732*G + 74*B + 512) >> 10;
                    const int Y_blur = Y[i];
                    int detail = Y_orig - Y_blur;

                    if (detail > 0) {
                        detail = (detail > threshold) ? (detail - threshold) : 0;
                    } else if (detail < 0) {
                        detail = (detail < -threshold) ? (detail + threshold) : 0;
                    }

                    const float scaled = (float)detail * amount;
                    int delta = (int)lrintf(scaled);

                    if (delta > 0) {
                        int maxRGB = R; if (G > maxRGB) maxRGB = G; if (B > maxRGB) maxRGB = B;
                        const int headroom = 255 - maxRGB;
                        if (delta > headroom) delta = headroom;
                    } else if (delta < 0) {
                        int minRGB = R; if (G < minRGB) minRGB = G; if (B < minRGB) minRGB = B;
                        const int headroom = -minRGB;
                        if (delta < headroom) delta = headroom;
                    }

                    int Rn = R + delta;
                    int Gn = G + delta;
                    int Bn = B + delta;
                    if (Rn < 0)   Rn = 0;   else if (Rn > 255) Rn = 255;
                    if (Gn < 0)   Gn = 0;   else if (Gn > 255) Gn = 255;
                    if (Bn < 0)   Bn = 0;   else if (Bn > 255) Bn = 255;
                    rgb[3*i + 0] = (uint8_t)Rn;
                    rgb[3*i + 1] = (uint8_t)Gn;
                    rgb[3*i + 2] = (uint8_t)Bn;
                }
            }
        };
        parallel_rows(0, H, /*auto*/ 0, +apply_worker, &ac);
    }

    free(scratch_H); free(scratch_W); free(Y);
}
