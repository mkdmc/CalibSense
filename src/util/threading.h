/*
 * threading.h — minimal pthread-based row-band dispatcher.
 *
 * Provides parallel_rows(), which splits a [y_start, y_end) range into
 * roughly equal bands and runs a caller-supplied worker on each band
 * concurrently. The last band runs on the caller's thread; the others
 * spawn pthreads which are joined before returning.
 *
 * Designed for embarrassingly-parallel image-processing passes where
 * each output row depends only on input data (read-only neighborhoods)
 * and writes are confined to a single output row. Demosaic, separable
 * convolution, and per-pixel transforms all fit this pattern.
 *
 * No locks, no allocator pressure, no template machinery — just
 * pthread_create / pthread_join.
 */

#ifndef THREADING_H
#define THREADING_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Worker callback signature. Process rows in [y0, y1).
 * `ctx` is opaque caller-provided context. */
typedef void (*row_worker_fn)(int y0, int y1, void *ctx);

/*
 * Run `worker` on rows [y_start, y_end) using up to `num_threads` threads.
 * If num_threads <= 0, auto-detect via sysconf(_SC_NPROCESSORS_ONLN),
 * clamped to [1, MAX_PARALLEL_THREADS].
 *
 * The function blocks until all bands complete. If thread creation fails
 * mid-spawn the function falls back to running the remaining work on the
 * caller's thread.
 */
void parallel_rows(int y_start, int y_end, int num_threads,
                   row_worker_fn worker, void *ctx);

/* Upper safety cap for num_threads. The actual count is auto-detected
 * via sysconf(_SC_NPROCESSORS_ONLN) — on the D5503 (Snapdragon 800 with
 * 4 Krait 400 cores) that returns 4. The cap of 8 just guards against
 * an unreasonable auto-detect result on multi-purpose desktop builds
 * where this code is also compiled and run. Going beyond the actual
 * core count would only add context-switch overhead. */
#define MAX_PARALLEL_THREADS 8

#ifdef __cplusplus
}
#endif

#endif /* THREADING_H */
