/*
 * threading.cpp — see threading.h.
 *
 * Implementation: spawn (N-1) pthread workers, run last band on the
 * caller's thread, join. The caller-thread band runs concurrently with
 * the spawned threads, so we get N-way parallelism with N-1 spawns.
 *
 * If thread creation fails partway through we run all remaining bands
 * on the caller's thread serially. This degrades performance but keeps
 * correctness guaranteed.
 */

#include "threading.h"

#include <pthread.h>
#include <unistd.h>
#include <stdlib.h>

namespace {

struct band_args {
    int            y0;
    int            y1;
    row_worker_fn  worker;
    void          *ctx;
};

static void* band_thread(void *arg) {
    band_args *a = (band_args*)arg;
    a->worker(a->y0, a->y1, a->ctx);
    return nullptr;
}

static int auto_thread_count() {
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n < 1) n = 1;
    if (n > MAX_PARALLEL_THREADS) n = MAX_PARALLEL_THREADS;
    return (int)n;
}

} /* anonymous namespace */

extern "C" void parallel_rows(int y_start, int y_end, int num_threads,
                                row_worker_fn worker, void *ctx)
{
    if (!worker) return;
    if (y_end <= y_start) return;

    if (num_threads <= 0) num_threads = auto_thread_count();
    if (num_threads < 1) num_threads = 1;
    if (num_threads > MAX_PARALLEL_THREADS) num_threads = MAX_PARALLEL_THREADS;

    const int total_rows = y_end - y_start;
    if (total_rows < num_threads) num_threads = total_rows;

    if (num_threads == 1) {
        worker(y_start, y_end, ctx);
        return;
    }

    pthread_t  tids[MAX_PARALLEL_THREADS];
    band_args  args[MAX_PARALLEL_THREADS];
    int spawned = 0;

    /* Compute band boundaries: rows[i] = y_start + i*total_rows/num_threads.
     * The last band runs on the caller thread. Threads 0..N-2 spawn.
     * This balances rows ±1 between bands. */
    for (int i = 0; i < num_threads - 1; ++i) {
        args[i].y0     = y_start + (int)((long)i      * total_rows / num_threads);
        args[i].y1     = y_start + (int)((long)(i+1)  * total_rows / num_threads);
        args[i].worker = worker;
        args[i].ctx    = ctx;
        if (pthread_create(&tids[i], nullptr, band_thread, &args[i]) != 0) {
            /* Spawn failure: run this band and remaining bands serially
             * on the caller thread. Already-spawned threads still in flight
             * will be joined below. */
            worker(args[i].y0, args[i].y1, ctx);
            /* run remaining bands on caller thread too */
            const int last_band_y0 = y_start + (int)((long)(num_threads - 1) * total_rows / num_threads);
            for (int j = i + 1; j < num_threads - 1; ++j) {
                int j0 = y_start + (int)((long)j     * total_rows / num_threads);
                int j1 = y_start + (int)((long)(j+1) * total_rows / num_threads);
                worker(j0, j1, ctx);
            }
            worker(last_band_y0, y_end, ctx);
            /* Join already-spawned (which are 0..i-1). */
            for (int j = 0; j < i; ++j) pthread_join(tids[j], nullptr);
            return;
        }
        ++spawned;
    }

    /* Caller thread runs last band concurrently with the spawned ones. */
    const int last_y0 = y_start + (int)((long)(num_threads - 1) * total_rows / num_threads);
    worker(last_y0, y_end, ctx);

    /* Join. */
    for (int i = 0; i < spawned; ++i) pthread_join(tids[i], nullptr);
}
