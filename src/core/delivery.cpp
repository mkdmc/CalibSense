// delivery.cpp — abstract Unix socket channel for DNG/JPEG → camera app delivery.
//
// Listens on an abstract-namespace AF_UNIX socket. The camera app connects
// at activity start and stays connected for the camera session. Each
// calibsense_process_frame() invocation streams a framed DNG message over the live
// connection.
//
// Security:
//   - Only one connection accepted at a time. A new connect closes any old.
//   - SO_PEERCRED verifies the connecting peer's uid against an expected
//     value before the connection is published. Wrong uid → connection
//     dropped immediately, never receives any bytes.
//   - All writes are non-blocking with MSG_DONTWAIT|MSG_NOSIGNAL: a slow
//     reader makes us drop the frame, never blocks the camera pipeline.

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/un.h>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#include <stdint.h>
#include <stddef.h>
#include <time.h>
#include <atomic>
using std::atomic;
using std::memory_order_relaxed;
#include <string.h>

#include "delivery.h"

// ---------- config ----------
//
// Abstract socket name. The leading NUL byte distinguishes the abstract
// namespace from filesystem paths in struct sockaddr_un — added at bind time.
static const char  SOCKET_NAME[]      = "calibsense_socket";
static const size_t SOCKET_NAME_LEN   = sizeof(SOCKET_NAME) - 1;

// Allowed peer uid range. Stock D5503 camera app: uid u0_a60 = 10060.
// We intentionally accept any uid in the "app" range (10000..19999) so the
// implementation works across firmware/build variations where the app uid
// can shift. The kernel-supplied uid via SO_PEERCRED is unforgeable; the
// only attackers we filter are non-app processes (system_server, mediaserver
// itself, root tools, etc.).
constexpr uid_t PEER_UID_MIN = 10000;
constexpr uid_t PEER_UID_MAX = 19999;

// ---------- shared state ----------
//
// `g_listen_fd`  : the bound listening socket. Created once at init.
// `g_client_fd`  : the currently-connected client fd, or -1 if none.
//                  Written by accept thread, read by calibsense_process_frame thread,
//                  accessed via atomics so neither tears.
// `g_accept_th`  : the accept thread (joined never; lives until process exit).
//
static int                  g_listen_fd     = -1;
static atomic<int>          g_client_fd{-1};
static pthread_t            g_accept_thread = 0;
static atomic<bool>         g_initialized{false};

// ---------- accept thread ----------

static void close_client_locked(void) {
    int old = g_client_fd.exchange(-1);
    if (old >= 0) {
        close(old);
    }
}

static void* accept_thread_main(void* /*arg*/) {
    // The thread services TWO event sources: new clients on g_listen_fd,
    // and disconnects on the currently-attached client. Using poll() lets
    // us notice the peer's close immediately (POLLHUP) and stop accepting
    // frames before the next caller invocation — otherwise an in-flight
    // send into a half-closed connection would block the producer.
    for (;;) {
        struct pollfd pfds[2];
        pfds[0].fd     = g_listen_fd;
        pfds[0].events = POLLIN;
        int npoll = 1;

        int cur_client = g_client_fd.load();
        if (cur_client >= 0) {
            pfds[1].fd     = cur_client;
            // POLLHUP fires when the peer closes its end. POLLERR/POLLNVAL
            // catch ungraceful tear-downs and fd-already-closed races.
            // POLLIN is requested too: if Java ever sends data we drain
            // and treat EOF as disconnect.
            pfds[1].events = POLLIN;
            npoll = 2;
        }

        int rc = poll(pfds, (nfds_t)npoll, -1);
        if (rc < 0) {
            if (errno == EINTR) continue;
            usleep(50 * 1000);
            continue;
        }
        if (rc == 0) continue;  // shouldn't happen with timeout=-1

        // ----- handle client-side disconnect first -----
        // If the client fd raced behind us (e.g. calibsense_delivery_send tore it
        // down between the load above and the poll), pfds[1].fd is stale.
        // POLLNVAL covers that case.
        if (npoll == 2) {
            short re = pfds[1].revents;
            if (re & (POLLHUP | POLLERR | POLLNVAL)) {
                // Verify the fd we polled is still the active client — a
                // racing accept could have replaced it between our snapshot
                // and now. Only close if it's still the same fd.
                if (g_client_fd.load() == pfds[1].fd) {
                    close_client_locked();
                }
            } else if (re & POLLIN) {
                // Drain any unsolicited data from Java. Treat EOF (recv==0)
                // as disconnect — peer half-closed.
                char drainbuf[64];
                ssize_t n = recv(pfds[1].fd, drainbuf, sizeof(drainbuf), MSG_DONTWAIT);
                if (n == 0) {
                    if (g_client_fd.load() == pfds[1].fd) {
                        close_client_locked();
                    }
                } else if (n < 0 && errno != EAGAIN && errno != EWOULDBLOCK) {
                    if (g_client_fd.load() == pfds[1].fd) {
                        close_client_locked();
                    }
                }
                // else: data drained, ignore (we don't expect Java to send us anything)
            }
        }

        // ----- handle new client connect -----
        if (!(pfds[0].revents & POLLIN)) {
            continue;
        }

        struct sockaddr_un peer;
        socklen_t plen = sizeof(peer);
        int c = accept(g_listen_fd, (struct sockaddr*)&peer, &plen);
        if (c < 0) {
            if (errno == EINTR) continue;
            usleep(50 * 1000);
            continue;
        }

        // Authenticate the peer by uid.
        struct ucred cred;
        socklen_t clen = sizeof(cred);
        if (getsockopt(c, SOL_SOCKET, SO_PEERCRED, &cred, &clen) < 0) {
            close(c);
            continue;
        }

        if (cred.uid < PEER_UID_MIN || cred.uid > PEER_UID_MAX) {
            close(c);
            continue;
        }


        // Switch to non-blocking; we never want a slow client to stall capture.
        int flags = fcntl(c, F_GETFL, 0);
        if (flags >= 0) fcntl(c, F_SETFL, flags | O_NONBLOCK);

        // Replace any existing client. Accept-only-newest semantics: if the
        // app reconnects (e.g. after activity restart) the old fd is closed
        // and the new one becomes active.
        int prev = g_client_fd.exchange(c);
        if (prev >= 0) {
            close(prev);
        }
    }
    return NULL;
}

// ---------- public API ----------

extern "C" int calibsense_delivery_start(void) {
    if (g_initialized.exchange(true)) {
        return 0;  // idempotent
    }

    int fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        return -1;
    }

    // Non-fatal: enable close-on-exec so any forks don't inherit.
    int fdflags = fcntl(fd, F_GETFD, 0);
    if (fdflags >= 0) fcntl(fd, F_SETFD, fdflags | FD_CLOEXEC);

    // Build abstract sockaddr: first byte of sun_path is NUL.
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    addr.sun_path[0] = '\0';
    memcpy(addr.sun_path + 1, SOCKET_NAME, SOCKET_NAME_LEN);
    socklen_t addrlen = (socklen_t)(offsetof(struct sockaddr_un, sun_path)
                                    + 1 + SOCKET_NAME_LEN);

    if (bind(fd, (struct sockaddr*)&addr, addrlen) < 0) {
        if (errno == EADDRINUSE) {
            // Another instance of libcalibsense in this process already bound
            // @calibsense_socket. That instance's accept thread is the active one;
            // we silently step aside so we don't double-listen. The first
            // bind wins for the lifetime of the process.
            close(fd);
            return 0;
        }
        close(fd);
        return -1;
    }

    if (listen(fd, 1) < 0) {
        close(fd);
        return -1;
    }

    g_listen_fd = fd;

    if (pthread_create(&g_accept_thread, NULL, accept_thread_main, NULL) != 0) {
        close(fd);
        g_listen_fd = -1;
        return -1;
    }
    // Don't join — let it run for the lifetime of the host process.
    pthread_detach(g_accept_thread);
    return 0;
}

// calibsense_delivery_send writes a framed message: 8-byte header then body.
//   header[0..3]  : magic (le u32)
//   header[4..7]  : body length (le u32)
//
// All writes are non-blocking. If the peer is slow / disconnected / has a
// full receive buffer, the frame is dropped and the connection is torn
// down. The capture path NEVER blocks here.
//
// Returns 0 on success (frame fully written), -1 on any failure (frame
// dropped).
extern "C" int calibsense_delivery_send(uint32_t magic, const void* body, size_t body_len) {
    int fd = g_client_fd.load();
    if (fd < 0) {
        return -1;  // no listener, fast skip — caller treats as "RAW disabled"
    }


    uint8_t hdr[8];
    hdr[0] = (uint8_t)(magic        & 0xFF);
    hdr[1] = (uint8_t)((magic >> 8)  & 0xFF);
    hdr[2] = (uint8_t)((magic >> 16) & 0xFF);
    hdr[3] = (uint8_t)((magic >> 24) & 0xFF);
    uint32_t blen = (uint32_t)body_len;
    hdr[4] = (uint8_t)(blen        & 0xFF);
    hdr[5] = (uint8_t)((blen >> 8)  & 0xFF);
    hdr[6] = (uint8_t)((blen >> 16) & 0xFF);
    hdr[7] = (uint8_t)((blen >> 24) & 0xFF);

    // Header
    ssize_t n = send(fd, hdr, sizeof(hdr), MSG_NOSIGNAL | MSG_DONTWAIT);
    if (n != (ssize_t)sizeof(hdr)) {
        close_client_locked();
        return -1;
    }

    // Body — possibly large (~41 MB DNG, ~4 MB JPG). Chunked send with
    // a per-chunk poll for writability. Returns -1 if Java doesn't drain
    // within the per-chunk timeout.
    const uint8_t* p = (const uint8_t*)body;
    size_t left = body_len;

    constexpr int    POLL_TIMEOUT_MS = 1000;
    constexpr size_t CHUNK_BYTES     = 64 * 1024;

    while (left > 0) {
        struct pollfd pfd;
        pfd.fd = fd;
        pfd.events = POLLOUT;
        pfd.revents = 0;

        int pr = poll(&pfd, 1, POLL_TIMEOUT_MS);

        if (pr <= 0 || !(pfd.revents & POLLOUT)) {
            // Java not draining within the timeout window.
            close_client_locked();
            return -1;
        }

        size_t want = left < CHUNK_BYTES ? left : CHUNK_BYTES;
        ssize_t s = send(fd, p, want, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (s <= 0) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) continue;
            close_client_locked();
            return -1;
        }
        p    += s;
        left -= (size_t)s;
    }

    return 0;
}

extern "C" int calibsense_delivery_has_peer(void) {
    return g_client_fd.load() >= 0 ? 1 : 0;
}
