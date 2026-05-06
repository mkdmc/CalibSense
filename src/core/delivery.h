// delivery.h — framed Unix-socket output channel.
//
// CalibSense delivers DNG and (optionally) JPEG frames over an abstract
// Unix socket named @calibsense_socket. The peer reads length-prefixed
// messages each prefixed with a 4-byte magic identifying the payload type.
//
// The listener is NOT brought up automatically — call calibsense_delivery_start()
// once during your application's startup to start the accept thread.
// After that, calibsense_process_frame() (in calibsense.h) will stream frames to any
// connected peer.

#ifndef CALIBSENSE_DELIVERY_H
#define CALIBSENSE_DELIVERY_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Frame magics — 4-byte LE tag at the start of every framed message.
// These are part of the on-the-wire contract; do NOT rename the symbols
// or change the values.
constexpr uint32_t FRAME_MAGIC_DNG = 0x31474E44u;  // 'DNG1' LE
constexpr uint32_t FRAME_MAGIC_JPG = 0x3147504Au;  // 'JPG1' LE — demosaiced JPEG

// Bring up the @calibsense_socket listener and spawn the accept thread.
// Idempotent (subsequent calls are no-ops). Returns 0 on success, -1 on
// socket / bind / listen / pthread_create failure.
int calibsense_delivery_start(void);

// Send a framed message. Non-blocking, drops frame on failure.
// Returns 0 on success, -1 on failure (no listener / send error / timeout).
int calibsense_delivery_send(uint32_t magic, const void* body, size_t body_len);

// Returns 1 if a peer is currently connected, 0 otherwise.
int calibsense_delivery_has_peer(void);

#ifdef __cplusplus
}
#endif

#endif /* CALIBSENSE_DELIVERY_H */
