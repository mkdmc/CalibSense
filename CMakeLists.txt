# CMakeLists.txt — libcalibsense for Sony D5503 (32-bit ARM Android)
#
# Build:
#   cmake -B build \
#         -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
#         -DANDROID_ABI=armeabi-v7a \
#         -DANDROID_PLATFORM=android-21 \
#         -DCMAKE_BUILD_TYPE=Release
#   cmake --build build
#
# Output:
#   build/libcalibsense.so

cmake_minimum_required(VERSION 3.13)
project(calibsense C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_compile_options(
    -Wall -Wextra
    -fvisibility=hidden
    -ffunction-sections -fdata-sections
    -fno-exceptions -fno-rtti
    -O2
)

# Enable NEON SIMD on ARMv7. AArch64 has NEON unconditionally so the
# flag is not needed (and would be rejected) on arm64-v8a.
#
# Without -mfpu=neon, the NDK's default for armeabi-v7a is vfpv3-d16
# (no NEON) and our `#if defined(__ARM_NEON)` paths compile out, falling
# back to scalar. The library still works, just slower.
if(ANDROID_ABI STREQUAL "armeabi-v7a")
    add_compile_options(-mfpu=neon)
endif()

add_link_options(
    -Wl,--gc-sections
    -Wl,--strip-all
    -Wl,--hash-style=sysv
    # NODELETE: prevent the host process from unmapping libcalibsense once
    # loaded. The accept thread spawned by calibsense_delivery_start() runs forever;
    # if our .so is dlclose'd while it's running, the next instruction it
    # executes is in unmapped memory and the host crashes.
    -Wl,-z,nodelete
)

# -----------------------------------------------------------------------------
# Internal STATIC libraries — one per src/ subdirectory.
#
# These are linked into the single libcalibsense.so output below. They exist
# only to make the dependency graph explicit and incremental builds faster;
# none of them are installed or used outside this project.
#
# Dependency layers (top depends on bottom):
#
#   core       (pipeline + init + delivery socket)
#       │
#   imaging    (demosaic, bayer_nr, sharpen, lsc)         jpeg
#       │                                                   │
#       └─────────────── util ───────────────────────────────┘
#                       (threading)
# -----------------------------------------------------------------------------

# ---- util ----
add_library(cs_util STATIC
    src/util/threading.cpp
)
target_include_directories(cs_util PUBLIC src/util)

# ---- imaging ----
add_library(cs_imaging STATIC
    src/imaging/demosaic.cpp
    src/imaging/bayer_nr.cpp
    src/imaging/sharpen.cpp
    src/imaging/lsc.cpp
)
target_include_directories(cs_imaging PUBLIC src/imaging)
target_link_libraries(cs_imaging PUBLIC cs_util)

# ---- jpeg ----
# jpeg_enc.cpp    = baseline 4:2:0 encoder (DCT, quantize, Huffman, output).
# jpeg_exif.cpp   = APP1 EXIF segment builder.
# jpeg_tables.cpp = standard Annex K quantization + Huffman tables.
add_library(cs_jpeg STATIC
    src/jpeg/jpeg_enc.cpp
    src/jpeg/jpeg_exif.cpp
    src/jpeg/jpeg_tables.cpp
)
target_include_directories(cs_jpeg PUBLIC src/jpeg)
target_link_libraries(cs_jpeg PUBLIC cs_util)

# ---- core ----
# pipeline.cpp    = the public calibsense_process_frame() entry; orchestrates DNG write
#                   and (optionally) the demosaic→sharpen→JPEG path.
# tiff_writer.cpp = TIFF tag/type constants + IfdBuilder used by pipeline.
# delivery.cpp    = the @calibsense_socket abstract Unix socket channel.
add_library(cs_core STATIC
    src/core/pipeline.cpp
    src/core/tiff_writer.cpp
    src/core/delivery.cpp
)
target_include_directories(cs_core
    PUBLIC
        include          # calibsense.h
        src/core         # delivery.h
)
target_link_libraries(cs_core PUBLIC cs_imaging cs_jpeg cs_util)

# -----------------------------------------------------------------------------
# Final shared library. Just an aggregator of the static libs above.
#
# The .cpp file is an empty stub — older CMake versions (and some NDK-bundled
# CMakes) reject add_library() targets that have no sources of their own,
# even when sources arrive transitively through target_link_libraries().
# Giving the target one trivial TU sidesteps the check and also lets CMake
# infer the linker language without needing an explicit LINKER_LANGUAGE hint.
# -----------------------------------------------------------------------------
add_library(calibsense SHARED
    src/libcalibsense.cpp
)

target_link_libraries(calibsense
    PRIVATE
        # --whole-archive forces every object from cs_core to be pulled in,
        # even if it has no callers visible inside libcalibsense.so. Required
        # because the public API entry points — calibsense_process_frame, calibsense_process_file,
        # calibsense_delivery_start / send / has_listener — have no internal callers
        # within the library itself; their consumers are external (whoever
        # links or dlopens this .so). Without --whole-archive, the static-lib
        # selection rule "pull in objects that resolve undefined references"
        # would drop pipeline.o and delivery.o entirely.
        -Wl,--whole-archive
        cs_core
        -Wl,--no-whole-archive
        cs_imaging
        cs_jpeg
        cs_util
)

# Pthreads (accept thread in delivery.cpp + worker pools in cs_util/threading).
find_package(Threads REQUIRED)
target_link_libraries(calibsense PRIVATE Threads::Threads)

# libm for powf (gamma LUT) and integer math helpers.
target_link_libraries(calibsense PRIVATE m)

# Default-visibility for symbols that need to be findable from outside the .so.
# Everything else stays hidden (the -fvisibility=hidden global flag).
#
# pipeline.cpp  exposes calibsense_process_frame / calibsense_process_file (the imaging entry).
# delivery.cpp  exposes calibsense_delivery_start / calibsense_delivery_send / calibsense_delivery_has_peer
#               (consumers call calibsense_delivery_start() once at startup, then bytes
#               from calibsense_process_frame flow over the socket to a connected peer).
# No other symbols need to escape the library.
set_source_files_properties(src/core/pipeline.cpp PROPERTIES COMPILE_FLAGS "-fvisibility=default")
set_source_files_properties(src/core/delivery.cpp PROPERTIES COMPILE_FLAGS "-fvisibility=default")
