// libcalibsense.cpp — placeholder source for the SHARED aggregator target.
//
// The libcalibsense.so target has no logic of its own; it exists only to
// link cs_core / cs_imaging / cs_jpeg / cs_util into a single shared
// library. CMake (depending on version) refuses to create a library
// target with zero sources, so this file is the stub that satisfies it.
//
// All real code lives in the static libraries pulled in by
// target_link_libraries(calibsense ...) in the top-level CMakeLists.txt.
