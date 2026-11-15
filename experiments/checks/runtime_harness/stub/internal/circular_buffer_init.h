// Stub of the circular buffers' set-up an overlay calls when it loads a program's
// buffers: not modelled (no scenario loads an overlay); it only compiles the call.
#pragma once

#include <stdint.h>

template <bool Read, bool Write, bool Init>
inline void setup_local_cb_read_write_interfaces(uint32_t*, uint32_t, uint32_t) {
}
