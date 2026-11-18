// Cross: what the reader's part and the writer's share (the leader's port 0: its
// reader says the crossing is done; the leader is in no product, so it is free).
#pragma once

#include "../../../kernels/ops.hpp"
#include "../../two.h"

constexpr uint32_t kCrossDonePort = 0;

template <typename Kernel>
inline volatile tt_l1_ptr uint32_t* cross_done(const Kernel& core) {
    return &km::port_words(core, kCrossDonePort)[KM_PORT_COUNT];
}
