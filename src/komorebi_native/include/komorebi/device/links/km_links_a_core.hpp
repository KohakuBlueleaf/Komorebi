// A link core's links, the one definition: the transport's and the stubs' of a build
// without page links (km_worker.hpp) alike.
#pragma once

#include <stdint.h>

namespace km {

    // A link core's links (link l: CPU l % kLinksACore of link core l / kLinksACore).
#if defined(KM_LINKS_A_CORE)
    constexpr uint32_t kLinksACore = KM_LINKS_A_CORE;  // the host's (chips links_a_core)
#else
    constexpr uint32_t kLinksACore = 2;
#endif
    static_assert(kLinksACore == 1 || kLinksACore == 2, "a link a CPU, two CPUs");

}  // namespace km
