// The service overlay's writer part on Wormhole: the paths after the work that writer
// level KM_WRITER_COLD moves out of the writer's 16 KiB, run from L1.
#include <stdint.h>

#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/tensors/km_extent.hpp"

uint32_t km_overlay(km::Writer&, const km::ColdCall& call) {
    switch (call.path) {
#if KM_WRITER_COLD >= 3
        case km::kColdComplete:
            return (
                km::complete(
                    *static_cast<const km::Core*>(call.object), *call.command, call.a
                )
            );
#endif
#if KM_WRITER_COLD >= 2
        case km::kColdFinish:
            return static_cast<km::Core*>(call.object)->finish(*call.command, call.a);
#endif
#if KM_GUARD
        case km::kColdBound:
            return km::bound_lookup(call.a, call.b);
#endif
        default:
            return 0;
    }
}
