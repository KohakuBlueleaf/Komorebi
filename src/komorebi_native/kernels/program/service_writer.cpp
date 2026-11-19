// The runtime's service overlay, its writer part on Wormhole (KM_WRITER_COLD): the
// writer's paths after the work, run from L1 out of the 16 KiB its CPU holds.
#include <stdint.h>

#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/tensors/km_extent.hpp"

uint32_t km_overlay(km::Writer&, const km::ColdCall& call) {
    switch (call.path) {
        case km::kColdComplete:
            return (
                km::complete(
                    *static_cast<const km::Core*>(call.object), *call.command, call.a
                )
            );
        case km::kColdFinish:
            return static_cast<km::Core*>(call.object)->finish(*call.command, call.a);
#if KM_GUARD
        case km::kColdBound:
            return km::bound_lookup(call.a, call.b);
#endif
        default:
            return 0;
    }
}
