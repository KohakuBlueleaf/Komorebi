// The service overlay's writer part on Wormhole: the paths after the work that writer
// level KM_WRITER_COLD moves out of the writer's 16 KiB, run from L1.
#include <stdint.h>

#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/tensors/km_extent.hpp"
#if KM_WRITER_COLD >= 4
#include "komorebi/device/movement/km_chain.hpp"
#endif

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
#if KM_WRITER_COLD >= 4
        case km::kColdChainMark: {  // a: the ChainDirect, b: 1 DRAM's too, 2 sent
            if (&km::chain_now == nullptr) {  // a kernel without a chain: none to mark
                return 0;
            }
            const auto& writer = *static_cast<const km::Writer*>(call.object);
            const auto& direct = *reinterpret_cast<const km::ChainDirect*>(call.a);
            const km::ChainShape shape{};
            return (
                (call.b & 2u) != 0
                    ? km::chain_mark<true>(
                        shape, writer, *call.command, direct, (call.b & 1u) != 0
                    )
                    : km::chain_mark<false>(
                        shape, writer, *call.command, direct, (call.b & 1u) != 0
                    )
            );
        }
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
