// Link operations, reader: OPS writes into another rank's L1 and adds to its word
// (km::write_l1, km::add_word); CLEAR and REPORT set and read a rank's area.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/tensors/km_ask.hpp"
#include "komorebi/device/links/km_far.hpp"
#include "komorebi/device/movement/km_port.hpp"
#include "komorebi/device/km_worker.hpp"
#include "ops.h"

// OPS (ops.h): every call in order; the calls done in the sender's own area.
void ops(km::Reader& reader, const km_command_t& command) {
    const uint32_t target = command.a0, calls = command.a1, port = command.a2;
    const uint32_t slots = command.a3, bytes = command.a4, kind = command.a5;
    const uint32_t area = get_write_ptr(kCbArea);
    const uint32_t source = get_write_ptr(kCbSource);
    volatile tt_l1_ptr uint32_t*
        words = (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(source));
    volatile tt_l1_ptr uint32_t* own = (
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area)
    );
    const int at = port == kNoPort ? -1 : static_cast<int>(port);
    uint32_t done = 0;
    for (uint32_t call = 0; call < calls; call++) {
        if (kind & kKindWrite) {
            for (uint32_t w = 0; w < bytes / 4; w++) {
                words[w] = ops_word(call, w, target);
            }
            const uint32_t slot = (
                area + kAreaRing + call % slots * ((bytes + 15) / 16 * 16)
            );
            if (!km::write_l1(
                reader,
                command,
                target,
                slot,
                reinterpret_cast<void*>(source),
                bytes,
                at
            )) {
                break;
            }
        }
        const uint32_t counter = area + kAreaCounter * 4;
        if (
            (kind & kKindAdd) &&
            !(
                kind & kKindPosted
                    ? km::add_word(reader, command, target, counter, 1, km::posted)
                    : km::add_word(reader, command, target, counter, 1)
            )
        ) {
            break;
        }
        done++;
    }
    if ((kind & kKindPosted) && done == calls && !km::put_wait(reader, command)) {
        done = 0;
    }
    noc_async_atomic_barrier();
    own[kAreaDone] = done;
}

void kernel_main() {
    constexpr auto args = TensorAccessorArgs<0>();  // the report tensors' (one shape)
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        const uint32_t area = get_write_ptr(kCbArea);
        volatile tt_l1_ptr uint32_t*
            words = (reinterpret_cast<volatile tt_l1_ptr uint32_t*>(area));
        if (km::handler(command) == kHandlerClear) {
            for (uint32_t w = 0; w < kAreaBytes / 4; w++) {
                words[w] = 0;
            }
            km::port_words(reader, command.a0)[KM_PORT_COUNT] = 0;
            return;
        }
        if (km::handler(command) == kHandlerOps) {
            ops(reader, command);
            return;
        }
        if (km::handler(command) == kHandlerReport) {
            invalidate_l1_cache();
            words[kAreaPort] = km::port_words(reader, command.a2)[KM_PORT_COUNT];
            km::write_pages(reader, command, args, command.a0, command.a1, 1, area);
            noc_async_write_barrier();
        }
    });
}
