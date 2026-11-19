// The Tensix dispatcher's inspect (the X280's grid/inspect.h), for "out": the cores'
// state, live words and trace read over the NoC, a core a pass, into kTbInspect.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"

namespace td {

    static_assert(
        kTbInspect + KMD_INSPECT_HEAD + KM_CORES_MAX * KMD_INSPECT_RECORD == kTbBytes,
        "the block ends with the inspect block"
    );

    // Whether [address, address + bytes) lies in a core's L1 above the firmware's part.
    inline bool in_l1(uint32_t address, uint32_t bytes) {
        return (
            address >= KMD_CONTROL_FLOOR && address <= MEM_L1_SIZE &&
            bytes <= MEM_L1_SIZE - address
        );
    }

    // An inspect of cores [first, first + count): a deadline's (its slot given up once
    // every core left the command after its abort word) or, later, the host's.
    struct Inspect {
        uint32_t block = 0, control = 0;
        bool active = false, leaving = false;
        uint32_t kind = 0, request = 0, first = 0, count = 0, next = 0;
        uint32_t slot = 0, sequence = 0;
        uint64_t start = 0, until = 0;

        volatile tt_l1_ptr uint32_t* header() const {
            return km::l1(block + kTbInspect);
        }

        uint32_t record(uint32_t index) const {
            return block + kTbInspect + KMD_INSPECT_HEAD + index * KMD_INSPECT_RECORD;
        }

        void begin(
            uint32_t kind_word,
            uint32_t request_word,
            uint32_t group,
            uint32_t slot_index,
            uint32_t sequence_word,
            uint64_t now
        ) {  // fmt: skip
            volatile tt_l1_ptr uint32_t* head = header();
            head[KMD_I_MARK] = 0;  // incomplete before any word changes
            head[KMD_I_KIND] = kind_word;
            head[KMD_I_REQUEST] = request_word;
            head[KMD_I_FIRST] = group & 0xFFFFu;
            head[KMD_I_COUNT] = group >> 16;
            head[KMD_I_DONE] = 0;
            head[KMD_I_SLOT] = slot_index;
            head[KMD_I_SEQUENCE] = sequence_word;
            head[KMD_I_START] = static_cast<uint32_t>(now);
            head[KMD_I_TICKS] = 0;
            head[KMD_I_RESET] = 0;
            head[KMD_I_RESET_FAILED] = 0;
            active = true;
            leaving = false;
            kind = kind_word;
            request = request_word;
            first = group & 0xFFFFu;
            count = group >> 16;
            next = 0;
            slot = slot_index;
            sequence = sequence_word;
            start = now;
        }

        // Record `index`: core `worker`'s state and live words, then the trace entries
        // of the inspect's sequence and the three before it, where its trace has them.
        template <typename Core>
        void read_core(
            uint32_t index, uint32_t worker, Core core, uint32_t coordinates
        ) {
            const uint32_t at = record(index);
            volatile tt_l1_ptr uint32_t* line = km::l1(at);
            for (uint32_t entry = 0; entry < KMD_INSPECT_ENTRIES; entry++) {
                line[KMD_IR_SEQUENCES + entry] = sequence - entry;
            }
            uint32_t flags = 0;
            if (in_l1(control, KM_CONTROL_BYTES)) {
                noc_async_read(
                    core(worker, control + KM_CONTROL_STATE),
                    at + KMD_INSPECT_STATE,
                    4 * KM_STATE_WORDS
                );
                noc_async_read(
                    core(worker, control + KM_CONTROL_LIVE),
                    at + KMD_INSPECT_LIVE,
                    KM_LIVE_READ
                );
                noc_async_read_barrier();
                invalidate_l1_cache();
                flags |= KMD_INSPECT_READ;
                volatile tt_l1_ptr uint32_t* live = km::l1(at + KMD_INSPECT_LIVE);
                const uint32_t trace = live[KM_LIVE_TRACE];
                const uint32_t cap = live[KM_LIVE_TRACE_CAP];
                if (
                    trace != 0 && trace % KM_TRACE_ENTRY == 0 && cap <= MEM_L1_SIZE &&
                    in_l1(trace, cap * KM_TRACE_ENTRY)
                ) {
                    for (uint32_t entry = 0; entry < KMD_INSPECT_ENTRIES; entry++) {
                        const uint32_t held = line[KMD_IR_SEQUENCES + entry];
                        const uint32_t to = (
                            at + KMD_INSPECT_TRACE + entry * KM_TRACE_ENTRY
                        );
                        if (held != 0 && held - 1 < cap) {
                            noc_async_read(
                                core(worker, trace + (held - 1) * KM_TRACE_ENTRY),
                                to,
                                KM_TRACE_ENTRY
                            );
                        } else {
                            for (uint32_t word = 0; word < KM_TRACE_ENTRY / 4; word++) {
                                km::l1(to)[word] = 0;
                            }
                        }
                    }
                    noc_async_read_barrier();
                    flags |= KMD_INSPECT_TRACED;
                }
            }
            line[KMD_IR_CORE] = worker;
            line[KMD_IR_COORDINATES] = coordinates;
            line[KMD_IR_FLAGS] = flags;
            line[KMD_IR_MTIME] = static_cast<uint32_t>(km::wall64());
        }

        // Whether core `worker` runs no part of the inspect's command any more: none of
        // its kernels' live sequence words holds it (read into `scratch`, 32 bytes).
        template <typename Core>
        bool left(uint32_t worker, Core core, uint32_t scratch) const {
            noc_async_read(core(worker, control + KM_CONTROL_LIVE + 16), scratch, 32);
            noc_async_read_barrier();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* live = km::l1(scratch);
            for (uint32_t role = 0; role < KM_LIVE_ROLES; role++) {
                if (live[KM_LIVE_SEQUENCE - 4 + role] == sequence) {
                    return false;
                }
            }
            return true;
        }

        // The block's records written: the ticks they took, then the mark.
        void mark(uint64_t now) {
            volatile tt_l1_ptr uint32_t* head = header();
            head[KMD_I_TICKS] = static_cast<uint32_t>(now - start);
            asm volatile("" ::: "memory");
            head[KMD_I_MARK] = KMD_INSPECT_MARK;
        }
    };

}  // namespace td
