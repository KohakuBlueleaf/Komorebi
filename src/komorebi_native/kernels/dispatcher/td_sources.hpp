// The Tensix dispatcher's sources of commands: one command ring each in this core's L1,
// under the X280 dispatcher's protocol with the host. A source added later is a row.
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/device/core/km_core.hpp"
#include "td_slots.hpp"

namespace td {

    // One source's ring: command i (from 1) is the 64-byte line i % depth, taken once
    // its four landing words show i.
    struct Source {
        volatile tt_l1_ptr uint32_t* ring = nullptr;
        uint32_t depth = 0;  // lines, a power of two
        uint32_t next = 1;   // the number of the next command to take

        void init(volatile tt_l1_ptr uint32_t* lines, uint32_t lines_depth) {
            ring = lines;
            depth = lines_depth;
            next = 1;
        }

        // The line command `next` lands in.
        volatile tt_l1_ptr uint32_t* line() const {
            return ring + (next & (depth - 1u)) * 16;
        }

        // Whether command `next` has landed: all four landing words show its number.
        bool landed() const {
            volatile tt_l1_ptr uint32_t* at = line();
            return at[0] == next && at[4] == next && at[8] == next && at[12] == next;
        }

        // Copies the landed command into `pending` and steps on; returns its number.
        uint32_t take(uint32_t* pending) {
            volatile tt_l1_ptr uint32_t* at = line();
            for (uint32_t word = 0; word < 16; word++) {
                pending[word] = at[word];
            }
            return next++;
        }
    };

    // A thread's area, laid out as the X280's work area (KMD_WORK_*), wherever its
    // writer runs: commands taken, events written, the events, then the lines to it.
    struct ThreadArea {
        uint32_t base = 0;         // L1 address, 64-byte aligned
        uint32_t event_depth = 0;  // events of 16 bytes, a power of two
        uint64_t remote = 0;       // the X280 thread's copy of the area (0: none)

#if !defined(COMPILE_FOR_TRISC)  // the copies are out's and in's (the engines: TRISC 0)
        // An X280 thread's area: its tile (x | y << 16) and the area's address there.
        void place(uint32_t tile, uint32_t address) {
            remote = tile == 0 ? 0 : get_noc_addr(tile & 0xFFFFu, tile >> 16, address);
        }

        // The area's `bytes` at `offset`, written here, copied to the X280's area.
        void mirror(uint32_t offset, uint32_t bytes) const {
            if (remote != 0) {
                noc_async_write(base + offset, remote + offset, bytes);
            }
        }
#endif

        // Written by out: the thread's commands taken (a line of the ring is free
        // again).
        volatile tt_l1_ptr uint32_t* consumed() const {
            return km::l1(base + KMD_WORK_CONSUMED);
        }

        // Written by in: events written into the ring below, the last one landed first.
        volatile tt_l1_ptr uint32_t* told() const {
            return km::l1(base + KMD_WORK_TOLD);
        }

        // Event `index` (from 1): {index, request, status, ticks}, as the host's ring.
        volatile tt_l1_ptr uint32_t* event(uint32_t index) const {
            return km::l1(base + KMD_WORK_EVENTS + (index & (event_depth - 1u)) * 16);
        }

        // Command line `number` (from 1) to the thread, until it answers it.
        volatile tt_l1_ptr uint32_t* in_line(uint32_t number) const {
            return (
                km::l1(
                    base + KMD_WORK_IN(event_depth) + (number % KMD_WORK_IN_DEPTH) * 64
                )
            );
        }

        uint32_t bytes() const {
            return KMD_WORK_IN(event_depth) + KMD_WORK_IN_DEPTH * 64;
        }
    };

    // The sources by index: the host 0, the thread (the user's code on this core, or a
    // core writing its ring over the NoC) 1, the engines 4, other islands' (td_link) 5.
    constexpr uint32_t kSourceHost = 0;
    constexpr uint32_t kSourceThread = 1;
    constexpr uint32_t kSourceEngine = 4;  // 2 and 3: threads on X280 clusters 2 and 3
    constexpr uint32_t kSourceLink = 5;
    constexpr uint32_t kSources = 6;
    constexpr uint32_t kClusters = 3;  // a thread on cluster c is source c

    // Source `source` is a thread's (clusters 1 to 3).
    inline bool is_thread(uint32_t source) {
        return source - kSourceThread < kClusters;
    }

    // Cluster c's thread area: 1 right after the block, 2 and 3 after the engines'.
    inline uint32_t thread_area(uint32_t block, uint32_t cluster) {
        return block + (cluster <= 1 ? kTbBytes : kTbFar + (cluster - 2) * kTaBytes);
    }

    // The argument of cluster c's thread tile (its work area's: the next one).
    inline uint32_t thread_tile_arg(uint32_t cluster) {
        return cluster <= 1 ? kTdThreadTile : kTdFarThread + 2 * (cluster - 2);
    }

    static_assert(kTdThreadTile + 1 == kTdThreadWork, "a tile, then its work area");

    // `ticks` of the wall clock in a thread's own clock, the X280's timer (kmx_time):
    // `scale` its ticks a wall-clock tick, times 2^32 (kTdThreadScale).
    inline uint32_t thread_ticks(uint32_t ticks, uint32_t scale) {
        return static_cast<uint32_t>(static_cast<uint64_t>(ticks) * scale >> 32);
    }

    // The source of a slot's record: its event goes back there.
    inline uint32_t record_source(uint32_t flags) {
        return (
            flags & kRecordThread
                ? kSourceThread + (flags / kRecordCluster) % 4
                : flags & kRecordEngine
                    ? kSourceEngine
                    : flags & kRecordLink
                        ? kSourceLink
                        : kSourceHost
        );
    }

    static_assert(
        kTaWork + KMD_WORK_IN(kTaEventDepth) + KMD_WORK_IN_DEPTH * 64 == kTaState
    );
    static_assert(kTeWork + KMD_WORK_EVENTS + kTeEventDepth * 16 == kTeTable);

}  // namespace td
