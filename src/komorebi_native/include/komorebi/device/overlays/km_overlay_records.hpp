// The overlay area's layout and records: each (part, slot) record, the service's own
// slot's, the epoch, a core's check. Needs km::l1 and tt_l1_ptr declared first.
#pragma once

#include <stdint.h>

#include "../../abi/km_abi.h"

// The slots of one part a run's programs have at most (the host's, a define of its
// kernels when over 4; else 4): records of slots 4 and up follow the overlay area.
#if !defined(KM_OVERLAY_SLOTS)
#define KM_OVERLAY_SLOTS 4
#endif

namespace km {

    // The overlay area: kOverlayAreaBytes after the control block, written by the
    // host at load (header, table), then by the cores (records, clock, lines).
    constexpr uint32_t kOverlayArea = KM_CONTROL_BYTES;
    constexpr uint32_t kOverlayParts = 5;  // reader, writer, unpack, math, pack
    constexpr uint32_t kOverlayPartReader = 0;
    constexpr uint32_t kOverlayPartWriter = 1;
    constexpr uint32_t kOverlayPartUnpack = 2;
    constexpr uint32_t kOverlaySlotsMax = KM_OVERLAY_SLOTS;  // slots of one part
    constexpr uint32_t kOverlaySlotsFixed = 4;  // of them, records in the area's block
    constexpr uint32_t kOverlaysMax = 64;       // numbers 1 to 63; 0 names none
    constexpr uint32_t kOverlayBanks = 8;       // core i reads its copy in bank i % 8
    // Header words: table entries; first slot, slots and slot bytes of each part.
    constexpr uint32_t kOverlayCount = 0;
    constexpr uint32_t kOverlayBase = 1;
    constexpr uint32_t kOverlaySlots = 6;
    constexpr uint32_t kOverlaySlotBytes = 11;
    // A 16-byte record per (part, slot 0 to 3; 4 on: kOverlayMoreRecords): the overlay
    // held (0: none), the part's own check (below), the calls running, its address.
    constexpr uint32_t kOverlayRecords = 64;
    constexpr uint32_t kOverlayHeld = 0;
    constexpr uint32_t kOverlayChecked = 1;
    constexpr uint32_t kOverlayCalling = 2;
    constexpr uint32_t kOverlayAddress = 3;
    // The check: the epoch, marked, at which the part's core found the tile holding
    // every part of the slot's overlay. While the epoch is that, load() is that slot.
    constexpr uint32_t kOverlayCheckedMark = 1u << 31;
    // The reader is the tile's loader; the writer and the compute cores (followers 0
    // to 3, parts 1 to 4) meet it parked in their load().
    constexpr uint32_t kOverlayFollowers = 4;
    // The clock line (32 bytes): the uses of overlays, then the epoch (24 bits), which
    // the reader steps after it changes any record's held word.
    constexpr uint32_t kOverlayClock = (
        kOverlayRecords + 16 * kOverlayParts * kOverlaySlotsFixed
    );
    constexpr uint32_t kOverlayEpoch = kOverlayClock + 4;
    constexpr uint32_t kOverlayEpochMask = 0xFFFFFFu;
    // The reader's meet count with each follower (a word each).
    constexpr uint32_t kOverlayMet = kOverlayClock + 32;
    // The reader's go to each follower (8 bytes each): token, slot.
    constexpr uint32_t kOverlayGo = kOverlayMet + 16;
    // Each follower's park line to the reader (8 bytes each): token, its count.
    constexpr uint32_t kOverlayPark = kOverlayGo + 8 * kOverlayFollowers;
    // The handlers added to the live run (run.add), a bit a number in two words: the
    // body of handler n is overlay n (enter()). Its service command sets the bit.
    constexpr uint32_t kOverlayHandlers = kOverlayPark + 8 * kOverlayFollowers;
    // The dynamic handlers added to the live run, a bit a number: a message of handler
    // n runs overlay n's compute part (message_number(), km_messages.hpp).
    constexpr uint32_t kOverlayMessages = kOverlayHandlers + 8;
    // The service overlay's own slot (host): its bytes, 0 for none (it shares the
    // reader's slots). Its record is one more than the reader's, kOverlayOwnSlot.
    constexpr uint32_t kOverlayServiceSlot = kOverlayMessages + 4;
    constexpr uint32_t kOverlayOwnSlot = kOverlaySlotsMax + 1;
    // The clock at the last use of each overlay (a word per number): the slot a load
    // replaces holds the least recently used one.
    constexpr uint32_t kOverlayUsed = 512;
    // The table: (DRAM address in every bank, bytes) of part p of overlay n at
    // kOverlayTable + 8 * (kOverlayParts * n + p); bytes 0: no such part.
    constexpr uint32_t kOverlayTable = kOverlayUsed + 4 * kOverlaysMax;
    // A load of a program (KM_HANDLER_LOAD), after the table: the switch line (the
    // program field held, its buffers, the reader's go, each follower's arrival).
    constexpr uint32_t kOverlaySwitch = (
        kOverlayTable + 8 * kOverlayParts * kOverlaysMax
    );
    constexpr uint32_t kSwitchHolding = 0;  // the program field held (host, reader)
    constexpr uint32_t kSwitchMask = 1;     // its buffer indices, a bit each
    constexpr uint32_t kSwitchGo = 2;       // the load's sequence: buffers set up
    constexpr uint32_t kSwitchArrived = 4;  // per follower: the load's sequence
    constexpr uint32_t kSwitchNone = 0xFFFFFFFFu;
    // The service overlay's own slot's record, in the switch line after its words.
    constexpr uint32_t kOverlayServiceRecord = kOverlaySwitch + 32;
    // The program's record, read by the reader from DRAM: its buffer mask and field,
    // its table entries, then (address, bytes, pages, page bytes) per buffer index.
    constexpr uint32_t kOverlayRecord = kOverlaySwitch + 64;
    constexpr uint32_t kRecordMask = 0;
    constexpr uint32_t kRecordField = 1;
    constexpr uint32_t kRecordTable = 2;
    // An added class's per-core args (komorebi.program.addon): its row table's address
    // (0: none; core c's row is page c), a row's bytes, and the row's L1 place.
    constexpr uint32_t kRecordArgs = 12;
    constexpr uint32_t kRecordArgsBytes = 13;
    constexpr uint32_t kRecordArgsL1 = 14;
    constexpr uint32_t kRecordBuffers = 16;
    constexpr uint32_t kRecordBytes = 4 * (kRecordBuffers + 4 * 32);
    constexpr uint32_t kOverlayAreaBytes = kOverlayRecord + kRecordBytes;
    // The records of slots 4 and up (a run with more than 4 slots of a part), after
    // the area in the run's control blocks: 16 * (more * part + slot - 4) from here.
    constexpr uint32_t kOverlayMoreRecords = kOverlayAreaBytes;
    constexpr uint32_t kOverlaySlotsMore = kOverlaySlotsMax - kOverlaySlotsFixed;
    static_assert(
        kOverlaySlotsMax >= kOverlaySlotsFixed, "KM_OVERLAY_SLOTS is 4 or more"
    );
    static_assert(
        kOverlayEpoch < kOverlayMet && kOverlayServiceSlot + 4 <= kOverlayUsed &&
            kOverlayTable % 64 == 0 && KM_CONTROL_BYTES % 64 == 0 &&
            kOverlayRecord % 64 == 0 && kRecordBytes % 64 == 0 &&
            4 * (kSwitchArrived + kOverlayFollowers) <= 32 &&
            kOverlayServiceRecord + 16 <= kOverlayRecord,
        "the overlay area's lines do not overlap and end before its table"
    );

    inline volatile tt_l1_ptr uint32_t* overlay_area(uint32_t control) {
        return l1(control + kOverlayArea);
    }

    inline volatile tt_l1_ptr uint32_t* overlay_record(
        uint32_t control, uint32_t part, uint32_t slot
    ) {
        if constexpr (kOverlaySlotsMore != 0) {
            if (slot >= kOverlaySlotsFixed) {
                return (
                    l1(
                        control + kOverlayArea + kOverlayMoreRecords +
                        16 * (kOverlaySlotsMore * part + slot - kOverlaySlotsFixed)
                    )
                );
            }
        }
        return (
            l1(
                control + kOverlayArea + kOverlayRecords +
                16 * (kOverlaySlotsFixed * part + slot)
            )
        );
    }

    inline volatile tt_l1_ptr uint32_t* overlay_own_record(uint32_t control) {
        return l1(control + kOverlayArea + kOverlayServiceRecord);
    }

    // The record of the reader's slot `slot`, kOverlayOwnSlot the service's own.
    inline volatile tt_l1_ptr uint32_t* overlay_slot_record(
        uint32_t control, uint32_t part, uint32_t slot
    ) {
        return (
            slot == kOverlayOwnSlot
                ? overlay_own_record(control)
                : overlay_record(control, part, slot)
        );
    }

    // The slot's address (the host's, in its record; the own slot's in its own).
    inline uint32_t overlay_slot_address(
        uint32_t control, uint32_t part, uint32_t slot
    ) {
        return overlay_slot_record(control, part, slot)[kOverlayAddress];
    }

    inline volatile tt_l1_ptr uint32_t* overlay_epoch(uint32_t control) {
        return l1(control + kOverlayArea + kOverlayEpoch);
    }

    // The record of the slot of `part` that holds `number` with this core's check at
    // the current epoch (no record changed since), or none: the fast path of load().
    inline volatile tt_l1_ptr uint32_t* overlay_checked_record(
        uint32_t control, uint32_t part, uint32_t number
    ) {
        const uint32_t checked = *overlay_epoch(control) | kOverlayCheckedMark;
        const uint32_t slots = overlay_area(control)[kOverlaySlots + part];
        for (uint32_t slot = 0; slot < slots && slot < kOverlaySlotsMax; slot++) {
            volatile tt_l1_ptr uint32_t* record = overlay_record(control, part, slot);
            if (record[kOverlayHeld] == number && record[kOverlayChecked] == checked) {
                return record;
            }
        }
        return nullptr;
    }

    // The service overlay's own slot's record when it holds `number` with this core's
    // check at the current epoch, or none (overlay_checked_record of that slot).
    inline volatile tt_l1_ptr uint32_t* overlay_own_checked(
        uint32_t control, uint32_t number
    ) {
        const uint32_t checked = *overlay_epoch(control) | kOverlayCheckedMark;
        volatile tt_l1_ptr uint32_t* record = overlay_own_record(control);
        return (
            record[kOverlayHeld] == number && record[kOverlayChecked] == checked
                ? record
                : nullptr
        );
    }

    // This core's check of its slot `slot` of `part` at `epoch` (read before the
    // records that showed every part held, or given by the reader's go).
    inline void overlay_check(
        uint32_t control, uint32_t part, uint32_t slot, uint32_t epoch
    ) {
        overlay_slot_record(control, part, slot)[kOverlayChecked] = epoch |
            kOverlayCheckedMark;
    }

}  // namespace km
