// Overlays: bodies built on the host, kept in DRAM, loaded into this core's L1 while
// the run is live (km::Overlay). On the core headers alone; kernels: km_overlay.hpp.
#pragma once

#include <stdint.h>

#include "internal/circular_buffer_init.h"
#include "../core/km_core.hpp"
#include "../core/km_command.hpp"
#include "km_overlay_records.hpp"

#if defined(TRISC_MATH)
#include "cmath_common.h"
#endif

// A program's own overlays in a run of several programs: the run's number of each is
// the program's own plus its shift (the host's, a define of its parts; else 0).
#if !defined(KM_OVERLAY_SHIFT)
#define KM_OVERLAY_SHIFT 0
#endif

namespace km {

    // The runtime's service overlay (PUSH, PULL, COPY: a reader part), in every run.
    constexpr uint32_t kOverlayService = kOverlaysMax - 1;

    // The overlay that holds program `program` of the run (a command's program field):
    // numbers from 62 down, below them the programs' own overlays.
    inline uint32_t program_overlay(uint32_t program) {
        return kOverlayService - 1 - program;
    }

    // An overlay's entry, at its slot's first byte: the kernel's object, the argument.
    using OverlayEntry = uint32_t (*)(void*, const void*);

#if defined(KM_OVERLAY_OWN)
    // Whether the service overlay has its own slot and `number`'s `part` goes there.
    inline bool overlay_own(uint32_t control, uint32_t part, uint32_t number) {
        return (
            number == kOverlayService && part == kOverlayPartReader &&
            overlay_area(control)[kOverlayServiceSlot / 4] != 0
        );
    }
#else
    // A program with no slots of its own: the service shares the reader's slots, and
    // the own slot's code is left out (session.own_slot_path).
    inline bool overlay_own(uint32_t, uint32_t, uint32_t) {
        return false;
    }
#endif

    // The table entry of part `part` of overlay `number`: {address, bytes}.
    inline volatile tt_l1_ptr uint32_t* overlay_entry(
        uint32_t control, uint32_t number, uint32_t part
    ) {
        return (
            l1(
                control + kOverlayArea + kOverlayTable + 8 * (kOverlayParts * number + part)
            )
        );
    }

    // Whether the command's handler was added to the live run: its body an overlay.
    // Only a load whose program names slot bytes (KM_ADDED_HANDLERS) has the path.
    inline bool entered(uint32_t control, const km_command_t& command) {
#if !defined(KM_ADDED_HANDLERS)
        (void)control;
        (void)command;
        return false;
#else
        const uint32_t number = handler(command);
        return (
            number < kOverlaysMax &&
            (
                (
                    overlay_area(control)[kOverlayHandlers / 4 + (number >> 5)] >>
                    (number & 31u)
                ) &
                1u
            ) != 0
        );
#endif
    }

    // Whether `number` is in the table and has a part `part`.
    inline bool overlay_has_part(uint32_t control, uint32_t number, uint32_t part) {
        return (
            number != 0 && number < overlay_area(control)[kOverlayCount] &&
            overlay_entry(control, number, part)[1] != 0
        );
    }

    // The slot of `part` holding `number`, or kOverlaySlotsMax.
    inline uint32_t overlay_held_slot(
        uint32_t control, uint32_t part, uint32_t number
    ) {
        const uint32_t slots = overlay_area(control)[kOverlaySlots + part];
        for (uint32_t slot = 0; slot < slots && slot < kOverlaySlotsMax; slot++) {
            if (overlay_record(control, part, slot)[kOverlayHeld] == number) {
                return slot;
            }
        }
        return kOverlaySlotsMax;
    }

    // The slot a load of `number`'s `part` takes: an empty one, else the least recently
    // used one not running; kOverlaySlotsMax: none.
    inline uint32_t overlay_free_slot(
        uint32_t control, uint32_t part, uint32_t number
    ) {
        const uint32_t slots = overlay_area(control)[kOverlaySlots + part];
        uint32_t found = kOverlaySlotsMax, oldest = 0;
        for (uint32_t slot = 0; slot < slots && slot < kOverlaySlotsMax; slot++) {
            volatile tt_l1_ptr uint32_t* record = overlay_record(control, part, slot);
            if (record[kOverlayCalling] != 0) {
                continue;
            }
            const uint32_t held = record[kOverlayHeld];
            if (held == 0) {
                return slot;
            }
            const uint32_t age = l1(control + kOverlayArea + kOverlayUsed)[held];
            if (found == kOverlaySlotsMax || age < oldest) {
                found = slot;
                oldest = age;
            }
        }
        return found;
    }

    // The slot's bytes (the own slot's: the service's).
    inline uint32_t overlay_slot_bytes(uint32_t control, uint32_t part, uint32_t slot) {
        volatile tt_l1_ptr uint32_t* area = overlay_area(control);
        return (
            slot == kOverlayOwnSlot
                ? area[kOverlayServiceSlot / 4]
                : area[kOverlaySlotBytes + part]
        );
    }

    // Marks the last use of overlay `number` with the clock (the reader only).
    inline void overlay_touch(uint32_t control, uint32_t number) {
        volatile tt_l1_ptr uint32_t* clock = l1(control + kOverlayArea + kOverlayClock);
        const uint32_t now = clock[0] + 1;
        clock[0] = now;
        l1(control + kOverlayArea + kOverlayUsed)[number] = now;
    }

    // The instruction cache bit of the core that runs `part` (the firmware's mask).
    inline uint32_t overlay_cache_bit(uint32_t part) {
        return (
            part == kOverlayPartReader
                ? 0x1u
                : part == kOverlayPartWriter
                    ? 0x10u
                    : 1u << (part - 1)
        );
    }

    // The followers (bit f: part kOverlayPartWriter + f) that overlay `number` has.
    inline uint32_t overlay_members(uint32_t control, uint32_t number) {
        uint32_t members = 0;
        for (uint32_t follower = 0; follower < kOverlayFollowers; follower++) {
            if (overlay_has_part(control, number, kOverlayPartWriter + follower)) {
                members |= 1u << follower;
            }
        }
        return members;
    }

    // Whether the tile holds every follower part of overlay `number`: then neither the
    // reader nor a follower meets (both decide it from the same records).
    inline bool overlay_members_held(uint32_t control, uint32_t number) {
        const uint32_t members = overlay_members(control, number);
        for (uint32_t follower = 0; follower < kOverlayFollowers; follower++) {
            const uint32_t part = kOverlayPartWriter + follower;
            if (
                ((members >> follower) & 1u) != 0 &&
                overlay_held_slot(control, part, number) == kOverlaySlotsMax
            ) {
                return false;
            }
        }
        return true;
    }

    // A follower's load() of `part` after its check missed: parks until the reader's go
    // unless the tile holds every follower part. The slot; kOverlaySlotsMax: gave up.
    inline uint32_t overlay_follow(
        uint32_t control,
        uint32_t number,
        uint32_t part,
        uint32_t limit,
        uint32_t& code,
        uint32_t& where
    ) {
        if (!overlay_has_part(control, number, part)) {
            code = kGaveOverlay;
            where = kWhereOverlayFind;
            return kOverlaySlotsMax;
        }
        const uint32_t epoch = *overlay_epoch(control);  // before the records
        if (overlay_members_held(control, number)) {
            const uint32_t slot = overlay_held_slot(control, part, number);
            overlay_check(control, part, slot, epoch);
            return slot;
        }
        const uint32_t follower = part - kOverlayPartWriter;
        volatile tt_l1_ptr uint32_t* park = (
            l1(control + kOverlayArea + kOverlayPark + 8 * follower)
        );
        volatile tt_l1_ptr uint32_t* go = (
            l1(control + kOverlayArea + kOverlayGo + 8 * follower)
        );
        const uint32_t count = park[1] + 1;
        const uint32_t token = count << 8 | number;
        park[1] = count;
        park[0] = token;
        code = poll_stop(
            l1(control + KM_CONTROL_STOP),
            [&] {
                return go[0] == token;
            },
            limit,
            0
        );
        if (code != kGaveNone) {
            where = kWhereOverlayMeet;
            return kOverlaySlotsMax;
        }
        const uint32_t slot = go[1] & 0xFFu;
        overlay_check(control, part, slot, go[1] >> 8);
        return slot;
    }

    inline volatile tt_l1_ptr uint32_t* program_line(uint32_t control) {
        return l1(control + kOverlayArea + kOverlaySwitch);
    }

    // The new program's buffers as the firmware's set-up reads them (the record's).
    inline uint32_t tt_l1_ptr* program_buffers(uint32_t control) {
        return (
            reinterpret_cast<uint32_t tt_l1_ptr*>(
                control + kOverlayArea + kOverlayRecord + 4 * kRecordBuffers
            )
        );
    }

    // A follower's part of a load (`follower` 0 the writer, 1 to 3 compute): marks its
    // arrival, its earlier commands done, and parks until the reader's go.
    inline bool program_follow(
        uint32_t control, uint32_t follower, uint32_t sequence, uint32_t limit
    ) {
        volatile tt_l1_ptr uint32_t* line = program_line(control);
        line[kSwitchArrived + follower] = sequence;
        const uint32_t code = poll_stop(
            l1(control + KM_CONTROL_STOP),
            [&] {
                return line[kSwitchGo] == sequence;
            },
            limit,
            0
        );
#if defined(KM_ADDED_ARGS)
        invalidate_l1_cache();  // an added class's row, written by the reader's read
#endif
        if (code != kGaveNone) {
            const uint32_t word = follower == 0 ? kStWriter : kStCompute;
            command_gave_up(control, word, code, kWhereProgramSwitch, 0, sequence);
            return false;
        }
        return true;
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
namespace km {

#if defined(COMPILE_FOR_NCRISC)
    const uint32_t kOverlayPartHere = kOverlayPartWriter;
    const uint32_t kOverlayStateWord = kStWriter;
#else
    const uint32_t kOverlayPartHere = kOverlayPartReader;
    const uint32_t kOverlayStateWord = kStReader;

    // Invalidates the instruction caches of the tile's cores in `mask` (the firmware's
    // register): only the reader writes it, for every part it has loaded.
    inline void overlay_invalidate(uint32_t mask) {
        volatile tt_reg_ptr uint32_t*
            config = (reinterpret_cast<volatile tt_reg_ptr uint32_t*>(TENSIX_CFG_BASE));
        config[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32] = mask;
        (void)config[RISCV_IC_INVALIDATE_InvalidateAll_ADDR32];
    }

    // Whether each buffer in `mask` is where `buffers` puts it: the start's load of
    // the program the cores already hold checks the host's layout against the live one.
    inline bool program_layout_live(const uint32_t tt_l1_ptr* buffers, uint32_t mask) {
        for (uint32_t cb = 0; cb < 32; cb++) {
            const LocalCBInterface& live = get_local_cb_interface(cb);
            const uint32_t* words = buffers + 4 * cb;
            if (
                ((mask >> cb) & 1u) != 0 &&
                (live.fifo_size != words[1] || live.fifo_limit != words[0] + words[1])
            ) {
                return false;
            }
        }
        return true;
    }

    // A load of a program (KM_HANDLER_LOAD) on the reader, all five cores parked: every
    // buffer of either program empty, counters zeroed, its own set-up, the table, go.
    template <typename Kernel>
    bool program_switch(Kernel& reader, const km_command_t& command) {
        const uint32_t control = reader.args().control;
        const uint32_t limit = reader.args().wait_limit;
        volatile tt_l1_ptr uint32_t* line = program_line(control);
        volatile tt_l1_ptr uint32_t* record = (
            l1(control + kOverlayArea + kOverlayRecord)
        );
        const uint32_t field = program(command);
        // the followers first: the last load's record is theirs until they arrive
        uint32_t code = reader.poll_at(
            kWhereProgramSwitch,
            [&] {
                for (uint32_t f = 0; f < kOverlayFollowers; f++) {
                    if (line[kSwitchArrived + f] != command.sequence) {
                        return false;
                    }
                }
                return true;
            },
            limit
        );
        noc_async_read(
            get_noc_addr_from_bank_id<true>(
                reader.args().me % kOverlayBanks, command.a0
            ),
            control + kOverlayArea + kOverlayRecord,
            kRecordBytes
        );
        if (!reader.flushed(kFlushReads, kStReader) && code == kGaveNone) {
            code = kGaveStop;  // recorded by flushed: the tile stops
        }
#if defined(KM_ADDED_ARGS)  // a class named ahead has per-core args (added_program)
        // An added class's row, landed before the go below: the four other CPUs park
        // until the go (program_follow), so none runs the class's next command before.
        if (code == kGaveNone && record[kRecordArgs] != 0) {
            const uint32_t me = reader.args().me, bytes = record[kRecordArgsBytes];
            noc_async_read(
                get_noc_addr_from_bank_id<true>(
                    me % kOverlayBanks, record[kRecordArgs] + me / kOverlayBanks * bytes
                ),
                record[kRecordArgsL1],
                bytes
            );
            if (!reader.flushed(kFlushReads, kStReader)) {
                code = kGaveStop;
            }
        }
#endif
        const uint32_t mask = record[kRecordMask], both = mask | line[kSwitchMask];
        if (code == kGaveNone && record[kRecordField] != field) {
            code = kGaveOverlay;
        }
        if (
            code == kGaveNone && line[kSwitchHolding] == field &&
            !program_layout_live(program_buffers(control), mask)
        ) {
            code = kGaveOverlay;
        }
        if (code == kGaveNone) {
            code = reader.poll_at(
                kWhereProgramSwitch,
                [&] {
                    for (uint32_t cb = 0; cb < 32; cb++) {
                        if (
                            ((both >> cb) & 1u) != 0 &&
                            (
                                (
                                    *get_cb_tiles_received_ptr(cb) ^
                                    *get_cb_tiles_acked_ptr(cb)
                                ) &
                                0xFFFFu
                            ) != 0
                        ) {
                            return false;
                        }
                    }
                    return true;
                },
                limit
            );
        }
        if (code != kGaveNone) {
            reader.gave_up(
                kStReader, code, kWhereProgramSwitch, slot(command), command.sequence
            );
            return false;
        }
        for (uint32_t cb = 0; cb < 32; cb++) {
            if (((both >> cb) & 1u) != 0) {
                *get_cb_tiles_received_ptr(cb) = 0;
                *get_cb_tiles_acked_ptr(cb) = 0;
            }
        }
        setup_local_cb_read_write_interfaces<true, true, false>(
            program_buffers(control), 0, mask
        );
        const uint32_t held = line[kSwitchHolding];
        for (uint32_t part = 0; part < kOverlayParts; part++) {
            if (held != kSwitchNone) {
                volatile tt_l1_ptr uint32_t* old = (
                    overlay_entry(control, program_overlay(held), part)
                );
                old[0] = old[1] = 0;
            }
            volatile tt_l1_ptr uint32_t* entry = (
                overlay_entry(control, program_overlay(field), part)
            );
            entry[0] = record[kRecordTable + 2 * part];
            entry[1] = record[kRecordTable + 2 * part + 1];
        }
        line[kSwitchHolding] = field;
        line[kSwitchMask] = mask;
        volatile tt_l1_ptr uint32_t* epoch = overlay_epoch(control);
        *epoch = (*epoch + 1) &
            kOverlayEpochMask;  // the table changed: every check again
        line[kSwitchGo] = command.sequence;
        return true;
    }
#endif

    // An overlay on a data-movement core: `Kernel` is the km::Reader or km::Writer of
    // this kernel. The reader is the tile's loader (see load()).
    template <typename Kernel>
    class Overlay {
    public:
        Overlay(Kernel& kernel, uint32_t number) :
            kernel_(kernel), number_(number + KM_OVERLAY_SHIFT) {
        }

        // Whether a slot of this core holds this kernel's part of the overlay.
        bool held() const {
            const uint32_t control = kernel_.args().control;
            invalidate_l1_cache();
            return (
                overlay_has_part(control, number_, kOverlayPartHere) &&
                (
                    overlay_held_slot(control, kOverlayPartHere, number_) <
                        kOverlaySlotsMax ||
                    (
                        overlay_own(control, kOverlayPartHere, number_) &&
                        overlay_own_record(control)[kOverlayHeld] == number_
                    )
                )
            );
        }

        // Reader: loads every part the tile does not hold (load_tile). Writer: meets
        // the reader's load() of it (overlay_follow). False: gave up (recorded).
        bool load() {
            const uint32_t control = kernel_.args().control;
#if defined(COMPILE_FOR_NCRISC)
            invalidate_l1_cache();
            if (place_checked(control)) {
                return true;
            }
            uint32_t code = kGaveNone, where = 0;
            const uint32_t slot = (
                overlay_follow(
                    control,
                    number_,
                    kOverlayPartHere,
                    kernel_.args().wait_limit,
                    code,
                    where
                )
            );
            if (slot == kOverlaySlotsMax) {
                return give_up(code, where);
            }
            return place(control, slot);
#else
            if (place_checked(control)) {
                overlay_touch(control, number_);
                return true;
            }
            return load_tile(control);
#endif
        }

        // Runs the overlay's entry with `argument` (handed by its address) and returns
        // its result; after a load() that returned true.
        template <typename Argument>
        uint32_t call(const Argument& argument) {
            if (slot_ == 0) {
                give_up(kGaveOverlay, kWhereOverlayFind);
                return 0;
            }
            record_[kOverlayCalling] = record_[kOverlayCalling] + 1;
            const uint32_t result = (
                reinterpret_cast<OverlayEntry>(slot_)(&kernel_, &argument)
            );
            record_[kOverlayCalling] = record_[kOverlayCalling] - 1;
            return result;
        }

    private:
        bool give_up(uint32_t code, uint32_t where) const {
            kernel_.gave_up(kOverlayStateWord, code, where, number_, 0);
            return false;
        }

        bool place(uint32_t control, uint32_t slot) {
            record_ = (
                kOverlayPartHere == kOverlayPartReader
                    ? overlay_slot_record(control, kOverlayPartHere, slot)
                    : overlay_record(control, kOverlayPartHere, slot)
            );
            slot_ = record_[kOverlayAddress];
            return true;
        }

        // The slot this core checked for the overlay at the current epoch, if any.
        bool place_checked(uint32_t control) {
            volatile tt_l1_ptr uint32_t* record = (
                overlay_checked_record(control, kOverlayPartHere, number_)
            );
            if (record == nullptr && overlay_own(control, kOverlayPartHere, number_)) {
                record = overlay_own_checked(control, number_);  // after a miss only
            }
            if (record == nullptr) {
                return false;
            }
            slot_ = record[kOverlayAddress];
            record_ = record;
            return true;
        }

#if !defined(COMPILE_FOR_NCRISC)
        // Starts the read of `part` into `slot` (the record says none until it lands).
        // False: the part does not fit the slot (recorded).
        bool start_read(uint32_t control, uint32_t part, uint32_t slot) const {
            volatile tt_l1_ptr uint32_t* entry = overlay_entry(control, number_, part);
            const uint32_t bytes = entry[1];
            if (bytes > overlay_slot_bytes(control, part, slot)) {
                return give_up(kGaveOverlay, kWhereOverlayFind);
            }
            overlay_slot_record(control, part, slot)[kOverlayHeld] = 0;
            volatile tt_l1_ptr uint32_t* epoch = overlay_epoch(control);
            *epoch = (*epoch + 1) & kOverlayEpochMask;  // after the record changed
            noc_async_read(
                get_noc_addr_from_bank_id<true>(
                    kernel_.args().me % kOverlayBanks, entry[0]
                ),
                overlay_slot_record(control, part, slot)[kOverlayAddress],
                bytes
            );
            return true;
        }

        // Starts the read of `part` unless a slot holds it: slot | loaded << 8, or
        // kOverlaySlotsMax (gave up).
        uint32_t fetch(uint32_t control, uint32_t part) const {
            const bool own = overlay_own(control, part, number_);  // its own slot
            volatile tt_l1_ptr uint32_t* mine = overlay_own_record(control);
            const uint32_t held = (
                own
                    ? (
                        mine[kOverlayHeld] == number_ ? kOverlayOwnSlot : kOverlaySlotsMax
                    )
                    : overlay_held_slot(control, part, number_)
            );
            if (held != kOverlaySlotsMax) {
                return held;
            }
            const uint32_t slot = (
                own
                    ? (mine[kOverlayCalling] == 0 ? kOverlayOwnSlot : kOverlaySlotsMax)
                    : overlay_free_slot(control, part, number_)
            );
            if (slot == kOverlaySlotsMax) {
#if KM_GUARD  // every slot runs a part: the word says so, with the kernel's slots
                kernel_.gave_up(
                    kOverlayStateWord,
                    kGaveOverlay,
                    own ? kWhereOverlayFind : kWhereOverlaySlots,
                    number_ |
                        (own ? 0 : overlay_area(control)[kOverlaySlots + part] << 8),
                    0
                );
#else
                give_up(kGaveOverlay, kWhereOverlayFind);
#endif
                return kOverlaySlotsMax;
            }
            return start_read(control, part, slot) ? slot | 1u << 8 : kOverlaySlotsMax;
        }

        // Timed wait until every read this core started has landed.
        bool landed() const {
            const uint32_t code = kernel_.poll_at(
                kWhereOverlayLoad,
                [] {
                    return ncrisc_noc_reads_flushed(noc_index);
                },
                kernel_.args().wait_limit
            );
            if (code != kGaveNone) {
                return give_up(code, kWhereOverlayLoad);
            }
            return true;
        }

        // Timed wait until every follower in `members` is parked in its next meet
        // with the reader, for this overlay.
        bool parked(uint32_t control, uint32_t members) const {
            volatile tt_l1_ptr uint32_t* park = (
                l1(control + kOverlayArea + kOverlayPark)
            );
            volatile tt_l1_ptr uint32_t* met = l1(control + kOverlayArea + kOverlayMet);
            bool mismatch = false;
            const uint32_t code = kernel_.poll_at(
                kWhereOverlayMeet,
                [&] {
                    mismatch = false;
                    for (
                        uint32_t follower = 0; follower < kOverlayFollowers; follower++
                    ) {
                        if (((members >> follower) & 1u) == 0) {
                            continue;
                        }
                        const uint32_t seen = park[2 * follower];
                        if ((seen >> 8) != ((met[follower] + 1) & 0xFFFFFFu)) {
                            return false;
                        }
                        mismatch = mismatch || (seen & 0xFFu) != number_;
                    }
                    return true;
                },
                kernel_.args().wait_limit
            );
            if (code != kGaveNone || mismatch) {
                return give_up(mismatch ? kGaveOverlay : code, kWhereOverlayMeet);
            }
            return true;
        }

#if KM_WRITER_COLD
    public:
        // At a load's start, before the writer runs L1 code (no meet): its part read
        // into its first slot, kept (marked running), its cache invalidated.
        bool load_kept(uint32_t control) {
            const uint32_t slot = fetch(control, kOverlayPartWriter);
            if (slot == kOverlaySlotsMax) {
                return false;
            }
            if ((slot & 0xFFu) != 0) {  // the first slot was not free at the start
                return give_up(kGaveOverlay, kWhereOverlaySlots);
            }
            if (!landed()) {
                return false;
            }
            overlay_invalidate(overlay_cache_bit(kOverlayPartWriter));
            volatile tt_l1_ptr uint32_t* record = (
                overlay_record(control, kOverlayPartWriter, 0)
            );
            record[kOverlayCalling] = 1;
            record[kOverlayHeld] = number_;
            return true;
        }

    private:
#endif

        // Reads the parts the tile does not hold (the followers' once they are parked),
        // invalidates their cores' caches in one write, then gives each follower go.
#if defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_BRISC)
        __attribute__((noinline))  // one window a kernel, however many loads it has
#endif
        bool load_tile(uint32_t control) {
#if defined(ARCH_BLACKHOLE) && defined(COMPILE_FOR_BRISC)
            // the branch predictor off (cfg0.DisBp) from before the first fetch to 33
            // instructions after the load: with it on, a core can stall after a load
            asm volatile("csrrsi zero, 0x7c0, 2" ::: "memory");
            const bool loaded = load_tile_parts(control);
            asm volatile(
                "li t1, 16\n"
                "1:\taddi t1, t1, -1\n\t"
                "bnez t1, 1b\n\t"
                "csrrci zero, 0x7c0, 2" ::
                    : "t1", "memory"
            );
            return loaded;
#else
            return load_tile_parts(control);
#endif
        }

        // load_tile's work, inside its window on Blackhole.
        bool load_tile_parts(uint32_t control) {
            const bool mine = overlay_has_part(control, number_, kOverlayPartReader);
            const uint32_t members = overlay_members(control, number_);
            if (!mine && members == 0) {
                return give_up(kGaveOverlay, kWhereOverlayFind);
            }
            const bool meet = members != 0 && !overlay_members_held(control, number_);
            if (meet && !parked(control, members)) {
                return false;
            }
            uint32_t slots[kOverlayParts] = {};
            uint32_t caches = 0;
            for (uint32_t part = 0; part < kOverlayParts; part++) {
                const bool wanted = (
                    part == kOverlayPartReader
                        ? mine
                        : ((members >> (part - kOverlayPartWriter)) & 1u) != 0
                );
                if (!wanted) {
                    continue;
                }
                slots[part] = (
                    part == kOverlayPartReader || meet
                        ? fetch(control, part)
                        : overlay_held_slot(control, part, number_)
                );
                if (slots[part] == kOverlaySlotsMax) {
                    return false;
                }
                if ((slots[part] >> 8) != 0) {
                    caches |= overlay_cache_bit(part);
                }
            }
            if (caches != 0) {
                if (!landed()) {
                    return false;
                }
                overlay_invalidate(caches);
            }
            for (uint32_t part = 0; part < kOverlayParts; part++) {
                const bool wanted = (
                    part == kOverlayPartReader
                        ? mine
                        : ((members >> (part - kOverlayPartWriter)) & 1u) != 0
                );
                if (wanted) {
                    overlay_slot_record(
                        control, part, slots[part] & 0xFFu
                    )[kOverlayHeld] = (number_);
                }
            }
            overlay_touch(control, number_);
            const uint32_t epoch = *overlay_epoch(control);
            if (meet) {
                volatile tt_l1_ptr uint32_t* go = (
                    l1(control + kOverlayArea + kOverlayGo)
                );
                volatile tt_l1_ptr uint32_t* met = (
                    l1(control + kOverlayArea + kOverlayMet)
                );
                for (uint32_t follower = 0; follower < kOverlayFollowers; follower++) {
                    if (((members >> follower) & 1u) != 0) {
                        const uint32_t count = met[follower] + 1;
                        met[follower] = count;
                        go[2 * follower + 1] = (
                            (slots[kOverlayPartWriter + follower] & 0xFFu) | epoch << 8
                        );
                        go[2 * follower] = count << 8 | number_;
                    }
                }
            }
            if (!mine) {
                return true;
            }
            const uint32_t slot = slots[kOverlayPartReader] & 0xFFu;
            overlay_check(control, kOverlayPartReader, slot, epoch);
            return place(control, slot);
        }
#endif

        Kernel& kernel_;
        uint32_t number_;
        uint32_t slot_ = 0;
        volatile tt_l1_ptr uint32_t* record_ = nullptr;
    };

#if !defined(COMPILE_FOR_NCRISC)
    // A PUSH, PULL or COPY: loads the service overlay unless a slot holds it, then
    // calls it with the command. A failed load gives up (recorded).
    template <typename Kernel>
    void serve(Kernel& reader, const km_command_t& command) {
        Overlay<Kernel> service(reader, kOverlayService);
        if (service.load()) {
            service.call(command);
        }
    }
#endif

#if (KM_WRITER_COLD >= 2 || (KM_WRITER_COLD && KM_GUARD)) && \
    defined(COMPILE_FOR_NCRISC) && !defined(KM_OVERLAY_PART)
    // The writer's start: waits for the reader's load of its part (load_kept, its first
    // writer slot) and keeps the entry; its after-the-work paths run there (cold_call).
    template <typename Kernel>
    bool cold_hold(Kernel& writer) {
        const uint32_t control = writer.args().control;
        volatile tt_l1_ptr uint32_t* record = (
            overlay_record(control, kOverlayPartWriter, 0)
        );
        const uint32_t number = kOverlayService + KM_OVERLAY_SHIFT;
        const uint32_t code = writer.poll_at(
            kWhereOverlayMeet,
            [&] {
                return record[kOverlayHeld] == number;
            },
            writer.args().wait_limit
        );
        if (code != kGaveNone) {
            writer.gave_up(kStWriter, code, kWhereOverlayMeet, number, 0);
            return false;
        }
        cold_entry = record[kOverlayAddress];
        return true;
    }
#elif KM_WRITER_COLD && !defined(COMPILE_FOR_NCRISC)
    // The reader's start: the service overlay's writer part loaded when it has one (the
    // host gives it one when the writer's code calls it), before any command.
    template <typename Kernel>
    bool cold_load(Kernel& reader) {
        const uint32_t control = reader.args().control;
        const uint32_t number = kOverlayService + KM_OVERLAY_SHIFT;
        if (!overlay_has_part(control, number, kOverlayPartWriter)) {
            return true;
        }
        Overlay<Kernel> service(reader, kOverlayService);
        return service.load_kept(control);
    }
#endif

    // An added handler (entered()): the reader loads its number's overlay, each
    // kernel with a part calls it with the command. False: gave up (recorded).
    template <typename Kernel>
    __attribute__((noinline)) bool enter(Kernel& kernel, const km_command_t& command) {
#if !defined(KM_ADDED_HANDLERS)
        (void)kernel;
        (void)command;
        return true;
#else
        const uint32_t number = handler(command);
        const bool mine = (
            overlay_has_part(kernel.args().control, number, kOverlayPartHere)
        );
#if defined(COMPILE_FOR_NCRISC)
        if (!mine) {
            return true;  // the reader loads the tile's parts
        }
#endif
        Overlay<Kernel> body(kernel, number);
        if (!body.load()) {
            return false;
        }
        if (mine) {
            body.call(command);
        }
        return true;
#endif
    }

}  // namespace km
#else
namespace km {

#if COMPILE_FOR_TRISC == 0
    const uint32_t kOverlayPartHere = kOverlayPartUnpack;
#elif COMPILE_FOR_TRISC == 1
    const uint32_t kOverlayPartHere = kOverlayPartUnpack + 1;  // math
#else
    const uint32_t kOverlayPartHere = kOverlayPartUnpack + 2;  // pack
#endif

    // An overlay on a compute core (`Kernel` is km::Compute): its load() parks until
    // the reader's load() of it has put the parts in place and invalidated the caches.
    template <typename Kernel>
    class Overlay {
    public:
        Overlay(Kernel& kernel, uint32_t number) :
            kernel_(kernel), number_(number + KM_OVERLAY_SHIFT) {
        }

        // Whether a slot of this compute core holds its part of the overlay.
        bool held() const {
            const uint32_t control = kernel_.control();
            invalidate_l1_cache();
            return (
                overlay_has_part(control, number_, kOverlayPartHere) &&
                overlay_held_slot(control, kOverlayPartHere, number_) < kOverlaySlotsMax
            );
        }

        // Unless the tile holds every follower part of the overlay: parks until the
        // reader's load() of it has put them in place. False: gave up (recorded).
        bool load() {
            const uint32_t control = kernel_.control();
            invalidate_l1_cache();
            volatile tt_l1_ptr uint32_t* checked = (
                overlay_checked_record(control, kOverlayPartHere, number_)
            );
            if (checked != nullptr) {
                slot_ = checked[kOverlayAddress];
                record_ = checked;
                return true;
            }
            uint32_t code = kGaveNone, where = 0;
            const uint32_t slot = (
                overlay_follow(
                    control, number_, kOverlayPartHere, kernel_.wait_limit(), code, where
                )
            );
            if (slot == kOverlaySlotsMax) {
                return give_up(code, where);
            }
            slot_ = overlay_slot_address(control, kOverlayPartHere, slot);
            record_ = overlay_record(control, kOverlayPartHere, slot);
            return true;
        }

        // Runs this compute core's part with `argument` and returns its result.
        template <typename Argument>
        uint32_t call(const Argument& argument) {
            if (slot_ == 0) {
                give_up(kGaveOverlay, kWhereOverlayFind);
                return 0;
            }
            record_[kOverlayCalling] = record_[kOverlayCalling] + 1;
            const uint32_t result = (
                reinterpret_cast<OverlayEntry>(slot_)(&kernel_, &argument)
            );
            record_[kOverlayCalling] = record_[kOverlayCalling] - 1;
            // the math library's cached flag value: the part may have changed it
            MATH((ckernel::math::_invalidate_src_zero_flag_state_()));
            return result;
        }

    private:
        bool give_up(uint32_t code, uint32_t where) const {
            command_gave_up(kernel_.control(), kStCompute, code, where, number_, 0);
            return false;
        }

        Kernel& kernel_;
        uint32_t number_;
        uint32_t slot_ = 0;
        volatile tt_l1_ptr uint32_t* record_ = nullptr;
    };

    // A handler added to the live run, on a compute core: its part, when it has one,
    // loaded (the reader's load() meets it) and called with the command.
    template <typename Kernel>
    __attribute__((noinline)) bool enter(Kernel& kernel, const km_command_t& command) {
#if !defined(KM_ADDED_HANDLERS)
        (void)kernel;
        (void)command;
        return true;
#else
        const uint32_t number = handler(command);
        if (!overlay_has_part(kernel.control(), number, kOverlayPartHere)) {
            return true;
        }
        Overlay<Kernel> body(kernel, number);
        if (!body.load()) {
            return false;
        }
        body.call(command);
        return true;
#endif
    }

}  // namespace km
#endif

#if defined(COMPILE_FOR_TRISC) && !defined(KM_OVERLAY_PART)
// The compute state the parts share with this kernel (the runtime's compute build
// defines it): kept and exported, so that a part links to these, not to copies.
extern uint32_t unp_cfg_context __attribute__((externally_visible));
extern uint32_t pack_sync_tile_dst_ptr __attribute__((externally_visible));
extern uint32_t math_sync_tile_dst_index __attribute__((externally_visible));
extern uint32_t gl_alu_format_spec_reg __attribute__((externally_visible));

namespace ckernel {
    extern volatile tt_reg_ptr uint* pc_buf_base __attribute__((externally_visible));
    extern volatile tt_reg_ptr uint* regfile __attribute__((externally_visible));
    extern volatile std::uint32_t tt_reg_ptr* mailbox_base[4]
        __attribute__((externally_visible));
}  // namespace ckernel

__attribute__((used)) volatile void* const km_overlay_exports[] = {
    &unp_cfg_context,
    &pack_sync_tile_dst_ptr,
    &math_sync_tile_dst_index,
    &gl_alu_format_spec_reg,
    &ckernel::pc_buf_base,
    &ckernel::regfile,
    &ckernel::mailbox_base,
};
#endif

#if defined(COMPILE_FOR_NCRISC) && !defined(KM_OVERLAY_PART)
// The NoC counters the runtime's writer build defines in this kernel: kept and
// exported, so that a writer part counts its reads and writes in these, not in copies.
extern uint32_t noc_reads_num_issued[NUM_NOCS] __attribute__((externally_visible));
extern uint32_t noc_nonposted_writes_num_issued[NUM_NOCS]
    __attribute__((externally_visible));
extern uint32_t noc_nonposted_writes_acked[NUM_NOCS]
    __attribute__((externally_visible));
extern uint32_t noc_nonposted_atomics_acked[NUM_NOCS]
    __attribute__((externally_visible));
extern uint32_t noc_posted_writes_num_issued[NUM_NOCS]
    __attribute__((externally_visible));

__attribute__((used)) volatile void* const km_overlay_writer_exports[] = {
    noc_reads_num_issued,
    noc_nonposted_writes_num_issued,
    noc_nonposted_writes_acked,
    noc_nonposted_atomics_acked,
    noc_posted_writes_num_issued,
};
#endif
