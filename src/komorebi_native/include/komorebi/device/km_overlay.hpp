// Overlays: kernel bodies built on the host, kept in DRAM and loaded into a slot of
// this core's L1 while the run is live. km::Overlay names one, loads it, calls it.
#pragma once

#include <stdint.h>

#include "km_worker.hpp"

#if defined(TRISC_MATH)
#include "cmath_common.h"
#endif

namespace km {

    // The overlay area: kOverlayAreaBytes after the control block, written by the
    // host at load (header, table), then by the cores (records, clock, lines).
    constexpr uint32_t kOverlayArea = KM_CONTROL_BYTES;
    constexpr uint32_t kOverlayParts = 5;  // reader, writer, unpack, math, pack
    constexpr uint32_t kOverlayPartReader = 0;
    constexpr uint32_t kOverlayPartWriter = 1;
    constexpr uint32_t kOverlayPartUnpack = 2;
    constexpr uint32_t kOverlaySlotsMax = 4;  // slots of one part
    constexpr uint32_t kOverlaysMax = 64;     // numbers 1 to 63; 0 names none
    constexpr uint32_t kOverlayBanks = 8;     // core i reads its copy in bank i % 8
    // Header words: table entries; first slot, slots and slot bytes of each part.
    constexpr uint32_t kOverlayCount = 0;
    constexpr uint32_t kOverlayBase = 1;
    constexpr uint32_t kOverlaySlots = 6;
    constexpr uint32_t kOverlaySlotBytes = 11;
    // A 16-byte record per (part, slot): the overlay it holds (0: none), the clock at
    // its last use, the calls running in it. The reader writes held and used.
    constexpr uint32_t kOverlayRecords = 64;
    constexpr uint32_t kOverlayHeld = 0;
    constexpr uint32_t kOverlayUsed = 1;
    constexpr uint32_t kOverlayCalling = 2;
    // The reader is the tile's loader; the writer and the compute cores (followers 0
    // to 3, parts 1 to 4) meet it parked in their load().
    constexpr uint32_t kOverlayFollowers = 4;
    // The clock line (32 bytes): word p counts the loads of part p.
    constexpr uint32_t kOverlayClock = (
        kOverlayRecords + 16 * kOverlayParts * kOverlaySlotsMax
    );
    // The reader's meeting count with each follower (a word each).
    constexpr uint32_t kOverlayMet = kOverlayClock + 32;
    // The reader's go to each follower (8 bytes each): token, slot.
    constexpr uint32_t kOverlayGo = kOverlayMet + 16;
    // Each follower's park line to the reader (8 bytes each): token, its count.
    constexpr uint32_t kOverlayPark = kOverlayGo + 8 * kOverlayFollowers;
    // The table: (DRAM address in every bank, bytes) of part p of overlay n at
    // kOverlayTable + 8 * (kOverlayParts * n + p); bytes 0: no such part.
    constexpr uint32_t kOverlayTable = 512;
    constexpr uint32_t kOverlayAreaBytes = (
        kOverlayTable + 8 * kOverlayParts * kOverlaysMax
    );
    static_assert(
        kOverlayClock + 4 * kOverlayParts <= kOverlayMet &&
            kOverlayPark + 8 * kOverlayFollowers <= kOverlayTable &&
            KM_CONTROL_BYTES % 64 == 0,
        "the overlay area's lines do not overlap and end before its table"
    );

    // An overlay's entry, at its slot's first byte: the kernel's object, the argument.
    using OverlayEntry = uint32_t (*)(void*, const void*);

    inline volatile tt_l1_ptr uint32_t* overlay_area(uint32_t control) {
        return l1(control + kOverlayArea);
    }

    inline volatile tt_l1_ptr uint32_t* overlay_record(
        uint32_t control, uint32_t part, uint32_t slot
    ) {
        return (
            l1(
                control + kOverlayArea + kOverlayRecords +
                16 * (kOverlaySlotsMax * part + slot)
            )
        );
    }

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

    // The slot a load of `part` replaces: an empty one, else the least recently used
    // one with no call running in it; kOverlaySlotsMax when every slot is running.
    inline uint32_t overlay_free_slot(uint32_t control, uint32_t part) {
        const uint32_t slots = overlay_area(control)[kOverlaySlots + part];
        uint32_t found = kOverlaySlotsMax, oldest = 0;
        for (uint32_t slot = 0; slot < slots && slot < kOverlaySlotsMax; slot++) {
            volatile tt_l1_ptr uint32_t* record = overlay_record(control, part, slot);
            if (record[kOverlayCalling] != 0) {
                continue;
            }
            if (record[kOverlayHeld] == 0) {
                return slot;
            }
            const uint32_t age = record[kOverlayUsed];
            if (found == kOverlaySlotsMax || age < oldest) {
                found = slot;
                oldest = age;
            }
        }
        return found;
    }

    inline uint32_t overlay_slot_address(
        uint32_t control, uint32_t part, uint32_t slot
    ) {
        volatile tt_l1_ptr uint32_t* area = overlay_area(control);
        return area[kOverlayBase + part] + slot * area[kOverlaySlotBytes + part];
    }

    // Marks the record's last use with the part's clock.
    inline void overlay_touch(uint32_t control, uint32_t part, uint32_t slot) {
        volatile tt_l1_ptr uint32_t* clock = l1(control + kOverlayArea + kOverlayClock);
        clock[part] = clock[part] + 1;
        overlay_record(control, part, slot)[kOverlayUsed] = clock[part];
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

    // A follower's load() of `part`: parks until the reader's go unless the tile holds
    // every follower part. The slot; kOverlaySlotsMax: gave up (`code`, `where`).
    inline uint32_t overlay_follow(
        uint32_t control,
        uint32_t number,
        uint32_t part,
        uint32_t limit,
        uint32_t& code,
        uint32_t& where
    ) {
        invalidate_l1_cache();
        if (!overlay_has_part(control, number, part)) {
            code = kGaveOverlay;
            where = kWhereOverlayFind;
            return kOverlaySlotsMax;
        }
        if (overlay_members_held(control, number)) {
            return overlay_held_slot(control, part, number);
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
            where = kWhereOverlayPair;
            return kOverlaySlotsMax;
        }
        return go[1] & 0xFFu;
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
#endif

    // An overlay on a data-movement core: `Kernel` is the km::Reader or km::Writer of
    // this kernel. The reader is the tile's loader (see load()).
    template <typename Kernel>
    class Overlay {
    public:
        Overlay(Kernel& kernel, uint32_t number) : kernel_(kernel), number_(number) {
        }

        // Whether a slot of this core holds this kernel's part of the overlay.
        bool held() const {
            const uint32_t control = kernel_.args().control;
            invalidate_l1_cache();
            return (
                overlay_has_part(control, number_, kOverlayPartHere) &&
                overlay_held_slot(control, kOverlayPartHere, number_) < kOverlaySlotsMax
            );
        }

        // Reader: loads every part the tile does not hold (load_tile). Writer: meets
        // the reader's load() of it (overlay_follow). False: gave up (recorded).
        bool load() {
            const uint32_t control = kernel_.args().control;
#if defined(COMPILE_FOR_NCRISC)
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
            slot_ = overlay_slot_address(control, kOverlayPartHere, slot);
            record_ = overlay_record(control, kOverlayPartHere, slot);
            return true;
        }

#if !defined(COMPILE_FOR_NCRISC)
        // Starts the read of `part` into `slot` (the record says none until it lands).
        // False: the part does not fit the slot (recorded).
        bool start_read(uint32_t control, uint32_t part, uint32_t slot) const {
            volatile tt_l1_ptr uint32_t* entry = overlay_entry(control, number_, part);
            const uint32_t bytes = entry[1];
            if (bytes > overlay_area(control)[kOverlaySlotBytes + part]) {
                return give_up(kGaveOverlay, kWhereOverlayFind);
            }
            overlay_record(control, part, slot)[kOverlayHeld] = 0;
            noc_async_read(
                get_noc_addr_from_bank_id<true>(kernel_.me() % kOverlayBanks, entry[0]),
                overlay_slot_address(control, part, slot),
                bytes
            );
            return true;
        }

        // Starts the read of `part` unless a slot holds it: slot | loaded << 8, or
        // kOverlaySlotsMax (gave up).
        uint32_t fetch(uint32_t control, uint32_t part) const {
            const uint32_t held = overlay_held_slot(control, part, number_);
            if (held != kOverlaySlotsMax) {
                return held;
            }
            const uint32_t slot = overlay_free_slot(control, part);
            if (slot == kOverlaySlotsMax) {
                give_up(kGaveOverlay, kWhereOverlayFind);
                return kOverlaySlotsMax;
            }
            return start_read(control, part, slot) ? slot | 1u << 8 : kOverlaySlotsMax;
        }

        // Timed wait until every read this core started has landed.
        bool landed() const {
            const uint32_t code = kernel_.poll(
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

        // Timed wait until every follower in `members` is parked in its next meeting
        // with the reader, for this overlay.
        bool parked(uint32_t control, uint32_t members) const {
            volatile tt_l1_ptr uint32_t* park = (
                l1(control + kOverlayArea + kOverlayPark)
            );
            volatile tt_l1_ptr uint32_t* met = l1(control + kOverlayArea + kOverlayMet);
            bool mismatch = false;
            const uint32_t code = kernel_.poll(
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
                return give_up(mismatch ? kGaveOverlay : code, kWhereOverlayPair);
            }
            return true;
        }

        // Reads the parts the tile does not hold (the followers' once they are parked),
        // invalidates their cores' caches in one write, then gives each follower go.
        bool load_tile(uint32_t control) {
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
                    const uint32_t slot = slots[part] & 0xFFu;
                    overlay_record(control, part, slot)[kOverlayHeld] = number_;
                    overlay_touch(control, part, slot);
                }
            }
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
                        go[
                            2 * follower + 1
                        ] = (slots[kOverlayPartWriter + follower] & 0xFFu);
                        go[2 * follower] = count << 8 | number_;
                    }
                }
            }
            return mine ? place(control, slots[kOverlayPartReader] & 0xFFu) : true;
        }
#endif

        Kernel& kernel_;
        uint32_t number_;
        uint32_t slot_ = 0;
        volatile tt_l1_ptr uint32_t* record_ = nullptr;
    };

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
        Overlay(Kernel& kernel, uint32_t number) : kernel_(kernel), number_(number) {
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
