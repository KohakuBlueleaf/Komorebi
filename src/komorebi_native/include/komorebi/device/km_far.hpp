// A write into any rank's L1 (a count at its port after) and an add to a word of any
// rank's L1: this island's ranks by the NoC, another island's by the link service.
#pragma once

#include <stdint.h>

#include "km_core.hpp"
#include "km_island.hpp"

#if !defined(COMPILE_FOR_TRISC)
#if defined(KM_PAGE_LINKS)
#include "km_pages.hpp"
#endif

namespace km {

    // km::add_word(..., km::posted): returns once its line is sent; a CPU's lines land
    // in the order sent (puts too), so its next confirmed call is the fence.
    struct Posted {};

    constexpr Posted posted{};

    // The give-up of a far call: the gave-up word's top byte is the rank's island.
    template <typename Kernel>
    inline bool far_gave_up(
        const Kernel& core, const km_command_t& command, uint32_t code, uint32_t island
    ) {
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        core.gave_up(
            word,
            code,
            kWhereFar,
            slot(command) | (island & 0xFFu) << 8,
            command.sequence
        );
        return false;
    }

    // The core index of `rank` on this island, KM_CORES_MAX when it is another
    // island's (`island` set) or no island's.
    template <typename Kernel>
    inline uint32_t far_core(const Kernel& core, uint32_t rank, uint32_t& island) {
        const Islands view = islands_at(core.args().control);
        const Resolved where = resolve(view, rank);
        island = where.island;
        if (!where.here) {
            return KM_CORES_MAX;
        }
        if (view.ranks() != 0 && rank - view.first() >= view.ranks()) {
            return KM_CORES_MAX;  // in no entry: not this island's either
        }
        const uint32_t index = core.core_of(rank);
        return index < KM_CORES_MAX ? index : KM_CORES_MAX;
    }

    // The one rule of a write on any island: bytes in whole 4s, both addresses
    // 16-aligned (a far rank's pieces need it; KM_GUARD refuses it on this island too).
    constexpr bool far_fits(uint32_t from, uint32_t address, uint32_t bytes) {
#if KM_GUARD
        return bytes % 4 == 0 && ((from | address) & 15u) == 0;
#else
        return ((from ^ address) & 15u) == 0;  // the NoC's own rule
#endif
    }

    // The code of a far call to another island in this build: nothing was sent.
    constexpr uint32_t far_refused() {
#if KM_GUARD && defined(KM_PAGE_LINKS)
        return kGaveRefused;  // the command alone, as a refused page line
#else
        return kGaveBadItem;
#endif
    }

#if defined(KM_PAGE_LINKS)
    // Waits until this CPU's lines are all confirmed (both parities). kGave* code.
    template <typename Kernel>
    inline uint32_t far_settled(const Kernel& core) {
        const uint32_t mine = islands_at(core.args().control).own_words();
        const volatile tt_l1_ptr uint32_t* own = (
            l1(mine + (core.role() == kRoleWriter ? KM_OWN_CPU_BYTES : 0))
        );
        return core.poll_at(
            kWhereAskConfirm,
            [&] {
                invalidate_l1_cache();
                return (
                    own[KM_OWN_PUT_NUMBER / 4] == own[KM_OWN_PUT_CONFIRMED / 4] &&
                    own[KM_OWN_PUT_NUMBER / 4 + 1] == own[KM_OWN_PUT_CONFIRMED / 4 + 1]
                );
            },
            core.args().wait_limit
        );
    }

    // What a far line carries besides its op: word 2 (piece bytes, the word, the value)
    // and 4 (pieces), the rank's L1 address, the asker's (pieces), the port + 1.
    struct FarLine {
        uint32_t ring;
        uint32_t rank;
        uint32_t holder;
        uint32_t page;
        uint32_t pieces;
        uint32_t to;
        uint32_t source;
        uint32_t port;
    };

    // One line of an L1 write or an add into the queue of the link core toward island
    // `far.holder` (a put's numbering, parity); `confirmed`: waits its confirmation.
    template <typename Kernel>
    inline uint32_t far_line(const Kernel& core, const FarLine& far, bool confirmed) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t mine = view.own_words();
        const uint32_t index = core.args().me - view.base();
#if defined(KM_FAR_PUTS)
        const bool reached = view.hops(far.holder) != 0;  // any links away
#else
        const bool reached = view.hops(far.holder) == 1;  // the next island
#endif
        if (area == 0 || mine == 0 || !reached || index >= KM_CORES_MAX) {
            return far_refused();  // nothing sent
        }
        const uint32_t link = view.base() + view.ranks() + view.link(far.holder);
        const uint32_t limit = core.args().wait_limit;
        const uint32_t cpu = core.role() == kRoleWriter ? 1u : 0u;
        const uint32_t block = mine + cpu * KM_OWN_CPU_BYTES;
        volatile tt_l1_ptr uint32_t* own = l1(block);
        invalidate_l1_cache();
        const uint32_t parity = own[KM_OWN_PUT_SENT / 4] & 1u;
        const uint32_t owner = (index * 2 + cpu) * 2 + parity;
        uint32_t code = core.poll_at(
            kWhereAskConfirm,
            [&] {
                invalidate_l1_cache();
                return (
                    own[KM_OWN_PUT_NUMBER / 4 + parity] ==
                    own[KM_OWN_PUT_CONFIRMED / 4 + parity]
                );
            },
            limit
        );
        if (code != kGaveNone) {
            return code;
        }
        volatile tt_l1_ptr uint32_t* returned = l1(block + KM_OWN_RETURNED);
        returned[0] = 0xFFFFFFFFu;  // the atomic's answer not yet in
        invalidate_l1_cache();      // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(link, area + KM_PUT_TAIL),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            block + KM_OWN_RETURNED
        );
        code = core.poll_at(
            kWhereAskNumber,
            [&] {
                invalidate_l1_cache();
                return (
                    ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                    returned[0] != 0xFFFFFFFFu
                );
            },
            limit
        );
        if (code != kGaveNone) {
            return code;
        }
        const uint32_t number = returned[0];
        PutRings rings{};
        code = put_rings_read(core, link, area, block + KM_OWN_BUILD, rings, number);
        if (code != kGaveNone) {
            return code;
        }
        volatile tt_l1_ptr uint32_t* line = l1(block + KM_OWN_LINE);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = 0;
        }
        line[KM_ASK_TENSOR] = far.rank;
        line[KM_ASK_PAGE] = far.page;
        line[KM_ASK_PAGES] = far.pieces;
        line[KM_ASK_CORE] = core.control(KM_CONTROL_CORES)[core.args().me];
        line[KM_ASK_TO] = far.to;
        line[KM_ASK_COUNT] = block + KM_OWN_FAR;
        line[KM_ASK_HOLDER] = far.holder;
        line[KM_ASK_FROM] = view.here();
        line[KM_ASK_SOURCE] = far.source;
        line[KM_ASK_RING] = far.ring;
        line[KM_ASK_OWNER] = owner;
        line[KM_ASK_PORT] = far.port;
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(
            block + KM_OWN_LINE,
            core.core_noc(
                link, rings.queue + KM_QUEUE_PUTS + number % KM_QUEUE_LINES * 64
            ),
            64
        );
        code = core.poll_at(
            kWhereAskSent,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        own[KM_OWN_PUT_SENT / 4] = own[KM_OWN_PUT_SENT / 4] + 1;
        own[KM_OWN_PUT_NUMBER / 4 + parity] = KM_OWN_CONFIRM(number, far.holder);
        if (code != kGaveNone || !confirmed) {
            return code;
        }
        return core.poll_at(
            kWhereAskConfirm,
            [&] {
                invalidate_l1_cache();
                return (
                    own[KM_OWN_PUT_CONFIRMED / 4 + parity] ==
                    KM_OWN_CONFIRM(number, far.holder)
                );
            },
            limit
        );
    }

    // write_l1 to another island's rank: whole pieces of the packet's bytes a line,
    // then 16s, then 4s inline; the line with the port after every other confirmed.
    template <typename Kernel>
    inline uint32_t far_write(
        const Kernel& core,
        uint32_t rank,
        uint32_t holder,
        uint32_t address,
        uint32_t source,
        uint32_t bytes,
        int port
    ) {
        const uint32_t piece = islands_at(core.args().control).packet() / 16 * 16;
        if (piece == 0) {
            return far_refused();
        }
        const uint32_t lines16 = bytes / 16 * 16;
        uint32_t done = 0, sent = 0;
        uint32_t code = kGaveNone;
        do {  // bytes 0: one line, the port alone
            FarLine far = {
                KM_ASK_OP_L1 << KM_ASK_OP_SHIFT,
                rank,
                holder,
                0,
                0,
                address + done,
                source + done,
                0
            };
            uint32_t length = bytes == 0 ? 0 : 4;  // 0: no piece, no word
            if (done < lines16) {
                const uint32_t left = lines16 - done;
                far.page = left >= piece ? piece : left;
                far.pieces = left >= piece ? left / piece : 1;
                length = far.page * far.pieces;
            } else if (bytes != 0) {
                far.ring |= KM_ASK_INLINE;
                invalidate_l1_cache();
                far.page = l1(source + done)[0];
            }
            const bool last = done + length == bytes;
            if (last && sent != 0) {
                code = far_settled(core);  // every earlier line done: then the port
            }
            far.port = last && port >= 0 ? static_cast<uint32_t>(port) + 1 : 0;
            if (code == kGaveNone) {
                code = far_line(core, far, last);
            }
            done += length;
            sent++;
        } while (code == kGaveNone && done < bytes);
        return code;
    }
#endif

    // `bytes` (4s, any count) from L1 `from` into L1 `address` of `rank` (16-aligned),
    // then 1 at its port (port >= 0). At return landed, `from` free. False: gave up.
    template <typename Kernel>
    inline bool write_l1(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        uint32_t address,
        const void* from,
        uint32_t bytes,
        int port = -1
    ) {
        const uint32_t source = reinterpret_cast<uint32_t>(from);
        uint32_t island = 0;
        const uint32_t index = far_core(core, rank, island);
        if (index == KM_CORES_MAX && island != islands_at(core.args().control).here()) {
            if (
                (port >= 0 && static_cast<uint32_t>(port) >= KM_PORTS) ||
                bytes % 4 != 0 || ((source | address) & 15u) != 0
            ) {
                return far_gave_up(core, command, kGaveBadItem, island);
            }
#if defined(KM_PAGE_LINKS)
            const uint32_t code = (
                bytes == 0 && port < 0
                    ? kGaveNone
                    : far_write(core, rank, island, address, source, bytes, port)
            );
            return code == kGaveNone || far_gave_up(core, command, code, island);
#else
            return far_gave_up(core, command, far_refused(), island);
#endif
        }
        if (
            index == KM_CORES_MAX ||
            (port >= 0 && static_cast<uint32_t>(port) >= KM_PORTS) ||
            !far_fits(source, address, bytes)
        ) {
            return far_gave_up(core, command, kGaveBadItem, island);
        }
        const uint32_t limit = core.args().wait_limit;
        uint32_t code = kGaveNone;
        if (bytes != 0) {
            noc_async_write(source, core.core_noc(index, address), bytes);
            code = core.poll_at(
                kWhereFar,
                [] {
                    return ncrisc_noc_nonposted_writes_flushed(noc_index);
                },
                limit
            );
        }
        if (code == kGaveNone && port >= 0) {
            const uint32_t count = (
                core.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
                KM_PORT_COUNT * 4
            );
            noc_semaphore_inc(core.core_noc(index, count), 1);
            code = core.poll_at(
                kWhereFar,
                [] {
                    return ncrisc_noc_nonposted_atomics_flushed(noc_index);
                },
                limit
            );
        }
        return code == kGaveNone || far_gave_up(core, command, code, island);
    }

    // add_word's steps; `confirmed`: done at return, else once its line is sent.
    template <typename Kernel>
    inline bool add_word_at(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        uint32_t address,
        uint32_t value,
        bool confirmed
    ) {
        uint32_t island = 0;
        const uint32_t index = far_core(core, rank, island);
        if (index == KM_CORES_MAX && island != islands_at(core.args().control).here()) {
            if (address % 4 != 0) {
                return far_gave_up(core, command, kGaveBadItem, island);
            }
#if defined(KM_PAGE_LINKS)
            const FarLine far = {
                KM_ASK_OP_ADD << KM_ASK_OP_SHIFT, rank, island, value, 0, address, 0, 0
            };
            const uint32_t code = far_line(core, far, confirmed);
            return code == kGaveNone || far_gave_up(core, command, code, island);
#else
            return far_gave_up(core, command, far_refused(), island);
#endif
        }
        if (index == KM_CORES_MAX || address % 4 != 0) {
            return far_gave_up(core, command, kGaveBadItem, island);
        }
        noc_semaphore_inc(core.core_noc(index, address), value);
        if (!confirmed) {
            return true;  // the next confirmed call's barrier covers it
        }
        const uint32_t code = core.poll_at(
            kWhereFar,
            [] {
                return ncrisc_noc_nonposted_atomics_flushed(noc_index);
            },
            core.args().wait_limit
        );
        return code == kGaveNone || far_gave_up(core, command, code, island);
    }

    // `value` added to the word at `address` (4-byte aligned) of rank `rank`'s L1.
    // Done when it returns. False: gave up (where `far`).
    template <typename Kernel>
    inline bool add_word(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        uint32_t address,
        uint32_t value
    ) {
        return add_word_at(core, command, rank, address, value, true);
    }

    // The same, posted: true once sent; the rank's adds land in the order made.
    template <typename Kernel>
    inline bool add_word(
        const Kernel& core,
        const km_command_t& command,
        uint32_t rank,
        uint32_t address,
        uint32_t value,
        Posted
    ) {
        return add_word_at(core, command, rank, address, value, false);
    }

}  // namespace km
#endif
