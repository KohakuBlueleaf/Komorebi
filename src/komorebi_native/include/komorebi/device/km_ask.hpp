// A kernel's read and write of a tensor's pages wherever they live: the local read or
// write for the pages its island holds; for pages another island holds, a line into
// its island's link core 0. A get: its link 0 CPU sends it to the holder's link core
// 0, whose page CPU pushes the pages into this core's L1 and counts them at its port.
// A put: its page CPU reads the pages from this core into the holder's DRAM, and the
// holder's link core 0 counts them at this core's port once written. On one island:
// exactly the local read or write.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_placement.hpp"
#include "km_port.hpp"

namespace km {

    constexpr uint32_t kAskNoAnswer = 0xFFFFFFFFu;  // the atomic's answer not yet in

    // A queue of lines in link core 0's link area (offsets): the count its island's
    // cores reserve lines with, the count it took, its lines; and the asking CPU's own
    // line and its two answer words, on its own core.
    struct AskQueue {
        uint32_t tail;
        uint32_t taken;
        uint32_t lines;
        uint32_t own;
        uint32_t returned;
        uint32_t read;
    };

    // The get queue (the reader's) and the put queue of CPU `role`.
    constexpr AskQueue kAskGets = {
        KM_ASK_TAIL,
        KM_ASK_TAKEN,
        KM_ASK_LINES,
        KM_ASK_OWN,
        KM_ASK_RETURNED,
        KM_ASK_READ
    };

    constexpr AskQueue ask_puts(uint32_t role) {
        const uint32_t own = KM_PUT_OWN + (role == kRoleWriter ? 0 : KM_PUT_OWN_BYTES);
        return {KM_PUT_TAIL, KM_PUT_TAKEN, KM_PUT_LINES, own, own + 0x40, own + 0x50};
    }

    // One line into `queue` for pages first .. first + pages - 1 of the tensor at bank
    // address `address` (`bytes` aligned page bytes), held by island `holder`, and L1
    // `at` of this core, counted at port `port`: a line reserved on link core 0 with a
    // local atomic, written once that line is free. False: gave up (recorded).
    template <typename Kernel>
    inline bool ask_line(
        const Kernel& core,
        const km_command_t& command,
        const AskQueue& queue,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        uint32_t at,
        uint32_t port
    ) {
        const Islands view = islands_at(core.args().control);
        const uint32_t area = view.link_area();
        const uint32_t named = slot(command) | holder << 8;
        const uint32_t word = core.role() == kRoleWriter ? kStWriter : kStReader;
        const auto give_up = [&](uint32_t code) {
            core.gave_up(word, code, kWhereAsk, named, command.sequence);
            return false;
        };
        if (
            area == 0 || view.hops(holder) != 1 || port >= KM_PORTS || bytes == 0 ||
            bytes % 16 != 0 || bytes > KM_PAGES_PACKET_MAX || at % 16 != 0
        ) {
            return give_up(kGaveBadItem);  // two islands a link apart, whole packets
        }
        // link core 0: the core after the island's ranks
        const uint32_t link = view.base() + view.ranks();
        const uint32_t limit = core.args().wait_limit;
        volatile tt_l1_ptr uint32_t* returned = l1(area + queue.returned);
        returned[0] = kAskNoAnswer;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            core.core_noc(link, area + queue.tail),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            area + queue.returned
        );
        uint32_t code = core.poll_at(
            kWhereAsk,
            [&] {
                invalidate_l1_cache();
                return (
                    ncrisc_noc_nonposted_atomics_flushed(noc_index) &&
                    returned[0] != kAskNoAnswer
                );
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        const uint32_t number = returned[0];
        // the line is free once link core 0 took the one KM_ASK_SLOTS before
        volatile tt_l1_ptr uint32_t* taken = l1(area + queue.read);
        bool reading = false;
        code = core.poll_at(
            kWhereAsk,
            [&] {
                if (reading) {
                    if (!ncrisc_noc_reads_flushed(noc_index)) {
                        return false;
                    }
                    reading = false;
                    invalidate_l1_cache();
                    if (number - taken[0] < KM_ASK_SLOTS) {
                        return true;
                    }
                }
                noc_async_read(
                    core.core_noc(link, area + queue.taken), area + queue.read, 16
                );
                reading = true;
                return false;
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        volatile tt_l1_ptr uint32_t* line = l1(area + queue.own);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = 0;
        }
        line[KM_ASK_TENSOR] = address;
        line[KM_ASK_PAGE] = bytes;
        line[KM_ASK_FIRST] = first;
        line[KM_ASK_PAGES] = pages;
        line[KM_ASK_CORE] = core.control(KM_CONTROL_CORES)[core.args().me];
        line[KM_ASK_TO] = at;
        line[KM_ASK_COUNT] = (
            core.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
            KM_PORT_COUNT * 4
        );
        line[KM_ASK_HOLDER] = holder;
        line[KM_ASK_FROM] = view.here();
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(
            area + queue.own,
            core.core_noc(link, area + queue.lines + number % KM_ASK_SLOTS * 64),
            64
        );
        code = core.poll_at(
            kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        return code == kGaveNone || give_up(code);
    }

    // Reader: one get line (the far island pushes the pages into L1 `to`).
    inline bool ask_pages(
        const Reader& reader,
        const km_command_t& command,
        uint32_t holder,
        uint32_t address,
        uint32_t bytes,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        uint32_t port
    ) {
        return ask_line(
            reader, command, kAskGets, holder, address, bytes, first, pages, to, port
        );
    }

    // Reader: pages first .. first + pages - 1 of `tensor` (bank address `address`, the
    // same on every island) into L1 `to` on, page k at to + k aligned pages. The pages
    // this island holds: read (the caller's read barrier waits for them); each run of
    // pages another island holds: one ask, all counted at port `port` and waited for
    // here (bounded). False: gave up; a miss names the island asked.
    template <uint32_t Base, typename Accessor>
    inline bool read_pages(
        const Reader& reader,
        const km_command_t& command,
        const Placement<Base>& placement,
        const Accessor& tensor,
        uint32_t address,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        uint32_t port
    ) {
        const uint32_t here = island();
        const uint32_t bytes = tensor.get_aligned_page_size();
        uint32_t asked = 0, holder = here;
        for (uint32_t k = 0; k < pages;) {
            const uint32_t at = island_of(placement, first + k);
            if (at == here) {
                noc_async_read_page(first + k, tensor, to + k * bytes);
                k++;
                continue;
            }
            uint32_t run = 1;
            while (k + run < pages && island_of(placement, first + k + run) == at) {
                run++;
            }
#if !defined(KM_ASKS_BY_HOST)
            if (
                !ask_pages(
                    reader,
                    command,
                    at,
                    address,
                    bytes,
                    first + k,
                    run,
                    to + k * bytes,
                    port
                )
            ) {
                return false;
            }
#endif
            asked += run;
            holder = at;
            k += run;
        }
        // asks by the host: the host's send brings them to `to` and counts them here
        return (
            asked == 0 ||
            port_wait_at(
                reader, command, port, asked, kWhereAsk, slot(command) | holder << 8
            )
        );
    }

    // Reader or writer: pages first .. first + pages - 1 of `tensor` (bank address
    // `address`, the same on every island) from L1 `from` on, page k at from + k
    // aligned pages. The pages this island holds: written as noc_async_write_page
    // writes them (the caller's write barrier waits for them); each run of pages
    // another island holds: one put line, whose pages link core 0's page CPU reads from
    // this core and sends into the holder's DRAM; all of them counted at port `port`
    // once written there, and waited for here (bounded): `from` is free on return.
    // False: gave up; a miss names the island.
    template <typename Kernel, uint32_t Base, typename Accessor>
    inline bool write_pages(
        const Kernel& core,
        const km_command_t& command,
        const Placement<Base>& placement,
        const Accessor& tensor,
        uint32_t address,
        uint32_t first,
        uint32_t pages,
        uint32_t from,
        uint32_t port
    ) {
        const uint32_t here = island();
        const uint32_t bytes = tensor.get_aligned_page_size();
        uint32_t put = 0, holder = here;
        for (uint32_t k = 0; k < pages;) {
            const uint32_t at = island_of(placement, first + k);
            if (at == here) {
                noc_async_write_page(first + k, tensor, from + k * bytes);
                k++;
                continue;
            }
            uint32_t run = 1;
            while (k + run < pages && island_of(placement, first + k + run) == at) {
                run++;
            }
            if (
                !ask_line(
                    core,
                    command,
                    ask_puts(core.role()),
                    at,
                    address,
                    bytes,
                    first + k,
                    run,
                    from + k * bytes,
                    port
                )
            ) {
                return false;
            }
            put += run;
            holder = at;
            k += run;
        }
        return (
            put == 0 ||
            port_wait_at(
                core, command, port, put, kWhereAsk, slot(command) | holder << 8
            )
        );
    }

}  // namespace km
