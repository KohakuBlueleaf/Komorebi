// A kernel's read of a tensor's pages wherever they live: the local read for the pages
// its island holds; for pages another island holds, an ask line into its island's
// link core 0, whose link 0 CPU sends it to the holder's link core 0, whose page CPU
// pushes the pages into this core's L1 and counts them at its port. On one island:
// exactly the local read.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "km_placement.hpp"
#include "km_port.hpp"

namespace km {

    constexpr uint32_t kAskNoAnswer = 0xFFFFFFFFu;  // the atomic's answer not yet in

    // Reader: one ask line for pages first .. first + pages - 1 of the tensor at bank
    // address `address` (`bytes` aligned page bytes), held by island `holder`, into L1
    // `to` of this core, counted at port `port`: a line reserved on link core 0 with
    // a local atomic, written once that line is free. False: gave up (recorded).
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
        const Islands view = islands_at(reader.args().control);
        const uint32_t area = view.link_area();
        const uint32_t named = slot(command) | holder << 8;
        const auto give_up = [&](uint32_t code) {
            reader.gave_up(kStReader, code, kWhereAsk, named, command.sequence);
            return false;
        };
        if (
            area == 0 || view.hops(holder) != 1 || port >= KM_PORTS || bytes == 0 ||
            bytes % 16 != 0 || bytes > KM_PAGES_PACKET_MAX || to % 16 != 0
        ) {
            return give_up(kGaveBadItem);  // two islands a link apart, whole packets
        }
        const uint32_t link = view.ranks();  // link core 0: the core after the ranks
        const uint32_t limit = reader.args().wait_limit;
        volatile tt_l1_ptr uint32_t* returned = l1(area + KM_ASK_RETURNED);
        returned[0] = kAskNoAnswer;
        invalidate_l1_cache();  // a fence: the store lands before the answer can
        noc_fast_atomic_increment<noc_mode, true>(
            noc_index,
            write_at_cmd_buf,
            reader.core_noc(link, area + KM_ASK_TAIL),
            NOC_UNICAST_WRITE_VC,
            1,
            31,
            false,
            false,
            area + KM_ASK_RETURNED
        );
        uint32_t code = reader.poll_at(
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
        volatile tt_l1_ptr uint32_t* taken = l1(area + KM_ASK_READ);
        bool reading = false;
        code = reader.poll_at(
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
                    reader.core_noc(link, area + KM_ASK_TAKEN), area + KM_ASK_READ, 16
                );
                reading = true;
                return false;
            },
            limit
        );
        if (code != kGaveNone) {
            return give_up(code);
        }
        volatile tt_l1_ptr uint32_t* line = l1(area + KM_ASK_OWN);
        for (uint32_t w = 0; w < 16; w++) {
            line[w] = 0;
        }
        line[KM_ASK_TENSOR] = address;
        line[KM_ASK_PAGE] = bytes;
        line[KM_ASK_FIRST] = first;
        line[KM_ASK_PAGES] = pages;
        line[KM_ASK_CORE] = reader.control(KM_CONTROL_CORES)[reader.args().me];
        line[KM_ASK_TO] = to;
        line[KM_ASK_COUNT] = (
            reader.args().control + KM_CONTROL_PORT + port * KM_PORT_STRIDE +
            KM_PORT_COUNT * 4
        );
        line[KM_ASK_HOLDER] = holder;
        line[KM_ASK_FROM] = view.here();
        line[KM_ASK_NUMBER] = number + 1;
        line[KM_ASK_END] = number + 1;
        noc_async_write(
            area + KM_ASK_OWN,
            reader.core_noc(link, area + KM_ASK_LINES + number % KM_ASK_SLOTS * 64),
            64
        );
        code = reader.poll_at(
            kWhereAsk,
            [] {
                return ncrisc_noc_nonposted_writes_flushed(noc_index);
            },
            limit
        );
        return code == kGaveNone || give_up(code);
    }

    // Reader: pages first .. of tensor `tensor` (its word; `args` its accessor's) into L1
    // `to` on; local pages read, each other island's run asked for. One card: the read.
    template <typename Args>
    inline bool read_pages(
        const Reader& reader,
        const km_command_t& command,
        const Args& args,
        uint32_t tensor,
        uint32_t first,
        uint32_t pages,
        uint32_t to,
        uint32_t port
    ) {
        const uint32_t here = island();
        const uint32_t bytes = TensorAccessor(args, 0).get_aligned_page_size();
        invalidate_l1_cache();  // the table may have grown since the last read
        uint32_t asked = 0, holder = here;
        for (uint32_t k = 0; k < pages;) {
            const PageAt at = page_at(tensor, first + k);
            if (at.island == here) {
                noc_async_read_page(
                    at.index, TensorAccessor(args, at.address), to + k * bytes
                );
                k++;
                continue;
            }
            uint32_t run = 1;
            while (k + run < pages) {
                const PageAt next = page_at(tensor, first + k + run);
                if (next.island != at.island || next.index != at.index + run) {
                    break;
                }
                run++;
            }
#if !defined(KM_ASKS_BY_HOST)
            if (
                !ask_pages(
                    reader,
                    command,
                    at.island,
                    at.address,
                    bytes,
                    at.index,
                    run,
                    to + k * bytes,
                    port
                )
            ) {
                return false;
            }
#endif
            asked += run;
            holder = at.island;
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

}  // namespace km
