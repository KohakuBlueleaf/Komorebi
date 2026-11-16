// An all-reduce of the ports' reduce: its sum cast back to every rank of the group in
// one command; the group's rank 0 casts it to its cores by multicast writes.
#pragma once

#include <stdint.h>

#include "km_group.hpp"
#include "km_port.hpp"

#if !defined(COMPILE_FOR_TRISC)
namespace km {

    // Reader: the reduce's arrivals, then (a rank other than the group's 0) the sum
    // cast back, its pages handed to compute in slot 0. False: gave up.
    inline bool all_reduce_arrive(
        const Reader& reader,
        const km_command_t& command,
        const Arrival& arrival,
        const Group& group,
        uint32_t form
    ) {
        if (!reduce_arrive(reader, command, arrival, group, form)) {
            return false;
        }
        return (
            group.rank == 0 || arrive(reader, command, arrival, KM_PORT_ALL_REDUCE, 1)
        );
    }

    // The writer's NoC writes and atomics flushed (bounded); a give-up names the port.
    // False: gave up.
    __attribute__((noinline)) inline bool cast_flushed(
        const Writer& writer, const km_command_t& command
    ) {
        const uint32_t code = writer.poll_at(
            kWhereEmit,
            [] {
                return (
                    ncrisc_noc_nonposted_writes_flushed(noc_index) &&
                    ncrisc_noc_nonposted_atomics_flushed(noc_index)
                );
            },
            writer.args().wait_limit
        );
        if (code != kGaveNone) {
            writer.gave_up(
                kStWriter,
                code,
                kWhereEmit,
                slot(command) | KM_PORT_ALL_REDUCE << 8,
                command.sequence
            );
            return false;
        }
        return true;
    }

    // One multicast of `bytes` from L1 `source` to L1 `target` on the cores of NoC
    // rectangle x0 .. x1, y0 .. y1 (`cores` of them); one core: a plain write.
    inline void cast_rectangle(
        uint32_t source,
        uint32_t target,
        uint32_t bytes,
        uint32_t x0,
        uint32_t y0,
        uint32_t x1,
        uint32_t y1,
        uint32_t cores
    ) {
        if (cores == 1) {
            noc_async_write(source, get_noc_addr(x0, y0, target), bytes);
            return;
        }
        // NoC 1 runs from the far corner
        const bool far = noc_index != 0;
        noc_async_write_multicast(
            source,
            get_noc_multicast_addr(
                far ? x1 : x0, far ? y1 : y0, far ? x0 : x1, far ? y0 : y1, target
            ),
            bytes,
            cores
        );
    }

    // `bytes` from `source` to `target` on flat ranks first + k * stride, k < ranks, of
    // this island: a multicast a NoC rectangle of whole runs (a NoC row, x one apart).
    inline void cast_ranks(
        const Writer& writer,
        uint32_t first,
        uint32_t stride,
        uint32_t ranks,
        uint32_t source,
        uint32_t target,
        uint32_t bytes
    ) {
        const volatile tt_l1_ptr uint32_t* table = writer.control(KM_CONTROL_CORES);
        const auto place = [&](uint32_t k) {
            return (
                core_entry(
                    table,
                    writer.core_of(first + k * stride),
                    writer.args().control,
                    writer.role()
                )
            );
        };
        // the rectangle so far: columns x0 .. x1, rows y0 .. y1
        uint32_t x0 = 0, x1 = 0, y0 = 0, y1 = 0, cores = 0;
        // the run being read: its row y, columns rx0 .. rx1
        uint32_t k = 0;
        while (k < ranks) {
            const uint32_t at = place(k);
            const uint32_t rx0 = at & 0xFFFFu, y = at >> 16;
            uint32_t rx1 = rx0;
            k++;
            while (k < ranks) {
                const uint32_t next = place(k);
                if (next >> 16 != y || (next & 0xFFFFu) != rx1 + 1) {
                    break;
                }
                rx1++, k++;
            }
            // a run under the rectangle with its columns grows it by a row
            if (cores != 0 && rx0 == x0 && rx1 == x1 && y == y1 + 1) {
                y1 = y, cores += rx1 - rx0 + 1;
                continue;
            }
            if (cores != 0) {
                cast_rectangle(source, target, bytes, x0, y0, x1, y1, cores);
            }
            x0 = rx0, x1 = rx1, y0 = y, y1 = y, cores = rx1 - rx0 + 1;
        }
        if (cores != 0) {
            cast_rectangle(source, target, bytes, x0, y0, x1, y1, cores);
        }
    }

    // Writer: rank 0 casts the sum (front of `cb`, kept there) into every rank's slot
    // 0, then 1 onto its port KM_PORT_ALL_REDUCE; the others emit. False: gave up.
    inline bool all_reduce_emit(
        const Writer& writer,
        const km_command_t& command,
        const Arrival& arrival,
        const Group& group,
        uint32_t form,
        uint32_t cb,
        uint32_t pages
    ) {
        if (group.rank != 0) {
            return reduce_emit(writer, command, arrival, group, form, cb, pages);
        }
        const uint32_t bytes = pages * get_local_cb_interface(cb).fifo_page_size;
        const uint32_t room = (
            arrival.per * get_local_cb_interface(arrival.cb).fifo_page_size
        );
        const Group part = group.local();
        if (bytes > room || part.size != group.size) {  // across islands: not yet
            writer.gave_up(
                kStWriter,
                kGaveBadItem,
                kWhereEmit,
                slot(command) | KM_PORT_ALL_REDUCE << 8,
                command.sequence
            );
            return false;
        }
        if (!writer.wait_front(cb, pages, kBehindPort)) {
            return false;
        }
        if (group.size < 2) {
            return true;
        }
        const uint32_t first = core_of(group, 1);
        cast_ranks(
            writer,
            first,
            group.stride,
            group.size - 1,
            get_read_ptr(cb),
            slot_addr(arrival, 0),
            bytes
        );
        if (!cast_flushed(writer, command)) {
            return false;
        }
        // the count after the pages, multicast as they were: 1 from a word at the same
        // 16-byte offset (KM_PORT_RIGHT's count, free); each port was cleared before
        const uint32_t count = (
            writer.args().control + KM_CONTROL_PORT +
            KM_PORT_ALL_REDUCE * KM_PORT_STRIDE + 4 * KM_PORT_COUNT
        );
        const uint32_t one = (
            writer.args().control + KM_CONTROL_PORT + KM_PORT_RIGHT * KM_PORT_STRIDE +
            4 * KM_PORT_COUNT
        );
        *l1(one) = 1;
        cast_ranks(writer, first, group.stride, group.size - 1, one, count, 4);
        return cast_flushed(writer, command);
    }

}  // namespace km

#else
namespace km {

    // Compute, its part packed: a rank but the group's 0 takes the sum cast back, use()
    // reads it from slot 0 (page i: i), released after. False: stopping.
    template <typename Use>
    inline bool all_reduce_take(
        const Compute& compute, const Arrival& arrival, const Group& group, Use use
    ) {
        if (group.rank == 0) {
            return true;
        }
        if (!take(compute, arrival)) {
            return false;
        }
        use();
        untake(arrival);
        return true;
    }

}  // namespace km
#endif
