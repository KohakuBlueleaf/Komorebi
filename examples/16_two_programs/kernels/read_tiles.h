// The two programs' reader: the core's partition of a1 tiles from tile a6 of the input
// (a0) into buffer `cb` (kCbIn), in blocks that never wrap the buffer.
#pragma once

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_program.hpp"
#include "two_programs.h"

template <uint32_t cb = kCbIn>
inline void read_tiles(km::Reader& reader, const km_command_t& command) {
    constexpr auto input_args = TensorAccessorArgs<0>();
    const km::Partition partition = (
        km::partition(command.a1, reader.me() - km::first(command), km::count(command))
    );
    const auto input = TensorAccessor(input_args, command.a0);
    for (uint32_t done = 0; done < partition.len;) {
        // the pages before the buffer's end: a part keeps no position of its own
        const LocalCBInterface& buffer = get_local_cb_interface(cb);
        const uint32_t to_end = (buffer.fifo_limit - buffer.fifo_wr_ptr) / kTileBytes;
        uint32_t block = partition.len - done < kBlock ? partition.len - done : kBlock;
        block = block < to_end ? block : to_end;
        if (!reader.reserve(cb, block)) {
            return;
        }
        const uint32_t destination = get_write_ptr(cb);
        for (uint32_t tile = 0; tile < block; tile++) {
            noc_async_read_page(
                command.a6 + partition.start + done + tile,
                input,
                destination + tile * kTileBytes
            );
        }
        noc_async_read_barrier();
        cb_push_back(cb, block);
        done += block;
    }
}
