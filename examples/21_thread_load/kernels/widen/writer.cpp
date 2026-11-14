// The widen program's writer: WIDEN hashes each of the core's float32 tiles and writes
// one line (the sum of the hashes) to line a3 + the core of the outs (a2).
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_program.hpp"
#include "../two_programs.h"
#include "../../thread_load.h"

bool km_program(km::Writer& writer, const km_command_t& command) {
    if (km::handler(command) != TL_HANDLER_WIDEN) {
        return true;
    }
    const km::Partition partition = (
        km::partition(command.a1, writer.me() - km::first(command), km::count(command))
    );
    uint32_t hashes = 0;
    for (uint32_t tile = 0; tile < partition.len; tile++) {
        if (!writer.wait_front(kCbWideOut, 1)) {
            return false;  // nothing to report: the tile is stopping
        }
        invalidate_l1_cache();
        const uint32_t* words = (
            reinterpret_cast<const uint32_t*>(get_read_ptr(kCbWideOut))
        );
        uint32_t hash = TL_HASH_START;
        for (uint32_t word = 0; word < 2 * kTileBytes / 4; word++) {
            hash = TL_MIX(hash, words[word]);
        }
        hashes += hash;
        cb_pop_front(kCbWideOut, 1);
    }
    const uint32_t buffer = get_write_ptr(TL_CB_WIDEN_LINE);
    volatile tt_l1_ptr uint32_t* line = km::l1(buffer);
    line[TL_LINE_HASH] = hashes;
    line[TL_LINE_TILES] = partition.len;
    line[TL_LINE_TAG] = command.a4;
    line[TL_LINE_PROGRAM] = TL_PROGRAM_WIDEN;
    line[TL_LINE_FIRST] = command.a6 + partition.start;
    const InterleavedAddrGen<true> outs = {
        .bank_base_address = command.a2, .page_size = TL_LINE_BYTES
    };
    noc_async_write(
        buffer, get_noc_addr(command.a3 + writer.me(), outs), TL_LINE_BYTES
    );
    noc_async_writes_flushed();
    return true;
}
