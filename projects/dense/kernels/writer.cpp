// Matrix product writer: PRODUCT stores each packed column block at Y[m, j0 ..);
// PROF stores this core's profile page.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dense.h"
#include "komorebi/device/km_worker.hpp"

void kernel_main() {
    constexpr auto tile_args = TensorAccessorArgs<0>();
    km::Writer writer;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbProf));
    writer.run([&](const km_command_t& command) {
        if (km::handler(command) == kHandlerProf) {
            const uint32_t profile_address = command.a0;
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = profile_address, .page_size = kProfBytes
            };
            noc_async_write(
                get_write_ptr(kCbProf), get_noc_addr(writer.me(), pages), kProfBytes
            );
            return flushed(writer, km::kStWriter, km::kFlushSent);
        }
        if (km::handler(command) == kHandlerLoad) {  // completes once W is in L1
            if (!writer.wait_front(kCbLoaded, 1)) {
                return false;
            }
            cb_pop_front(kCbLoaded, 1);
            return true;
        }
        const bool layer = km::handler(command) == kHandlerLayer;
        const uint32_t held_buffers = command.a8;  // LAYER only: A | B << 8 | Y << 16
        const uint32_t mode = command.a7;
        if (layer && ((held_buffers >> 16) & 0xFF) != kDram) {
            return true;  // Y stays in a held buffer
        }
        if (
            !layer && (km::handler(command) != kHandlerProduct || mode == kModeReadOnly)
        ) {
            return true;
        }
        const uint32_t started = km::wall();
        const uint32_t y_address = command.a1, m_tiles = command.a2;
        const uint32_t n_tiles = command.a4;
        // LAYER: a5 is the column block and every block is stored.
        const uint32_t block_width = layer ? command.a5 : command.a6;
        const km::Partition owned_rows = (
            km::partition(m_tiles, writer.me() - km::first(command), km::count(command))
        );
        const bool store = layer || mode == kModeFull;
        const auto y_matrix = TensorAccessor(tile_args, y_address);
        uint32_t wait = 0, write = 0, tiles_written = 0;
        for (
            uint32_t row = owned_rows.start; row < owned_rows.start + owned_rows.len;
            ++row
        ) {
            for (uint32_t column = 0; column < n_tiles; column += block_width) {
                const uint32_t asked = km::wall();
                if (!writer.wait_front(kCbOut, block_width)) {
                    return false;  // nothing to report: the tile is stopping
                }
                const uint32_t ready = km::wall();
                if (store) {
                    uint32_t source = get_read_ptr(kCbOut);
                    for (uint32_t tile = 0; tile < block_width; ++tile) {
                        noc_async_write_page(
                            row * n_tiles + column + tile, y_matrix, source
                        );
                        source += kTileBytes;
                    }
                    if (!flushed(writer, km::kStWriter, km::kFlushSent)) {
                        return false;
                    }
                    tiles_written += block_width;
                }
                cb_pop_front(kCbOut, block_width);
                wait += ready - asked;
                write += km::wall() - ready;
            }
        }
        km::add_sat(profile + kPrWrWait, wait);
        km::add_sat(profile + kPrWrWrite, write);
        km::add_sat(profile + kPrWrTiles, tiles_written);
        km::add_sat(profile + kPrWrBody, km::wall() - started);
        return true;
    });
}
