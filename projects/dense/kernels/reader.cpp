// Matrix product reader: LOAD reads all of W into the W buffer; PRODUCT reads the rows
// of A this core owns (and, in the streamed form, W's chunks), in buffer items.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "dense.h"
#include "komorebi/device/km_worker.hpp"

namespace {

    // Reads `rows` x `width` tiles of accessor from tile `first` (rows `stride` tiles
    // apart) into the reserved item of cb, then pushes it. False: the reads gave up.
    template <typename Accessor>
    bool read_item(
        const km::Reader& reader,
        uint32_t cb,
        const Accessor& accessor,
        uint32_t first,
        uint32_t rows,
        uint32_t width,
        uint32_t stride
    ) {
        uint32_t destination = get_write_ptr(cb);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t column = 0; column < width; ++column) {
                noc_async_read_page(
                    first + row * stride + column, accessor, destination
                );
                destination += kTileBytes;
            }
        }
        if (!flushed(reader, km::kStReader, km::kFlushReads)) {
            return false;
        }
        cb_push_back(cb, rows * width);
        return true;
    }

    // LAYER: A's row from DRAM per owned row when A is not held (B is always held).
    template <typename Accessor>
    void layer(
        const km::Reader& reader,
        const km_command_t& command,
        const Accessor& a_matrix,
        volatile tt_l1_ptr uint32_t* profile
    ) {
        if ((command.a8 & 0xFF) != kDram) {
            return;
        }
        const km::Partition owned_rows = (
            km::partition(
                command.a2, reader.me() - km::first(command), km::count(command)
            )
        );
        const uint32_t k_tiles = command.a3;
        uint32_t wait = 0, read = 0, tiles_read = 0;
        for (
            uint32_t row = owned_rows.start; row < owned_rows.start + owned_rows.len;
            ++row
        ) {
            const uint32_t asked = km::wall();
            if (!reader.reserve(kCbA, k_tiles)) {
                return;
            }
            const uint32_t reserved = km::wall();
            if (!read_item(reader, kCbA, a_matrix, row * k_tiles, 1, k_tiles, 0)) {
                return;
            }
            wait += reserved - asked;
            read += km::wall() - reserved;
            tiles_read += k_tiles;
        }
        km::add_sat(profile + kPrRdWait, wait);
        km::add_sat(profile + kPrRdRead, read);
        km::add_sat(profile + kPrRdTiles, tiles_read);
    }

}  // namespace

void kernel_main() {
    constexpr auto tile_args = TensorAccessorArgs<0>();  // every operand: bf16 tiles in DRAM
    km::Reader reader;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbProf));
    for (uint32_t word = 0; word < kProfWords; ++word) {
        profile[word] = 0;
    }
    reader.run([&](const km_command_t& command) {
        const uint32_t started = km::wall();
        const uint32_t k_tiles = command.a3, n_tiles = command.a4;
        if (km::handler(command) == kHandlerLoad) {
            const uint32_t w_address = command.a0, w_buffer = command.a1;
            if (!reader.reserve(w_buffer, k_tiles * n_tiles)) {
                return;
            }
            const auto w_matrix = TensorAccessor(tile_args, w_address);
            if (
                read_item(reader, w_buffer, w_matrix, 0, 1, k_tiles * n_tiles, 0) &&
                reader.reserve(kCbLoaded, 1)
            ) {
                cb_push_back(kCbLoaded, 1);
            }
            return;
        }
        if (km::handler(command) == kHandlerLayer) {
            const uint32_t a_address = command.a0;
            layer(reader, command, TensorAccessor(tile_args, a_address), profile);
            km::add_sat(profile + kPrRdBody, km::wall() - started);
            return;
        }
        const uint32_t mode = command.a7;
        if (km::handler(command) != kHandlerProduct || mode == kModeComputeOnly) {
            return;
        }
        const uint32_t a_address = command.a0, m_tiles = command.a2;
        const uint32_t chunk_tiles = command.a5, block_width = command.a6;
        const uint32_t form = command.a8, w_address = command.a9;
        const km::Partition owned_rows = (
            km::partition(m_tiles, reader.me() - km::first(command), km::count(command))
        );
        const auto a_matrix = TensorAccessor(tile_args, a_address);
        const auto w_matrix = TensorAccessor(tile_args, w_address);
        const bool stream = form == kFormStream;
        // The row is one item when it is one chunk or when W is streamed.
        const bool row_item = stream || chunk_tiles == k_tiles;
        uint32_t wait = 0, read = 0, stream_wait = 0, stream_read = 0, tiles_read = 0;
        for (
            uint32_t row = owned_rows.start; row < owned_rows.start + owned_rows.len;
            ++row
        ) {
            for (uint32_t column = 0; column < n_tiles; column += block_width) {
                for (uint32_t chunk = 0; chunk < k_tiles; chunk += chunk_tiles) {
                    if (!row_item || (column == 0 && chunk == 0)) {
                        const uint32_t length = row_item ? k_tiles : chunk_tiles;
                        const uint32_t asked = km::wall();
                        if (!reader.reserve(kCbA, length)) {
                            return;
                        }
                        const uint32_t reserved = km::wall();
                        if (!read_item(
                            reader,
                            kCbA,
                            a_matrix,
                            row * k_tiles + chunk,
                            1,
                            length,
                            0
                        )) {
                            return;
                        }
                        wait += reserved - asked;
                        read += km::wall() - reserved;
                        tiles_read += length;
                    }
                    if (stream) {
                        const uint32_t asked = km::wall();
                        if (!reader.reserve(kCbWs, chunk_tiles * block_width)) {
                            return;
                        }
                        const uint32_t reserved = km::wall();
                        if (!read_item(
                            reader,
                            kCbWs,
                            w_matrix,
                            chunk * n_tiles + column,
                            chunk_tiles,
                            block_width,
                            n_tiles
                        )) {
                            return;
                        }
                        stream_wait += reserved - asked;
                        stream_read += km::wall() - reserved;
                        tiles_read += chunk_tiles * block_width;
                    }
                }
            }
        }
        km::add_sat(profile + kPrRdWait, wait);
        km::add_sat(profile + kPrRdRead, read);
        km::add_sat(profile + kPrRdWsWait, stream_wait);
        km::add_sat(profile + kPrRdWsRead, stream_read);
        km::add_sat(profile + kPrRdTiles, tiles_read);
        km::add_sat(profile + kPrRdBody, km::wall() - started);
    });
}
