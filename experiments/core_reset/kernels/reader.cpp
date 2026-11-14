// The map/collect example's reader (examples/01_mapcollect) with SPIN: rank a0 loops
// for ever in its reader. A core reset ladder's stuck core.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_ask.hpp"
#include "komorebi/device/km_port.hpp"
#include "komorebi/device/km_worker.hpp"
#include "spin.h"

void kernel_main() {
    using Input = km::Tensor<0>;
    km::Reader reader;
    uint32_t write_page = 0;  // write position in kCbIn, in pages: a block never wraps
    reader.run([&](const km_command_t& command) {
        const uint32_t index = reader.me() - km::first(command);
        switch (km::handler(command)) {
            case kHandlerMap: {
                const uint32_t input_address = command.a0;
                const uint32_t tiles = command.a1;
                const uint32_t first_tile = command.a6;
                const km::Partition partition = (
                    km::partition(tiles, index, km::count(command))
                );
                const Input input(input_address);  // its reference word
                for (uint32_t done = 0; done < partition.len;) {
                    uint32_t block_tiles = (
                        partition.len - done < kBlock ? partition.len - done : kBlock
                    );
                    block_tiles = (
                        block_tiles < kInPages - write_page
                            ? block_tiles
                            : kInPages - write_page
                    );
                    if (!reader.reserve(kCbIn, block_tiles)) {
                        return;
                    }
                    const uint32_t destination = get_write_ptr(kCbIn);
                    // the tiles another island holds are asked for, then waited for
                    if (!km::read_pages(
                        reader,
                        command,
                        input,
                        first_tile + partition.start + done,
                        block_tiles,
                        destination,
                        kAskPort
                    )) {
                        return;
                    }
                    noc_async_read_barrier();
                    cb_push_back(kCbIn, block_tiles);
                    done += block_tiles;
                    write_page = (write_page + block_tiles) % kInPages;
                }
                break;
            }
            case kHandlerBadItem: {  // from page `write_page`: past the end unless 0
                const uint32_t faulty_rank = command.a0;
                const uint32_t faulty_role = command.a1;
                if (
                    index == faulty_rank && faulty_role == 1 &&
                    !reader.reserve(kCbIn, kInPages)
                ) {
                    return;
                }
                break;
            }
            case kHandlerDot: {
                const uint32_t word_sum = command.a1;
                const km::Block block = reader.block(command);
                uint32_t sum = 0;
                for (uint32_t word = 0; word < block.bytes / 4; word++) {
                    sum += block.words[word];
                }
                if (sum != word_sum) {
                    reader.gave_up(
                        km::kStReader,
                        km::kGaveBadItem,
                        km::kWhereData,
                        km::slot(command),
                        command.sequence
                    );
                }
                break;
            }
            case kHandlerStall: {
                const uint32_t stalled_rank = command.a0;
                if (index == stalled_rank) {
                    reader.wait_stop(command);
                }
                break;
            }
            case kHandlerHang: {
                const uint32_t hung_rank = command.a0;
                const uint32_t hang_mode = command.a1;
                if (index == hung_rank && hang_mode == kHangPort) {
                    km::arrive(reader, command, km::Arrival{kCbIn, 1}, 0, 1);
                } else if (index == hung_rank && hang_mode == kHangInput) {
                    if (!reader.reserve(kCbIn, 1)) {
                        return;
                    }
                    cb_push_back(kCbIn, 1);  // one of the two tiles compute waits for
                    write_page = (write_page + 1) % kInPages;
                }
                break;
            }
            case kHandlerSpin: {
                const uint32_t spinning_rank = command.a0;
                const uint32_t spin_where = command.a1;
                if (index == spinning_rank && spin_where == kSpinReader) {
                    for (volatile uint32_t turn = 0;; turn = turn + 1) {
                        ;
                    }
                }
                break;
            }
            default:
                break;
        }
    });
}
