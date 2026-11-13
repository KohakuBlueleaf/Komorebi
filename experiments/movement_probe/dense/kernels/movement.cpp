// The matrix product's data movement, one source for both movers: PRODUCT runs A's
// rows and Y's blocks as streams through the engine; LOAD and PROF as in main.
#include <stdint.h>

#include "../../kernels/movement.hpp"
#include "api/dataflow/dataflow_api.h"
#include "dense.h"
#include "komorebi/device/km_worker.hpp"

namespace {

#if defined(COMPILE_FOR_BRISC)
    using Runtime = km::Reader;
    constexpr uint32_t kGaveUpWord = km::kStReader;
    constexpr uint32_t kProfileBusy = kPrRdRead;
    constexpr uint32_t kProfileBody = kPrRdBody;
    constexpr uint32_t kProfileTiles = kPrRdTiles;
#else
    using Runtime = km::Writer;
    constexpr uint32_t kGaveUpWord = km::kStWriter;
    constexpr uint32_t kProfileBusy = kPrWrWrite;
    constexpr uint32_t kProfileBody = kPrWrBody;
    constexpr uint32_t kProfileTiles = kPrWrTiles;
#endif
    constexpr bool kReaderRole = movement::kMover == movement::kMoverA;

    // LOAD: the reader role reads all of W into the W buffer and tells the writer role,
    // which completes the command once W has landed.
    template <typename Arguments>
    bool load(
        const Runtime& runtime, const km_command_t& command, const Arguments& tiles
    ) {
        const uint32_t weights = command.a0;
        const uint32_t buffer = command.a1;
        const uint32_t pages = command.a3 * command.a4;
        const uint32_t limit = runtime.args().wait_limit;
        if constexpr (kReaderRole) {
            const auto tensor = TensorAccessor(tiles, weights);
            const uint32_t code = runtime.poll(
                [&] {
                    return cb_pages_reservable_at_back(buffer, pages);
                },
                limit
            );
            if (code != km::kGaveNone) {
                runtime.gave_up(kGaveUpWord, code, km::kWhereData, 0, 0);
                return false;
            }
            cb_reserve_back(buffer, pages);
            uint32_t address = get_write_ptr(buffer);
            for (uint32_t page = 0; page < pages; page++) {
                noc_async_read_page(page, tensor, address);
                address += kTileBytes;
            }
            if (!movement::settle(runtime, kGaveUpWord, true, false)) {
                return false;
            }
            cb_push_back(buffer, pages);
            cb_reserve_back(kCbLoaded, 1);
            cb_push_back(kCbLoaded, 1);
            return true;
        } else {
            const uint32_t code = runtime.poll(
                [&] {
                    return cb_pages_available_at_front(kCbLoaded, 1);
                },
                limit
            );
            if (code != km::kGaveNone) {
                runtime.gave_up(kGaveUpWord, code, km::kWhereData, 0, 0);
                return false;
            }
            cb_pop_front(kCbLoaded, 1);
            return true;
        }
    }

    // PRODUCT, resident form, the full call: A's rows (one block a row, mover A) and
    // Y's column blocks (owner by Y_PATTERN) through the engine.
    template <typename Arguments>
    bool product(
        const Runtime& runtime,
        const km_command_t& command,
        const Arguments& tiles,
        volatile tt_l1_ptr uint32_t* profile
    ) {
        const uint32_t start = km::wall();
        const uint32_t row_tiles = command.a2;
        const uint32_t inner_tiles = command.a3;
        const uint32_t column_tiles = command.a4;
        const uint32_t chunk_tiles = command.a5;
        const uint32_t block_tiles = command.a6;
        const uint32_t mode = command.a7;
        const uint32_t form = command.a8;
        if (mode != kModeFull || form != kFormResident || chunk_tiles != inner_tiles) {
            runtime.gave_up(kGaveUpWord, km::kGaveBadItem, km::kWhereData, 0, 0);
            return false;
        }
        const km::Partition rows = (
            km::partition(
                row_tiles, runtime.me() - km::first(command), km::count(command)
            )
        );
        using Accessor = decltype(TensorAccessor(tiles, 0));
        const movement::Stream<Accessor> streams[2] = {
            {
                movement::kRead,
                TensorAccessor(tiles, command.a0),
                rows.start * inner_tiles,
                rows.len * inner_tiles,
                inner_tiles,
                1,
                {kCbA, kCbA},
                movement::pattern_of(1)
            },
            {
                movement::kWrite,
                TensorAccessor(tiles, command.a1),
                rows.start * column_tiles,
                rows.len * column_tiles,
                block_tiles,
                1,
                {kOutputMoverA, kCbOut},
                movement::pattern_of(Y_PATTERN)
            },
        };
        movement::Busy busy = {0, 0};
        const bool done = (
            rows.len == 0 || movement::run(runtime, kGaveUpWord, streams, busy)
        );
        km::add_sat(profile + kProfileBusy, busy.cycles);
        km::add_sat(profile + kProfileTiles, busy.tiles);
        km::add_sat(profile + kProfileBody, km::wall() - start);
        return done;
    }

}  // namespace

void kernel_main() {
    constexpr auto tiles = TensorAccessorArgs<0>();  // every operand: bf16 tiles in DRAM
    Runtime runtime;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kCbProf));
    for (uint32_t word = 0; word < kProfWords; word++) {
        profile[word] = 0;
    }
    runtime.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler == kHandlerLoad) {
            return load(runtime, command, tiles);
        }
        if (handler == kHandlerProduct) {
            return product(runtime, command, tiles, profile);
        }
        if (handler != kHandlerProf || kReaderRole) {
            return true;
        }
        const uint32_t profile_tensor = command.a0;
        const InterleavedAddrGen<true> pages = {
            .bank_base_address = profile_tensor, .page_size = kProfBytes
        };
        noc_async_write(
            get_write_ptr(kCbProf), get_noc_addr(runtime.me(), pages), kProfBytes
        );
        return movement::settle(runtime, kGaveUpWord, false, true);
    });
}
