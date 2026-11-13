// The probe's data-movement kernel, one source for both movers: RUN moves its streams
// through the engine or plays this CPU's classic role; PROFILE stores the page.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "komorebi/device/km_worker.hpp"
#include "movement.hpp"
#include "probe.h"

namespace {

#if defined(COMPILE_FOR_BRISC)
    using Runtime = km::Reader;
    constexpr uint32_t kGaveUpWord = km::kStReader;
    constexpr uint32_t kProfileMover = kProfileMoverA;
#else
    using Runtime = km::Writer;
    constexpr uint32_t kGaveUpWord = km::kStWriter;
    constexpr uint32_t kProfileMover = kProfileMoverB;
#endif

    // Timed until `buffer` has room (reads) or data (writes) for `pages`.
    bool wait_buffer(
        const Runtime& runtime, uint32_t buffer, uint32_t pages, bool room
    ) {
        const LocalCBInterface& ring = get_local_cb_interface(buffer);
        const uint32_t pointer = room ? ring.fifo_wr_ptr : ring.fifo_rd_ptr;
        if (!km::item_fits(buffer, pointer, pages)) {
            runtime.gave_up(kGaveUpWord, km::kGaveBadItem, km::kWhereData, 0, 0);
            return false;
        }
        const uint32_t code = runtime.poll(
            [&] {
                return (
                    room
                        ? cb_pages_reservable_at_back(buffer, pages)
                        : cb_pages_available_at_front(buffer, pages)
                );
            },
            runtime.args().wait_limit
        );
        if (code != km::kGaveNone) {
            runtime.gave_up(kGaveUpWord, code, km::kWhereData, 0, 0);
            return false;
        }
        return true;
    }

    // The classic role: every block of the stream on this CPU, blocking calls (wait
    // for the buffer, issue, wait at the barrier, push or pop).
    template <typename Accessor>
    bool classic(
        const Runtime& runtime,
        const movement::Stream<Accessor>& stream,
        uint32_t buffer,
        movement::Busy& busy
    ) {
        const uint32_t pages = stream.tiles_per_block;
        const uint32_t blocks = stream.tiles / pages * stream.repetitions;
        const bool read = stream.direction == movement::kRead;
        for (uint32_t block = 0; block < blocks; block++) {
            if (!wait_buffer(runtime, buffer, pages, read)) {
                return false;
            }
            const uint32_t start = km::wall();
            uint32_t page = stream.first_page + block % (stream.tiles / pages) * pages;
            if (read) {
                cb_reserve_back(buffer, pages);
                uint32_t address = get_write_ptr(buffer);
                for (uint32_t tile = 0; tile < pages; tile++) {
                    noc_async_read_page(page + tile, stream.tensor, address);
                    address += kTileBytes;
                }
            } else {
                cb_wait_front(buffer, pages);
                uint32_t address = get_read_ptr(buffer);
                for (uint32_t tile = 0; tile < pages; tile++) {
                    noc_async_write_page(page + tile, stream.tensor, address);
                    address += kTileBytes;
                }
            }
            if (!movement::settle(runtime, kGaveUpWord, read, !read)) {
                return false;
            }
            if (read) {
                cb_push_back(buffer, pages);
            } else {
                cb_pop_front(buffer, pages);
            }
            busy.tiles += pages;
            busy.cycles += km::wall() - start;
        }
        if (read) {
            return true;
        }
        const uint32_t start = km::wall();
        const bool settled = movement::settle(runtime, kGaveUpWord, false, true, true);
        busy.cycles += km::wall() - start;
        return settled;
    }

    // Fills this mover's whole half with stamps: the page `position` pages after the
    // read pointer holds salt << 20 | mover << 19 | position << 9 | word.
    void stamp(uint32_t buffer, uint32_t salt) {
        const LocalCBInterface& ring = get_local_cb_interface(buffer);
        for (uint32_t position = 0; position < ring.fifo_num_pages; position++) {
            uint32_t address = ring.fifo_rd_ptr + position * kTileBytes;
            if (address >= ring.fifo_limit) {
                address -= ring.fifo_size;
            }
            volatile tt_l1_ptr uint32_t* words = km::l1(address);
            const uint32_t head = salt << 20 | movement::kMover << 19 | position << 9;
            for (uint32_t word = 0; word < kTileBytes / 4; word++) {
                words[word] = head | word;
            }
        }
    }

    // RUN on this mover. False: a wait gave up.
    template <typename Arguments>
    bool run(
        const Runtime& runtime,
        const km_command_t& command,
        const Arguments& arguments,
        volatile tt_l1_ptr uint32_t* profile
    ) {
        const uint32_t input = command.a0;
        const uint32_t output = command.a1;
        const uint32_t tiles = command.a2;
        const uint32_t tiles_per_block = command.a3;
        const uint32_t pattern_word = command.a4;
        const uint32_t repetitions = command.a5;
        const uint32_t program = command.a6;
        const uint32_t layout = command.a7;
        const uint32_t salt = command.a8;
        const uint32_t rank = runtime.me() - km::first(command);
        const uint32_t first_page = rank * tiles;
        // A buffer keeps its filling and draining CPUs for the whole load (compute
        // counts pages itself), so copies use halves of their own.
        const movement::Pair pair = program == kProgramCopy
            ? movement::Pair{kBufferCopyA, kBufferCopyB}
            : movement::Pair{kBufferA, kBufferB};
        const movement::Pattern pattern = movement::pattern_of(pattern_word);
        using Accessor = decltype(TensorAccessor(arguments, 0));
        const movement::Stream<Accessor> read = {
            movement::kRead,
            TensorAccessor(arguments, input),
            first_page,
            tiles,
            tiles_per_block,
            repetitions,
            pair,
            pattern
        };
        const movement::Stream<Accessor> write = {
            movement::kWrite,
            TensorAccessor(arguments, output),
            first_page,
            tiles,
            tiles_per_block,
            repetitions,
            pair,
            pattern
        };

        movement::Busy busy = {0, 0};
        const uint32_t start = km::wall();
        bool done = true;
        if (layout == kLayoutClassic) {
            // A classic copy passes through a buffer the reader fills, the writer
            // drains.
            const bool reader = movement::kMover == movement::kMoverA;
            if (program == kProgramRead && reader) {
                done = classic(runtime, read, kBufferA, busy);
            } else if (program == kProgramWrite && !reader) {
                if (salt != 0) {
                    stamp(kBufferB, salt);
                }
                done = classic(runtime, write, kBufferB, busy);
            } else if (program == kProgramCopy) {
                done = (
                    reader
                        ? classic(runtime, read, kBufferClassicCopy, busy)
                        : classic(runtime, write, kBufferClassicCopy, busy)
                );
            }
        } else if (program == kProgramRead) {
            const movement::Stream<Accessor> streams[1] = {read};
            done = movement::run(runtime, kGaveUpWord, streams, busy);
        } else if (program == kProgramWrite) {
            if (salt != 0) {
                stamp(movement::half(pair, movement::kMover), salt);
            }
            const movement::Stream<Accessor> streams[1] = {write};
            done = movement::run(runtime, kGaveUpWord, streams, busy);
        } else {
            const movement::Stream<Accessor> streams[2] = {read, write};
            done = movement::run(runtime, kGaveUpWord, streams, busy);
        }
        profile[kProfileMover + kProfileStart] = start;
        profile[kProfileMover + kProfileEnd] = km::wall();
        profile[kProfileMover + kProfileTiles] = busy.tiles;
        profile[kProfileMover + kProfileBusy] = busy.cycles;
        return done;
    }

}  // namespace

void kernel_main() {
    constexpr auto arguments = TensorAccessorArgs<0>();  // every tensor: 2048-B pages
    Runtime runtime;
    volatile tt_l1_ptr uint32_t* profile = km::l1(get_write_ptr(kBufferProfile));
    for (uint32_t word = 0; word < kProfileWords; word++) {
        profile[word] = 0;
    }
    runtime.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler == kHandlerRun) {
            return run(runtime, command, arguments, profile);
        }
        if (handler != kHandlerProfile || movement::kMover != movement::kMoverB) {
            return true;
        }
        const uint32_t profile_tensor = command.a0;
        const InterleavedAddrGen<true> pages = {
            .bank_base_address = profile_tensor, .page_size = kProfileBytes
        };
        noc_async_write(
            get_write_ptr(kBufferProfile),
            get_noc_addr(runtime.me(), pages),
            kProfileBytes
        );
        return movement::settle(runtime, kGaveUpWord, false, true);
    });
}
