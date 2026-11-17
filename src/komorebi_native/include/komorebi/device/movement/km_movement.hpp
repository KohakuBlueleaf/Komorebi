// Both data-movement CPUs of a core (the movers) carrying one stream: a pattern names
// the mover of each block, and each mover moves its blocks through its half of a pair.
#pragma once

#include <stdint.h>

#include <type_traits>

#include "../core/km_core.hpp"
#include "../km_worker.hpp"

namespace km {

    // Mover A is RISCV_0 (the reader kernel's CPU), mover B is RISCV_1 (the writer
    // kernel's); each on the NoC its program gives it (NoC 0 and NoC 1 by default).
    constexpr uint32_t kMoverA = 0;
    constexpr uint32_t kMoverB = 1;
    constexpr uint32_t kPatternLengthMax = 24;

    // Which mover owns block k: a length (1 to 24) and one bit per position, set when
    // mover B owns it. The word: the length in the low byte, the bits from bit 8.
    struct Pattern {
        uint32_t length;
        uint32_t owners;

        static constexpr Pattern of(uint32_t word) {
            return {word & 0xFFu, word >> 8};
        }

        constexpr uint32_t word() const {
            return length | owners << 8;
        }

        constexpr bool valid() const {
            return (
                length >= 1 && length <= kPatternLengthMax && (owners >> length) == 0
            );
        }

        constexpr uint32_t owner(uint32_t block) const {
            return (owners >> (block % length)) & 1u;
        }
    };

    // The pattern of a word of letters A and B ("AAAB"); a pattern that is not valid()
    // for any other text.
    constexpr Pattern pattern(const char* letters) {
        Pattern made = {0, 0};
        for (; letters[made.length] != 0; made.length++) {
            const char letter = letters[made.length];
            if (made.length == kPatternLengthMax || (letter != 'A' && letter != 'B')) {
                return {0, 0};
            }
            made.owners |= (letter == 'B' ? 1u : 0u) << made.length;
        }
        return made;
    }

    constexpr Pattern kPatternA = pattern("A");
    constexpr Pattern kPatternB = pattern("B");
    constexpr Pattern kPatternAB = pattern("AB");
    constexpr Pattern kPatternAAAB = pattern("AAAB");
    constexpr Pattern kPatternBBBBBAAA = pattern("BBBBBAAA");

    // A logical buffer as two circular buffers: mover A's half and mover B's half. Each
    // half has one filling CPU and one draining CPU for the whole life of a load.
    struct BufferPair {
        uint32_t mover_a;
        uint32_t mover_b;

        constexpr uint32_t half(uint32_t mover) const {
            return mover == kMoverA ? mover_a : mover_b;
        }
    };

    // Compute: the half that holds block `block` of a stream under `pattern`. Compute
    // takes the blocks in order, each a whole item of the half.
    constexpr uint32_t block_half(
        const BufferPair& pair, const Pattern& pattern, uint32_t block
    ) {
        return pair.half(pattern.owner(block));
    }

    // The blocks before block `block` of a stream that half `owner` holds under
    // `pattern` (a half's ring takes its blocks in order).
    constexpr uint32_t blocks_before(
        const Pattern& pattern, uint32_t owner, uint32_t block
    ) {
        const uint32_t all = (1u << pattern.length) - 1u;
        const uint32_t mine = owner != 0 ? pattern.owners : ~pattern.owners & all;
        const uint32_t part = mine & ((1u << (block % pattern.length)) - 1u);
        return (
            block / pattern.length * __builtin_popcount(mine) + __builtin_popcount(part)
        );
    }

}  // namespace km

#if !defined(COMPILE_FOR_TRISC)
#include "../tensors/km_ask.hpp"

namespace km {

#if defined(COMPILE_FOR_BRISC)
    constexpr uint32_t kMover = kMoverA;
    constexpr uint32_t kStMover = kStReader;
    using MoverRuntime = Reader;
#else
    constexpr uint32_t kMover = kMoverB;
    constexpr uint32_t kStMover = kStWriter;
    using MoverRuntime = Writer;
#endif
    static_assert(noc_mode != DM_DYNAMIC_NOC, "the movers use one NoC each");

    // A read moves tensor pages into this mover's half; a write moves pages of its
    // half to the tensor.
    constexpr uint32_t kMoveRead = 0;
    constexpr uint32_t kMoveWrite = 1;

    // The blocks of a stream, a two-level strided walk: block b starts at tensor page
    // first_page + b * block_step and holds `rows` rows of `columns` pages.
    struct Walk {
        uint32_t first_page;
        uint32_t blocks;       // blocks in one repetition
        uint32_t block_step;   // tensor pages from one block's first page to the next
        uint32_t rows;         // rows of a block
        uint32_t row_step;     // tensor pages from one row to the next
        uint32_t columns;      // pages of a row
        uint32_t column_step;  // tensor pages from one page of a row to the next
        uint32_t row_pitch;    // L1 pages from one row of a block to the next
        uint32_t tiles_per_block;  // the item: pages taken and handed on per block
        uint32_t last_rows;        // the last block of a repetition: rows and columns
        uint32_t last_columns;     // moved (a ragged end); the item stays whole

        // `tiles` consecutive pages from `first_page` in blocks of `tiles_per_block`;
        // the last block holds the rest.
        static Walk run(uint32_t first_page, uint32_t tiles, uint32_t tiles_per_block) {
            const uint32_t blocks = (
                tiles_per_block == 0
                    ? 0
                    : (tiles + tiles_per_block - 1) / tiles_per_block
            );
            return {
                first_page,
                blocks,
                tiles_per_block,
                1,
                0,
                tiles_per_block,
                1,
                tiles_per_block,
                tiles_per_block,
                1,
                blocks == 0 ? 0 : tiles - (blocks - 1) * tiles_per_block
            };
        }
    };

    // What a stream keeps of a tensor: a km::Tensor whose island holds every page, its
    // accessor (one island: the stream is the plain one); else the tensor itself.
    template <typename T>
    struct StreamTensorOf {
        using Type = T;
    };

    template <uint32_t Base>
    struct StreamTensorOf<Tensor<Base>> {
        using Type = std::conditional_t<
            Tensor<Base>::kLocal,
            typename Tensor<Base>::Accessor,
            Tensor<Base>>;
    };

    template <typename T>
    using StreamTensor = typename StreamTensorOf<T>::Type;

    // A stream's tensor whose DRAM page holds K tiles along a row: the walk stays in
    // tiles, and a block's tiles are read (or written) one NoC call a page.
    template <uint32_t K, typename T>
    struct PagesOf {
        T tensor;
    };

    template <typename T>
    constexpr uint32_t kTilesAPage = 1;
    template <uint32_t K, typename T>
    constexpr uint32_t kTilesAPage<PagesOf<K, T>> = K;

    template <uint32_t K, typename T>
    inline auto pages_of(const T& tensor) {
        using Kept = StreamTensor<T>;
        if constexpr (std::is_same_v<Kept, T>) {
            return PagesOf<K, Kept>{tensor};
        } else {
            return PagesOf<K, Kept>{tensor.accessor};
        }
    }

    // The tensor a stream's pages are read from: a PagesOf's own, else itself.
    template <typename T>
    inline const auto& paged_tensor(const T& tensor) {
        if constexpr (kTilesAPage<T> > 1) {
            return tensor.tensor;
        } else {
            return tensor;
        }
    }

    // A cast's read of tiles first .. of a tensor in pages of K tiles: its pages
    // first / K .., each K tiles at `to`. A split page gives up.
    template <typename Kernel, uint32_t K, typename T>
    inline bool read_pages(
        const Kernel& core,
        const km_command_t& command,
        const PagesOf<K, T>& paged,
        uint32_t first,
        uint32_t pages,
        uint32_t to
    ) {
        if (first % K != 0 || pages % K != 0) {
            core.gave_up(
                core.role() == kRoleWriter ? kStWriter : kStReader,
                kGaveBadItem,
                kWherePageRange,
                slot(command) | (first & 0xFFu) << 8,
                command.sequence
            );
            return false;  // KM_GUARD
        }
        return read_pages(core, command, paged.tensor, first / K, pages / K, to);
    }

    // Whether a walk of tiles is whole pages of K tiles: rows of whole pages, K apart
    // in the tensor, blocks of whole pages.
    template <uint32_t K>
    inline bool walk_paged(const Walk& walk) {
        return (
            walk.first_page % K == 0 && walk.block_step % K == 0 &&
            walk.row_step % K == 0 && walk.columns % K == 0 &&
            walk.last_columns % K == 0 && walk.column_step == 1 &&
            walk.tiles_per_block % K == 0
        );
    }

    // A read stream's tensor: `placed`, its blocks already where the halves' rings
    // take them (km_chain.hpp InPlaceLanding), no read issued; else read `tensor`.
    template <typename Accessor>
    struct InPlace {
        Accessor tensor;
        bool placed;

        uint64_t get_noc_addr(
            uint32_t id, uint32_t offset = 0, uint8_t noc = noc_index
        ) const {
            return tensor_accessor::detail::transfer_noc_addr(tensor, id, offset, noc);
        }

        uint32_t get_aligned_page_size() const {
            return tensor.get_aligned_page_size();
        }
    };

    template <typename T>
    constexpr bool kInPlace = false;
    template <typename Accessor>
    constexpr bool kInPlace<InPlace<Accessor>> = true;

    template <typename T>
    inline auto in_place(const T& tensor, bool placed) {
        if constexpr (kIsTensor<T>) {
            static_assert(T::kLocal, "an in-place input reads one island's tensor");
        }
        using Accessor = std::decay_t<decltype(accessor_of(tensor))>;
        return InPlace<Accessor>{accessor_of(tensor), placed};
    }

    template <typename Kept, typename T>
    inline const Kept& kept_tensor(const T& tensor) {
        if constexpr (std::is_same_v<Kept, T>) {
            return tensor;
        } else {
            return tensor.accessor;
        }
    }

    // Items of `page_size` bytes packed in a tensor's pages, `per` a page: item i is at
    // page i / per, byte (i % per) * page_size. A stream of it moves items as pages.
    template <typename Accessor>
    struct Items {
        Accessor accessor;
        uint32_t page_size;
        uint32_t per;

        uint64_t get_noc_addr(
            uint32_t id, uint32_t offset = 0, uint8_t noc = noc_index
        ) const {
            return (
                tensor_accessor::detail::transfer_noc_addr(
                    accessor, id / per, (id % per) * page_size + offset, noc
                )
            );
        }
    };

    // The items of a tensor whose island holds every page (an items stream does not
    // cross islands).
    template <typename T>
    inline auto items(const T& tensor, uint32_t size, uint32_t per) {
        if constexpr (kIsTensor<T>) {
            static_assert(T::kLocal, "an items stream reads one island's tensor");
        }
        using Accessor = std::decay_t<decltype(accessor_of(tensor))>;
        return Items<Accessor>{accessor_of(tensor), size, per};
    }

    // One stream: direction, tensor (a km::Tensor or a local accessor), blocks,
    // repetitions, pair, pattern.
    template <typename Accessor>
    struct Stream {
        uint32_t direction;
        Accessor tensor;
        Walk walk;
        uint32_t repetitions;
        BufferPair pair;
        Pattern pattern;

        template <typename T>
        Stream(
            uint32_t direction,
            const T& tensor,
            const Walk& walk,
            uint32_t repetitions,
            BufferPair pair,
            Pattern pattern
        ) :
            direction(direction),
            tensor(kept_tensor<Accessor>(tensor)),
            walk(walk),
            repetitions(repetitions),
            pair(pair),
            pattern(pattern) {
        }

        // The plain case: `tiles` pages from `first_page`, `repetitions` times.
        template <typename T>
        Stream(
            uint32_t direction,
            const T& tensor,
            uint32_t first_page,
            uint32_t tiles,
            uint32_t tiles_per_block,
            uint32_t repetitions,
            BufferPair pair,
            Pattern pattern
        ) :
            Stream(
                direction,
                tensor,
                Walk::run(first_page, tiles, tiles_per_block),
                repetitions,
                pair,
                pattern
            ) {
        }
    };

    template <typename T, typename... Rest>
    Stream(uint32_t, const T&, Rest...) -> Stream<StreamTensor<T>>;

    // What one mover spent in a move: tiles moved, cycles in passes that moved a block.
    struct Moved {
        uint32_t tiles;
        uint32_t cycles;
    };

    // Halves this mover filled and drained in this load, one bit per buffer index: a
    // half whose roles change gives up (compute's page counts would no longer agree).
#if defined(KM_OVERLAY_PART)
    // A part: the kernel's words (linked to its symbols) if it moves, else its own.
    [[gnu::weak]] extern uint32_t move_filled;
    [[gnu::weak]] extern uint32_t move_drained;

    inline uint32_t& move_roles(uint32_t* kernel, uint32_t& own) {
        return kernel != nullptr ? *kernel : own;
    }

    inline uint32_t& move_filled_word() {
        static uint32_t own = 0;
        return move_roles(&move_filled, own);
    }

    inline uint32_t& move_drained_word() {
        static uint32_t own = 0;
        return move_roles(&move_drained, own);
    }
#else
    inline uint32_t move_filled = 0;
    inline uint32_t move_drained = 0;

    inline uint32_t& move_filled_word() {
        return move_filled;
    }

    inline uint32_t& move_drained_word() {
        return move_drained;
    }
#endif

    // Where a mover is in a stream: the next block, that block's index in its
    // repetition, its position in the pattern and its first page.
    struct MoveCursor {
        uint32_t block;
        uint32_t in_repetition;
        uint32_t position;
        uint32_t page;
        uint32_t blocks;  // blocks of the whole stream
    };

    template <typename Accessor>
    inline void move_step(MoveCursor& cursor, const Stream<Accessor>& stream) {
        cursor.block++;
        cursor.position = (
            cursor.position + 1 == stream.pattern.length ? 0 : cursor.position + 1
        );
        cursor.in_repetition++;
        cursor.page += stream.walk.block_step;
        if (cursor.in_repetition == stream.walk.blocks) {
            cursor.in_repetition = 0;
            cursor.page = stream.walk.first_page;
        }
    }

    // Steps to the first block from the cursor on that this mover owns.
    template <typename Accessor>
    inline void move_seek(MoveCursor& cursor, const Stream<Accessor>& stream) {
        while (
            cursor.block < cursor.blocks &&
            ((stream.pattern.owners >> cursor.position) & 1u) != kMover
        ) {
            move_step(cursor, stream);
        }
    }

    // Checks a stream against its half and adds the half to the filled or drained
    // bits. False: a bad pattern or walk, or an item the half does not divide.
    template <typename Accessor>
    inline bool move_begin(
        const Stream<Accessor>& stream,
        MoveCursor& cursor,
        uint32_t& filled,
        uint32_t& drained
    ) {
        const Walk& walk = stream.walk;
        const uint32_t buffer = stream.pair.half(kMover);
        if (buffer >= 32) {
            return false;
        }
        const uint32_t bit = 1u << buffer;
        const LocalCBInterface& ring = get_local_cb_interface(buffer);
        const bool shaped = (
            walk.tiles_per_block != 0 && walk.rows != 0 && walk.last_rows != 0 &&
            walk.columns <= walk.row_pitch && walk.last_columns <= walk.row_pitch &&
            (walk.rows - 1) * walk.row_pitch + walk.columns <= walk.tiles_per_block &&
            (walk.last_rows - 1) * walk.row_pitch + walk.last_columns <=
                walk.tiles_per_block
        );
        const bool read = stream.direction == kMoveRead;
        if (
            !stream.pattern.valid() || !shaped ||
            (ring.fifo_num_pages % walk.tiles_per_block != 0 && !padded_ring(buffer))
        ) {
            return false;
        }
        if constexpr (kTilesAPage<Accessor> > 1) {  // a page is never split
            if (!walk_paged<kTilesAPage<Accessor>>(walk)) {
                return false;
            }
        }
        (read ? filled : drained) |= bit;
        cursor = {0, 0, 0, walk.first_page, walk.blocks * stream.repetitions};
        move_seek(cursor, stream);
        return true;
    }

    // Issues the cursor's block at this mover's half (a tensor that may cross: a row at
    // a time through km::read_pages, km::write_pages). False: gave up (recorded).
    template <typename Accessor>
    inline bool move_issue(
        const Core& core, const Stream<Accessor>& stream, const MoveCursor& cursor
    ) {
        const Walk& walk = stream.walk;
        const uint32_t buffer = stream.pair.half(kMover);
        const uint32_t page_bytes = get_local_cb_interface(buffer).fifo_page_size;
        const bool last = cursor.in_repetition + 1 == walk.blocks;
        const uint32_t rows = last ? walk.last_rows : walk.rows;
        const uint32_t columns = last ? walk.last_columns : walk.columns;
        const bool read = stream.direction == kMoveRead;
        const uint32_t base = read ? get_write_ptr(buffer) : get_read_ptr(buffer);
        if constexpr (kInPlace<Accessor>) {  // placed: the block where the ring is
            if (stream.tensor.placed) {
                const LocalCBInterface& ring = get_local_cb_interface(buffer);
                const uint32_t at = (
                    ring.fifo_limit - ring.fifo_size +
                    blocks_before(stream.pattern, kMover, cursor.block) *
                        walk.tiles_per_block * page_bytes
                );
                if (base != at || stream.repetitions != 1) {
                    core.gave_up(kStMover, kGaveBadItem, kWhereData, 0xC5u, 0);
                    return false;
                }
                return true;
            }
        }
        constexpr uint32_t kTiles = kTilesAPage<Accessor>;  // tiles a tensor page
        const auto& paged = paged_tensor(stream.tensor);
        using Paged = std::decay_t<decltype(paged)>;
        if constexpr (kMayCross<Paged>) {
            const uint32_t slot = core.running_slot();
            const uint32_t sequence = core.running_sequence();
            // a row of consecutive pages is one call; else one call a page
            const uint32_t run = walk.column_step == 1 ? columns : 1;
            for (uint32_t row = 0; row < rows; row++) {
                uint32_t page = cursor.page + row * walk.row_step;
                uint32_t address = base + row * walk.row_pitch * page_bytes;
                for (uint32_t column = 0; column < columns; column += run) {
                    if (!(read ? read_pages_at(
                        core,
                        slot,
                        sequence,
                        paged,
                        page / kTiles,
                        run / kTiles,
                        address
                    )
                               : write_pages_at(
                                   core,
                                   slot,
                                   sequence,
                                   paged,
                                   page / kTiles,
                                   run / kTiles,
                                   address
                               ))) {
                        return false;
                    }
                    page += run * walk.column_step;
                    address += run * page_bytes;
                }
            }
            return true;
        } else {
            const auto& tensor = accessor_of(paged);
            for (uint32_t row = 0; row < rows; row++) {
                uint32_t page = cursor.page + row * walk.row_step;
                uint32_t address = base + row * walk.row_pitch * page_bytes;
                for (uint32_t column = 0; column < columns; column += kTiles) {
                    if (read) {
                        noc_async_read_page(page / kTiles, tensor, address);
                    } else {
                        noc_async_write_page(page / kTiles, tensor, address);
                    }
                    page += kTiles * walk.column_step;
                    address += kTiles * page_bytes;
                }
            }
            return true;
        }
    }

    // One pass over a stream: if this mover has a block left and its half has room
    // (reads) or data (writes), takes the item and issues the block. False: gave up.
    template <typename Accessor>
    inline bool move_offer(
        const Core& core,
        const Stream<Accessor>& stream,
        const MoveCursor& cursor,
        bool& issued,
        bool& open,
        bool& reads,
        bool& writes
    ) {
        if (cursor.block >= cursor.blocks) {
            return true;
        }
        open = true;
        const uint32_t buffer = stream.pair.half(kMover);
        const uint32_t pages = stream.walk.tiles_per_block;
        const LocalCBInterface& ring = get_local_cb_interface(buffer);
        const bool read = stream.direction == kMoveRead;
        const uint32_t pointer = read ? ring.fifo_wr_ptr : ring.fifo_rd_ptr;
#if defined(KM_RING_PADDED)
        const uint32_t pad = ring_pad(buffer, pointer, pages);
#else
        constexpr uint32_t pad = 0;
#endif
        if (pad == 0 && !item_fits(buffer, pointer, pages)) {
            return false;
        }
        if (read) {
#if defined(KM_RING_PADDED)
            if (pad != 0) {  // the pages to the limit: padding the consumer pops
                if (!cb_pages_reservable_at_back(buffer, pad)) {
                    return true;
                }
                cb_reserve_back(buffer, pad);
                cb_push_back(buffer, pad);
            }
#endif
            if (!cb_pages_reservable_at_back(buffer, pages)) {
                return true;
            }
            cb_reserve_back(buffer, pages);
            reads = true;
        } else {
#if defined(KM_RING_PADDED)
            if (pad != 0) {  // the producer's padding to the limit
                if (!cb_pages_available_at_front(buffer, pad)) {
                    return true;
                }
                cb_wait_front(buffer, pad);
                cb_pop_front(buffer, pad);
            }
#endif
            if (!cb_pages_available_at_front(buffer, pages)) {
                return true;
            }
            cb_wait_front(buffer, pages);
            writes = true;
        }
        issued = true;
        return move_issue(core, stream, cursor);
    }

    // After the barrier: push (reads) or pop (writes) an issued block, go to the next.
    template <typename Accessor>
    inline void move_hand_on(
        const Stream<Accessor>& stream, MoveCursor& cursor, bool& issued, Moved& moved
    ) {
        if (!issued) {
            return;
        }
        const uint32_t buffer = stream.pair.half(kMover);
        const uint32_t pages = stream.walk.tiles_per_block;
        if (stream.direction == kMoveRead) {
            cb_push_back(buffer, pages);
        } else {
            cb_pop_front(buffer, pages);
        }
        moved.tiles += pages;
        move_step(cursor, stream);
        move_seek(cursor, stream);
        issued = false;
    }

    // A read stream held while `gate.holds(bits)` (km_chain.hpp's ChainGate: a record's
    // input), the move's other streams going on; `gate.give_up` records a give-up then.
    template <typename S, typename Gate>
    struct Gated {
        const S& stream;
        Gate& gate;
        uint32_t bits;  // the stream's tensors, for the gate
        uint32_t direction;

        Gated(const S& stream, Gate& gate, uint32_t bits) :
            stream(stream), gate(gate), bits(bits), direction(stream.direction) {
        }
    };

    template <typename S, typename Gate>
    Gated(const S&, Gate&, uint32_t) -> Gated<S, Gate>;

    template <typename T>
    inline constexpr bool kGated = false;

    template <typename S, typename Gate>
    inline constexpr bool kGated<Gated<S, Gate>> = true;

    template <typename S, typename Gate>
    inline bool move_begin(
        const Gated<S, Gate>& gated,
        MoveCursor& cursor,
        uint32_t& filled,
        uint32_t& drained
    ) {
        return move_begin(gated.stream, cursor, filled, drained);
    }

    // move_offer of a gated stream: nothing offered while its gate holds.
    template <typename S, typename Gate>
    inline bool move_offer(
        const Core& core,
        const Gated<S, Gate>& gated,
        const MoveCursor& cursor,
        bool& issued,
        bool& open,
        bool& reads,
        bool& writes
    ) {
        if (cursor.block < cursor.blocks && gated.gate.holds(gated.bits)) {
            open = true;
            return true;
        }
        return move_offer(core, gated.stream, cursor, issued, open, reads, writes);
    }

    template <typename S, typename Gate>
    inline void move_hand_on(
        const Gated<S, Gate>& gated, MoveCursor& cursor, bool& issued, Moved& moved
    ) {
        move_hand_on(gated.stream, cursor, issued, moved);
    }

    // A give-up met in a move: recorded by a gate that still holds, else false.
    template <typename T>
    inline bool move_held_gave_up(const Core&, const T&, uint32_t) {
        return false;
    }

    template <typename S, typename Gate>
    inline bool move_held_gave_up(
        const Core& core, const Gated<S, Gate>& gated, uint32_t code
    ) {
        if (!gated.gate.holds(gated.bits)) {
            return false;
        }
        gated.gate.give_up(core, code);
        return true;
    }

    // Timed wait until this mover's reads have landed and its writes have left L1
    // (`acknowledged`: until they are acknowledged). False: gave up (recorded).
    inline bool move_settle(
        const Core& core, bool reads, bool writes, bool acknowledged = false
    ) {
        const uint32_t code = core.poll_at(
            kWhereData,
            [&] {
                const bool read_done = !reads || ncrisc_noc_reads_flushed(noc_index);
                if (!writes) {
                    return read_done;
                }
                return (
                    read_done &&
                    (
                        acknowledged
                            ? ncrisc_noc_nonposted_writes_flushed(noc_index)
                            : ncrisc_noc_nonposted_writes_sent(noc_index)
                    )
                );
            },
            core.args().wait_limit,
            core.args().wait_limit
        );
        if (code != kGaveNone) {
            core.gave_up(kStMover, code, kWhereData, 0, 0);
        }
        return code == kGaveNone;
    }

#if KM_GUARD
    // Whether `T` is a bank-addressed accessor (a tensor of one island's banks).
    template <typename T, typename = decltype(&T::get_bank_base_address)>
    constexpr bool walk_banked(int) {
        return true;
    }

    template <typename T>
    constexpr bool walk_banked(...) {
        return false;
    }

    // page_range_ok of a walk, first page to last (steps never go back), over a
    // banked tensor at `word` of `bytes` pages (a crossing one: the page calls').
    __attribute__((noinline)) inline bool walk_inside(
        const Core& core, const Walk& walk, uint32_t word, uint32_t bytes
    ) {
        if (walk.blocks == 0) {
            return true;
        }
        uint32_t last = (
            (walk.blocks - 1u) * walk.block_step +
            (walk.last_rows - 1u) * walk.row_step +
            (walk.last_columns - 1u) * walk.column_step
        );
        if (walk.blocks > 1u) {
            const uint32_t full = (
                (walk.blocks - 2u) * walk.block_step +
                (walk.rows - 1u) * walk.row_step +
                (walk.columns - 1u) * walk.column_step
            );
            last = full > last ? full : last;
        }
        return page_range_ok(core, word, walk.first_page, last + 1u, bytes);
    }

    template <typename Plain>
    inline bool walk_inside(const Core& core, const Walk& walk, const Plain& tensor) {
        if constexpr (walk_banked<Plain>(0)) {
            return (
                walk_inside(
                    core,
                    walk,
                    tensor.get_bank_base_address(),
                    tensor.get_aligned_page_size()
                )
            );
        }
        return true;
    }

    // walk_inside of a stream (one copy a kind of stream: a stream's call is short).
    template <typename Accessor>
    __attribute__((noinline)) bool move_inside(
        const Core& core, const Stream<Accessor>& stream
    ) {
        if constexpr (kInPlace<Accessor>) {
            return (
                stream.tensor.placed ||
                walk_inside(core, stream.walk, stream.tensor.tensor)
            );
        } else if constexpr (kTilesAPage<Accessor> > 1) {
            if constexpr (kMayCross<decltype(stream.tensor.tensor)>) {
                return true;
            } else {  // the walk in the tensor's pages
                constexpr uint32_t kTiles = kTilesAPage<Accessor>;
                Walk pages = stream.walk;
                pages.first_page /= kTiles;
                pages.block_step /= kTiles;
                pages.row_step /= kTiles;
                pages.columns /= kTiles;
                pages.last_columns /= kTiles;
                return walk_inside(core, pages, accessor_of(stream.tensor.tensor));
            }
        } else if constexpr (!kMayCross<Accessor>) {
            return walk_inside(core, stream.walk, accessor_of(stream.tensor));
        } else {
            return true;
        }
    }

    template <typename S, typename Gate>
    inline bool move_inside(const Core& core, const Gated<S, Gate>& gated) {
        return move_inside(core, gated.stream);
    }

    // Whether the dispatcher gave up the command this mover runs (its abort word): then
    // recorded, kGaveDeadline in kWhereMove.
    __attribute__((noinline)) inline bool move_given_up(const Core& core) {
        const uint32_t slot = core.running_slot();
        const uint32_t sequence = core.running_sequence();
        if (!given_up(core.args().control, slot, sequence)) {
            return false;
        }
        core.gave_up(kStMover, kGaveDeadline, kWhereMove, slot, sequence);
        return true;
    }
#endif

    // An idle pass of a move with a gated stream: stop, wait limit, abort; a give-up
    // recorded by a gate that holds (as chain_input's), else as move's. False: gave up.
    template <typename... Streams>
    __attribute__((noinline)) bool move_held_idle(
        const Core& core, uint64_t deadline, const Streams&... streams
    ) {
        uint32_t code = kGaveNone;
        if (core.stop()) {
            code = kGaveStop;
        } else if (wall64() > deadline) {
            code = kGaveLimit;
        }
#if KM_GUARD
        const uint32_t slot = core.running_slot();
        const uint32_t sequence = core.running_sequence();
        if (code == kGaveNone && given_up(core.args().control, slot, sequence)) {
            code = kGaveDeadline;
        }
#endif
        if (code == kGaveNone) {
            return true;
        }
        if (!(move_held_gave_up(core, streams, code) || ...)) {
            if (code == kGaveDeadline) {
                KM_GUARDED(core.gave_up(kStMover, code, kWhereMove, slot, sequence));
            } else {
                core.gave_up(kStMover, code, kWhereData, 0, 0);
            }
        }
        return false;
    }

    // The engine, on this mover: per pass, issue the next block of each stream whose
    // half has room or data (gated: open), one barrier, push or pop. False: gave up.
    template <typename... Streams>
    bool move(Moved& moved, const Core& core, const Streams&... streams) {
        constexpr uint32_t kCount = sizeof...(Streams);
        static_assert(kCount >= 1, "move needs a stream");
        MoveCursor cursors[kCount];
        bool issued[kCount] = {};
        bool fine = true, any_write = false;
        uint32_t index = 0, filled = 0, drained = 0;
        (
            (
                fine = fine && move_begin(streams, cursors[index], filled, drained),
                any_write = any_write || streams.direction == kMoveWrite,
                index++
            ),
            ...
        );
        // A half keeps the roles it had in this mover's first move of the load.
        uint32_t& kept_filled = move_filled_word();
        uint32_t& kept_drained = move_drained_word();
        const uint32_t known = (filled | drained) & (kept_filled | kept_drained);
        fine = (
            fine && (known & ((filled ^ kept_filled) | (drained ^ kept_drained))) == 0
        );
        kept_filled |= filled;
        kept_drained |= drained;
        if (!fine) {
            core.gave_up(kStMover, kGaveBadItem, kWhereData, 0, 0);
            return false;
        }
#if KM_GUARD
        if (!(move_inside(core, streams) && ...)) {  // a range past a tensor's end
            return false;
        }
#endif
        const uint64_t limit = static_cast<uint64_t>(core.args().wait_limit) << 10;
        uint64_t deadline = wall64() + limit;
#if KM_GUARD
        // shown live while it moves; its idle passes read the command's abort word
        volatile tt_l1_ptr uint32_t* live = core.control(KM_CONTROL_LIVE);
        const uint32_t before = live[kLvWhere + core.role()];
        live[kLvWhere + core.role()] = kWhereMove;
#endif
        for (uint32_t pass = 0;; pass++) {
            invalidate_l1_cache();
            bool open = false, reads = false, writes = false;
            const uint32_t start = wall();
            index = 0;
            (
                (
                    fine = fine &&
                        move_offer(
                            core,
                            streams,
                            cursors[index],
                            issued[index],
                            open,
                            reads,
                            writes
                        ),
                    index++
                ),
                ...
            );
            if (!fine) {
                core.gave_up(kStMover, kGaveBadItem, kWhereData, 0, 0);
                return false;
            }
            if (!open) {
                break;
            }
            if (!reads && !writes) {
                if constexpr ((kGated<Streams> || ...)) {  // a gate's: chain_input's
                    if (
                        (pass & 15) == 15 && !move_held_idle(core, deadline, streams...)
                    ) {
                        return false;
                    }
                    continue;
                }
                if ((pass & 15) == 15) {
                    if (core.stop()) {
                        core.gave_up(kStMover, kGaveStop, kWhereData, 0, 0);
                        return false;
                    }
                    if (wall64() > deadline) {
                        core.gave_up(kStMover, kGaveLimit, kWhereData, 0, 0);
                        return false;
                    }
                    KM_GUARDED(if (move_given_up(core)) { return false; });
                }
                continue;
            }
            if (!move_settle(core, reads, writes)) {
                return false;
            }
            index = 0;
            (
                (move_hand_on(streams, cursors[index], issued[index], moved), index++),
                ...
            );
            moved.cycles += wall() - start;
            deadline = wall64() + limit;
        }
        KM_GUARDED(live[kLvWhere + core.role()] = before);
        if (!any_write) {
            return true;
        }
        const uint32_t start = wall();
        const bool settled = move_settle(core, false, true, true);
        moved.cycles += wall() - start;
        return settled;
    }

    template <typename... Streams>
    bool move(const Core& core, const Streams&... streams) {
        Moved moved = {0, 0};
        return move(moved, core, streams...);
    }

    // A source compiled for both movers: on mover A the reader's runtime, on mover B
    // the writer's, plus this mover's share of streams. One body serves both CPUs.
    class Movement : public MoverRuntime {
    public:
        static constexpr uint32_t mover() {
            return kMover;
        }

        template <typename... Streams>
        bool move(const Streams&... streams) const {
            return km::move(*this, streams...);
        }

        template <typename... Streams>
        bool move(Moved& moved, const Streams&... streams) const {
            return km::move(moved, *this, streams...);
        }
    };

}  // namespace km
#endif
