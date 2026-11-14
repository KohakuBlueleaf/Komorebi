// Ray tracing reader: LOAD copies the tree image into the tree buffer; the other
// handlers send a header page (first page, count), then stream this core's pages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "timed.hpp"
#include "dir_noc.hpp"
#include "komorebi/device/km_worker.hpp"
#include "komorebi/device/km_ask.hpp"
#include "raytrace.h"

// The tree's and the treelets' pages, then the directory's (km::Tensor): compile
// arguments from 0.
using TreePages = km::Tensor<0>;
using Directories = km::Tensor<TreePages::next_compile_time_args_offset()>;

namespace {

    constexpr uint32_t kMaxSlots = 8;
    constexpr uint16_t kDone = 0x8000;  // a treelet's group of this round is submitted

    uint32_t g_tree = 0;  // L1 address of the tree (or top tree) image, after LOAD

    // Stores to L1 finished before a push: a load of the last word written, used.
    void settle(volatile tt_l1_ptr uint32_t* last_word) {
        const uint32_t value = *last_word;
        asm volatile("and x0, x0, %0" : : "r"(value) : "memory");
    }

    uint32_t in_page_bytes(uint32_t handler) {
        if (handler == kHandlerMath) {
            return kMaInVectors * kVec * 4;
        }
        return handler == kHandlerArith ? kArInVectors * kVec * 4 : kPacketBytes;
    }

    // Packet pages go kPagePackets to a DRAM page; the check pages one to a page.
    uint32_t packets_per_page(uint32_t handler) {
        return (
            handler == kHandlerArith || handler == kHandlerMath ||
                handler == kHandlerRecip
                ? 1
                : kPagePackets
        );
    }

    // Page `page` of a stream of pages holding `page_packets` items of `item_bytes`
    // each, into the input buffer, pushed; the wait for room is added to `wait`.
    bool read_page(
        km::Reader& reader,
        uint32_t base_address,
        uint32_t item_bytes,
        uint32_t page_packets,
        uint32_t page,
        uint32_t& wait
    ) {
        const uint32_t started = km::wall();
        if (!reader.reserve(kCbIn, 1)) {
            return false;
        }
        wait += km::wall() - started;
        const InterleavedAddrGen<true> addresses = {
            .bank_base_address = base_address, .page_size = item_bytes * page_packets
        };
        noc_async_read(
            get_noc_addr(
                page / page_packets, addresses, (page % page_packets) * item_bytes
            ),
            get_write_ptr(kCbIn),
            item_bytes
        );
        if (!timed::reads_done(reader, km::kStReader)) {
            return false;
        }
        cb_push_back(kCbIn, 1);
        return true;
    }

    // Treelets in L1 slots, least recently used replaced; a slot is free again once
    // compute has finished the last work page that names it.
    struct Slots {
        uint32_t base, bytes, count;
        uint32_t treelet[kMaxSlots];
        uint32_t last[kMaxSlots];  // work pages submitted when it was last named
        uint32_t used[kMaxSlots];  // last use
        uint32_t clock = 0;

        int32_t find(uint32_t wanted) const {
            for (uint32_t slot = 0; slot < count; ++slot) {
                if (treelet[slot] == wanted) {
                    return static_cast<int32_t>(slot);
                }
            }
            return -1;
        }

        uint32_t victim() const {
            uint32_t oldest = 0;
            for (uint32_t slot = 1; slot < count; ++slot) {
                oldest = used[slot] < used[oldest] ? slot : oldest;
            }
            return oldest;
        }
    };

    // A command's treelet schedule per batch: rounds (entry `round` of every list) or
    // once; packets binned by treelet, a treelet read into a slot unless present.
    class Schedule {
    public:
        Schedule(
            km::Reader& reader,
            const km_command_t& command,
            uint32_t packets,
            volatile tt_l1_ptr uint32_t* directory
        ) :
            reader_(reader),
            command_(command),
            treelet_images_address_(command.a5),
            list_entries_(command.a6),
            slot_bytes_(command.a7),
            slot_limit_(command.a8),
            policy_word_(command.a9),
            packets_(packets),
            directory_(directory) {
        }

        // Buffers and empty slots; false when the lists' scratch does not fit.
        bool init() {
            const uint32_t* top = reinterpret_cast<const uint32_t*>(g_tree);
            treelet_count_ = top[kTreeTreelets];
            const uint32_t treelets = treelet_count_;
            treelet_table_ = top + top[kTreeTableOff];
            lists_ = km::l1(get_write_ptr(kCbLists));
            uint32_t* counts = (
                reinterpret_cast<uint32_t*>(
                    get_write_ptr(kCbLists) + 4 * (kLsHead + 2 * packets_) +
                    2 * list_entries_
                )
            );
            counts_ = counts;
            present_ = reinterpret_cast<uint16_t*>(counts + treelets);
            order_ = present_ + treelets;
            const uint32_t most = list_entries_ > packets_ ? list_entries_ : packets_;
            const uint32_t end = reinterpret_cast<uint32_t>(order_ + most);
            if (!timed::fits(kCbLists, end - get_write_ptr(kCbLists))) {
                reader_.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 3, end);
                return false;
            }
            slots_.base = get_write_ptr(kCbSlots);
            slots_.bytes = slot_bytes_;
            slots_.count = slot_limit_ < kMaxSlots ? slot_limit_ : kMaxSlots;
            for (uint32_t slot = 0; slot < slots_.count; ++slot) {
                slots_.treelet[slot] = 0xFFFFFFFF;
                slots_.last[slot] = slots_.used[slot] = 0;
            }
            for (uint32_t treelet = 0; treelet < treelets; ++treelet) {
                counts[treelet] = 0;
            }
            return true;
        }

        // The current lists (one batch); then kLsMore: packets left for another batch.
        bool batch() {
            const volatile uint32_t* heads = lists_ + kLsHead;
            const volatile uint16_t* pool = (
                reinterpret_cast<const volatile uint16_t*>(
                    lists_ + kLsHead + 2 * packets_
                )
            );
            uint32_t* counts = counts_;
            uint16_t* present = present_;
            uint16_t* order = order_;
            // Rounds: entry `round` of every list in that round. Once: every entry in
            // one round.
            const bool once = (policy_word_ & 0xFF) == kPolicyOnce;
            uint32_t rounds = 0;
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                rounds = (
                    heads[2 * packet + 1] > rounds ? heads[2 * packet + 1] : rounds
                );
            }
            rounds = once && rounds > 0 ? 1 : rounds;
            directory_[kRdWords + kRdRounds] += rounds;
            for (uint32_t round = 0; round < rounds; ++round) {
                const uint32_t started = km::wall();
                uint32_t present_count = 0;
                for (uint32_t packet = 0; packet < packets_; ++packet) {
                    const uint32_t length = heads[2 * packet + 1];
                    const uint32_t first_entry = heads[2 * packet];
                    const uint32_t entry_end = (
                        once ? length : (length > round ? round + 1 : 0)
                    );
                    for (
                        uint32_t entry = once ? 0 : round; entry < entry_end; ++entry
                    ) {
                        const uint32_t treelet = pool[first_entry + entry];
                        if (counts[treelet]++ == 0) {
                            present[present_count++] = treelet;
                        }
                    }
                }
                uint32_t sum = 0;
                for (uint32_t index = 0; index < present_count; ++index) {
                    const uint32_t count = counts[present[index]];
                    counts[present[index]] = sum;
                    sum += count;
                }
                for (uint32_t packet = 0; packet < packets_; ++packet) {
                    const uint32_t length = heads[2 * packet + 1];
                    const uint32_t first_entry = heads[2 * packet];
                    const uint32_t entry_end = (
                        once ? length : (length > round ? round + 1 : 0)
                    );
                    for (
                        uint32_t entry = once ? 0 : round; entry < entry_end; ++entry
                    ) {
                        order[counts[pool[first_entry + entry]]++] = packet;
                    }
                }
                directory_[kRdWords + kRdBin] += km::wall() - started;
                // Treelets already in a slot first (marked done), then the others.
                for (uint32_t pass = 0; pass < 2; ++pass) {
                    uint32_t begin = 0;
                    for (uint32_t index = 0; index < present_count; ++index) {
                        const uint32_t treelet = present[index] & ~kDone;
                        const uint32_t end = counts[treelet];
                        const bool now = (
                            pass == 0
                                ? slots_.find(treelet) >= 0
                                : (present[index] & kDone) == 0
                        );
                        if (now && !group(treelet, order + begin, end - begin)) {
                            return false;
                        }
                        present[index] |= now ? kDone : 0;
                        begin = end;
                    }
                }
                for (uint32_t index = 0; index < present_count; ++index) {
                    counts[present[index] & ~kDone] = 0;
                }
            }
            return true;
        }

        // A batch's end page: kWorkEnd (after the counters are written: the writer
        // reads them next) or kWorkNext.
        bool finish(uint32_t marker) {
            if (!reader_.reserve(kCbWork, 1)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* page = km::l1(get_write_ptr(kCbWork));
            page[kWorkCount] = marker;
            settle(page + kWorkCount);
            cb_push_back(kCbWork, 1);
            return true;
        }

        bool more() const {
            return lists_[kLsMore] != 0;
        }

    private:
        bool group(uint32_t treelet, const uint16_t* ids, uint32_t packets) {
            int32_t slot = slots_.find(treelet);
            if (slot >= 0) {
                directory_[kRdWords + kRdHits]++;
            } else {
                slot = static_cast<int32_t>(slots_.victim());
                if (!wait_done(slots_.last[slot]) || !fetch(treelet, slot)) {
                    return false;
                }
            }
            for (uint32_t i = 0; i < packets; i += kWorkMax) {
                const uint32_t chunk = packets - i < kWorkMax ? packets - i : kWorkMax;
                const uint32_t started = km::wall();
                if (!reader_.reserve(kCbWork, 1)) {
                    return false;
                }
                directory_[kRdWords + kRdWorkWait] += km::wall() - started;
                const uint32_t page = get_write_ptr(kCbWork);
                volatile tt_l1_ptr uint32_t* work = km::l1(page);
                work[kWorkSlot] = slots_.base + slot * slots_.bytes;
                work[kWorkTreelet] = treelet;
                work[kWorkCount] = chunk;
                volatile tt_l1_ptr uint16_t*
                    ids_out = (
                        reinterpret_cast<volatile tt_l1_ptr uint16_t*>(work + kWorkIds)
                    );
                for (uint32_t j = 0; j < chunk; ++j) {
                    ids_out[j] = ids[i + j];
                }
                settle(work + kWorkIds + (chunk - 1) / 2);
                cb_push_back(kCbWork, 1);
                items_++;
                directory_[kRdWords + kRdItems] = items_;
                directory_[kRdWords + kRdVisits] += chunk;
            }
            slots_.last[slot] = items_;
            slots_.used[slot] = ++slots_.clock;
            directory_[kRdWords + kRdGroups]++;
            return true;
        }

        // Until compute has finished `need` work pages.
        bool wait_done(uint32_t need) {
            const uint32_t started = km::wall();
            const uint32_t code = reader_.poll(
                [&] {
                    return lists_[kLsDone] >= need;
                },
                reader_.args().wait_limit
            );
            directory_[kRdWords + kRdSlotWait] += km::wall() - started;
            if (code != km::kGaveNone) {
                reader_.gave_up(km::kStReader, code, km::kWhereData, 0, 0);
                return false;
            }
            return true;
        }

        bool fetch(uint32_t treelet, uint32_t slot) {
            const uint32_t started = km::wall();
            const uint32_t first_page = treelet_table_[2 * treelet];
            const uint32_t pages = treelet_table_[2 * treelet + 1];
            const uint32_t destination = slots_.base + slot * slots_.bytes;
            if (
                pages * kTreePageBytes > slots_.bytes ||
                !timed::fits(kCbSlots, (slot + 1) * slots_.bytes)
            ) {
                reader_.gave_up(
                    km::kStReader, km::kGaveLimit, km::kWhereData, 1, treelet
                );
                return false;
            }
            if (!km::read_pages(
                reader_,
                command_,
                TreePages(treelet_images_address_),
                first_page,
                pages,
                destination
            ) ||
                !timed::reads_done(reader_, km::kStReader)) {
                return false;
            }
            slots_.treelet[slot] = treelet;
            directory_[kRdWords + kRdFetch] += km::wall() - started;
            directory_[kRdWords + kRdReads]++;
            directory_[kRdWords + kRdPages] += pages;
            return true;
        }

        km::Reader& reader_;
        const km_command_t& command_;
        const uint32_t treelet_images_address_;
        const uint32_t list_entries_;
        const uint32_t slot_bytes_;
        const uint32_t slot_limit_;
        const uint32_t policy_word_;
        const uint32_t packets_;
        volatile tt_l1_ptr uint32_t* directory_;
        volatile tt_l1_ptr uint32_t* lists_ = nullptr;
        const uint32_t* treelet_table_ = nullptr;
        uint32_t treelet_count_ = 0;
        uint32_t* counts_ = nullptr;
        uint16_t* present_ = nullptr;
        uint16_t* order_ = nullptr;
        Slots slots_;
        uint32_t items_ = 0;
    };

    // SORT and UNSORT in L1 (the raw slot buffer), a vector of every packet at a time;
    // order[i] is the lane (packet * 32 + lane) that goes to place i.
    class Sorter {
    public:
        Sorter(
            km::Reader& reader,
            const km_command_t& command,
            uint32_t first_packet,
            uint32_t packets,
            volatile tt_l1_ptr uint32_t* directory,
            uint32_t unsort_vectors
        ) :
            reader_(reader),
            input_address_(command.a1),
            output_address_(command.a2),
            order_address_(command.a3),
            vectors_(unsort_vectors),
            corner_{command.a6, command.a7, command.a8},
            key_word_(command.a9),
            first_packet_(first_packet),
            packets_(packets),
            lanes_(packets * kVec),
            directory_(directory) {
        }

        // Key and order arrays, two vectors of every packet, the radix counts and the
        // per-packet signs: the bytes from the slot buffer's write pointer.
        static uint32_t scratch_bytes(uint32_t packets) {
            return (6 * packets * kVec + kSortHist + packets) * 4;
        }

        bool sort() {
            if (!layout()) {
                return false;
            }
            const uint32_t started = km::wall();
            for (uint32_t lane = 0; lane < lanes_; ++lane) {
                key_[lane] = 0;
                index_[lane] = lane;
            }
            const uint32_t page_bytes = kPacketBytes;
            // Keys: the octant from the direction's signs, the origin's Morton cell.
            if (!gather_in(input_address_, page_bytes, kPkBest)) {
                return false;
            }
            uint32_t* __restrict key = key_;
            const uint32_t* __restrict source = source_;
            const uint32_t lanes = lanes_;
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                key[lane] = source[lane] == kMinusOneBits ? kSortInvalid : 0;
            }
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if (!gather_in(input_address_, page_bytes, kPkD + axis)) {
                    return false;
                }
                const uint32_t bit = 3 * kSortCellBits + axis;
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    key[lane] |= (source[lane] >> 31) << bit;
                }
            }
            const uint32_t fraction_bits = key_word_ & 0xFF;
            const uint32_t shift = (key_word_ >> 8) & 0xFF;
            for (uint32_t axis = 0; axis < 3; ++axis) {
                if (!gather_in(input_address_, page_bytes, kPkO + axis)) {
                    return false;
                }
                const uint32_t low = corner_[axis];
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    key[lane] |= (
                        spread(cell(source[lane], fraction_bits, low, shift)) << axis
                    );
                }
            }
            for (uint32_t lane = 0; lane < lanes; ++lane) {
                key[lane] = (key[lane] & kSortInvalid) ? kSortInvalid : key[lane];
            }
            const uint32_t radix_started = km::wall();
            radix();
            directory_[kRdWords + 1] += km::wall() - radix_started;
            // The order pages, then every vector through the order; the header last.
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                write_lanes(
                    order_address_, kVec * 4, packet, 0, l1(order_ + packet * kVec)
                );
            }
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                signs_[packet] = 0;
            }
            const uint32_t vector_order[kPkHead] = {
                kPkBest,
                kPkD,
                kPkD + 1,
                kPkD + 2,
                kPkO,
                kPkO + 1,
                kPkO + 2,
                kPkInv,
                kPkInv + 1,
                kPkInv + 2
            };
            for (uint32_t step = 0; step < kPkHead; ++step) {
                const uint32_t vector = vector_order[step];
                if (!gather_in(input_address_, page_bytes, vector)) {
                    return false;
                }
                permute(destination_, source_, order_, lanes_, true);
                if (vector >= kPkD && vector < kPkD + 3) {
                    count_signs(vector - kPkD);
                }
                if (vector == kPkBest) {
                    for (uint32_t lane = 0; lane < lanes_; ++lane) {
                        valid_[lane] = destination_[lane] != kMinusOneBits;
                    }
                }
                if (!scatter_out(output_address_, page_bytes, vector)) {
                    return false;
                }
            }
            for (uint32_t lane = 0; lane < lanes_; ++lane) {
                destination_[lane] = 0;
            }
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                destination_[packet * kVec + kHeadSigns] = signs_[packet];
            }
            if (!scatter_out(output_address_, page_bytes, kPkHead)) {
                return false;
            }
            directory_[kRdWords + 2] = km::wall() - started;
            return true;
        }

        bool unsort() {
            if (!layout()) {
                return false;
            }
            const uint32_t started = km::wall();
            const uint32_t vectors = vectors_, page_bytes = vectors * kVec * 4;
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                read_lanes(
                    order_address_, kVec * 4, packet, 0, l1(order_ + packet * kVec)
                );
            }
            if (!timed::reads_done(reader_, km::kStReader)) {
                return false;
            }
            for (uint32_t lane = 0; lane < lanes_; ++lane) {
                if (order_[lane] >= lanes_) {
                    reader_.gave_up(
                        km::kStReader, km::kGaveLimit, km::kWhereData, 4, lane
                    );
                    return false;
                }
            }
            for (uint32_t vector = 0; vector < vectors; ++vector) {
                if (!gather_in(input_address_, page_bytes, vector)) {
                    return false;
                }
                permute(destination_, source_, order_, lanes_, false);
                if (!scatter_out(output_address_, page_bytes, vector)) {
                    return false;
                }
            }
            directory_[kRdWords + 2] = km::wall() - started;
            return true;
        }

    private:
        static constexpr uint32_t kSortHist = 1u << kSortDigitBits;
        static constexpr uint32_t kMinusOneBits = 0xBF800000;

        static uint32_t l1(const uint32_t* pointer) {
            return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(pointer));
        }

        bool layout() {
            if (!timed::fits(kCbSlots, scratch_bytes(packets_))) {
                reader_.gave_up(
                    km::kStReader, km::kGaveLimit, km::kWhereData, 3, packets_
                );
                return false;
            }
            uint32_t* scratch = reinterpret_cast<uint32_t*>(get_write_ptr(kCbSlots));
            key_ = scratch;
            order_ = key_ + lanes_;
            key_swap_ = order_ + lanes_;
            index_swap_ = key_swap_ + lanes_;
            source_ = index_swap_ + lanes_;
            destination_ = source_ + lanes_;
            histogram_ = destination_ + lanes_;
            signs_ = histogram_ + kSortHist;
            valid_ = reinterpret_cast<uint8_t*>(key_swap_);  // free after the sort
            index_ = order_;
            return true;
        }

        // The cell of a float coordinate: its fixed-point value (fraction_bits fraction
        // bits, truncated) less the low corner, shifted down, clamped to the grid.
        static uint32_t cell(
            uint32_t coordinate, uint32_t fraction_bits, uint32_t low, uint32_t shift
        ) {
            const int32_t exponent = static_cast<int32_t>((coordinate >> 23) & 0xFF);
            int32_t fixed = 0;
            if (exponent != 0) {
                const uint32_t mantissa = (coordinate & 0x7FFFFF) | 0x800000;
                const int32_t exponent_shift = (
                    exponent - 150 + static_cast<int32_t>(fraction_bits)
                );
                const uint32_t magnitude = (
                    exponent_shift >= 8
                        ? 0x7FFFFFFF
                        : exponent_shift >= 0
                            ? mantissa << exponent_shift
                            : exponent_shift > -32
                                ? mantissa >> -exponent_shift
                                : 0
                );
                fixed = (
                    (coordinate >> 31)
                        ? -static_cast<int32_t>(magnitude)
                        : static_cast<int32_t>(magnitude)
                );
            }
            const int32_t cell_index = (fixed - static_cast<int32_t>(low)) >> shift;
            const int32_t last_cell = (1 << kSortCellBits) - 1;
            return (
                static_cast<uint32_t>(
                    cell_index < 0
                        ? 0
                        : cell_index > last_cell
                            ? last_cell
                            : cell_index
                )
            );
        }

        // Bit `bit` of `value` (10 bits at most) to bit 3 * bit.
        static uint32_t spread(uint32_t value) {
            value &= 0x3FF;
            value = (value | value << 16) & 0x030000FF;
            value = (value | value << 8) & 0x0300F00F;
            value = (value | value << 4) & 0x030C30C3;
            return (value | value << 2) & 0x09249249;
        }

        // destination[i] = source[order[i]] (gather) or destination[order[i]] =
        // source[i], four at a time.
        static void permute(
            uint32_t* __restrict destination,
            const uint32_t* __restrict source,
            const uint32_t* __restrict order,
            uint32_t lanes,
            bool gather
        ) {
            uint32_t i = 0;
            if (gather) {
                for (; i + 4 <= lanes; i += 4) {
                    const uint32_t first = source[order[i]];
                    const uint32_t second = source[order[i + 1]];
                    const uint32_t third = source[order[i + 2]];
                    const uint32_t fourth = source[order[i + 3]];
                    destination[i] = first;
                    destination[i + 1] = second;
                    destination[i + 2] = third;
                    destination[i + 3] = fourth;
                }
                for (; i < lanes; ++i) {
                    destination[i] = source[order[i]];
                }
                return;
            }
            for (; i + 4 <= lanes; i += 4) {
                const uint32_t first_place = order[i], second_place = order[i + 1];
                const uint32_t third_place = order[i + 2], fourth_place = order[i + 3];
                const uint32_t first = source[i];
                const uint32_t second = source[i + 1];
                const uint32_t third = source[i + 2];
                const uint32_t fourth = source[i + 3];
                destination[first_place] = first;
                destination[second_place] = second;
                destination[third_place] = third;
                destination[fourth_place] = fourth;
            }
            for (; i < lanes; ++i) {
                destination[order[i]] = source[i];
            }
        }

        // Least significant digit first, stable: (key, index) through (key_swap,
        // index_swap) and back.
        void radix() {
            uint32_t *keys = key_, *indices = index_, *keys_next = key_swap_,
                     *indices_next = index_swap_;
            uint32_t* __restrict histogram = histogram_;
            const uint32_t lanes = lanes_;
            for (uint32_t pass = 0; pass < 3; ++pass) {
                const uint32_t digit_shift = pass * kSortDigitBits;
                for (uint32_t digit = 0; digit < kSortHist; ++digit) {
                    histogram[digit] = 0;
                }
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    histogram[(keys[lane] >> digit_shift) & (kSortHist - 1)]++;
                }
                uint32_t sum = 0;
                for (uint32_t digit = 0; digit < kSortHist; ++digit) {
                    const uint32_t count = histogram[digit];
                    histogram[digit] = sum;
                    sum += count;
                }
                for (uint32_t lane = 0; lane < lanes; ++lane) {
                    const uint32_t key = keys[lane];
                    const uint32_t place = (
                        histogram[(key >> digit_shift) & (kSortHist - 1)]++
                    );
                    keys_next[place] = key;
                    indices_next[place] = indices[lane];
                }
                uint32_t* swap = keys;
                keys = keys_next;
                keys_next = swap;
                swap = indices;
                indices = indices_next;
                indices_next = swap;
            }
            // Three passes leave the result in (key_swap, index_swap): the order is
            // index_swap.
            order_ = indices;
        }

        // Per packet of the order: bit `axis` of its signs when most of its rays point
        // down axis `axis` (the walk's child order; any value is correct).
        void count_signs(uint32_t axis) {
            const uint8_t* __restrict valid_flags = valid_;
            const uint32_t* __restrict destination = destination_;
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                uint32_t negative = 0, valid_count = 0;
                for (
                    uint32_t lane = packet * kVec; lane < (packet + 1) * kVec; ++lane
                ) {
                    valid_count += valid_flags[lane];
                    negative += valid_flags[lane] & (destination[lane] >> 31);
                }
                signs_[packet] |= (2 * negative > valid_count ? 1u : 0u) << axis;
            }
        }

        void read_lanes(
            uint32_t base_address,
            uint32_t page_bytes,
            uint32_t packet,
            uint32_t vector,
            uint32_t destination
        ) {
            const InterleavedAddrGen<true> addresses = {
                .bank_base_address = base_address,
                .page_size = page_bytes * kPagePackets
            };
            const uint32_t page = first_packet_ + packet;
            noc_async_read(
                get_noc_addr(
                    page / kPagePackets,
                    addresses,
                    (page % kPagePackets) * page_bytes + vector * kVec * 4
                ),
                destination,
                kVec * 4
            );
        }

        void write_lanes(
            uint32_t base_address,
            uint32_t page_bytes,
            uint32_t packet,
            uint32_t vector,
            uint32_t source
        ) {
            const InterleavedAddrGen<true> addresses = {
                .bank_base_address = base_address,
                .page_size = page_bytes * kPagePackets
            };
            const uint32_t page = first_packet_ + packet;
            noc_async_write(
                source,
                get_noc_addr(
                    page / kPagePackets,
                    addresses,
                    (page % kPagePackets) * page_bytes + vector * kVec * 4
                ),
                kVec * 4
            );
        }

        // Vector `vector` of every packet of a stream of `page_bytes` pages into
        // source_.
        bool gather_in(uint32_t base_address, uint32_t page_bytes, uint32_t vector) {
            const uint32_t started = km::wall();
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                read_lanes(
                    base_address,
                    page_bytes,
                    packet,
                    vector,
                    l1(source_ + packet * kVec)
                );
            }
            const bool done = timed::reads_done(reader_, km::kStReader);
            directory_[kRdWords + 0] += km::wall() - started;
            return done;
        }

        // destination_ as vector `vector` of every packet of a stream of `page_bytes`
        // pages.
        bool scatter_out(uint32_t base_address, uint32_t page_bytes, uint32_t vector) {
            const uint32_t started = km::wall();
            for (uint32_t packet = 0; packet < packets_; ++packet) {
                write_lanes(
                    base_address,
                    page_bytes,
                    packet,
                    vector,
                    l1(destination_ + packet * kVec)
                );
            }
            const bool done = timed::writes_sent(reader_, km::kStReader);
            directory_[kRdWords + 4] += km::wall() - started;
            return done;
        }

        km::Reader& reader_;
        const uint32_t input_address_;
        const uint32_t output_address_;
        const uint32_t order_address_;
        const uint32_t vectors_;
        const uint32_t corner_[3];
        const uint32_t key_word_;
        const uint32_t first_packet_, packets_, lanes_;
        volatile tt_l1_ptr uint32_t* directory_;
        uint32_t *key_ = nullptr, *index_ = nullptr, *order_ = nullptr;
        uint32_t *key_swap_ = nullptr, *index_swap_ = nullptr, *source_ = nullptr,
                 *destination_ = nullptr;
        uint32_t *histogram_ = nullptr, *signs_ = nullptr;
        uint8_t* valid_ = nullptr;
    };

}  // namespace

void kernel_main() {
    km::Reader reader;
    reader.run([&](const km_command_t& command) {
        const uint32_t handler = km::handler(command);
        if (handler == kHandlerLoad) {
            const uint32_t tree_address = command.a0, tree_bytes = command.a1;
            const uint32_t pages = (tree_bytes + kTreePageBytes - 1) / kTreePageBytes;
            if (!reader.reserve(kCbTree, pages)) {
                return;
            }
            const uint32_t base = get_write_ptr(kCbTree);
            if (!timed::fits(kCbTree, pages * kTreePageBytes)) {
                reader.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 2, pages);
                return;
            }
            if (!km::read_pages(
                reader, command, TreePages(tree_address), 0, pages, base
            ) ||
                !timed::reads_done(reader, km::kStReader)) {
                return;
            }
            g_tree = base;
            cb_push_back(kCbTree, pages);
            return;
        }
        if (handler == kHandlerSort) {
            const uint32_t directory_address = command.a0;
            const uint32_t profile_address = command.a4;
            const uint32_t unsort_vectors = command.a5;
            const uint32_t directory_space = get_write_ptr(kCbDirReader);
            volatile tt_l1_ptr uint32_t* directory = km::l1(directory_space);
            if (!km::read_pages(
                reader,
                command,
                Directories(directory_address),
                reader.me(),
                1,
                directory_space
            ) ||
                !timed::reads_done(reader, km::kStReader)) {
                return;
            }
            for (uint32_t counter = 0; counter < kRdCounters; ++counter) {
                directory[kRdWords + counter] = 0;
            }
            directory[kRdWords + 3] = directory[kDirCount] * kVec;
            Sorter sorter(
                reader,
                command,
                directory[kDirFirst],
                directory[kDirCount],
                directory,
                unsort_vectors
            );
            if (!(unsort_vectors == 0 ? sorter.sort() : sorter.unsort())) {
                return;
            }
            // The reader's counters to its place in this core's profile page.
            const InterleavedAddrGen<true> profile_pages = {
                .bank_base_address = profile_address, .page_size = kOutPageBytes
            };
            noc_async_write(
                directory_space + 4 * kRdWords,
                get_noc_addr(reader.me(), profile_pages, 4 * kProfReader),
                4 * kRdCounters
            );
            // The reader marks the command done without a barrier: the streams land
            // first.
            timed::writes_landed(reader, km::kStReader);
            return;
        }
        if (
            handler != kHandlerTrace && handler != kHandlerArith &&
            handler != kHandlerRecip && handler != kHandlerTreelets &&
            handler != kHandlerProbe && handler != kHandlerMath && !is_stage(handler)
        ) {
            return;
        }
        const uint32_t started = km::wall();
        const uint32_t directory_space = get_write_ptr(kCbDirReader);
        volatile tt_l1_ptr uint32_t* directory = km::l1(directory_space);
        if (
            !read_directory<Directories>(reader, command, handler, directory_space) ||
            !timed::reads_done(reader, km::kStReader)
        ) {
            return;
        }
        for (uint32_t counter = 0; counter < kRdCounters; ++counter) {
            directory[kRdWords + counter] = 0;
        }
        const uint32_t first_packet = directory[kDirFirst];
        const uint32_t packets = directory[kDirCount];
        if (!reader.reserve(kCbIn, 1)) {
            return;
        }
        volatile tt_l1_ptr uint32_t* header = km::l1(get_write_ptr(kCbIn));
        header[kDirFirst] = first_packet;
        header[kDirCount] = packets;
        cb_push_back(kCbIn, 1);
        uint32_t wait = 0, read = 0;
        if (is_stage(handler)) {
            // The constant page, then each packet's page of every input stream.
            const uint32_t constants_address = command.a3;
            const Stage stage = stage_of(handler);
            if (!read_page(
                reader, constants_address, kCnVectors * kVec * 4, 1, 0, wait
            )) {
                return;
            }
            for (uint32_t packet = 0; packet < packets; ++packet) {
                for (uint32_t stream = 0; stream < stage.inputs; ++stream) {
                    const uint32_t read_started = km::wall();
                    const uint32_t page_bytes = stage.vectors[stream] * kVec * 4;
                    if (!read_page(
                        reader,
                        stream_arg(command, stream),
                        page_bytes,
                        kPagePackets,
                        first_packet + packet,
                        wait
                    )) {
                        return;
                    }
                    read += km::wall() - read_started;
                }
            }
            directory[kRdWords + 0] = wait;
            directory[kRdWords + 1] = read - wait;
            directory[kRdWords + 3] = packets * stage.inputs;
            directory[kRdWords + 2] = km::wall() - started;
            return;
        }
        const uint32_t input_address = command.a1;
        const uint32_t page_bytes = in_page_bytes(handler);
        const uint32_t page_packets = packets_per_page(handler);
        const uint32_t pages = handler == kHandlerRecip ? 0 : packets;
        for (uint32_t page = 0; page < pages; ++page) {
            const uint32_t read_started = km::wall();
            if (!read_page(
                reader,
                input_address,
                page_bytes,
                page_packets,
                first_packet + page,
                wait
            )) {
                return;
            }
            read += km::wall() - read_started;
        }
        read -= wait;
        directory[kRdWords + 0] = wait;
        directory[kRdWords + 1] = read;
        directory[kRdWords + 3] = pages;
        if (handler == kHandlerTreelets) {
            // Per batch: the top walks' end, the schedule, the batch's end page. At
            // most packets + 1 batches: each one admits a packet (compute sees to it).
            Schedule schedule(reader, command, packets, directory);
            for (uint32_t batch = 0; batch <= packets; ++batch) {
                const uint32_t signal_started = km::wall();
                const uint32_t code = reader.poll(
                    [] {
                        return cb_pages_available_at_front(kCbSignal, 1);
                    },
                    reader.args().wait_limit
                );
                if (code != km::kGaveNone) {
                    reader.gave_up(km::kStReader, code, km::kWhereData, 0, 0);
                    return;
                }
                cb_wait_front(kCbSignal, 1);
                cb_pop_front(kCbSignal, 1);
                directory[kRdWords + kRdSignal] += km::wall() - signal_started;
                if ((batch == 0 && !schedule.init()) || !schedule.batch()) {
                    return;
                }
                const bool more = schedule.more() && batch < packets;
                if (!more) {
                    directory[kRdWords + 2] = km::wall() - started;
                }
                if (!schedule.finish(more ? kWorkNext : kWorkEnd) || !more) {
                    return;
                }
            }
            return;
        }
        directory[kRdWords + 2] = km::wall() - started;
    });
}
