// Ray tracing reader: LOAD copies the tree image into the tree buffer; the other
// segments send a header page (first page, count), then stream this core's pages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "bounded.hpp"
#include "dir_noc.hpp"
#include "komorebi/device/km_worker.hpp"
#include "raytrace.h"

namespace {

    constexpr uint32_t kMaxSlots = 8;
    constexpr uint16_t kDone = 0x8000;  // a treelet's group of this round is issued

    uint32_t g_tree = 0;  // L1 address of the tree (or top tree) image, after LOAD

    // Stores to L1 finished before a push: a load of the last word written, used.
    void settle(volatile tt_l1_ptr uint32_t* w) {
        const uint32_t v = *w;
        asm volatile("and x0, x0, %0" : : "r"(v) : "memory");
    }

    uint32_t in_page_bytes(uint32_t seg) {
        if (seg == kSegMath) {
            return kMaInVectors * kVec * 4;
        }
        return seg == kSegArith ? kArInVectors * kVec * 4 : kPacketBytes;
    }

    // Packet pages go kPagePackets to a DRAM page; the check pages one to a page.
    uint32_t per_page(uint32_t seg) {
        return (
            seg == kSegArith || seg == kSegMath || seg == kSegRecip ? 1 : kPagePackets
        );
    }

    // Page g of a stream of `bytes` pages into the input buffer, pushed; the wait for
    // room is added to `wait`.
    bool read_page(
        km::Reader& rd,
        uint32_t base,
        uint32_t bytes,
        uint32_t per,
        uint32_t g,
        uint32_t& wait
    ) {
        const uint32_t a = km::wall();
        if (!rd.reserve(kCbIn, 1)) {
            return false;
        }
        wait += km::wall() - a;
        const InterleavedAddrGen<true> pages = {
            .bank_base_address = base, .page_size = bytes * per
        };
        noc_async_read(
            get_noc_addr(g / per, pages, (g % per) * bytes), get_write_ptr(kCbIn), bytes
        );
        if (!bounded::reads_done(rd, km::kStReader)) {
            return false;
        }
        cb_push_back(kCbIn, 1);
        return true;
    }

    // Treelets in L1 slots, least recently used replaced; a slot is free again once
    // compute has finished the last work page that names it.
    struct Slots {
        uint32_t base, bytes, n;
        uint32_t treelet[kMaxSlots];
        uint32_t last[kMaxSlots];  // work pages issued when it was last named
        uint32_t used[kMaxSlots];  // use stamp
        uint32_t clock = 0;

        int32_t find(uint32_t t) const {
            for (uint32_t k = 0; k < n; ++k) {
                if (treelet[k] == t) {
                    return static_cast<int32_t>(k);
                }
            }
            return -1;
        }

        uint32_t victim() const {
            uint32_t v = 0;
            for (uint32_t k = 1; k < n; ++k) {
                v = used[k] < used[v] ? k : v;
            }
            return v;
        }
    };

    // A record's treelet schedule, batch by batch: rounds (r-th list entries) or once;
    // packets binned by treelet, a treelet read into a slot unless present.
    class Schedule {
    public:
        Schedule(
            km::Reader& rd,
            const km_record_t& r,
            uint32_t count,
            volatile tt_l1_ptr uint32_t* d
        )
            : rd_(rd), r_(r), count_(count), d_(d) {
        }

        // Buffers and empty slots; false when the lists' scratch does not fit.
        bool init() {
            const uint32_t* top = reinterpret_cast<const uint32_t*>(g_tree);
            n_treelets_ = top[kTreeTreelets];
            const uint32_t n_treelets = n_treelets_;
            table_ = top + top[kTreeTableOff];
            ls_ = km::l1(get_write_ptr(kCbLists));
            uint32_t* cnt = (
                reinterpret_cast<uint32_t*>(
                    get_write_ptr(kCbLists) + 4 * (kLsHead + 2 * count_) + 2 * r_.a6
                )
            );
            cnt_ = cnt;
            present_ = reinterpret_cast<uint16_t*>(cnt + n_treelets);
            order_ = present_ + n_treelets;
            const uint32_t most = r_.a6 > count_ ? r_.a6 : count_;
            const uint32_t end = reinterpret_cast<uint32_t>(order_ + most);
            if (!bounded::fits(kCbLists, end - get_write_ptr(kCbLists))) {
                rd_.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 3, end);
                return false;
            }
            slots_.base = get_write_ptr(kCbSlots);
            slots_.bytes = r_.a7;
            slots_.n = r_.a8 < kMaxSlots ? r_.a8 : kMaxSlots;
            for (uint32_t k = 0; k < slots_.n; ++k) {
                slots_.treelet[k] = 0xFFFFFFFF;
                slots_.last[k] = slots_.used[k] = 0;
            }
            for (uint32_t t = 0; t < n_treelets; ++t) {
                cnt[t] = 0;
            }
            return true;
        }

        // The current lists (one batch); then kLsMore: packets left for another batch.
        bool batch() {
            const volatile uint32_t* dir = ls_ + kLsHead;
            const volatile uint16_t* pool = (
                reinterpret_cast<const volatile uint16_t*>(ls_ + kLsHead + 2 * count_)
            );
            uint32_t* cnt = cnt_;
            uint16_t* present = present_;
            uint16_t* order = order_;
            // Rounds: entry r of every list in round r. Once: every entry in one round.
            const bool once = (r_.a9 & 0xFF) == kPolicyOnce;
            uint32_t rounds = 0;
            for (uint32_t p = 0; p < count_; ++p) {
                rounds = dir[2 * p + 1] > rounds ? dir[2 * p + 1] : rounds;
            }
            rounds = once && rounds > 0 ? 1 : rounds;
            d_[kRdWords + kRdRounds] += rounds;
            for (uint32_t round = 0; round < rounds; ++round) {
                const uint32_t a = km::wall();
                uint32_t n_present = 0;
                for (uint32_t p = 0; p < count_; ++p) {
                    const uint32_t len = dir[2 * p + 1], at = dir[2 * p];
                    const uint32_t hi = once ? len : (len > round ? round + 1 : 0);
                    for (uint32_t j = once ? 0 : round; j < hi; ++j) {
                        const uint32_t t = pool[at + j];
                        if (cnt[t]++ == 0) {
                            present[n_present++] = t;
                        }
                    }
                }
                uint32_t sum = 0;
                for (uint32_t k = 0; k < n_present; ++k) {
                    const uint32_t c = cnt[present[k]];
                    cnt[present[k]] = sum;
                    sum += c;
                }
                for (uint32_t p = 0; p < count_; ++p) {
                    const uint32_t len = dir[2 * p + 1], at = dir[2 * p];
                    const uint32_t hi = once ? len : (len > round ? round + 1 : 0);
                    for (uint32_t j = once ? 0 : round; j < hi; ++j) {
                        order[cnt[pool[at + j]]++] = p;
                    }
                }
                d_[kRdWords + kRdBin] += km::wall() - a;
                // Treelets already in a slot first (marked done), then the others.
                for (uint32_t pass = 0; pass < 2; ++pass) {
                    uint32_t begin = 0;
                    for (uint32_t k = 0; k < n_present; ++k) {
                        const uint32_t t = present[k] & ~kDone;
                        const uint32_t end = cnt[t];
                        const bool now =
                            (pass == 0 ? slots_.find(t) >= 0
                                       : (present[k] & kDone) == 0);
                        if (now && !group(t, order + begin, end - begin)) {
                            return false;
                        }
                        present[k] |= now ? kDone : 0;
                        begin = end;
                    }
                }
                for (uint32_t k = 0; k < n_present; ++k) {
                    cnt[present[k] & ~kDone] = 0;
                }
            }
            return true;
        }

        // A batch's end page: kWorkEnd (after the counters are written: the writer
        // reads them next) or kWorkNext.
        bool finish(uint32_t word) {
            if (!rd_.reserve(kCbWork, 1)) {
                return false;
            }
            volatile tt_l1_ptr uint32_t* w = km::l1(get_write_ptr(kCbWork));
            w[kWorkCount] = word;
            settle(w + kWorkCount);
            cb_push_back(kCbWork, 1);
            return true;
        }

        bool more() const {
            return ls_[kLsMore] != 0;
        }

    private:
        bool group(uint32_t t, const uint16_t* ids, uint32_t n) {
            int32_t k = slots_.find(t);
            if (k >= 0) {
                d_[kRdWords + kRdHits]++;
            } else {
                k = static_cast<int32_t>(slots_.victim());
                if (!wait_done(slots_.last[k]) || !fetch(t, k)) {
                    return false;
                }
            }
            for (uint32_t i = 0; i < n; i += kWorkMax) {
                const uint32_t m = n - i < kWorkMax ? n - i : kWorkMax;
                const uint32_t a = km::wall();
                if (!rd_.reserve(kCbWork, 1)) {
                    return false;
                }
                d_[kRdWords + kRdWorkWait] += km::wall() - a;
                const uint32_t page = get_write_ptr(kCbWork);
                volatile tt_l1_ptr uint32_t* w = km::l1(page);
                w[kWorkSlot] = slots_.base + k * slots_.bytes;
                w[kWorkTreelet] = t;
                w[kWorkCount] = m;
                volatile tt_l1_ptr uint16_t*
                    out = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(w + kWorkIds);
                for (uint32_t j = 0; j < m; ++j) {
                    out[j] = ids[i + j];
                }
                settle(w + kWorkIds + (m - 1) / 2);
                cb_push_back(kCbWork, 1);
                items_++;
                d_[kRdWords + kRdItems] = items_;
                d_[kRdWords + kRdVisits] += m;
            }
            slots_.last[k] = items_;
            slots_.used[k] = ++slots_.clock;
            d_[kRdWords + kRdGroups]++;
            return true;
        }

        // Until compute has finished `need` work pages.
        bool wait_done(uint32_t need) {
            const uint32_t a = km::wall();
            const uint32_t code = rd_.poll(
                [&] {
                    return ls_[kLsDone] >= need;
                },
                rd_.args().wait_limit
            );
            d_[kRdWords + kRdSlotWait] += km::wall() - a;
            if (code != km::kGaveNone) {
                rd_.gave_up(km::kStReader, code, km::kWhereData, 0, 0);
                return false;
            }
            return true;
        }

        bool fetch(uint32_t t, uint32_t k) {
            const uint32_t a = km::wall();
            const InterleavedAddrGen<true> images = {
                .bank_base_address = r_.a5, .page_size = kTreePageBytes
            };
            const uint32_t first = table_[2 * t], pages = table_[2 * t + 1];
            const uint32_t dst = slots_.base + k * slots_.bytes;
            if (pages * kTreePageBytes > slots_.bytes ||
                !bounded::fits(kCbSlots, (k + 1) * slots_.bytes)) {
                rd_.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 1, t);
                return false;
            }
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read(
                    get_noc_addr(first + p, images),
                    dst + p * kTreePageBytes,
                    kTreePageBytes
                );
            }
            if (!bounded::reads_done(rd_, km::kStReader)) {
                return false;
            }
            slots_.treelet[k] = t;
            d_[kRdWords + kRdFetch] += km::wall() - a;
            d_[kRdWords + kRdReads]++;
            d_[kRdWords + kRdPages] += pages;
            return true;
        }

        km::Reader& rd_;
        const km_record_t& r_;
        const uint32_t count_;
        volatile tt_l1_ptr uint32_t* d_;
        volatile tt_l1_ptr uint32_t* ls_ = nullptr;
        const uint32_t* table_ = nullptr;
        uint32_t n_treelets_ = 0;
        uint32_t* cnt_ = nullptr;
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
            km::Reader& rd,
            const km_record_t& r,
            uint32_t first,
            uint32_t count,
            volatile tt_l1_ptr uint32_t* d
        )
            : rd_(rd), r_(r), first_(first), count_(count), n_(count * kVec), d_(d) {
        }

        // Key and order arrays, two vectors of every packet, the radix counts and the
        // per-packet signs: the bytes from the slot buffer's write pointer.
        static uint32_t bytes(uint32_t count) {
            return (6 * count * kVec + kSortHist + count) * 4;
        }

        bool sort() {
            if (!layout()) {
                return false;
            }
            const uint32_t a = km::wall();
            for (uint32_t i = 0; i < n_; ++i) {
                key_[i] = 0;
                idx_[i] = i;
            }
            const uint32_t bytes = kPacketBytes;
            // Keys: the octant from the direction's signs, the origin's Morton cell.
            if (!gather_in(r_.a1, bytes, kPkBest)) {
                return false;
            }
            uint32_t* __restrict key = key_;
            const uint32_t* __restrict src = src_;
            const uint32_t n = n_;
            for (uint32_t i = 0; i < n; ++i) {
                key[i] = src[i] == kMinusOneBits ? kSortInvalid : 0;
            }
            for (uint32_t k = 0; k < 3; ++k) {
                if (!gather_in(r_.a1, bytes, kPkD + k)) {
                    return false;
                }
                const uint32_t at = 3 * kSortCellBits + k;
                for (uint32_t i = 0; i < n; ++i) {
                    key[i] |= (src[i] >> 31) << at;
                }
            }
            const uint32_t frac = r_.a9 & 0xFF, shift = (r_.a9 >> 8) & 0xFF;
            const uint32_t lo[3] = {r_.a6, r_.a7, r_.a8};
            for (uint32_t k = 0; k < 3; ++k) {
                if (!gather_in(r_.a1, bytes, kPkO + k)) {
                    return false;
                }
                const uint32_t low = lo[k];
                for (uint32_t i = 0; i < n; ++i) {
                    key[i] |= spread(cell(src[i], frac, low, shift)) << k;
                }
            }
            for (uint32_t i = 0; i < n; ++i) {
                key[i] = (key[i] & kSortInvalid) ? kSortInvalid : key[i];
            }
            const uint32_t b = km::wall();
            radix();
            d_[kRdWords + 1] += km::wall() - b;
            // The order pages, then every vector through the order; the header last.
            for (uint32_t q = 0; q < count_; ++q) {
                write_lanes(r_.a3, kVec * 4, q, 0, l1(order_ + q * kVec));
            }
            for (uint32_t q = 0; q < count_; ++q) {
                signs_[q] = 0;
            }
            const uint32_t order[kPkHead] = {
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
            for (uint32_t j = 0; j < kPkHead; ++j) {
                const uint32_t v = order[j];
                if (!gather_in(r_.a1, bytes, v)) {
                    return false;
                }
                permute(dst_, src_, order_, n_, true);
                if (v >= kPkD && v < kPkD + 3) {
                    count_signs(v - kPkD);
                }
                if (v == kPkBest) {
                    for (uint32_t i = 0; i < n_; ++i) {
                        valid_[i] = dst_[i] != kMinusOneBits;
                    }
                }
                if (!scatter_out(r_.a2, bytes, v)) {
                    return false;
                }
            }
            for (uint32_t i = 0; i < n_; ++i) {
                dst_[i] = 0;
            }
            for (uint32_t q = 0; q < count_; ++q) {
                dst_[q * kVec + kHeadSigns] = signs_[q];
            }
            if (!scatter_out(r_.a2, bytes, kPkHead)) {
                return false;
            }
            d_[kRdWords + 2] = km::wall() - a;
            return true;
        }

        bool unsort() {
            if (!layout()) {
                return false;
            }
            const uint32_t a = km::wall();
            const uint32_t vectors = r_.a5, bytes = vectors * kVec * 4;
            for (uint32_t q = 0; q < count_; ++q) {
                read_lanes(r_.a3, kVec * 4, q, 0, l1(order_ + q * kVec));
            }
            if (!bounded::reads_done(rd_, km::kStReader)) {
                return false;
            }
            for (uint32_t i = 0; i < n_; ++i) {
                if (order_[i] >= n_) {
                    rd_.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 4, i);
                    return false;
                }
            }
            for (uint32_t v = 0; v < vectors; ++v) {
                if (!gather_in(r_.a1, bytes, v)) {
                    return false;
                }
                permute(dst_, src_, order_, n_, false);
                if (!scatter_out(r_.a2, bytes, v)) {
                    return false;
                }
            }
            d_[kRdWords + 2] = km::wall() - a;
            return true;
        }

    private:
        static constexpr uint32_t kSortHist = 1u << kSortDigitBits;
        static constexpr uint32_t kMinusOneBits = 0xBF800000;

        static uint32_t l1(const uint32_t* p) {
            return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
        }

        bool layout() {
            if (!bounded::fits(kCbSlots, bytes(count_))) {
                rd_.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 3, count_);
                return false;
            }
            uint32_t* base = reinterpret_cast<uint32_t*>(get_write_ptr(kCbSlots));
            key_ = base;
            order_ = key_ + n_;
            key2_ = order_ + n_;
            idx2_ = key2_ + n_;
            src_ = idx2_ + n_;
            dst_ = src_ + n_;
            hist_ = dst_ + n_;
            signs_ = hist_ + kSortHist;
            valid_ = reinterpret_cast<uint8_t*>(key2_);  // free after the sort
            idx_ = order_;
            return true;
        }

        // The cell of a float coordinate: its fixed-point value (frac fraction bits,
        // truncated) less the low corner, shifted down, clamped to the grid.
        static uint32_t cell(uint32_t f, uint32_t frac, uint32_t lo, uint32_t shift) {
            const int32_t e = static_cast<int32_t>((f >> 23) & 0xFF);
            int32_t v = 0;
            if (e != 0) {
                const uint32_t m = (f & 0x7FFFFF) | 0x800000;
                const int32_t sh = e - 150 + static_cast<int32_t>(frac);
                const uint32_t u =
                    (sh >= 8        ? 0x7FFFFFFF
                         : sh >= 0  ? m << sh
                         : sh > -32 ? m >> -sh
                                    : 0);
                v = (f >> 31) ? -static_cast<int32_t>(u) : static_cast<int32_t>(u);
            }
            const int32_t c = (v - static_cast<int32_t>(lo)) >> shift;
            const int32_t top = (1 << kSortCellBits) - 1;
            return static_cast<uint32_t>(c < 0 ? 0 : c > top ? top : c);
        }

        // Bit b of c (10 bits at most) to bit 3 b.
        static uint32_t spread(uint32_t c) {
            c &= 0x3FF;
            c = (c | c << 16) & 0x030000FF;
            c = (c | c << 8) & 0x0300F00F;
            c = (c | c << 4) & 0x030C30C3;
            return (c | c << 2) & 0x09249249;
        }

        // dst[i] = src[order[i]] (gather) or dst[order[i]] = src[i], four at a time.
        static void permute(
            uint32_t* __restrict dst,
            const uint32_t* __restrict src,
            const uint32_t* __restrict order,
            uint32_t n,
            bool gather
        ) {
            uint32_t i = 0;
            if (gather) {
                for (; i + 4 <= n; i += 4) {
                    const uint32_t a = src[order[i]], b = src[order[i + 1]];
                    const uint32_t c = src[order[i + 2]], e = src[order[i + 3]];
                    dst[i] = a;
                    dst[i + 1] = b;
                    dst[i + 2] = c;
                    dst[i + 3] = e;
                }
                for (; i < n; ++i) {
                    dst[i] = src[order[i]];
                }
                return;
            }
            for (; i + 4 <= n; i += 4) {
                const uint32_t o0 = order[i], o1 = order[i + 1];
                const uint32_t o2 = order[i + 2], o3 = order[i + 3];
                const uint32_t a = src[i], b = src[i + 1], c = src[i + 2],
                               e = src[i + 3];
                dst[o0] = a;
                dst[o1] = b;
                dst[o2] = c;
                dst[o3] = e;
            }
            for (; i < n; ++i) {
                dst[order[i]] = src[i];
            }
        }

        // Least significant digit first, stable: (key, idx) through (key2, idx2) and
        // back.
        void radix() {
            uint32_t *k0 = key_, *i0 = idx_, *k1 = key2_, *i1 = idx2_;
            uint32_t* __restrict hist = hist_;
            const uint32_t n = n_;
            for (uint32_t pass = 0; pass < 3; ++pass) {
                const uint32_t sh = pass * kSortDigitBits;
                for (uint32_t h = 0; h < kSortHist; ++h) {
                    hist[h] = 0;
                }
                for (uint32_t i = 0; i < n; ++i) {
                    hist[(k0[i] >> sh) & (kSortHist - 1)]++;
                }
                uint32_t sum = 0;
                for (uint32_t h = 0; h < kSortHist; ++h) {
                    const uint32_t c = hist[h];
                    hist[h] = sum;
                    sum += c;
                }
                for (uint32_t i = 0; i < n; ++i) {
                    const uint32_t key = k0[i];
                    const uint32_t at = hist[(key >> sh) & (kSortHist - 1)]++;
                    k1[at] = key;
                    i1[at] = i0[i];
                }
                uint32_t* t = k0;
                k0 = k1;
                k1 = t;
                t = i0;
                i0 = i1;
                i1 = t;
            }
            // Three passes leave the result in (key2, idx2): the order is idx2.
            order_ = i0;
        }

        // Per packet of the order: bit k of its signs when most of its rays point down
        // axis k (the walk's child order; any value is correct).
        void count_signs(uint32_t k) {
            const uint8_t* __restrict ok = valid_;
            const uint32_t* __restrict dst = dst_;
            for (uint32_t q = 0; q < count_; ++q) {
                uint32_t neg = 0, valid = 0;
                for (uint32_t i = q * kVec; i < (q + 1) * kVec; ++i) {
                    valid += ok[i];
                    neg += ok[i] & (dst[i] >> 31);
                }
                signs_[q] |= (2 * neg > valid ? 1u : 0u) << k;
            }
        }

        void read_lanes(
            uint32_t base, uint32_t bytes, uint32_t q, uint32_t v, uint32_t to
        ) {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = base, .page_size = bytes * kPagePackets
            };
            const uint32_t g = first_ + q;
            noc_async_read(
                get_noc_addr(
                    g / kPagePackets, pages, (g % kPagePackets) * bytes + v * kVec * 4
                ),
                to,
                kVec * 4
            );
        }

        void write_lanes(
            uint32_t base, uint32_t bytes, uint32_t q, uint32_t v, uint32_t from
        ) {
            const InterleavedAddrGen<true> pages = {
                .bank_base_address = base, .page_size = bytes * kPagePackets
            };
            const uint32_t g = first_ + q;
            noc_async_write(
                from,
                get_noc_addr(
                    g / kPagePackets, pages, (g % kPagePackets) * bytes + v * kVec * 4
                ),
                kVec * 4
            );
        }

        // Vector v of every packet of a stream of `bytes` pages into src.
        bool gather_in(uint32_t base, uint32_t bytes, uint32_t v) {
            const uint32_t a = km::wall();
            for (uint32_t q = 0; q < count_; ++q) {
                read_lanes(base, bytes, q, v, l1(src_ + q * kVec));
            }
            const bool ok = bounded::reads_done(rd_, km::kStReader);
            d_[kRdWords + 0] += km::wall() - a;
            return ok;
        }

        // dst as vector v of every packet of a stream of `bytes` pages.
        bool scatter_out(uint32_t base, uint32_t bytes, uint32_t v) {
            const uint32_t a = km::wall();
            for (uint32_t q = 0; q < count_; ++q) {
                write_lanes(base, bytes, q, v, l1(dst_ + q * kVec));
            }
            const bool ok = bounded::writes_sent(rd_, km::kStReader);
            d_[kRdWords + 4] += km::wall() - a;
            return ok;
        }

        km::Reader& rd_;
        const km_record_t& r_;
        const uint32_t first_, count_, n_;
        volatile tt_l1_ptr uint32_t* d_;
        uint32_t *key_ = nullptr, *idx_ = nullptr, *order_ = nullptr;
        uint32_t *key2_ = nullptr, *idx2_ = nullptr, *src_ = nullptr, *dst_ = nullptr;
        uint32_t *hist_ = nullptr, *signs_ = nullptr;
        uint8_t* valid_ = nullptr;
    };

}  // namespace

void kernel_main() {
    km::Reader rd;
    rd.run([&](const km_record_t& r) {
        const uint32_t seg = km::seg(r);
        if (seg == kSegLoad) {
            const uint32_t pages = (r.a1 + kTreePageBytes - 1) / kTreePageBytes;
            if (!rd.reserve(kCbTree, pages)) {
                return;
            }
            const InterleavedAddrGen<true> tree = {
                .bank_base_address = r.a0, .page_size = kTreePageBytes
            };
            const uint32_t base = get_write_ptr(kCbTree);
            if (!bounded::fits(kCbTree, pages * kTreePageBytes)) {
                rd.gave_up(km::kStReader, km::kGaveLimit, km::kWhereData, 2, pages);
                return;
            }
            for (uint32_t p = 0; p < pages; ++p) {
                noc_async_read(
                    get_noc_addr(p, tree), base + p * kTreePageBytes, kTreePageBytes
                );
            }
            if (!bounded::reads_done(rd, km::kStReader)) {
                return;
            }
            g_tree = base;
            cb_push_back(kCbTree, pages);
            return;
        }
        if (seg == kSegSort) {
            const uint32_t dir = get_write_ptr(kCbDirReader);
            volatile tt_l1_ptr uint32_t* d = km::l1(dir);
            const InterleavedAddrGen<true> dirs = {
                .bank_base_address = r.a0, .page_size = kDirBytes
            };
            noc_async_read(get_noc_addr(rd.me(), dirs), dir, kDirBytes);
            if (!bounded::reads_done(rd, km::kStReader)) {
                return;
            }
            for (uint32_t k = 0; k < kRdCounters; ++k) {
                d[kRdWords + k] = 0;
            }
            d[kRdWords + 3] = d[kDirCount] * kVec;
            Sorter so(rd, r, d[kDirFirst], d[kDirCount], d);
            if (!(r.a5 == 0 ? so.sort() : so.unsort())) {
                return;
            }
            // The reader's counters to its place in this core's profile page.
            const InterleavedAddrGen<true> profs = {
                .bank_base_address = r.a4, .page_size = kOutPageBytes
            };
            noc_async_write(
                dir + 4 * kRdWords,
                get_noc_addr(rd.me(), profs, 4 * kProfReader),
                4 * kRdCounters
            );
            // The reader marks the record done without a barrier: the streams land
            // first.
            bounded::writes_landed(rd, km::kStReader);
            return;
        }
        if (seg != kSegTrace && seg != kSegArith && seg != kSegRecip &&
            seg != kSegTreelets && seg != kSegProbe && seg != kSegMath &&
            !is_stage(seg)) {
            return;
        }
        const uint32_t t0 = km::wall();
        const uint32_t dir = get_write_ptr(kCbDirReader);
        volatile tt_l1_ptr uint32_t* d = km::l1(dir);
        noc_async_read(dir_addr(r, seg, rd.me()), dir, kDirBytes);
        if (!bounded::reads_done(rd, km::kStReader)) {
            return;
        }
        for (uint32_t k = 0; k < kRdCounters; ++k) {
            d[kRdWords + k] = 0;
        }
        const uint32_t first = d[kDirFirst], count = d[kDirCount];
        if (!rd.reserve(kCbIn, 1)) {
            return;
        }
        volatile tt_l1_ptr uint32_t* h = km::l1(get_write_ptr(kCbIn));
        h[kDirFirst] = first;
        h[kDirCount] = count;
        cb_push_back(kCbIn, 1);
        uint32_t wait = 0, read = 0;
        if (is_stage(seg)) {
            // The constant page, then each packet's page of every input stream.
            const Stage st = stage_of(seg);
            if (!read_page(rd, r.a3, kCnVectors * kVec * 4, 1, 0, wait)) {
                return;
            }
            for (uint32_t p = 0; p < count; ++p) {
                for (uint32_t s = 0; s < st.inputs; ++s) {
                    const uint32_t a = km::wall();
                    const uint32_t bytes = st.vectors[s] * kVec * 4;
                    if (!read_page(
                            rd, stream_arg(r, s), bytes, kPagePackets, first + p, wait
                        )) {
                        return;
                    }
                    read += km::wall() - a;
                }
            }
            d[kRdWords + 0] = wait;
            d[kRdWords + 1] = read - wait;
            d[kRdWords + 3] = count * st.inputs;
            d[kRdWords + 2] = km::wall() - t0;
            return;
        }
        const uint32_t bytes = in_page_bytes(seg), per = per_page(seg);
        const uint32_t n = seg == kSegRecip ? 0 : count;
        for (uint32_t p = 0; p < n; ++p) {
            const uint32_t a = km::wall();
            if (!read_page(rd, r.a1, bytes, per, first + p, wait)) {
                return;
            }
            read += km::wall() - a;
        }
        read -= wait;
        d[kRdWords + 0] = wait;
        d[kRdWords + 1] = read;
        d[kRdWords + 3] = n;
        if (seg == kSegTreelets) {
            // Per batch: the top walks' end, the schedule, the batch's end page. At
            // most count + 1 batches: each one admits a packet (compute sees to it).
            Schedule s(rd, r, count, d);
            for (uint32_t k = 0; k <= count; ++k) {
                const uint32_t a = km::wall();
                const uint32_t code = rd.poll(
                    [] {
                        return cb_pages_available_at_front(kCbSignal, 1);
                    },
                    rd.args().wait_limit
                );
                if (code != km::kGaveNone) {
                    rd.gave_up(km::kStReader, code, km::kWhereData, 0, 0);
                    return;
                }
                cb_wait_front(kCbSignal, 1);
                cb_pop_front(kCbSignal, 1);
                d[kRdWords + kRdSignal] += km::wall() - a;
                if ((k == 0 && !s.init()) || !s.batch()) {
                    return;
                }
                const bool more = s.more() && k < count;
                if (!more) {
                    d[kRdWords + 2] = km::wall() - t0;
                }
                if (!s.finish(more ? kWorkNext : kWorkEnd) || !more) {
                    return;
                }
            }
            return;
        }
        d[kRdWords + 2] = km::wall() - t0;
    });
}
