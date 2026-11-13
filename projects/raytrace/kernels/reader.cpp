// Ray tracing reader: LOAD copies the tree image into the tree buffer; the other
// segments send a header page (first page, count), then stream this core's pages.
#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "bounded.hpp"
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
            const bool once = r_.a9 == kPolicyOnce;
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
        if (seg != kSegTrace && seg != kSegArith && seg != kSegRecip &&
            seg != kSegTreelets && seg != kSegProbe && seg != kSegMath &&
            !is_stage(seg)) {
            return;
        }
        const uint32_t t0 = km::wall();
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
