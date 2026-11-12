// Records in the ring: header and team fields, sequence order, membership, the share of
// a range, and the scan that takes the next record of a core. Free of tt-metal headers.
#pragma once

#include <stdint.h>

#include "../abi/km_abi.h"

namespace km {

    static_assert(sizeof(km_record_t) == 64, "a record is one 64-byte line");
    static_assert(sizeof(km_event_t) == 16, "an event is one 16-byte chunk");

    constexpr uint32_t kRecordWords = 16;

    inline uint32_t seg(const km_record_t& r) {
        return r.hdr & 0xFFu;
    }

    inline uint32_t flags(const km_record_t& r) {
        return (r.hdr >> 8) & 0xFFu;
    }

    inline uint32_t slot(const km_record_t& r) {
        return (r.hdr >> 16) & 0xFFu;
    }

    inline uint32_t first(const km_record_t& r) {
        return r.team & 0xFFFFu;
    }

    inline uint32_t count(const km_record_t& r) {
        return r.team >> 16;
    }

    inline uint32_t make_hdr(uint32_t seg, uint32_t flags, uint32_t slot) {
        return seg | flags << 8 | slot << 16;
    }

    inline uint32_t make_team(uint32_t first, uint32_t count) {
        return first | count << 16;
    }

    // True when sequence `a` comes after `b` (signed difference: valid across wrap).
    inline bool newer(uint32_t a, uint32_t b) {
        return static_cast<int32_t>(a - b) > 0;
    }

    // True when core `me` is in the record's team.
    inline bool in_team(const km_record_t& r, uint32_t me) {
        return me - first(r) < count(r);  // unsigned: me < first wraps to a large value
    }

    // True when teams `a` and `b` (first | count << 16) share a core: the test that
    // holds a KM_F_EXCL record while an overlapping team is in flight.
    inline bool teams_overlap(uint32_t a, uint32_t b) {
        const uint32_t fa = a & 0xFFFFu, na = a >> 16, fb = b & 0xFFFFu, nb = b >> 16;
        return na != 0 && nb != 0 && fa < fb + nb && fb < fa + na;
    }

    // Member `index` of `count` takes items [start, start + len) of `n`: contiguous
    // parts covering every item once, the first n % count members one item more.
    struct Share {
        uint32_t start;
        uint32_t len;
    };

    inline Share share(uint32_t n, uint32_t index, uint32_t count) {
        const uint32_t base = n / count, extra = n % count;
        return {index * base + (index < extra ? index : extra), base + (index < extra)};
    }

    // Where core `me` reports record r (`parent`; me == first: the dispatcher) and how
    // many reports it waits for first (`want`). A team inside one grid row of `row`
    // cores (or row == 0): one level, every member to the first member. A team that
    // spans rows: two levels, members to the first team member of their row, row
    // leaders to the first member.
    struct Report {
        uint32_t parent;
        uint32_t want;
    };

    inline Report report_to(const km_record_t& r, uint32_t me, uint32_t row) {
        const uint32_t f = first(r), n = count(r), end = f + n;
        if (row == 0 || f / row == (end - 1) / row) {
            return {f, me == f ? n - 1 : 0};
        }
        const uint32_t row0 = me / row * row;
        const uint32_t lead = row0 > f ? row0 : f;
        if (me != lead) {
            return {lead, 0};
        }
        const uint32_t row_end = row0 + row < end ? row0 + row : end;
        const uint32_t rows = me == f ? (end - 1) / row - f / row : 0;
        return {f, row_end - lead - 1 + rows};
    }

    // Copies a record word by word (from a ring slot or into the job copy).
    template <typename Src, typename Dst>
    inline void copy_record(const Src* src, Dst* dst) {
        const volatile uint32_t* s = reinterpret_cast<const volatile uint32_t*>(src);
        volatile uint32_t* d = reinterpret_cast<volatile uint32_t*>(dst);
        for (uint32_t w = 0; w < kRecordWords; w++) {
            d[w] = s[w];
        }
    }

    // What one pass over the ring found.
    struct Pass {
        int32_t pick;       // slot of the record to run (smallest sequence), -1: none
        uint32_t pick_seq;  // its sequence number
        bool partial;       // a newer record was seen that has not fully landed
    };

    // One pass over slots [0, depth): newer records not for `me` are consumed; the
    // newer landed EXIT or own record with the smallest sequence is picked (checked on
    // a copy).
    template <typename Word>
    inline Pass scan(
        const volatile km_record_t* ring, Word* last, uint32_t depth, uint32_t me
    ) {
        Pass p = {-1, 0, false};
        for (uint32_t j = 0; j < depth; j++) {
            const uint32_t s = ring[j].seq;
            if (!newer(s, last[j])) {
                continue;
            }
            km_record_t r;
            copy_record(&ring[j], &r);
            if (r.seq != s || !KM_LANDED(&r)) {
                p.partial = true;
                continue;
            }
            if (seg(r) != KM_SEG_EXIT && !in_team(r, me)) {
                last[j] = s;
                continue;
            }
            if (p.pick < 0 || newer(p.pick_seq, s)) {
                p.pick = static_cast<int32_t>(j);
                p.pick_seq = s;
            }
        }
        return p;
    }

}  // namespace km
