// Commands in the ring: header and group fields, sequence order, group test, a rank's
// partition of a range, the scan for a core's next command. Free of tt-metal headers.
#pragma once

#include <stdint.h>

#include "../../abi/km_abi.h"

namespace km {

    static_assert(sizeof(km_command_t) == 64, "a command is one 64-byte line");
    static_assert(sizeof(km_event_t) == 16, "an event is one 16-byte chunk");

    constexpr uint32_t kCommandWords = 16;

    inline uint32_t handler(const km_command_t& command) {
        return command.hdr & 0xFFu;
    }

    inline uint32_t flags(const km_command_t& command) {
        return (command.hdr >> 8) & 0xFFu;
    }

    inline uint32_t slot(const km_command_t& command) {
        return (command.hdr >> 16) & 0xFFu;
    }

    // The command's program: 0 the run's own (first, or only) program.
    inline uint32_t program(const km_command_t& command) {
        return command.hdr >> 24;
    }

    inline uint32_t first(const km_command_t& command) {
        return command.group & 0xFFFFu;
    }

    inline uint32_t count(const km_command_t& command) {
        return command.group >> 16;
    }

    inline uint32_t make_hdr(
        uint32_t handler, uint32_t flags, uint32_t slot, uint32_t program = 0
    ) {
        return handler | flags << 8 | slot << 16 | program << 24;
    }

    inline uint32_t make_group(uint32_t first, uint32_t count) {
        return first | count << 16;
    }

    // True when `sequence` comes after `other` (signed difference: valid across wrap).
    inline bool newer(uint32_t sequence, uint32_t other) {
        return static_cast<int32_t>(sequence - other) > 0;
    }

    // True when rank `me` (an island's first rank + its local core) is in the
    // command's group.
    inline bool in_group(const km_command_t& command, uint32_t me) {
        return me - first(command) <
            count(command);  // unsigned: me < first wraps to a large value
    }

    // True when `group` and `other` (first | count << 16) share a core: the test that
    // holds a KM_FLAG_EXCLUSIVE command while an overlapping group is in flight.
    inline bool groups_overlap(uint32_t group, uint32_t other) {
        const uint32_t first = group & 0xFFFFu;
        const uint32_t count = group >> 16;
        const uint32_t other_first = other & 0xFFFFu;
        const uint32_t other_count = other >> 16;
        return (
            count != 0 && other_count != 0 && first < other_first + other_count &&
            other_first < first + count
        );
    }

    // Rank `index` of `count` takes items [start, start + len) of `items`: contiguous
    // parts covering every item once, the first items % count cores one item more.
    struct Partition {
        uint32_t start;
        uint32_t len;
    };

    inline Partition partition(uint32_t items, uint32_t index, uint32_t count) {
        const uint32_t base = items / count, extra = items % count;
        return {index * base + (index < extra ? index : extra), base + (index < extra)};
    }

    // Where core `me` reports a group [first_core, + size) of cores (`parent`; first
    // core: dispatcher), after `want` reports. One row of `row` cores: one level.
    struct Report {
        uint32_t parent;
        uint32_t want;
    };

    inline Report report_in(
        uint32_t first_core, uint32_t size, uint32_t me, uint32_t row
    ) {
        const uint32_t end = first_core + size;
        if (row == 0 || first_core / row == (end - 1) / row) {
            return {first_core, me == first_core ? size - 1 : 0};
        }
        const uint32_t row_first = me / row * row;
        const uint32_t lead = row_first > first_core ? row_first : first_core;
        if (me != lead) {
            return {lead, 0};
        }
        const uint32_t row_end = row_first + row < end ? row_first + row : end;
        const uint32_t rows = me == first_core ? (end - 1) / row - first_core / row : 0;
        return {first_core, row_end - lead - 1 + rows};
    }

    // The same for the command's whole group.
    inline Report report_to(const km_command_t& command, uint32_t me, uint32_t row) {
        return report_in(first(command), count(command), me, row);
    }

    // Copies a command word by word (from a ring slot or into the job copy).
    template <typename Source, typename Destination>
    inline void copy_command(const Source* source, Destination* destination) {
        const volatile uint32_t*
            source_words = reinterpret_cast<const volatile uint32_t*>(source);
        volatile uint32_t* destination_words = (
            reinterpret_cast<volatile uint32_t*>(destination)
        );
        for (uint32_t word = 0; word < kCommandWords; word++) {
            destination_words[word] = source_words[word];
        }
    }

    // What one pass over the ring found.
    struct Pass {
        int32_t pick;  // slot of the command to run (smallest sequence), -1: none
        uint32_t pick_sequence;  // its sequence number
        bool partial;            // a newer command was seen that has not fully landed
    };

    // The core's own landed commands not yet taken, in the reader's memory: bit s of
    // `mask` set, slot s holds `sequence[s]`, and a pass reads only that slot's word 0.
    struct Waiting {
        uint32_t sequence[KM_RING_SLOTS];
        uint64_t mask;
    };

    // One pass over slots [0, depth): newer commands not for `me` (or `link`) are
    // consumed; the landed EXIT or own command of the smallest sequence is picked.
    template <typename Word>
    inline Pass scan(
        const volatile km_command_t* ring,
        Word* last,
        Waiting& waiting,
        uint32_t depth,
        uint32_t me,
        uint32_t link = 0
    ) {
        Pass found = {-1, 0, false};
        for (uint32_t slot_index = 0; slot_index < depth; slot_index++) {
            const volatile km_command_t& line = ring[slot_index];
            const uint32_t sequence = line.sequence;
            if (!newer(sequence, last[slot_index])) {
                continue;
            }
            const uint64_t bit = static_cast<uint64_t>(1) << slot_index;
            if (!(waiting.mask & bit) || waiting.sequence[slot_index] != sequence) {
                // Looked at afresh: the deciding words in place, word 0 last again.
                waiting.mask &= ~bit;
                km_command_t head;
                head.land1 = line.land1;
                head.land2 = line.land2;
                head.land3 = line.land3;
                head.hdr = line.hdr;
                head.group = line.group;
                head.sequence = line.sequence;
                if (head.sequence != sequence || !KM_LANDED(&head)) {
                    found.partial = true;
                    continue;
                }
                const uint32_t name = KM_GROUP_IS_LINK(first(head)) ? link : me;
                if (handler(head) != KM_HANDLER_EXIT && !in_group(head, name)) {
                    last[slot_index] = sequence;
                    continue;
                }
                waiting.sequence[slot_index] = sequence;
                waiting.mask |= bit;
            }
            if (found.pick < 0 || newer(found.pick_sequence, sequence)) {
                found.pick = static_cast<int32_t>(slot_index);
                found.pick_sequence = sequence;
            }
        }
        return found;
    }

    // The same with nothing remembered: every newer slot is looked at afresh.
    template <typename Word>
    inline Pass scan(
        const volatile km_command_t* ring,
        Word* last,
        uint32_t depth,
        uint32_t me,
        uint32_t link = 0
    ) {
        Waiting none = {};
        return scan(ring, last, none, depth, me, link);
    }

}  // namespace km
