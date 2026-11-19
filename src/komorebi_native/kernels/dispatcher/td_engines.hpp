// Out's side of the resident engines (td_engine.hpp, TRISC 0): the host's RUN and
// CANCEL handed to them, their POST and DROP, and the graph table.

// The X280's engine_port.h and graph_port.h. Their other commands are source 4's lines
// (dispatcher.cpp), their events go back through in (td_in.hpp).
#pragma once

#include <stdint.h>

#include "dispatcher.h"
#include "km_dispatch.h"
#include "komorebi/abi/km_graph.h"
#include "komorebi/abi/km_stream.h"
#include "komorebi/device/core/km_core.hpp"
#include "td_queue.hpp"
#include "td_sources.hpp"

namespace td {

    // Out to the engines: {table entry, kEventGraphRun, command}: run its graph.
    constexpr uint32_t kEventGraphRun = 0x18;  // the X280's KMD_EVENT_GRAPH_RUN
    constexpr uint32_t kGraphEntries = KM_GRAPH_MAX_KEPT + KM_GRAPH_MAX_ACTIVE;
    // A table entry's state (KMD_GT_*).
    constexpr uint32_t kEntryFree = 0, kEntryRunning = 1, kEntryKept = 2;

    // A graph of the table, one line (kmd_gentry_t): out writes it, TRISC 0 reads it
    // after its kEventGraphRun.
    struct GraphEntry {
        uint32_t state;
        uint32_t request, offset, bytes;
        uint32_t binding_offset, binding_count;
        uint32_t bind_ok;  // the binding block lies in the region
        uint32_t keep;     // the header's keep bit
        uint32_t full;     // the kept table had no room at the RUN
        uint32_t again;    // a run of a kept graph: it stays kept, whatever its end
        uint32_t pad[6];
    };

    static_assert(sizeof(GraphEntry) == 64, "a table entry is one line");
    static_assert(kTeTable + kGraphEntries * sizeof(GraphEntry) <= kTeState);

    // A stream out let run (kmd_run_t).
    struct EngineRun {
        uint32_t active, request, offset, bytes;
    };

    // Out's side: what runs, the events owed to the engines, their deadline.
    struct Engines {
        uint32_t area = 0;     // their area (kTe*)
        uint32_t caused = 0;   // events to them, written or owed
        uint32_t taken = 0;    // their jobs taken into the queue, or rejected
        uint32_t streams = 0;  // streams running
        uint32_t graphs = 0;   // graphs running
        uint32_t told_queued = 0, told_taken = 0;
        uint64_t deadline = 0;  // ticks of each command of theirs, submit to ack
        EngineRun runs[KM_STREAM_MAX_ACTIVE];

        // Zeroes an earlier load's lines, events, words and table.
        void init(uint32_t at, uint64_t ticks) {
            area = at;
            deadline = ticks;
            for (uint32_t i = 0; i < kTeState / 4; i++) {
                km::l1(area)[i] = 0;
            }
            for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
                runs[i].active = 0;
            }
        }

        volatile tt_l1_ptr uint32_t* consumed() const {
            return km::l1(area + kTeWork + KMD_WORK_CONSUMED);
        }

        volatile tt_l1_ptr GraphEntry* entry(uint32_t index) const {
            return (
                reinterpret_cast<volatile tt_l1_ptr GraphEntry*>(
                    area + kTeTable + index * sizeof(GraphEntry)
                )
            );
        }

        // Whether one more event fits their ring.
        bool room() const {
            return caused - km::l1(area + kTeRead)[0] < kTeEventDepth;
        }

        bool busy() const {
            return streams != 0 || graphs != 0;
        }

        // The upload region's L1 address.
        uint32_t upload() const {
            return area + kTeUpload;
        }

        // Word `offset` of the upload region with byte loads (it may be unaligned), 0
        // outside.
        uint32_t word(uint32_t offset) const {
            if (offset > kTeUploadBytes - 4u) {
                return 0;
            }
            const volatile tt_l1_ptr uint8_t*
                bytes = (
                    reinterpret_cast<volatile tt_l1_ptr uint8_t*>(upload() + offset)
                );
            return bytes[0] | bytes[1] << 8 | bytes[2] << 16 | bytes[3] << 24;
        }

        // The host's deadline (microseconds), for their streams that name none.
        void host_deadline(uint32_t microseconds) const {
            km::l1(area + kTeOut)[kToDeadline] = microseconds;
        }

        // The queue's length and their jobs taken, when changed.
        void tell(uint32_t queued) {
            if (queued != told_queued || taken != told_taken) {
                volatile tt_l1_ptr uint32_t* out = km::l1(area + kTeOut);
                out[kToQueued] = told_queued = queued;
                out[kToTaken] = told_taken = taken;
            }
        }

        // Whether a graph of the table has `request` or a range that overlaps
        // [offset, offset + bytes).
        bool claims(uint32_t offset, uint32_t bytes, uint32_t request) const {
            for (uint32_t i = 0; i < kGraphEntries; i++) {
                volatile tt_l1_ptr GraphEntry* at = entry(i);
                if (
                    at->state != kEntryFree &&
                    (
                        at->request == request ||
                        (offset < at->offset + at->bytes && at->offset < offset + bytes)
                    )
                ) {
                    return true;
                }
            }
            return false;
        }

        // Whether a stream that runs has `request` or overlaps the range.
        bool overlaps(uint32_t offset, uint32_t bytes, uint32_t request) const {
            for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
                const EngineRun& run = runs[i];
                if (
                    run.active &&
                    (
                        run.request == request ||
                        (offset < run.offset + run.bytes && run.offset < offset + bytes)
                    )
                ) {
                    return true;
                }
            }
            return false;
        }

        // A line of theirs out serves itself: the host's RUN or CANCEL, their POST or
        // DROP.
        static bool own(uint32_t source, uint32_t handler) {
            return (
                source == kSourceHost
                    ? handler == KM_HANDLER_RUN || handler == KM_HANDLER_CANCEL
                    : source == kSourceEngine &&
                        (handler == KMD_HANDLER_POST || handler == KMD_HANDLER_DROP)
            );
        }

        // Serves `line`, command `request` of `source` (own() said so): false while it
        // waits. `posts`: the free posts; host_owed, host_room(): the host's events.
        template <typename Post, typename HostRoom>
        bool serve(
            const uint32_t* line,
            uint32_t request,
            uint32_t source,
            bool exited,
            Queue& queue,
            uint64_t& before,
            uint32_t posts,
            uint32_t& host_owed,
            Post post,
            HostRoom host_room
        ) {  // fmt: skip
            const uint32_t handler = line[1] & 0xFFu;
            if (posts == 0) {
                return false;
            }
            if (source == kSourceEngine && handler == KMD_HANDLER_DROP) {
                return drop(line[3], queue, before, posts, post);
            }
            if (source == kSourceEngine) {
                return post_line(line, host_owed, post, host_room);
            }
            if (handler == KM_HANDLER_CANCEL) {
                return cancel(request, line[3], host_owed, post);
            }
            if (line[6] != 0) {
                return graph_run(line, request, exited, host_owed, post);
            }
            return run(request, line[3], line[5], exited, host_owed, post);
        }

        // RUN of a stream (`offset`, `bytes`): rejected, ended with a header error, or
        // handed to TRISC 0, its end's room held.
        template <typename Post>
        bool run(
            uint32_t command,
            uint32_t offset,
            uint32_t bytes,
            bool exited,
            uint32_t& host_owed,
            Post post
        ) {  // fmt: skip
            const uint32_t stream_request = (
                offset <= kTeUploadBytes - sizeof(km_stream_t) && bytes >= 28u
                    ? word(offset + 24u)
                    : 0u
            );
            if (exited || streams == KM_STREAM_MAX_ACTIVE) {
                const uint32_t why = exited ? KMD_REJECT_EXITED : KMD_REJECT_STREAMS;
                post(command, KM_EVENT_REJECTED | why << 8, 0, kSourceHost);
                host_owed++;
                return true;
            }
            const bool bad = (
                (offset & 63u) || offset > kTeUploadBytes ||
                bytes > kTeUploadBytes - offset || bytes < sizeof(km_stream_t) ||
                overlaps(offset, bytes, stream_request) ||
                claims(offset, bytes, stream_request)
            );
            if (bad) {
                const uint32_t why = KM_STREAM_ERROR_HEADER;
                post(stream_request, KM_EVENT_STREAM_ERROR | why << 8, 0, kSourceHost);
                host_owed++;
                return true;
            }
            if (!room()) {
                return false;
            }
            post(offset, KMD_EVENT_RUN, bytes, kSourceEngine);
            caused++;
            for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
                if (!runs[i].active) {
                    runs[i] = {1, stream_request, offset, bytes};
                    break;
                }
            }
            streams++;
            host_owed++;  // its end's
            return true;
        }

        // RUN of a graph (a2 the form): rejected, ended with a header error, or handed
        // to TRISC 0 through table entry, its end's room held.
        template <typename Post>
        bool graph_run(
            const uint32_t* line,
            uint32_t command,
            bool exited,
            uint32_t& host_owed,
            Post post
        ) {  // fmt: skip
            const uint32_t at = line[3], bytes = line[5], form = line[6];
            const uint32_t binding_offset = line[7], binding_count = line[9];
            uint32_t why = 0;
            if (form > KM_GRAPH_RUN_KEPT) {
                why = KMD_REJECT_HANDLER;
            } else if (exited || graphs == KM_GRAPH_MAX_ACTIVE) {
                why = exited ? KMD_REJECT_EXITED : KMD_REJECT_STREAMS;
            }
            uint32_t kept = 0, request = at, index = kGraphEntries;
            volatile tt_l1_ptr GraphEntry* chosen = nullptr;
            for (uint32_t i = 0; i < kGraphEntries; i++) {
                volatile tt_l1_ptr GraphEntry* candidate = entry(i);
                kept += (
                    candidate->state == kEntryKept ||
                    (
                        candidate->state == kEntryRunning &&
                        (candidate->again || (candidate->keep && !candidate->full))
                    )
                );
                if (
                    why == 0 && form == KM_GRAPH_RUN_KEPT && chosen == nullptr &&
                    candidate->state == kEntryKept && candidate->request == at
                ) {
                    chosen = candidate;
                    index = i;
                }
            }
            if (why == 0 && form == KM_GRAPH_RUN_KEPT && chosen == nullptr) {
                why = KMD_REJECT_UNKNOWN;
            }
            if (why != 0) {
                post(command, KM_EVENT_REJECTED | why << 8, 0, kSourceHost);
                host_owed++;
                return true;
            }
            if (form != KM_GRAPH_RUN_KEPT) {
                request = bytes >= 28u ? word(at + 24u) : 0u;
                const bool bad = (
                    (at & 63u) || at > kTeUploadBytes || bytes > kTeUploadBytes - at ||
                    overlaps(at, bytes, request) || claims(at, bytes, request)
                );
                for (
                    uint32_t i = 0; i < kGraphEntries && !bad && chosen == nullptr; i++
                ) {
                    chosen = entry(i)->state == kEntryFree ? entry(i) : nullptr;
                    index = i;
                }
                if (bad || chosen == nullptr) {
                    const uint32_t error = (
                        KM_STREAM_ERROR_HEADER | KM_GRAPH_NO_NODE << 8
                    );
                    post(request, KM_EVENT_GRAPH_ERROR | error << 8, 0, kSourceHost);
                    host_owed++;
                    return true;
                }
            }
            if (!room()) {
                return false;
            }
            if (form == KM_GRAPH_RUN_KEPT) {
                chosen->again = 1;
                chosen->full = 0;
            } else {
                const uint32_t flags = bytes >= 48u ? word(at + 44u) : 0u;
                chosen->request = request;
                chosen->offset = at;
                chosen->bytes = bytes;
                chosen->keep = flags & KM_GRAPH_KEEP;
                chosen->full = chosen->keep && kept >= KM_GRAPH_MAX_KEPT;
                chosen->again = 0;
            }
            chosen->binding_offset = binding_offset;
            chosen->binding_count = binding_count;
            chosen->bind_ok = (
                binding_count == 0 ||
                (
                    !(binding_offset & 63u) && binding_offset <= kTeUploadBytes &&
                    binding_count <= (kTeUploadBytes - binding_offset) / 4u
                )
            );
            chosen->state = kEntryRunning;
            post(index, kEventGraphRun, command, kSourceEngine);
            caused++;
            graphs++;
            host_owed++;  // its end's
            return true;
        }

        // CANCEL of `request`: a running graph or stream is told, a kept graph dropped
        // (its error event); else rejected.
        template <typename Post>
        bool cancel(
            uint32_t command, uint32_t request, uint32_t& host_owed, Post post
        ) {
            bool running = false;
            for (uint32_t i = 0; i < kGraphEntries; i++) {
                volatile tt_l1_ptr GraphEntry* at = entry(i);
                if (at->state == kEntryFree || at->request != request) {
                    continue;
                }
                if (at->state == kEntryKept) {
                    at->state = kEntryFree;
                    const uint32_t why = (
                        KM_STREAM_ERROR_CANCELLED | KM_GRAPH_NO_NODE << 8
                    );
                    post(request, KM_EVENT_GRAPH_ERROR | why << 8, 0, kSourceHost);
                    host_owed++;
                    return true;
                }
                running = true;
            }
            for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
                running = running || (runs[i].active && runs[i].request == request);
            }
            if (!running) {
                post(
                    command, KM_EVENT_REJECTED | KMD_REJECT_UNKNOWN << 8, 0, kSourceHost
                );
                host_owed++;
                return true;
            }
            if (!room()) {
                return false;
            }
            post(command, KMD_EVENT_CANCEL, request, kSourceEngine);
            caused++;
            return true;
        }

        // Their POST {a0 request, a1 status, a2 ticks, a3 kept}: an end's event uses
        // its held room (its stream or entry freed or kept), others need room.
        template <typename Post, typename HostRoom>
        bool post_line(
            const uint32_t* line, uint32_t& host_owed, Post post, HostRoom host_room
        ) {
            const uint32_t request = line[3], status = line[5], ticks = line[6];
            const uint32_t kind = status & 0xFFu;
            if (kind == KM_EVENT_GRAPH_END || kind == KM_EVENT_GRAPH_ERROR) {
                for (uint32_t i = 0; i < kGraphEntries; i++) {
                    volatile tt_l1_ptr GraphEntry* at = entry(i);
                    if (at->state == kEntryRunning && at->request == request) {
                        const bool kept = at->again || line[7] != 0;
                        at->state = kept ? kEntryKept : kEntryFree;
                        graphs--;
                        post(request, status, ticks, kSourceHost);
                        return true;
                    }
                }
            }
            if (kind == KM_EVENT_STREAM_END || kind == KM_EVENT_STREAM_ERROR) {
                for (uint32_t i = 0; i < KM_STREAM_MAX_ACTIVE; i++) {
                    if (runs[i].active && runs[i].request == request) {
                        runs[i].active = 0;
                        streams--;
                        post(request, status, ticks, kSourceHost);
                        return true;
                    }
                }
            }
            if (!host_room()) {
                return false;
            }
            post(request, status, ticks, kSourceHost);
            host_owed++;
            return true;
        }

        // DROP of tag `tag`: their queued jobs of it leave the queue, each answered
        // REJECTED, reason DROPPED (its room held since it was queued).
        template <typename Post>
        bool drop(
            uint32_t tag, Queue& queue, uint64_t& before, uint32_t posts, Post post
        ) {
            for (uint32_t position = 0; position < queue.length;) {
                volatile tt_l1_ptr uint32_t* job = queue.job(queue.order[position]);
                if (job[kJobSource] != kSourceEngine || (job[2] & 0xFFFFu) != tag) {
                    position++;
                    continue;
                }
                if (posts == 0) {
                    return false;  // the rest at a later pass
                }
                post(
                    job[kJobRequest],
                    KM_EVENT_REJECTED | KMD_REJECT_DROPPED << 8,
                    0,
                    kSourceEngine
                );
                posts--;
                queue.remove(position, before);
            }
            return true;
        }
    };

}  // namespace td
