/* Komorebi ABI: the definitions shared by the Tensix kernels, the X280 dispatcher
   and the host. Plain C, usable from C and C++. */
#ifndef KOMOREBI_ABI_KM_ABI_H
#define KOMOREBI_ABI_KM_ABI_H

#include <stdint.h>

#if defined(__cplusplus)
#define KM_STATIC_ASSERT(c, msg) static_assert(c, msg)
#else
#define KM_STATIC_ASSERT(c, msg) _Static_assert(c, msg)
#endif

/* Command slots of every core's ring, 64 bytes each: the most commands in flight. The
   ring is the last part of the control block. */
#define KM_RING_SLOTS 64u
#define KM_CORES_MAX 128u
#define KM_HANDLER_NOP 0u
#define KM_HANDLER_EXIT 255u /* every core leaves its kernels */
#define KM_HANDLER_PING 254u /* host -> dispatcher only: answer with an event */
#define KM_HANDLER_STOP 253u /* host -> dispatcher only: stop the grid and finish */
/* host -> dispatcher only: a0 = microseconds for later commands */
#define KM_HANDLER_DEADLINE 252u
#define KM_HANDLER_USER_MAX 239u /* handlers 1..239 belong to the program */
/* host -> dispatcher only: a0 = offset of a stream upload, a1 = its bytes */
#define KM_HANDLER_RUN 251u
/* host -> dispatcher only: a0 = request id of the stream to cancel */
#define KM_HANDLER_CANCEL 250u
/* host only: the group's cores hold the program in header bits 24 to 31 from now on;
 * a0 = the DRAM address of its record (its table entries and buffers) */
#define KM_HANDLER_LOAD 240u
/* host only, served by the runtime's service overlay: the group's cores push slices of
 * DRAM pages into the session's region of the host channel, or pull them into DRAM */
#define KM_HANDLER_PUSH 241u
#define KM_HANDLER_PULL 242u
/* Handlers 241 to 248 are the runtime's: the reader's service overlay serves them, and
 * no writer or compute kernel runs a body for one (one range check in each loop). */
#define KM_HANDLER_SERVED_FIRST 241u
#define KM_HANDLER_SERVED_LAST 248u
/* A transfer's words a0 to a9: the tensor (bank address, aligned page bytes, first
 * page, pages), the channel (offsets of the data and of the word slots, the region): */
#define KM_TRANSFER_ADDRESS 0u
#define KM_TRANSFER_PAGE 1u
#define KM_TRANSFER_FIRST 2u
#define KM_TRANSFER_PAGES 3u
#define KM_TRANSFER_OFFSET 4u
#define KM_TRANSFER_WORDS 5u
#define KM_TRANSFER_REGION 6u
#define KM_TRANSFER_NONCE 7u
#define KM_TRANSFER_BUFFER 8u /* L1: two chunks, then the word slot */
#define KM_TRANSFER_CHUNK 9u  /* bytes, whole pages */
/* A chunk of KM_TRANSFER_L1: a tensor in L1, one page a rank at `address` in that
 * rank's own L1, moved as one write or read; the word slot at `buffer` */
#define KM_TRANSFER_L1 0u
/* a rank's slot: its sequence word (nonce, bytes, rank, the mark ^ nonce), then the
 * wall clock at its start and after its data, written once its data is done */
#define KM_TRANSFER_WORD_BYTES 64u
#define KM_TRANSFER_MARK 0x50555348u
#define KM_CHANNEL_BYTES 0x40000000u /* the host channel: no offset reaches past it */
/* each of a run's two L1 chunks on every core */
#define KM_TRANSFER_CHUNK_BYTES 16384u
/* host or a work cluster's thread, served by the service overlay: the group's cores
 * copy their slices of DRAM pages through L1 into other DRAM pages (km_copy.h) */
#define KM_HANDLER_COPY 245u
/* host, served by the service overlay: each core of the group writes the table entry
 * of an overlay added to the live run (run.add) in its own overlay area */
#define KM_HANDLER_ENTRY 246u
#define KM_ENTRY_NUMBER 0u /* the overlay's number | KM_ENTRY_HANDLER */
/* the handler of that number is the overlay's parts */
#define KM_ENTRY_HANDLER 0x100u
#define KM_ENTRY_MESSAGE 0x200u /* the dynamic handler of that number is its part */
#define KM_ENTRY_TABLE 0x400u   /* a tensor's table row: a1 its word, a2 the row */
#define KM_ENTRY_ADDRESS 1u     /* a1 to a5: each part's DRAM address in every bank */
#define KM_ENTRY_BYTES 6u       /* a6 to a8: each part's bytes / 64, two parts a word */
/* host, served by the service overlay: every rank of the group calls km::all_gather
 * or km::all_reduce (km_ask.hpp) on pages of a copy tensor, with these words */
#define KM_HANDLER_GATHER 243u
#define KM_HANDLER_REDUCE 244u
#define KM_COLL_TENSOR 0u  /* its reference word */
#define KM_COLL_FIRST 1u   /* first page */
#define KM_COLL_PAGES 2u   /* pages */
#define KM_COLL_SCRATCH 3u /* reduce: a copy of 2 x pages pages, the same page size */
#define KM_COLL_SUM 4u     /* reduce: km::Sum, 0 int32, 1 float32, 2 bfloat16 */
#define KM_COLL_BUFFER 5u  /* reduce: L1 of every rank for the folds */
#define KM_COLL_ROOM 6u    /* reduce: its bytes, two pages or more */
/* served on link cores (km_pages.hpp): host: each CPU sends its pages share over its
 * link, or the reader waits a count; dispatcher: link 0 sends a staged packet */
#define KM_HANDLER_PAGES 248u
#define KM_PAGES_FROM 0u  /* send: the source's bank address (interleaved DRAM) */
#define KM_PAGES_PAGE 1u  /* send: aligned page bytes, at most the island's packet */
#define KM_PAGES_FIRST 2u /* send: first page */
/* send: pages of all CPUs; range: pages a rank; wait: the count waited for */
#define KM_PAGES_PAGES 3u
#define KM_PAGES_TO 4u /* send: the target's bank address, or L1 of its first page */
/* send: x | y << 8 of the target core (KM_PAGES_DRAM: DRAM), range: the first rank's
   far core, chunk / 64 << 16: the L1 bytes of a CPU */
#define KM_PAGES_TO_CORE 5u
/* L1 address of the count (wait: here; open: far hello; range: on each rank) */
#define KM_PAGES_COUNT 6u
/* x | y << 8 of the counted core (open: the far link core; range: the ranks), hops
   << 16, mode << 24 */
#define KM_PAGES_COUNTER 7u
/* byte k: Ethernet channel of link k; CPU c of link core r sends on link 2r + c.
   Link 0 carries the lines (LINE); pages go on links 1 on, on link 0 when alone. */
#define KM_PAGES_LINKS 8u
/* send: L1 of CPU c at buffer + c * chunk (bits 0..23), the push's shape << 24 */
#define KM_PAGES_BUFFER 9u
/* a send's push shape (0: the loop's; others measure): pages a read 4 ** bits 0..1,
 * slots 15 << bits 2..3 - 1, bit 4 no read, 5 no L1 wait, 6 link 0's CPU shares */
#define KM_PAGES_SHAPE_SHIFT 24u
#define KM_PAGES_SHAPE_NO_READ 0x10u
#define KM_PAGES_SHAPE_NO_WAIT 0x20u
#define KM_PAGES_SHAPE_LINK0 0x40u
#define KM_PAGES_SHAPE_MEASURE 0x3Fu /* the bits that measure the loop */
#define KM_PAGES_DRAM 0xFFFFu /* page p into page p of the tensor at KM_PAGES_TO */
/* a page CPU's DRAM side in bank b's range (page i at address + i * page): b | this */
#define KM_PAGES_BANK 0xFF00u
#define KM_PAGES_NO_LINK 0xFFu
#define KM_PAGES_SEND 0u
#define KM_PAGES_WAIT 1u
/* each CPU opens its connection, held until its kernel ends, and says hello to the far
 * link core's same CPU: port KM_PAGES_HELLO_PORT + c of both, 1 up, 2 both up */
#define KM_PAGES_OPEN 2u
#define KM_PAGES_HELLO_PORT 5u
/* dispatcher -> link core 0 of its island: CPU 0 (link 0) sends the packet staged in
 * the link area, then its count line, into the other island's dispatcher */
#define KM_PAGES_LINE 3u
/* pages a rank for each of ranks r (far cores from KM_PAGES_TO_CORE): pages first + r *
 * pages on into L1 KM_PAGES_TO of core r, counted at KM_PAGES_COUNT */
#define KM_PAGES_RANGE 4u
/* link 0's CPU of link core 0 serves the messages of the RUN whose group word is a0
 * across its link until that RUN ends (km_link.hpp) */
#define KM_PAGES_MESSAGES 5u
/* each CPU of the link cores named: its row of what crossed (KM_LINK_CROSS) into page
 * 2 k + c of the DRAM tensor at KM_PAGES_TO (k its link core, c the CPU) */
#define KM_PAGES_COUNTS 6u
/* the largest packet payload the routers take (Blackhole): the bound of an island's
 * packet word (KM_ISLAND_PACKET) */
#define KM_PACKET_LARGEST 15232u
/* LINE's words: a0 stage, a1 packet number, a2 far tile x | y << 16, a3 far packet
 * address, a4 count line's, a5 link area, a6 its island; a7 as above */
#define KM_LINE_STAGE 0u
#define KM_LINE_NUMBER 1u
#define KM_LINE_TILE 2u
#define KM_LINE_PACKET 3u
#define KM_LINE_COUNT 4u
#define KM_LINE_AREA 5u
#define KM_LINE_ISLAND 6u
/* The link area, in link core 0's L1 (a run over islands; offsets): two packet headers,
 * the stages (a command line and a crossing line each), then the count line */
#define KM_LINK_HEADERS 0x40u /* 64 bytes each: the packet's, the count line's */
#define KM_LINK_STAGE 0x100u
#define KM_LINK_STAGES 4u
#define KM_LINK_PACKET_BYTES 128u
#define KM_LINK_STAGE_BYTES 192u
/* Then asks for pages another island holds (km_ask.hpp): counts and answers on link
 * core 0, a page CPU's answers, the lines, the far asks */
#define KM_PAGE_CPUS 4u
/* a link core's own L1 (KM_LINK_QUEUE): the put queue, then with KM_LINK_CHUNK two
 * chunks; an index above every other (the trace's is 31; a Blackhole has 64) */
#define KM_LINK_CB 32u
/* The put queue, on link core 0 (KM_LINK_CB, at KM_PUT_QUEUE): up to KM_QUEUE_LINES
 * unconfirmed put lines, the far lines as they arrive, pages landed per owner */
#define KM_QUEUE_LINES (4u * KM_CORES_MAX) /* a core's two CPUs, two lines each */
#define KM_QUEUE_WORDS 0u
#define KM_QUEUE_PENDING_COUNT 0u /* a word of KM_QUEUE_WORDS */
#define KM_QUEUE_PUTS 0x40u
#define KM_QUEUE_ARRIVALS (KM_QUEUE_PUTS + KM_QUEUE_LINES * 64u)
#define KM_QUEUE_ARRIVED (KM_QUEUE_ARRIVALS + KM_QUEUE_LINES * 64u)
#define KM_QUEUE_PENDING (KM_QUEUE_ARRIVED + KM_QUEUE_LINES * 4u)
#define KM_QUEUE_ENTRY 32u
#define KM_QUEUE_E_OWNER 0u
#define KM_QUEUE_E_PAGES 1u
#define KM_QUEUE_E_CORE 2u   /* the asker, x | y << 16 */
#define KM_QUEUE_E_COUNT 3u  /* its port count's L1 address */
#define KM_QUEUE_E_NUMBER 4u /* the line's number word */
#define KM_QUEUE_E_KEEPER 5u /* a gather's keeper (KM_ASK_KEEPER), 0: none */
#define KM_QUEUE_BYTES (KM_QUEUE_PENDING + KM_QUEUE_LINES * KM_QUEUE_ENTRY)
#define KM_ASK_TAIL (KM_LINK_STAGE + KM_LINK_STAGES * KM_LINK_STAGE_BYTES)
#define KM_ASK_TAKEN (KM_ASK_TAIL + 0x10u)
#define KM_ASK_SERVED (KM_ASK_TAIL + 0x20u)   /* + 4 j: page CPU j's */
#define KM_ASK_ANSWERED (KM_ASK_TAIL + 0x30u) /* + 4 cpu: the core's CPU's */
#define KM_ASK_BAD (KM_ASK_TAIL + 0x38u)
#define KM_ASK_OWN (KM_ASK_TAIL + 0x40u)
#define KM_ASK_RETURNED (KM_ASK_OWN + 0x40u)
#define KM_ASK_READ (KM_ASK_OWN + 0x50u)
#define KM_ASK_HEADERS (KM_ASK_OWN + 0x80u)
#define KM_ASK_LINES (KM_ASK_HEADERS + 0x80u)
#define KM_ASK_SLOTS 32u
#define KM_ASK_INBOX (KM_ASK_LINES + KM_ASK_SLOTS * 64u)
/* A line of the link service's one queue (64 bytes; words 0 and 15 its number): bank
 * address, page bytes, first page, pages, asker's tile, L1 and port, islands, kind */
#define KM_ASK_NUMBER 0u
#define KM_ASK_TENSOR 1u
#define KM_ASK_PAGE 2u
#define KM_ASK_FIRST 3u
#define KM_ASK_PAGES 4u
#define KM_ASK_CORE 5u
#define KM_ASK_TO 6u
#define KM_ASK_COUNT 7u
#define KM_ASK_HOLDER 8u /* the destination island */
#define KM_ASK_FROM 9u
#define KM_ASK_DRAM 10u
#define KM_ASK_DRAM_FIRST 11u
#define KM_ASK_RING 12u /* the kind: op << KM_ASK_OP_SHIFT | form (KM_ASK_FORM) */
#define KM_ASK_OP_SHIFT 8u
/* plain 0, into the rings 1, bytes of a page 2, a batch 3 */
#define KM_ASK_FORM 0xFFu
#define KM_ASK_OP_WRITE 3u /* op 3: pages or bytes into the holder's DRAM */
#define KM_ASK_OP_READ 4u  /* op 4: pages or bytes of the holder's into the asker */
/* op 1, reserved: an L1 write; word 1 the target rank, 4 the bytes, 6 the target's L1
 * address, 7 the asker's count (0: none), 8 the rank's island; the rest as op 3 */
#define KM_ASK_OP_L1 1u
/* op 2, reserved: an add to a word; word 1 the target rank, 4 the value, 6 the word's
 * L1 address, 7 the asker's count (0: none), 8 the rank's island */
#define KM_ASK_OP_ADD 2u
#define KM_ASK_OWNER 13u /* a put: (core * 2 + its CPU) * 2 + parity, the asker's */
#define KM_ASK_END 15u
/* A batch line (km::put_bytes): its batch, page KM_ASK_FIRST of the landing, follows
 * in the rings; the far lead page CPU writes its records, then confirms (no page) */
#define KM_ASK_RING_BATCH 3u
/* Then what crossed, a row a CPU of the core, counted only on a link core by that CPU:
 * messages, command lines, asks, pages, relayed parts, bytes (low, high) it sent */
#define KM_LINK_CROSS (KM_ASK_INBOX + KM_ASK_SLOTS * 64u)
#define KM_CROSS_ROW 32u
#define KM_CROSS_MESSAGES 0u
#define KM_CROSS_LINES 1u
#define KM_CROSS_ASKS 2u
#define KM_CROSS_PAGES 3u
#define KM_CROSS_PARTS 4u
#define KM_CROSS_BYTES 5u /* the low word; the high word after it */
#define KM_CROSS_PUTS 7u  /* put lines and their confirmations */
/* A get line of part of a page (km::read_bytes): KM_ASK_RING KM_ASK_GET_BYTES, 1 page
 * (it counts 1), KM_ASK_GET_SPAN offset | bytes << 16, the bytes into KM_ASK_TO. */
#define KM_ASK_GET_BYTES 2u
#define KM_ASK_GET_SPAN 14u
/* A put line from DRAM may name keepers (word 14, km::keeper): once its pages are all
 * written, keeper k of the far island counts share k at its port. 0: none. */
#define KM_ASK_KEEPER 14u
#define KM_KEEPER_RANKS_SHIFT 8u
#define KM_KEEPER_PORT_SHIFT 16u
/* Then a core's puts into another island's DRAM (km::write_pages), as the asks: link
 * core 0's counts and each ring's address, each CPU's ring words, the put lines */
#define KM_PUT_TAIL (KM_LINK_CROSS + 2u * KM_CROSS_ROW)
#define KM_PUT_TAKEN (KM_PUT_TAIL + 0x10u) /* + 4 j: page CPU j's */
#define KM_PUT_DONE (KM_PUT_TAIL + 0x20u)
#define KM_PUT_CONFIRMED (KM_PUT_TAIL + 0x24u)
#define KM_PUT_BAD (KM_PUT_TAIL + 0x28u)
#define KM_PUT_CPUS (KM_PUT_TAIL + 0x2Cu) /* read with the counts taken: 32 bytes */
#define KM_PUT_RING (KM_PUT_TAIL + 0x30u) /* + 4 j: page CPU j's ring */
#define KM_PUT_RING_CORE (KM_PUT_TAIL + 0x40u)  /* + 4 j */
#define KM_PUT_RING_SLOTS (KM_PUT_TAIL + 0x50u) /* | 1 << 31 once published */
#define KM_PUT_RING_FAR (KM_PUT_TAIL + 0x54u)   /* the far link core 0, x | y << 16 */
/* the put queue's L1 address, 0: none yet */
#define KM_PUT_QUEUE (KM_PUT_TAIL + 0x58u)
#define KM_PUT_LANDING (KM_PUT_TAIL + 0x5Cu) /* the host's: the far landing */
#define KM_PUT_LANDED (KM_PUT_TAIL + 0x60u)  /* the host's: this island's */
/* A CPU's batch: used, records, island, sequence; records of address, page, page
 * bytes, offset | bytes << 20, the bytes at offset % 16. Landing: island*lines+owner */
#define KM_PUT_BATCH_BYTES 0x800u
#define KM_BATCH_USED 0u
#define KM_BATCH_RECORDS 1u
#define KM_BATCH_HOLDER 2u
#define KM_BATCH_SEQUENCE 3u
#define KM_BATCH_HEAD 16u /* the batch's header and a record's */
#define KM_BATCH_SPAN_SHIFT 20u
#define KM_PUT_RING_READ 0x50u /* what a putting core reads, from KM_PUT_TAKEN */
#define KM_PUT_OWN (KM_PUT_TAIL + 0x80u) /* + 0x80 the second CPU's */
#define KM_PUT_OWN_BYTES 0x80u           /* unused (a core's own words, KM_OWN_*) */
/* The lead page CPU's wall clock at a put line's steps (in link core 0's unused
 * KM_PUT_OWN), then its loop's passes */
#define KM_PUT_STAMPS KM_PUT_OWN
#define KM_STAMP_TAKEN 0u
#define KM_STAMP_ACROSS 1u
#define KM_STAMP_PUSHED 2u
#define KM_STAMP_ARRIVED 3u
#define KM_STAMP_CONFIRMED 4u
#define KM_STAMP_PASSES 5u
#define KM_STAMP_WORDS 8u
#define KM_PUT_RING_WORDS (KM_PUT_OWN + 2u * KM_PUT_OWN_BYTES) /* + 16 a CPU */
#define KM_PUT_RING_TAIL 0u
#define KM_PUT_RING_SENT 4u
#define KM_PUT_RING_SENDING 8u /* cycles its turns sent (COUNTS reads them) */
#define KM_PUT_RING_ROUTER 12u /* of them, cycles waiting for the router's slots */
/* + 16 a CPU: the cycles between its turns, by its ring's first slot at the next: */
#define KM_PUT_RING_IDLE (KM_PUT_RING_WORDS + 32u)
#define KM_PUT_RING_LATE 0u    /* ready (the CPU served something else) */
#define KM_PUT_RING_FILLING 4u /* reserved, its room shown, not yet written */
#define KM_PUT_RING_TURNS 8u   /* (not cycles) the turns that sent */
#define KM_PUT_RING_EMPTY 12u  /* not reserved */
#define KM_PUT_HEADERS (KM_PUT_OWN + 2u * KM_PUT_OWN_BYTES + 0x40u)
#define KM_PUT_LINES (KM_PUT_HEADERS + 0xC0u)
#define KM_PUT_INBOX (KM_PUT_LINES + KM_ASK_SLOTS * 64u)
/* the headers a CPU builds for a put into the rings (KM_OWN_BUILD) */
#define KM_PUT_BUILD_BYTES 0x800u
/* Then a page CPU's own (0x100 a CPU): its ask line and put line copies, its answer
 * header; on link core 0 the pages landed of far put line i (KM_PUT_ARRIVED) */
#define KM_PAGE_CPU (KM_PUT_INBOX + KM_ASK_SLOTS * 64u)
#define KM_PAGE_CPU_BYTES 0x100u
#define KM_PAGE_CPU_ASK 0u
#define KM_PAGE_CPU_PUT 0x40u
#define KM_PAGE_CPU_HEADER 0x80u
#define KM_PUT_ARRIVED(i) \
    (KM_PAGE_CPU + ((i) >> 4) * KM_PAGE_CPU_BYTES + 0xC0u + ((i) & 15u) * 4u)
#define KM_LINK_AREA_BYTES (KM_PAGE_CPU + 2u * KM_PAGE_CPU_BYTES)
/* Flags: bit 0 is reserved and zero. Every command reports completion. */
/* dispatcher: submitted only when no in-flight group overlaps the command's group */
#define KM_FLAG_EXCLUSIVE 2u
/* group leader: acks with {ack, result} as one 8-byte write; the kernel keeps it */
#define KM_FLAG_RESULT 4u
/* dispatcher: a queued job; `first` is ignored, the group allocator chooses it */
#define KM_FLAG_QUEUE 8u
/* dispatcher: one island's part of a group split over islands (the host sends the flat
 * line to each island): run here, never crossed. A spanning line crosses. */
#define KM_FLAG_PART 16u
#define KM_GROUP_ALL(n) ((uint32_t)(n) << 16) /* first 0, count n */
/* A command to the X280 thread of work cluster c (1 to 3): first is
   KM_GROUP_CLUSTER_BASE + c, count 1. No core has such a first. */
#define KM_GROUP_CLUSTER_BASE 0xFF00u
#define KM_GROUP_CLUSTER(c) ((KM_GROUP_CLUSTER_BASE + (uint32_t)(c)) | 1u << 16)
/* A command to link cores k .. k + n - 1 of the island that takes it: first
 * KM_GROUP_LINK_BASE + k, count n; never cut or crossed. Ranks are below the base. */
#define KM_GROUP_LINK_BASE 0xFE00u
#define KM_GROUP_IS_LINK(first)               \
    ((uint32_t)(first) - KM_GROUP_LINK_BASE < \
     KM_GROUP_CLUSTER_BASE - KM_GROUP_LINK_BASE)
/* Whether a group (first, count) names a work cluster's thread, cluster 1 to 3. */
#define KM_GROUP_IS_CLUSTER(first, count) \
    ((count) == 1u && (uint32_t)(first) - (KM_GROUP_CLUSTER_BASE + 1u) < 3u)

/* One command: four 16-byte chunks. Word 0 of each chunk carries the sequence number;
   a command has landed when all four agree (KM_LANDED). */
#define KM_LANDED(r)                                               \
    ((r)->land1 == (r)->sequence && (r)->land2 == (r)->sequence && \
     (r)->land3 == (r)->sequence)

typedef struct {
    uint32_t sequence; /* chunk 0: sequence number, starts at 1, increases */
    uint32_t hdr;      /* handler | flags << 8 | slot << 16 | program << 24 */
    uint32_t
        group; /* first | count << 16: cores [first, first + count) in core order */
    uint32_t a0;
    uint32_t land1, a1, a2, a3; /* chunk 1 */
    uint32_t land2, a4, a5, a6; /* chunk 2 */
    uint32_t land3, a7, a8, a9; /* chunk 3 */
} km_command_t;                 /* sizeof == 64; ten argument words */

KM_STATIC_ASSERT(sizeof(km_command_t) == 64, "a command is one 64-byte line");
KM_STATIC_ASSERT(
    KM_GROUP_CLUSTER_BASE >= KM_CORES_MAX && KM_GROUP_CLUSTER_BASE % 256u == 0,
    "a cluster's group is no core's, and its low 8 bits are the cluster"
);

/* Control block in L1: the same address on every core. Offsets in bytes. The command
   ring is last, so the words the kernels use most sit at small offsets. */
#define KM_CONTROL_RESERVED 0x000u /* reserved, 16 bytes */
/* nonzero: every wait gives up */
#define KM_CONTROL_STOP (KM_CONTROL_RESERVED + 0x10u)
/* KM_RING_SLOTS counters, 16 bytes apart: cores done */
#define KM_CONTROL_DONE (KM_CONTROL_STOP + 0x30u)
/* start-up check-in area (column groups, as today) */
#define KM_CONTROL_BOOT (KM_CONTROL_DONE + KM_RING_SLOTS * 16u)
/* per core: last sequence consumed and reported per slot, gave-up codes, counters */
#define KM_CONTROL_STATE (KM_CONTROL_BOOT + 0x100u)
#define KM_STATE_WORDS (2u * KM_RING_SLOTS + 16u) /* km::kStateWords */
/* the command being run, for the compute and writer kernels */
#define KM_CONTROL_JOB (KM_CONTROL_STATE + 4u * KM_STATE_WORDS)
/* scratch for outgoing 4-byte NoC writes */
#define KM_CONTROL_OUT (KM_CONTROL_JOB + 0x40u)
/* core index -> NoC x | y << 16, KM_CORES_MAX words */
#define KM_CONTROL_CORES (KM_CONTROL_OUT + 0x40u)
/* KM_PORT_SLOTS port slots of 16 bytes: count (bumped by senders), then the count a
   wait wants (nonzero only while waiting, or after the wait gave up) */
#define KM_CONTROL_PORT (KM_CONTROL_CORES + 4u * KM_CORES_MAX)

KM_STATIC_ASSERT(
    KM_RING_SLOTS >= 1u && KM_RING_SLOTS <= 64u,
    "a slot is one bit of the dispatcher's 64-bit masks and fits the header's byte"
);
KM_STATIC_ASSERT(
    KM_CONTROL_BOOT == KM_CONTROL_DONE + 16u * KM_RING_SLOTS &&
        KM_CONTROL_JOB == KM_CONTROL_STATE + 4u * KM_STATE_WORDS,
    "the DONE counters and the state words follow the slot count"
);
KM_STATIC_ASSERT(
    KM_CONTROL_DONE % 16u == 0 && KM_CONTROL_BOOT % 16u == 0 &&
        KM_CONTROL_STATE % 16u == 0 && KM_CONTROL_JOB % 64u == 0 &&
        KM_CONTROL_OUT % 16u == 0 && KM_CONTROL_CORES % 16u == 0,
    "16-byte slots, the job copy on a line"
);

/* Ports: what one core receives; a kernel names 0 to KM_PORTS - 1, the page calls
   alone count at the reader's KM_PORT_PAGES and at the writer's, the one after. */
#define KM_PORTS 8u
#define KM_PORT_PAGES KM_PORTS
/* the collectives' counts of pages from the island before and after on the line; word
   KM_PORT_MET of KM_PORT_LEFT's slot counts an island's ranks' meetings */
#define KM_PORT_LEFT (KM_PORTS + 2u)
#define KM_PORT_RIGHT (KM_PORTS + 3u)
#define KM_PORT_SLOTS (KM_PORTS + 4u)
#define KM_PORT_STRIDE 16u
#define KM_PORT_COUNT 0u /* word offsets inside a port slot */
#define KM_PORT_WANT 1u
#define KM_PORT_MET 2u
/* word KM_PORT_GO of KM_PORT_LEFT's slot: the leader's release of a whole island, a
   marker it multicasts (km_ask.hpp); every core of the island's rectangles takes it */
#define KM_PORT_GO 3u
/* page port words (guard): pages asked or put not yet waited for; the bank address of
   a line a give-up left unserved */
#define KM_PORT_FLIGHT 2u
#define KM_PORT_LOST_TENSOR 3u
/* word 3 of ports 0 to KM_PORTS - 1 (a sender's): the credits a reduce repeated in one
   command gives its senders, one a turn after the first (km_port.hpp) */
#define KM_PORT_CREDIT 3u
/* a port or emit give-up: the gave-up word's top byte names the port */
#define KM_GAVE_PORT_SHIFT 24u
/* KM_BROADCASTS areas, KM_BROADCAST_BYTES each; on a 64-byte line after the ports */
#define KM_CONTROL_BROADCAST \
    (KM_CONTROL_PORT + (KM_PORT_SLOTS * KM_PORT_STRIDE + 63u) / 64u * 64u)

KM_STATIC_ASSERT(
    KM_CONTROL_PORT == KM_CONTROL_CORES + 4u * KM_CORES_MAX, "ports after cores"
);
KM_STATIC_ASSERT(
    KM_CONTROL_PORT + KM_PORT_SLOTS * KM_PORT_STRIDE <= KM_CONTROL_BROADCAST,
    "ports end"
);
KM_STATIC_ASSERT(KM_CONTROL_PORT % 16u == 0 && KM_PORT_STRIDE == 16u, "16-byte slots");

/* Broadcast: one area per broadcast, offsets inside it (16-byte slots). */
#define KM_BROADCASTS 2u
#define KM_BROADCAST_RANKS 16u   /* DONE-BY words of ranks 1 to 15; rank 0 sends */
#define KM_BROADCAST_VALID 0x00u /* sequence << 16 | blocks valid, from the sender */
#define KM_BROADCAST_OUT 0x10u   /* source slot of this core's VALID or DONE write */
/* first give-up: where | area << 8, block, sequence */
#define KM_BROADCAST_GAVE 0x20u
#define KM_BROADCAST_STATS 0x30u /* blocks, wait cycles low and high words */
#define KM_BROADCAST_DONE 0x40u  /* rank r's DONE-BY word: KM_BROADCAST_DONE + 16 r */
#define KM_BROADCAST_BYTES 0x140u
#define KM_BROADCAST_GAVE_WORDS 3u   /* words of the KM_BROADCAST_GAVE slot in use */
#define KM_BROADCAST_STATS_WORDS 3u  /* words of the KM_BROADCAST_STATS slot in use */
#define KM_BROADCAST_DONE_STRIDE 16u /* bytes between two ranks' DONE-BY words */
/* word offsets: in VALID (one 16-byte write) the cast's word and 1 if it carries one */
#define KM_BROADCAST_WORD 1u
#define KM_BROADCAST_CARRIES 2u
/* in rank r's DONE-BY slot: the marker of the blocks whose word it took */
#define KM_BROADCAST_ARRIVED 1u

KM_STATIC_ASSERT(
    KM_CONTROL_BROADCAST == KM_CONTROL_CORES + 4u * KM_CORES_MAX + 0xC0u, "ports room"
);
KM_STATIC_ASSERT(
    KM_BROADCAST_DONE + KM_BROADCAST_DONE_STRIDE * KM_BROADCAST_RANKS ==
        KM_BROADCAST_BYTES,
    "broadcast area"
);
KM_STATIC_ASSERT(
    KM_BROADCAST_GAVE_WORDS * 4u <= KM_BROADCAST_STATS - KM_BROADCAST_GAVE, "gave slot"
);
KM_STATIC_ASSERT(
    KM_BROADCAST_STATS_WORDS * 4u <= KM_BROADCAST_DONE - KM_BROADCAST_STATS,
    "stats slot"
);
KM_STATIC_ASSERT(
    KM_CONTROL_BROADCAST % 16u == 0 && KM_BROADCAST_BYTES % 16u == 0, "16-byte slots"
);
KM_STATIC_ASSERT(
    KM_BROADCAST_CARRIES * 4u < KM_BROADCAST_OUT - KM_BROADCAST_VALID &&
        KM_BROADCAST_ARRIVED * 4u < KM_BROADCAST_DONE_STRIDE,
    "word slots"
);

/* routed messages: queue counters, idle words, status */
#define KM_CONTROL_MESSAGES (KM_CONTROL_BROADCAST + KM_BROADCASTS * KM_BROADCAST_BYTES)

/* Routed messages: offsets in KM_CONTROL_MESSAGES, one 16-byte slot per word. */
/* reader: its tail as last read from its counter tile */
#define KM_MESSAGES_IN_TAIL 0x00u
#define KM_MESSAGES_IN_HEAD 0x10u /* reader */
/* writer: messages sent to any queue, new ones included */
#define KM_MESSAGES_SENT 0x20u
#define KM_MESSAGES_CONSUMED 0x30u /* compute: messages of the packets it finished */
#define KM_MESSAGES_QUOTA 0x40u    /* compute: sends left */
/* idle words of reader, compute, writer, 16 bytes apart */
#define KM_MESSAGES_FLAGS 0x50u
#define KM_MESSAGES_OUT 0x80u      /* source slot of the status line */
#define KM_MESSAGES_GAVE 0x90u     /* first give-up: where, position, run, word seen */
#define KM_MESSAGES_RETURNED 0xA0u /* writer: the reservation's returned position */
/* reader: its inbox's head (the X280 ranks' messages to it), kept over a load's runs */
#define KM_MESSAGES_INBOX_HEAD 0xB0u
#define KM_MESSAGES_BYTES 0xC0u
/* Counter tiles: the last K cores of a dynamic run's group hold the cores' tails and
 * never submit an atomic; offsets inside KM_CONTROL_TAILS. */
#define KM_TAILS_READY 0x00u  /* counter tile: {run, 1, cores it serves, 0} */
#define KM_TAILS_READER 0x10u /* reader: a counter's READY line read here */
#define KM_TAILS_WRITER 0x20u /* writer: the same */
#define KM_TAILS_LINE 0x30u   /* + 16 c: group core c's tail, the atomic adds' target */
#define KM_TAILS_BYTES (KM_TAILS_LINE + 16u * KM_CORES_MAX)
/* routed messages: a counter tile's tail lines */
#define KM_CONTROL_TAILS (KM_CONTROL_MESSAGES + KM_MESSAGES_BYTES)
#define KM_COUNTERS_MAX 4u
/* the submitter ends a dynamic run with this command to its group; a0 = run's sequence
 */
#define KM_HANDLER_MESSAGES_END 249u
/* A program's dynamic handlers, one per byte of a word (0: none); whether `handler` is
   one of them. A program with one dynamic handler has its number as the word. */
#define KM_DYNAMIC_MAX 4u
#define KM_IS_DYNAMIC(handlers, handler)                                             \
    ((handler) != 0u &&                                                              \
     (((handlers) & 0xFFu) == (handler) || ((handlers) >> 8 & 0xFFu) == (handler) || \
      ((handlers) >> 16 & 0xFFu) == (handler) || ((handlers) >> 24) == (handler)))

/* Status line of a core in a dynamic run, 16 bytes at the submitter, one store. */
#define KM_MESSAGES_LINE_RUN 0u /* word offsets: the run's sequence */
#define KM_MESSAGES_LINE_STATE 1u
#define KM_MESSAGES_LINE_SENT 2u
#define KM_MESSAGES_LINE_CONSUMED 3u
#define KM_MESSAGES_BUSY 1u /* line states */
#define KM_MESSAGES_IDLE 2u
#define KM_MESSAGES_GAVE_UP 3u

/* X280 ranks: X280 cores 0 to n - 1 of one work cluster, ranks after the group's cores.
 * Rank r has the tail line, status line and destination byte of core index: */
#define KM_X280_RANKS_MAX 4u
/* KM_X280_RANK_CORE(r), the last indices: no core of a grid of at most 124 cores */
#define KM_X280_RANK_CORE(r) (KM_CORES_MAX - KM_X280_RANKS_MAX + (uint32_t)(r))
/* The run line: one 16-byte store of the dispatcher into the X280 ranks' thread at a
 * dynamic run's RUN and end command, {run, KM_X280_RUN_*, group, its handler}. */
#define KM_X280_RUN_SEQUENCE 0u /* word offsets */
#define KM_X280_RUN_STATE 1u
#define KM_X280_RUN_GROUP 2u
#define KM_X280_RUN_HANDLER 3u
#define KM_X280_RUN_START 1u /* states */
#define KM_X280_RUN_END 2u
/* A core's credit line in the X280 ranks' cluster (16 bytes, the core's reader writes
 * it): word 0 its inbox's head, the positions it has taken. */
#define KM_X280_CREDIT_BYTES 16u

KM_STATIC_ASSERT(
    KM_X280_RANK_CORE(KM_X280_RANKS_MAX - 1u) == KM_CORES_MAX - 1u &&
        KM_X280_RANK_CORE(KM_X280_RANKS_MAX - 1u) <= 0xFFu,
    "an X280 rank's index is a core index and fits a destination byte"
);
KM_STATIC_ASSERT(
    KM_MESSAGES_INBOX_HEAD + 16u <= KM_MESSAGES_BYTES &&
        KM_MESSAGES_INBOX_HEAD >= KM_MESSAGES_RETURNED + 16u,
    "the inbox head's slot"
);

KM_STATIC_ASSERT(
    KM_CONTROL_MESSAGES == KM_CONTROL_CORES + 4u * KM_CORES_MAX + 0x340u,
    "after broadcasts"
);
KM_STATIC_ASSERT(
    KM_CONTROL_BROADCAST + KM_BROADCASTS * KM_BROADCAST_BYTES == KM_CONTROL_MESSAGES,
    "broadcast end"
);
KM_STATIC_ASSERT(
    KM_CONTROL_MESSAGES + KM_MESSAGES_BYTES == KM_CONTROL_TAILS, "dynamic area ends"
);
KM_STATIC_ASSERT(KM_CONTROL_TAILS % 16u == 0, "tails");
KM_STATIC_ASSERT(
    KM_CONTROL_MESSAGES % 16u == 0 && KM_MESSAGES_RETURNED + 16u <= KM_MESSAGES_BYTES,
    "slots"
);
KM_STATIC_ASSERT(
    KM_MESSAGES_IN_TAIL % 16u == 0 && KM_MESSAGES_OUT % 16u == 0, "16-byte slots"
);
KM_STATIC_ASSERT(
    KM_HANDLER_MESSAGES_END > KM_HANDLER_USER_MAX, "not a program handler"
);

/* A command's block: a7 its DRAM bank address (64-byte aligned), a8 bytes | field << 16
   (field 0: a copy in every bank, a core reads bank core % 8; b + 1: bank b only). */
#define KM_BLOCK_ADDRESS_WORD 7u /* a7 */
#define KM_BLOCK_SIZE_WORD 8u    /* a8 */
#define KM_BLOCK_BYTES 1024u     /* the largest block */
#define KM_BLOCK_BYTES_MASK 0xFFFFu
#define KM_BLOCK_BANK_SHIFT 16u
#define KM_BLOCK_BANKS 8u
/* an area: a line (word 0: the sequence read in), then the block */
#define KM_BLOCK_AREA_BYTES (64u + KM_BLOCK_BYTES)

/* X280 port: 16-byte lines {sequence, w0, w1, w2}, each with one writer, offsets inside
   KM_CONTROL_X280. A line lands whole; sequence names it (a tell's latest wins). */
#define KM_X280_TELL 0x00u   /* the X280 core's tell: sequence counts its tells here */
#define KM_X280_ANSWER 0x10u /* the X280 core's answer: sequence repeats the ask's */
#define KM_X280_OUT 0x20u    /* source of this core's stores to its X280 core */
/* the host's, at load: X280 core tile x | y << 16, mailbox */
#define KM_X280_TO 0x30u
#define KM_X280_BYTES 0x40u
/* X280 port: the lines shared with the serving X280 core, on a 64-byte line */
#define KM_CONTROL_X280 ((KM_CONTROL_TAILS + KM_TAILS_BYTES + 63u) / 64u * 64u)
/* a command's block: the reader's copy, then the writer's (KM_BLOCK_AREA_BYTES each) */
#define KM_CONTROL_BLOCK (KM_CONTROL_X280 + KM_X280_BYTES)
/* abort word of slot j at + 4 j: the sequence of the slot's command its cores give up
   (the dispatcher writes it at the deadline; a command's waits poll it) */
#define KM_CONTROL_ABORT (KM_CONTROL_BLOCK + 2u * KM_BLOCK_AREA_BYTES)
/* the kernels' live words (where each waits, on which command) and what a give-up
   clears (km::kLv*); the dispatcher's inspect reads its first KM_LIVE_READ bytes */
#define KM_CONTROL_LIVE (KM_CONTROL_ABORT + 4u * KM_RING_SLOTS)
#define KM_LIVE_BYTES 0x200u
#define KM_LIVE_READ 0x80u
#define KM_LIVE_SEQUENCE 5u   /* live words 5 to 9: the command each kernel runs */
#define KM_LIVE_ROLES 5u      /* kernels with live words: reader, writer, compute's 3 */
#define KM_LIVE_TRACE 20u     /* live word: the trace's L1 address (0: none) */
#define KM_LIVE_TRACE_CAP 21u /* live word: the sequences it holds */
#define KM_TRACE_ENTRY 16u /* bytes of one sequence's trace entry (km::kTraceWords) */
/* Islands: sets of cores that reach each other only through links, ranks flat and
   contiguous by island. A header {island, islands, first rank, ranks, first core} */
#define KM_CONTROL_ISLANDS (KM_CONTROL_LIVE + KM_LIVE_BYTES)
/* the island table (KM_ISLAND_TABLE): an entry an island {first rank u16, count u16,
   next link u8 (own link number), hops u8, ports u8 (bit p: shortest path), 0 u8} */
#define KM_ISLAND_ENTRY_BYTES 8u
#ifndef KM_ISLANDS_MAX /* per-island arrays; a load of more islands defines it */
#define KM_ISLANDS_MAX 16u
#endif
#define KM_ISLAND_HEADER_BYTES 32u
#define KM_ISLAND_NO_LINK 0xFFu /* next link of the holder's own entry */
#define KM_ISLAND_HERE 0u       /* header words: this island (0: island 0 of one) */
#define KM_ISLAND_COUNT 1u      /* islands in the table (0 read as 1) */
#define KM_ISLAND_FIRST 2u      /* this island's first rank */
#define KM_ISLAND_RANKS 3u      /* this island's ranks */
#define KM_ISLAND_BASE 4u       /* its first core of the card: core = base + local */
/* the payload bytes of a packet on this island's links, as the routers were told (the
 * session's; 0: no links) */
#define KM_ISLAND_PACKET 5u
/* the links toward the other of two islands a relay ask may take (0 or 1: link 0
 * alone); a source rank takes link rank % links, always the same (km_relay.hpp) */
#define KM_ISLAND_LINKS 6u
/* header word: the island table's L1 address on this island's cores (0: one island,
 * its one entry in the area at KM_ISLAND_PORTS) */
#define KM_ISLAND_TABLE 7u
/* the port table in the area: port p's link l at (p * KM_ISLAND_PORT_LINKS + l) * 8:
 * {link core u8 (0xFF: none), CPU u8, far island u8, links of the port u8, far tile} */
#define KM_ISLAND_PORTS KM_ISLAND_HEADER_BYTES
#define KM_ISLAND_PORTS_MAX 4u
#define KM_ISLAND_PORT_LINKS 4u
#define KM_ISLAND_PORT_BYTES 8u
#define KM_ISLAND_NO_CORE 0xFFu
/* after the ports: a word, the link area's L1 address on this island's cores (0: no
   link area) */
#define KM_ISLAND_LINK_AREA \
    (KM_ISLAND_PORTS +      \
     KM_ISLAND_PORTS_MAX * KM_ISLAND_PORT_LINKS * KM_ISLAND_PORT_BYTES)
#define KM_ISLANDS_BYTES ((KM_ISLAND_LINK_AREA + 32u + 63u) / 64u * 64u)
/* words after it: the relay area's L1 address on this island (0: no relay), the far
 * island's, and the far link core 0's NoC tile x | y << 16 (km_relay.hpp) */
#define KM_ISLAND_RELAY (KM_ISLAND_LINK_AREA / 4u + 1u)
#define KM_ISLAND_RELAY_FAR (KM_ISLAND_RELAY + 1u)
#define KM_ISLAND_RELAY_TILE (KM_ISLAND_RELAY + 2u)
/* The relay area, on link cores (every core when they run the program's kernels): a
 * core's ask to link core 0 (64 bytes), then link core 0's words */
#define KM_RELAY_ASKS 256u /* ask lines: 2 a core */
#define KM_RELAY_ASK 0x0000u
#define KM_RELAY_DONE 0x4000u
#define KM_RELAY_ASKED 0x4020u   /* link core: asks posted (asking cores add 1) */
#define KM_RELAY_CREDIT 0x4030u  /* link core: the far side's taken count */
#define KM_RELAY_LANDED 0x4040u  /* link core: requests landed from the far side */
#define KM_RELAY_STATE 0x4050u   /* link core: its six words (km_relay.hpp) */
#define KM_RELAY_LINES 0x4070u   /* link core: 16-byte sources (taken, a done word) */
#define KM_RELAY_HEADERS 0x40C0u /* link core: four packet headers, 64 bytes each */
#define KM_RELAY_DESC 0x41C0u    /* link core: an ask as it crosses, 64 bytes */
#define KM_RELAY_STAGE 0x4200u   /* link core: one chunk on its way out */
#define KM_RELAY_CHUNK 4096u
#define KM_RELAY_INBOX (KM_RELAY_STAGE + KM_RELAY_CHUNK) /* the far ask, its bytes */
#define KM_RELAY_MAX 16384u /* the most bytes one ask carries */
/* link core: its second CPU's relay words, at the offsets from KM_RELAY_ASKED of
 * its first's (asks posted, credit, landed, state, lines) */
#define KM_RELAY_SECOND (KM_RELAY_INBOX + 64u + KM_RELAY_MAX)
#define KM_RELAY_BYTES (KM_RELAY_SECOND + 128u)
/* then a word (byte offset): the tensor table's L1 address on this island (0: none);
 * a build without one (KM_TENSOR_TABLE): the session's extent list (km_extent.hpp) */
#define KM_ISLAND_TENSORS ((KM_ISLAND_RELAY_TILE + 1u) * 4u)
/* then a word (byte offset): the forward area's L1 address on this island (0: none) */
#define KM_ISLAND_FORWARD (KM_ISLAND_TENSORS + 4u)
/* then a word (byte offset): the own words' L1 address on this island (0: none) */
#define KM_ISLAND_OWN (KM_ISLAND_FORWARD + 4u)
/* then a word (byte offset): the island's cores for a multicast release, grid columns
 * | its rows << 8 | panel 1's first column << 16 (0: one panel) | 1 << 24; 0: none */
#define KM_ISLAND_GO (KM_ISLAND_OWN + 4u)
#define KM_ISLAND_GO_ROWS 8u
#define KM_ISLAND_GO_SPLIT 16u
#define KM_ISLAND_GO_ON (1u << 24)
/* A core's own words (every core of a run with page links): per data-movement CPU its
 * asks' scratch, put lines and put headers */
#define KM_OWN_LINE 0x00u
#define KM_OWN_RETURNED 0x40u
#define KM_OWN_READ 0x50u
#define KM_OWN_DONE 0x70u
#define KM_OWN_PUT_SENT 0x80u
#define KM_OWN_PUT_NUMBER 0x84u
#define KM_OWN_PUT_CONFIRMED 0x8Cu
#define KM_OWN_BUILD 0xC0u
#define KM_OWN_CPU_BYTES (KM_OWN_BUILD + KM_PUT_BUILD_BYTES)
#define KM_OWN_BYTES (2u * KM_OWN_CPU_BYTES)
/* then the reader's batch of byte records and the writer's (KM_BATCH_*), from the
 * next 64-byte boundary on: what the session's own words hold in all */
#define KM_OWN_BATCH KM_OWN_BYTES
#define KM_OWN_ALL_BYTES (KM_OWN_BATCH + 2u * KM_PUT_BATCH_BYTES + 64u)
/* The forward area (km_transport.hpp), a link core's: packets for another island, sent
 * on toward it. Part 0's writer: far link core; part 1 + i: link core i */
#define KM_FORWARD_PARTS 3u
#define KM_FORWARD_SLOTS 2u
#define KM_FORWARD_LANDED 0x00u /* 16 bytes a part: its packets landed */
/* 16 bytes a destination (0 across): packets it took */
#define KM_FORWARD_CREDIT 0x30u
#define KM_FORWARD_STATE 0x60u /* taken a part, sent a destination, failed */
#define KM_FORWARD_FAILED (2u * KM_FORWARD_PARTS) /* state word */
#define KM_FORWARD_HEADERS 0x80u                  /* two packet headers */
#define KM_FORWARD_STAGE 0x100u /* the head of a packet this core starts */
#define KM_FORWARD_SLOT 0x180u  /* slots: {island, bytes, KM_CROSS_* field} */
/* a forwarded packet's largest payload: its slot's (the session's packets over more
 * than two islands are no larger) */
#define KM_FORWARD_PAYLOAD 4352u
#define KM_FORWARD_SLOT_BYTES (128u + KM_FORWARD_PAYLOAD) /* header 64, payload 128 */
#define KM_FORWARD_BYTES \
    (KM_FORWARD_SLOT + KM_FORWARD_PARTS * KM_FORWARD_SLOTS * KM_FORWARD_SLOT_BYTES)
#define KM_FORWARD_NO_FIELD 0xFFu
/* a reference word below KM_TENSORS_MAX names an entry, any other a bank address; a
 * load of more entries (km.Session(tensors=n)) defines it as n */
#ifndef KM_TENSORS_MAX
#define KM_TENSORS_MAX 32u
#endif
/* an entry: {aligned page bytes, steps of the split dimension, pages a step, copy |
 * its pages << 2 (km_extent.hpp's bound)} */
#define KM_TENSOR_HEADER_BYTES 16u
/* then each island's part: {its bank address, first step, steps, 0} */
#define KM_TENSOR_PART_BYTES 16u
/* a part's word 3 (the 0 above): its bytes in each bank, km::SizedTensor's bound */
#define KM_TENSOR_PART_BANK_BYTES 3u
/* header word 3 of a range of one bank (km::BankRange): {holder, bank, bytes, 2}, the
 * holder's part {its bank address, 0, 0, 0}, the other parts 0 */
#define KM_TENSOR_BANK 2u
/* a table of more entries lives in each island's DRAM, a row a page of 64 B lines;
 * its L1 place holds a head (word 0: that table's address) */
#define KM_TABLE_CACHE_ENTRIES 32u
#define KM_TABLE_HEAD_BYTES 64u
/* KM_RING_SLOTS commands, the last part of the block */
#define KM_CONTROL_RING (KM_CONTROL_ISLANDS + KM_ISLANDS_BYTES)
#define KM_CONTROL_BYTES (KM_CONTROL_RING + KM_RING_SLOTS * 64u)
#define KM_X280_TO_TILE 0u /* word offsets of the KM_X280_TO line */
#define KM_X280_TO_BOX 1u  /* LIM address of this core's mailbox line; 0: not served */
/* A core's mailbox line in its X280 core's LIM (64 bytes, the core writes it). */
#define KM_BOX_ASK 0x00u  /* the core's ask: sequence counts its asks */
#define KM_BOX_TELL 0x10u /* the core's tell: sequence counts its tells */
#define KM_BOX_BYTES 0x40u
/* The pull: an ask {KM_PULL_ASK} answered {first unit, count (0: none left), pulls}. */
#define KM_PULL_ASK 0x50554C4Cu /* "PULL" */

KM_STATIC_ASSERT(KM_CONTROL_TAILS + KM_TAILS_BYTES <= KM_CONTROL_X280, "tails end");
KM_STATIC_ASSERT(
    KM_CONTROL_X280 % 64u == 0 && KM_CONTROL_X280 + KM_X280_BYTES == KM_CONTROL_BLOCK,
    "X280 core lines"
);
KM_STATIC_ASSERT(
    KM_CONTROL_BLOCK % 64u == 0 && KM_BLOCK_BYTES % 64u == 0 &&
        KM_BLOCK_BYTES <= KM_BLOCK_BYTES_MASK &&
        KM_CONTROL_BLOCK + 2u * KM_BLOCK_AREA_BYTES == KM_CONTROL_ABORT,
    "block areas: whole lines, after the X280 lines"
);
KM_STATIC_ASSERT(
    KM_CONTROL_ABORT % 64u == 0 && KM_CONTROL_LIVE % 64u == 0 &&
        KM_LIVE_READ <= KM_LIVE_BYTES,
    "abort words and live words: whole lines"
);
KM_STATIC_ASSERT(
    KM_CONTROL_ISLANDS == KM_CONTROL_LIVE + KM_LIVE_BYTES &&
        KM_CONTROL_ISLANDS % 64u == 0 && KM_ISLANDS_MAX >= 1u &&
        KM_ISLANDS_MAX <= 256u &&
        KM_CONTROL_ISLANDS + KM_ISLANDS_BYTES == KM_CONTROL_RING,
    "the island area: whole lines after the live words; an island is a byte"
);
KM_STATIC_ASSERT(
    KM_ISLAND_LINK_AREA + 4u <= KM_ISLANDS_BYTES && KM_ASK_LINES % 64u == 0 &&
        KM_ISLAND_HEADER_BYTES + KM_ISLAND_ENTRY_BYTES <= KM_ISLAND_LINK_AREA &&
        KM_ASK_READ + 16u <= KM_ASK_HEADERS && KM_PUT_HEADERS % 64u == 0 &&
        KM_PUT_LINES % 64u == 0 && KM_ASK_SLOTS == 32u &&
        KM_PUT_QUEUE + 4u <= KM_PUT_TAKEN + KM_PUT_RING_READ &&
        KM_PUT_TAKEN + KM_PUT_RING_READ <= KM_PUT_OWN && KM_PUT_TAKEN % 16u == 0 &&
        KM_ASK_OWNER < KM_ASK_END && KM_PUT_RING_READ <= KM_PUT_BUILD_BYTES &&
        KM_QUEUE_PUTS % 64u == 0 && KM_QUEUE_BYTES % 64u == 0 &&
        KM_PUT_RING_IDLE + 32u <= KM_PUT_HEADERS &&
        2u * KM_PUT_BUILD_BYTES <= 2u * KM_ASK_SLOTS * 64u && KM_ASK_RING < KM_ASK_END,
    "the link area word after the island table; the ask lines whole lines; the rings"
);
KM_STATIC_ASSERT(
    KM_PUT_LANDING > KM_PUT_QUEUE &&
        KM_PUT_LANDING + 4u <= KM_PUT_TAKEN + KM_PUT_RING_READ &&
        KM_PUT_LANDED >= KM_PUT_TAKEN + KM_PUT_RING_READ &&
        KM_PUT_LANDED + 4u <= KM_PUT_OWN && KM_PUT_BATCH_BYTES % 64u == 0 &&
        KM_PUT_BATCH_BYTES - 3u * KM_BATCH_HEAD < (1u << (32u - KM_BATCH_SPAN_SHIFT)),
    "the batches' landing in the block a putting core reads; a record's bytes fit"
);
KM_STATIC_ASSERT(
    KM_ISLAND_RELAY * 4u >= KM_ISLAND_LINK_AREA + 4u &&
        KM_ISLAND_TENSORS == (KM_ISLAND_RELAY_TILE + 1u) * 4u &&
        KM_ISLAND_GO == KM_ISLAND_OWN + 4u && KM_ISLAND_GO + 4u <= KM_ISLANDS_BYTES &&
        KM_OWN_RETURNED >= KM_OWN_LINE + 64u && KM_OWN_READ >= KM_OWN_RETURNED + 16u &&
        KM_OWN_DONE >= KM_OWN_READ + 32u && KM_OWN_PUT_SENT >= KM_OWN_DONE + 16u &&
        KM_OWN_PUT_NUMBER >= KM_OWN_PUT_SENT + 4u &&
        KM_OWN_PUT_CONFIRMED >= KM_OWN_PUT_NUMBER + 8u &&
        KM_OWN_BUILD >= KM_OWN_PUT_CONFIRMED + 8u && KM_OWN_BUILD % 64u == 0 &&
        KM_OWN_CPU_BYTES % 64u == 0 &&
        KM_FORWARD_STATE >= KM_FORWARD_CREDIT + 16u * KM_FORWARD_PARTS &&
        KM_FORWARD_HEADERS >= KM_FORWARD_STATE + 4u * (KM_FORWARD_FAILED + 1u) &&
        KM_FORWARD_SLOT % 64u == 0 && KM_FORWARD_SLOT_BYTES % 64u == 0 &&
        KM_RELAY_DONE >= KM_RELAY_ASK + 64u * KM_RELAY_ASKS &&
        KM_RELAY_ASKED >= KM_RELAY_DONE + 16u * 2u &&
        KM_RELAY_LINES >= KM_RELAY_STATE + 4u * 6u &&
        KM_RELAY_HEADERS >= KM_RELAY_LINES + 32u && KM_RELAY_SECOND % 64u == 0 &&
        KM_RELAY_LINES + 32u - KM_RELAY_ASKED <= KM_RELAY_BYTES - KM_RELAY_SECOND,
    "the relay and tensor table words after the island table; asks and done words"
);
KM_STATIC_ASSERT(
    KM_CONTROL_RING % 64u == 0 &&
        KM_CONTROL_RING + 64u * KM_RING_SLOTS == KM_CONTROL_BYTES,
    "the ring: whole lines, the last part of the block"
);

/* Words the Tensix grid pushes into the X280's local memory. */
#define KM_ACK_STRIDE 16u /* ack word of slot j at ack_base + j * KM_ACK_STRIDE */
#define KM_ACK(sequence) ((uint32_t)(sequence)) /* the whole sequence number */
#define KM_READY 0xFFFFFFFFu /* READY words are not in the ack area */

/* Host -> dispatcher: one command is one command, its number in `sequence` and the
   landing words. Handlers 252..254 are commands to the kernel and are not forwarded. */

/* Dispatcher -> host: one 16-byte event per finished command, in a host-memory ring.
   DONE: auxiliary = first core (of its island's part, local) | sequence low 16 << 8 */
#define KM_EVENT_DONE 0u
/* auxiliary = slot | sequence low 16 << 8; cores: their state words */
#define KM_EVENT_TIMEOUT 1u
#define KM_EVENT_PONG 2u
/* consumed, not run; auxiliary = reason (KMD_REJECT_*) */
#define KM_EVENT_REJECTED 3u

typedef struct {
    uint32_t index;  /* event number, starts at 1: the ring slot is index % depth */
    uint32_t req;    /* the host's command number */
    uint32_t status; /* KM_EVENT_* | auxiliary << 8 */
    uint32_t ticks;  /* submit to completion, in the X280's timer ticks */
} km_event_t;        /* sizeof == 16: one chunk, lands whole */

KM_STATIC_ASSERT(sizeof(km_event_t) == 16, "an event is one 16-byte chunk");

/* A DONE or TIMEOUT event's auxiliary: the first core (its island's local core) or the
   slot in its low 8 bits, the low 16 bits of the command's sequence number above. */
#define KM_EVENT_AUXILIARY_LOW 0xFFu
#define KM_EVENT_SEQUENCE_SHIFT 8u
#define KM_EVENT_SEQUENCE_MASK 0xFFFFu
KM_STATIC_ASSERT(
    KM_CORES_MAX <= KM_EVENT_AUXILIARY_LOW + 1u, "a first core fits in 8 bits"
);
/* A TIMEOUT of a command whose part on another island did not come back by its
   deadline: its low 8 bits are KM_EVENT_ISLAND + that island (no slot reaches it) */
#define KM_EVENT_ISLAND 0x80u
KM_STATIC_ASSERT(
    KM_RING_SLOTS <= KM_EVENT_ISLAND && KM_EVENT_ISLAND + KM_ISLANDS_MAX <= 0x100u,
    "an island's timeout is no slot's"
);

#endif /* KOMOREBI_ABI_KM_ABI_H */
