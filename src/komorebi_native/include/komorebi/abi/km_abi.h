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
/* served on link cores (km_pages.hpp): from the host, each data-movement CPU of the
 * group sends its share of DRAM pages over its own link, or the reader waits for a
 * count; from the dispatcher, link core 0's link 0 CPU sends a staged packet across */
#define KM_HANDLER_PAGES 248u
#define KM_PAGES_FROM 0u  /* send: the source's bank address (interleaved DRAM) */
#define KM_PAGES_PAGE 1u  /* send: aligned page bytes, at most KM_PAGES_PACKET_MAX */
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
#define KM_PAGES_BUFFER 9u    /* send: L1 of CPU c at buffer + c * chunk */
#define KM_PAGES_DRAM 0xFFFFu /* page p into page p of the tensor at KM_PAGES_TO */
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
/* pages a rank to each of ranks r (the far island's cores from KM_PAGES_TO_CORE on):
 * rank r's pages first + r * pages on into L1 address KM_PAGES_TO of its core, counted
 * at its KM_PAGES_COUNT; the CPUs split the ranks */
#define KM_PAGES_RANGE 4u
/* link 0's CPU of link core 0 serves the messages of the RUN whose group word is a0
 * across its link until that RUN ends (km_link.hpp) */
#define KM_PAGES_MESSAGES 5u
#define KM_PAGES_PACKET_MAX 4352u /* a fabric packet's largest payload */
/* LINE's words: a0 the stage, a1 the packet's number (each line of the stage holds it
 * in word 0, the count line in word 1), a2 the far tile x | y << 16, a3 the far address
 * of the packet, a4 that of the count line, a5 the link area; a7 as KM_PAGES_COUNTER */
#define KM_LINE_STAGE 0u
#define KM_LINE_NUMBER 1u
#define KM_LINE_TILE 2u
#define KM_LINE_PACKET 3u
#define KM_LINE_COUNT 4u
#define KM_LINE_AREA 5u
/* The link area, in link core 0's L1 (a run over islands; its offsets): two packet
 * headers, then the stages, each a packet (the command line and its crossing line) and
 * the count line written after */
#define KM_LINK_HEADERS 0x40u /* 64 bytes each: the packet's, the count line's */
#define KM_LINK_STAGE 0x100u
#define KM_LINK_STAGES 4u
#define KM_LINK_PACKET_BYTES 128u
#define KM_LINK_STAGE_BYTES 192u
/* Then the asks of cores for pages another island holds (km_ask.hpp), on every core
 * of the area: on link core 0 the count its island's cores reserve lines with (a local
 * atomic), the lines it took, the asks the far island answered, the far asks this one
 * answered, the bad ones; a core's own line and two answer words; the two headers of
 * the link CPUs (the far ask, the answer's count); the lines, then the far asks */
#define KM_ASK_TAIL (KM_LINK_STAGE + KM_LINK_STAGES * KM_LINK_STAGE_BYTES)
#define KM_ASK_TAKEN (KM_ASK_TAIL + 0x10u)
#define KM_ASK_SERVED (KM_ASK_TAIL + 0x20u)
#define KM_ASK_ANSWERED (KM_ASK_TAIL + 0x30u)
#define KM_ASK_BAD (KM_ASK_TAIL + 0x34u)
#define KM_ASK_OWN (KM_ASK_TAIL + 0x40u)
#define KM_ASK_RETURNED (KM_ASK_OWN + 0x40u)
#define KM_ASK_READ (KM_ASK_OWN + 0x50u)
#define KM_ASK_HEADERS (KM_ASK_OWN + 0x80u)
#define KM_ASK_LINES (KM_ASK_HEADERS + 0x80u)
#define KM_ASK_SLOTS 32u
#define KM_ASK_INBOX (KM_ASK_LINES + KM_ASK_SLOTS * 64u)
/* An ask line's words (64 bytes; words 0 and 15 its number, the reserved count + 1):
 * the tensor's bank address (the same on every island), its aligned page bytes, the
 * first page, pages, the asker's x | y << 16 (core table), its L1 address, its port
 * count's L1 address, the holder island, the asker's island */
#define KM_ASK_NUMBER 0u
#define KM_ASK_TENSOR 1u
#define KM_ASK_PAGE 2u
#define KM_ASK_FIRST 3u
#define KM_ASK_PAGES 4u
#define KM_ASK_CORE 5u
#define KM_ASK_TO 6u
#define KM_ASK_COUNT 7u
#define KM_ASK_HOLDER 8u
#define KM_ASK_FROM 9u
#define KM_ASK_END 15u
#define KM_LINK_AREA_BYTES (KM_ASK_INBOX + KM_ASK_SLOTS * 64u)
/* Flags: bit 0 is reserved and zero. Every command reports completion. */
/* dispatcher: submitted only when no in-flight group overlaps the command's group */
#define KM_FLAG_EXCLUSIVE 2u
/* group leader: acks with {ack, result} as one 8-byte write; the kernel keeps it */
#define KM_FLAG_RESULT 4u
/* dispatcher: a queued job; `first` is ignored, the group allocator chooses it */
#define KM_FLAG_QUEUE 8u
/* dispatcher: one island's part of a group its submitter split over islands (the host
 * sends the flat line to each island it touches): the dispatcher runs its own part
 * and never crosses the line to another island. A spanning line without it crosses. */
#define KM_FLAG_PART 16u
#define KM_GROUP_ALL(n) ((uint32_t)(n) << 16) /* first 0, count n */
/* A command to the X280 thread of work cluster c (1 to 3): first is
   KM_GROUP_CLUSTER_BASE + c, count 1. No core has such a first. */
#define KM_GROUP_CLUSTER_BASE 0xFF00u
#define KM_GROUP_CLUSTER(c) ((KM_GROUP_CLUSTER_BASE + (uint32_t)(c)) | 1u << 16)
/* A command to link cores k to k + n - 1 of the island that takes it: first is
   KM_GROUP_LINK_BASE + k, count n; never cut by island, never crossing. A rank is
   below KM_GROUP_LINK_BASE; a link core's rank in it is its link-core index - k. */
#define KM_GROUP_LINK_BASE 0xFE00u
#define KM_GROUP_IS_LINK(first) \
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
/* KM_PORTS port slots of 16 bytes: count (bumped by senders), then the count a wait
   wants (nonzero only while waiting, or after the wait gave up) */
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

/* Ports: the receiving side of a value sent to one core. */
#define KM_PORTS 8u
#define KM_PORT_STRIDE 16u
#define KM_PORT_COUNT 0u /* word offsets inside a port slot */
#define KM_PORT_WANT 1u
/* a port or emit give-up: the gave-up word's top byte names the port */
#define KM_GAVE_PORT_SHIFT 24u
/* KM_BROADCASTS areas, KM_BROADCAST_BYTES each */
#define KM_CONTROL_BROADCAST (KM_CONTROL_PORT + KM_PORTS * KM_PORT_STRIDE)

KM_STATIC_ASSERT(
    KM_CONTROL_PORT == KM_CONTROL_CORES + 4u * KM_CORES_MAX, "ports after cores"
);
KM_STATIC_ASSERT(
    KM_CONTROL_PORT + KM_PORTS * KM_PORT_STRIDE == KM_CONTROL_BROADCAST, "ports end"
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
    KM_CONTROL_BROADCAST == KM_CONTROL_CORES + 4u * KM_CORES_MAX + 0x80u, "ports room"
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
 * dynamic run's RUN and at its end command, {run, KM_X280_RUN_*, group, its handler}.
 */
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
    KM_CONTROL_MESSAGES == KM_CONTROL_CORES + 4u * KM_CORES_MAX + 0x300u,
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
   contiguous by island. A header line {island, islands, its first rank, its ranks} */
#define KM_CONTROL_ISLANDS (KM_CONTROL_LIVE + KM_LIVE_BYTES)
/* then the island table: an entry per island {first rank u16, count u16, next link u8
   (this island's own link number toward it), hops u8, 0 u16}; the host's, at load */
#define KM_ISLANDS_MAX 16u
#define KM_ISLAND_ENTRY_BYTES 8u
#define KM_ISLAND_HEADER_BYTES 16u
#define KM_ISLAND_NO_LINK 0xFFu /* next link of the holder's own entry */
#define KM_ISLAND_HERE 0u       /* header words: this island (0: island 0 of one) */
#define KM_ISLAND_COUNT 1u      /* islands in the table (0 read as 1) */
#define KM_ISLAND_FIRST 2u      /* this island's first rank */
#define KM_ISLAND_RANKS 3u      /* this island's ranks */
#define KM_ISLANDS_BYTES                                                             \
    ((KM_ISLAND_HEADER_BYTES + KM_ISLANDS_MAX * KM_ISLAND_ENTRY_BYTES + 63u) / 64u * \
     64u)
/* after the table: a word, the link area's L1 address on this island's cores (0: no
   link area) */
#define KM_ISLAND_LINK_AREA \
    (KM_ISLAND_HEADER_BYTES + KM_ISLANDS_MAX * KM_ISLAND_ENTRY_BYTES)
/* words after it: the relay area's L1 address on this island (0: no relay), the far
 * island's, and the far link core 0's NoC tile x | y << 16 (km_relay.hpp) */
#define KM_ISLAND_RELAY 37u
#define KM_ISLAND_RELAY_FAR 38u
#define KM_ISLAND_RELAY_TILE 39u
/* The relay area, on every core: a leader's ask to link core 0 (64 bytes a ring slot),
 * the link core's done word back to the leader (16 bytes a slot), then link core 0's */
#define KM_RELAY_ASK 0x0000u
#define KM_RELAY_DONE 0x1000u
#define KM_RELAY_ASKED 0x1400u  /* link core: asks posted (leaders add 1) */
#define KM_RELAY_CREDIT 0x1410u /* link core: the far side's taken count */
#define KM_RELAY_LANDED 0x1420u /* link core: requests landed from the far side */
#define KM_RELAY_STATE 0x1430u  /* link core: served, sent, taken, failed */
#define KM_RELAY_LINES 0x1440u  /* link core: 16-byte sources (taken, a done word) */
#define KM_RELAY_HEADERS 0x1480u /* link core: four packet headers, 64 bytes each */
#define KM_RELAY_DESC 0x1580u    /* link core: an ask as it crosses, 64 bytes */
#define KM_RELAY_STAGE 0x1600u   /* link core: one chunk on its way out */
#define KM_RELAY_CHUNK 4096u
#define KM_RELAY_INBOX (KM_RELAY_STAGE + KM_RELAY_CHUNK) /* the far ask, its bytes */
#define KM_RELAY_MAX 16384u /* the most bytes one ask carries */
#define KM_RELAY_BYTES (KM_RELAY_INBOX + 64u + KM_RELAY_MAX)
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
        KM_ISLANDS_MAX <= 32u &&
        KM_CONTROL_ISLANDS + KM_ISLANDS_BYTES == KM_CONTROL_RING,
    "the island area: whole lines after the live words; an island is a bit of a word"
);
KM_STATIC_ASSERT(
    KM_ISLAND_LINK_AREA + 4u <= KM_ISLANDS_BYTES && KM_ASK_LINES % 64u == 0 &&
        KM_ASK_READ + 16u <= KM_ASK_HEADERS,
    "the link area word after the island table; the ask lines whole lines"
);
KM_STATIC_ASSERT(
    KM_ISLAND_RELAY * 4u >= KM_ISLAND_LINK_AREA + 4u &&
        (KM_ISLAND_RELAY_TILE + 1u) * 4u <= KM_ISLANDS_BYTES &&
        KM_RELAY_DONE >= KM_RELAY_ASK + 64u * KM_RING_SLOTS &&
        KM_RELAY_ASKED >= KM_RELAY_DONE + 16u * KM_RING_SLOTS,
    "the relay words after the island table; an ask and a done word a ring slot"
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
