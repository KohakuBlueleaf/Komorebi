/* An X280 core as a rank, the thread (xr.h, README.md): it answers its queue's messages
 * into the senders' inboxes and writes its stand-in's status line at the dispatcher. */
#include <stdint.h>

#include "xr.h"
#include "kmx.h"
#include "noc.h"

/* X280 core h's non-posted window onto the status lines: + h */
#define STATUS_WINDOW 152u
#define WAIT_TICKS KMX_TICKS_PER_SECOND
/* queue looks between looks at the stand-in's tell and the stop */
#define LOOK_EVERY 64u
#define TAKE_MAX 8u /* messages taken at once */
#define WINDOW_BYTES 0x200000ull

typedef struct {
    uint32_t w[XR_RES_STRIDE];
} xstat_t;

static xstat_t g_xs[4] __attribute__((section(".data"))) = {{{0}}};

static uint64_t cycles(void) {
    uint64_t count;
    __asm__ volatile("csrr %0, mcycle" : "=r"(count));
    return count;
}

/* One 16-byte vector store. */
static void store16(volatile uint8_t* destination, const uint32_t* source) {
    __asm__ volatile(
        ".option push\n.option arch, +v\n"
        "vsetivli zero, 4, e32, m1, ta, ma\n"
        "vle32.v v8, (%0)\n"
        "vse32.v v8, (%1)\n"
        ".option pop\n"
        :
        : "r"(source), "r"(destination)
        : "memory", "v8"
    );
}

/* Core `core`'s L1 through the library's port window onto it (opened at the start,
 * aimed at L1 address 0, posted). */
static volatile uint8_t* l1_of(int core) {
    return (
        (volatile uint8_t*)(
            NOC_2M_WINDOW_BASE +
            (uint64_t)(KMX_PORT_WINDOW + (uint32_t)core) * NOC_2M_WINDOW_STRIDE
        )
    );
}

/* The stand-in of this X280 core, or XR_NONE. */
static uint32_t stand_in_of(int cluster, int cpu) {
    const kmx_args* arguments = kmx_args_get();
    const uint32_t index = (
        XR_WORD_STANDIN + 4u * (uint32_t)(cluster - 1) + (uint32_t)cpu
    );
    return index < arguments->nwords ? arguments->words[index] : XR_NONE;
}

/* The status line of `stand_in` at the dispatcher, through a non-posted window: a store
 * and a fence return once it landed. 0 when the ring does not name the dispatcher. */
static volatile uint8_t* status_line(int cluster, int cpu, uint32_t stand_in) {
    const uint8_t* lim = (const uint8_t*)kmx_args_get();
    const kmx_ring_descriptor* ring = (const kmx_ring_descriptor*)(lim + KMX_LIM_RING);
    if (!(ring->flags & KMX_RING_REMOTE)) {
        return 0;
    }
    const uint64_t lines = (
        ring->cmd - kmx_args_get()->words[XR_WORD_BACK + cluster - 1]
    );
    const uint32_t tile = ring->reserved[0];
    noc_tlb_2m_t programmed;
    volatile uint8_t* window = (
        (volatile uint8_t*)noc_configure_tlb_2m_checked(
            STATUS_WINDOW + (uint32_t)cpu,
            tile & 0xFFFFu,
            tile >> 16,
            lines & ~(WINDOW_BYTES - 1),
            NOC_ORDER_DEFAULT,
            &programmed
        )
    );
    if (window == 0) {
        return 0;
    }
    return window + (lines & (WINDOW_BYTES - 1)) + 16u * stand_in;
}

static void status(
    volatile uint8_t* line,
    uint32_t run,
    uint32_t state,
    uint32_t sent,
    uint32_t consumed
) {
    const uint32_t words[4] = {run, state, sent, consumed};
    store16(line, words);
    x280_fence_ow();
}

/* One X280 core: HELLO to its cores, then runs until FINISH, the stop or the deadline.
 */
static void serve(int cluster, int cpu, uint64_t arg) {
    (void)arg;
    xstat_t* stats = &g_xs[cpu];
    const uint32_t stand_in = stand_in_of(cluster, cpu);
    if (stand_in == XR_NONE) {
        return;
    }
    const kmx_args* arguments = kmx_args_get();
    const uint32_t inbox = arguments->words[XR_WORD_INBOX];
    const uint64_t idle_ticks = arguments->words[XR_WORD_IDLE];
    uint64_t region_bytes;
    uint8_t* region = (uint8_t*)kmx_region(0, &region_bytes);
    if (region == 0 || region_bytes < XR_REGION_BYTES) {
        stats->w[XR_RES_GAVE] = XR_X280_GAVE_REGION;
        return;
    }
    const uint64_t area = (uint64_t)cpu * XR_AREA;
    const volatile uint32_t* queue = (const volatile uint32_t*)(region + area);
    const volatile uint32_t* credits = (
        (const volatile uint32_t*)(region + area + XR_CREDITS)
    );
    volatile uint8_t* line = status_line(cluster, cpu, stand_in);
    if (line == 0) {
        stats->w[XR_RES_GAVE] = XR_X280_GAVE_WINDOW;
        return;
    }
    /* The stack and the image's data are in the cluster's local memory (uncached, about
     * 30 cycles a load): the ask loop walks the list of this core's cores only. */
    uint8_t mine[KMX_PORTS_CORES], list[KMX_PORTS_CORES];
    uint32_t out[KMX_PORTS_CORES], credit[KMX_PORTS_CORES], ranks = 0;
    for (int core = 0; core < KMX_PORTS_CORES; core++) {
        out[core] = credit[core] = 0;
        mine[core] = (
            core < kmx_port_cores() && kmx_port_owner(core) == cpu &&
            (uint32_t)core != stand_in
        );
        if (mine[core]) {
            list[ranks] = (uint8_t)core;
            kmx_port_tell(
                core,
                XR_TELL_HELLO,
                (uint32_t)arguments->region[0].offset + (uint32_t)area,
                arguments->words[XR_WORD_BANK + cluster - 1]
            );
            ranks++;
        }
    }
    const uint32_t me = 0x100u | (uint32_t)cluster << 2 | (uint32_t)cpu;
    uint32_t run = 0, mode = 0, head = 0, sent = 0, consumed = 0, idle_shown = 0;
    uint64_t last = kmx_time();
    for (uint32_t pass = 0;; pass++) {
        if (pass % LOOK_EVERY == 0) {
            if (kmx_stop_requested()) {
                break;
            }
            kmx_message told;
            if (kmx_port_told((int)stand_in, &told) == KMX_OK) {
                if (told.w[0] == XR_TELL_FINISH) {
                    break;
                }
                if (told.w[0] == XR_TELL_START) {
                    run = told.w[1];
                    mode = told.w[2];
                    sent = consumed = 0;
                    idle_shown = mode == XR_MODE_ASK;
                    if (idle_shown) { /* an ask is no message of the run */
                        status(line, run, KM_MESSAGES_IDLE, 0, 0);
                    }
                    last = kmx_time();
                    stats->w[XR_RES_RUNS]++;
                } else if (told.w[0] == XR_TELL_END) {
                    run = 0;
                }
            }
        }
        if (run == 0) {
            continue;
        }
        if (mode == XR_MODE_ASK) {
            for (uint32_t index = 0; index < ranks; index++) {
                const int core = list[index];
                kmx_message asked;
                if (kmx_port_asked(core, &asked) == KMX_OK) {
                    kmx_port_answer(
                        core, asked.sequence, XR_ANSWER(asked.w[0], asked.w[1]), 0, 0
                    );
                    stats->w[XR_RES_ASKS]++;
                }
            }
            continue;
        }
        /* The prefix of valid marks from the head: one line looked at while the queue
         * is empty, up to TAKE_MAX once the first is valid. */
        const uint64_t at = area + 64u * (head & (XR_QUEUE - 1));
        kmx_invalidate_at(0, at, 64);
        if (queue[16u * (head & (XR_QUEUE - 1)) + XR_W_MARK] != head + 1u) {
            stats->w[XR_RES_POLLS]++;
            if (!idle_shown && kmx_time() - last >= idle_ticks) {
                status(line, run, KM_MESSAGES_IDLE, sent, consumed);
                idle_shown = 1;
            }
            continue;
        }
        const uint64_t started = cycles();
        if (idle_shown) { /* busy has landed before the first message is taken */
            status(line, run, KM_MESSAGES_BUSY, 0, 0);
            idle_shown = 0;
        }
        uint32_t valid = 1;
        const uint32_t room = XR_QUEUE - (head & (XR_QUEUE - 1));
        const uint32_t more = (TAKE_MAX < room ? TAKE_MAX : room) - 1u;
        if (more != 0) {
            kmx_invalidate_at(0, at + 64u, 64u * more);
            while (
                valid <= more &&
                queue[16u * ((head + valid) & (XR_QUEUE - 1)) + XR_W_MARK] ==
                    head + valid + 1u
            ) {
                valid++;
            }
        }
        /* Each reply: its words 4 to 15 into the sender's inbox, then after a fence the
         * first 16 bytes with its mark. */
        uint64_t store_cycles = 0;
        uint32_t first[TAKE_MAX][4];
        volatile uint8_t* where[TAKE_MAX];
        for (uint32_t i = 0; i < valid; i++) {
            const volatile uint32_t*
                message = (queue + 16u * ((head + i) & (XR_QUEUE - 1)));
            const uint32_t from = message[XR_W_FROM], k = message[XR_W_INDEX];
            const uint32_t value = message[XR_W_VALUE];
            if (message[XR_W_CHECK] != XR_CHECK(from, k, value)) {
                stats->w[XR_RES_BAD]++;
            }
            if (from >= KMX_PORTS_CORES || !mine[from]) {
                stats->w[XR_RES_GAVE] = XR_X280_GAVE_FOREIGN;
                where[i] = 0;
                continue;
            }
            if (out[from] - credit[from] >= XR_INBOX) { /* the inbox may be full */
                const uint64_t until = kmx_time() + WAIT_TICKS;
                do {
                    kmx_invalidate_at(0, area + XR_CREDITS + 64u * from, 64);
                    credit[from] = credits[16u * from];
                } while (out[from] - credit[from] >= XR_INBOX && kmx_time() < until);
            }
            if (out[from] - credit[from] >= XR_INBOX) {
                stats->w[XR_RES_GAVE] = XR_X280_GAVE_CREDIT;
                where[i] = 0;
                continue;
            }
            const uint32_t reply = XR_REPLY(value, k);
            const uint32_t rest[12] = {
                XR_CHECK(me, k, reply), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
            };
            where[i] = l1_of((int)from) + inbox + 64u * (out[from] & (XR_INBOX - 1));
            const uint64_t store_started = cycles();
            store16(where[i] + 16, rest);
            store16(where[i] + 32, rest + 4);
            store16(where[i] + 48, rest + 8);
            store_cycles += cycles() - store_started;
            first[i][XR_W_MARK] = out[from] + 1u;
            first[i][XR_W_FROM] = me;
            first[i][XR_W_INDEX] = k;
            first[i][XR_W_VALUE] = reply;
            out[from]++;
        }
        const uint64_t stored = cycles();
        x280_fence_ow();
        for (uint32_t i = 0; i < valid; i++) {
            if (where[i] != 0) {
                store16(where[i], first[i]);
                sent++;
                stats->w[XR_RES_SENT]++;
            }
        }
        x280_fence_ow();
        store_cycles += cycles() - stored;
        stats->w[XR_RES_STORE] += (uint32_t)(store_cycles >> 4);
        stats->w[XR_RES_TAKES]++;
        consumed += valid;
        stats->w[XR_RES_CONSUMED] += valid;
        head += valid;
        last = kmx_time();
        stats->w[XR_RES_TAKE] += (uint32_t)((cycles() - started) >> 4);
    }
    noc_clear_tlb_2m(STATUS_WINDOW + (uint32_t)cpu);
}

int kmx_main(int cpu) {
    (void)cpu;
    uint32_t words[XR_RES_WORDS];
    for (uint32_t i = 0; i < XR_RES_WORDS; i++) {
        words[i] = 0;
    }
    int status_code = KMX_OK;
    if (kmx_cluster() != kmx_lead()) {
        status_code = kmx_serve();
    } else {
        status_code = kmx_parallel_all(serve, 0);
    }
    for (uint32_t h = 0; h < 4; h++) {
        for (uint32_t i = 0; i < XR_RES_STRIDE; i++) {
            words[XR_RES_CORE + XR_RES_STRIDE * h + i] = g_xs[h].w[i];
        }
    }
    kmx_result(
        status_code == KMX_OK ? 0u : 1u + (uint32_t)status_code, words, XR_RES_WORDS
    );
    return 0;
}
