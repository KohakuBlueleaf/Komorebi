/* One cluster: kmx_spawn and kmx_join with their rejections, kmx_parallel beside a busy
 * core, kmx_parallel_all with no other cluster, a kmx_push into its own bank. */
#include <stdint.h>

#include "kmx.h"

/* Set by core 0 to let core 1's spawn end; each core's mark (globals need .data). */
static volatile uint32_t g_go __attribute__((section(".data"))) = 0;
static volatile uint32_t g_mark[4][16] __attribute__((section(".data"))) = {{0}};
static uint32_t g_source[1024] __attribute__((section(".data"), aligned(64))) = {1};

static void wait_go(int core, void* arg) {
    (void)arg;
    while (!g_go && !kmx_stop_requested()) {
        ;
    }
    g_mark[core][0] = 100u + (uint32_t)core;
}

static void mark(int core, void* arg) {
    g_mark[core][0] = (uint32_t)(uintptr_t)arg + (uint32_t)core;
}

static void mark_all(int cluster, int core, uint64_t arg) {
    g_mark[core][0] = (uint32_t)arg + 10u * (uint32_t)cluster + (uint32_t)core;
}

int kmx_main(int core) {
    (void)core;
    uint32_t words[24];
    uint32_t word_count = 0;
    words[word_count++] = (uint32_t)kmx_spawn(1, wait_go, 0); /* OK */
    words[word_count++] = (uint32_t)kmx_spawn(1, mark, 0);    /* FULL: core 1 is busy */
    words[word_count++] = (uint32_t)kmx_parallel(mark, 0);    /* FULL */
    words[word_count++] = (uint32_t)kmx_spawn(0, mark, 0);    /* BAD */
    words[word_count++] = (uint32_t)kmx_join(4);              /* BAD */
    g_go = 1;
    words[word_count++] = (uint32_t)kmx_join(1); /* OK */
    words[word_count++] = g_mark[1][0];          /* 101 */
    words[word_count++] = (uint32_t)kmx_spawn(2, mark, (void*)200);
    words[word_count++] = (uint32_t)kmx_spawn(3, mark, (void*)300);
    words[word_count++] = (uint32_t)kmx_join(2);
    words[word_count++] = (uint32_t)kmx_join(3);
    words[word_count++] = g_mark[2][0]; /* 202 */
    words[word_count++] = g_mark[3][0]; /* 303 */
    words[word_count++] = (uint32_t)kmx_parallel(mark, (void*)400);
    words[word_count++] = g_mark[0][0] + g_mark[1][0] + g_mark[2][0] +
        g_mark[3][0];                            /* 1606 */
    words[word_count++] = kmx_clusters();        /* 2 */
    words[word_count++] = (uint32_t)kmx_lead();  /* 1 */
    words[word_count++] = (uint32_t)kmx_serve(); /* NONE */
    words[word_count++] = (uint32_t)kmx_parallel_all(mark_all, 500);
    words[word_count++] = g_mark[0][0] + g_mark[3][0]; /* 510 + 513 */
    for (uint32_t i = 0; i < 1024; i++) {
        g_source[i] = i * 2654435761u + 7u;
    }
    words[word_count++] = (uint32_t)kmx_push(
        2, 0, 0, g_source, 4096
    ); /* BAD: not in the program */
    words[word_count++] = (uint32_t)kmx_push(
        1, 0, 32, g_source, 64
    ); /* BAD: not a line */
    words[word_count++] = (uint32_t)kmx_push(1, 0, 4096, g_source, 4096);
    kmx_invalidate_at(0, 4096, 4096);
    const uint32_t* pushed = (const uint32_t*)kmx_region(0, 0) + 1024;
    uint32_t wrong = 0;
    for (uint32_t i = 0; i < 1024; i++) {
        wrong += pushed[i] != g_source[i];
    }
    words[word_count++] = wrong; /* 0 */
    kmx_result(0, words, word_count);
    return 0;
}
