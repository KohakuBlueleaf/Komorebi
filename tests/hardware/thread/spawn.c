/* One cluster: kmx_spawn and kmx_join with their refusals, kmx_par beside a busy hart,
 * kmx_par_all with no other cluster, and a kmx_push into the thread's own bank. */
#include <stdint.h>

#include "kmx.h"

/* Set by hart 0 to let hart 1's spawn end; each hart's mark (globals need .data). */
static volatile uint32_t g_go __attribute__((section(".data"))) = 0;
static volatile uint32_t g_mark[4][16] __attribute__((section(".data"))) = {{0}};
static uint32_t g_src[1024] __attribute__((section(".data"), aligned(64))) = {1};

static void wait_go(int hart, void* arg) {
    (void)arg;
    while (!g_go && !kmx_stop_requested()) {
        ;
    }
    g_mark[hart][0] = 100u + (uint32_t)hart;
}

static void mark(int hart, void* arg) {
    g_mark[hart][0] = (uint32_t)(uintptr_t)arg + (uint32_t)hart;
}

static void mark_all(int cluster, int hart, uint64_t arg) {
    g_mark[hart][0] = (uint32_t)arg + 10u * (uint32_t)cluster + (uint32_t)hart;
}

int kmx_main(int hart) {
    (void)hart;
    uint32_t w[24];
    uint32_t n = 0;
    w[n++] = (uint32_t)kmx_spawn(1, wait_go, 0); /* OK */
    w[n++] = (uint32_t)kmx_spawn(1, mark, 0);    /* FULL: hart 1 is busy */
    w[n++] = (uint32_t)kmx_par(mark, 0);         /* FULL */
    w[n++] = (uint32_t)kmx_spawn(0, mark, 0);    /* BAD */
    w[n++] = (uint32_t)kmx_join(4);              /* BAD */
    g_go = 1;
    w[n++] = (uint32_t)kmx_join(1); /* OK */
    w[n++] = g_mark[1][0];          /* 101 */
    w[n++] = (uint32_t)kmx_spawn(2, mark, (void*)200);
    w[n++] = (uint32_t)kmx_spawn(3, mark, (void*)300);
    w[n++] = (uint32_t)kmx_join(2);
    w[n++] = (uint32_t)kmx_join(3);
    w[n++] = g_mark[2][0]; /* 202 */
    w[n++] = g_mark[3][0]; /* 303 */
    w[n++] = (uint32_t)kmx_par(mark, (void*)400);
    w[n++] = g_mark[0][0] + g_mark[1][0] + g_mark[2][0] + g_mark[3][0]; /* 1606 */
    w[n++] = kmx_clusters();                                            /* 2 */
    w[n++] = (uint32_t)kmx_lead();                                      /* 1 */
    w[n++] = (uint32_t)kmx_serve();                                     /* NONE */
    w[n++] = (uint32_t)kmx_par_all(mark_all, 500);
    w[n++] = g_mark[0][0] + g_mark[3][0]; /* 510 + 513 */
    for (uint32_t i = 0; i < 1024; i++) {
        g_src[i] = i * 2654435761u + 7u;
    }
    w[n++] = (uint32_t)kmx_push(2, 0, 0, g_src, 4096); /* BAD: not in the program */
    w[n++] = (uint32_t)kmx_push(1, 0, 32, g_src, 64);  /* BAD: not a line */
    w[n++] = (uint32_t)kmx_push(1, 0, 4096, g_src, 4096);
    kmx_take_at(0, 4096, 4096);
    const uint32_t* r = (const uint32_t*)kmx_region(0, 0) + 1024;
    uint32_t wrong = 0;
    for (uint32_t i = 0; i < 1024; i++) {
        wrong += r[i] != g_src[i];
    }
    w[n++] = wrong; /* 0 */
    kmx_result(0, w, n);
    return 0;
}
