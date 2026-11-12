/* Work-cluster example: the decision (see decide.h). No libc, no globals: it runs as
 * it is in the X280 image and in the host's shared library. */
#include "decide.h"

#include "komorebi/abi/km_abi.h"

uint32_t wc_start(wc_state* s, const uint32_t* words) {
    for (int i = 0; i < WC_WORDS; i++) {
        s->word[i] = words[i];
    }
    s->round = 0;
    s->first = 0;
    s->sum = 0;
    s->trail = 2166136261u;
    const uint32_t cores = words[WC_CORES], count = words[WC_COUNT];
    const int bad =
        (cores == 0 || cores > KM_CORES_MAX || count == 0 || count > cores ||
         words[WC_MAX_ROUNDS] == 0);
    return bad ? 1u : 0u;
}

void wc_round(const wc_state* s, wc_record out[2]) {
    out[0].seg = WC_SEG_STEP;
    out[0].flags = 0;
    out[0].first = s->first;
    out[0].count = s->word[WC_COUNT];
    out[0].a0 = s->word[WC_ADDR];
    out[0].a1 = s->round + 1;
    out[1].seg = WC_SEG_SUM;
    out[1].flags = KM_F_EXCL | KM_F_RESULT;
    out[1].first = 0;
    out[1].count = s->word[WC_CORES];
    out[1].a0 = s->word[WC_ADDR];
    out[1].a1 = s->word[WC_CORES];
}

uint32_t wc_result(wc_state* s, uint32_t sum) {
    s->sum = sum;
    s->round += 1;
    s->trail = (s->trail ^ s->first) * 16777619u;
    if (sum >= s->word[WC_LIMIT]) {
        return WC_END;
    }
    if (s->round >= s->word[WC_MAX_ROUNDS]) {
        return WC_NO_END;
    }
    /* the value from the card picks where the next STEP lands */
    s->first = sum % (s->word[WC_CORES] - s->word[WC_COUNT] + 1);
    return WC_AGAIN;
}
