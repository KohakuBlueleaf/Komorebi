/* Work-cluster example: the decision (see decide.h). No libc, no globals: it runs as
 * it is in the X280 image and in the host's shared library. */
#include "decide.h"

#include "komorebi/abi/km_abi.h"

uint32_t wc_start(wc_state* state, const uint32_t* words) {
    for (int i = 0; i < WC_WORDS; i++) {
        state->word[i] = words[i];
    }
    state->round = 0;
    state->first = 0;
    state->sum = 0;
    state->trail = 2166136261u;
    const uint32_t cores = words[WC_CORES], count = words[WC_COUNT];
    const uint32_t base = words[WC_BASE];
    const int bad = (
        cores == 0 || base > KM_CORES_MAX || cores > KM_CORES_MAX - base ||
        count == 0 || count > cores || words[WC_MAX_ROUNDS] == 0
    );
    return bad ? 1u : 0u;
}

void wc_round(const wc_state* state, wc_command commands[2]) {
    commands[0].handler = WC_HANDLER_STEP;
    commands[0].flags = 0;
    commands[0].first = state->word[WC_BASE] + state->first;
    commands[0].count = state->word[WC_COUNT];
    commands[0].a0 = state->word[WC_ADDR];
    commands[0].a1 = state->round + 1;
    commands[1].handler = WC_HANDLER_SUM;
    commands[1].flags = KM_FLAG_EXCLUSIVE | KM_FLAG_RESULT;
    commands[1].first = state->word[WC_BASE];
    commands[1].count = state->word[WC_CORES];
    commands[1].a0 = state->word[WC_ADDR];
    commands[1].a1 = state->word[WC_CORES];
}

uint32_t wc_result(wc_state* state, uint32_t sum) {
    state->sum = sum;
    state->round += 1;
    state->trail = (state->trail ^ state->first) * 16777619u;
    if (sum >= state->word[WC_LIMIT]) {
        return WC_END;
    }
    if (state->round >= state->word[WC_MAX_ROUNDS]) {
        return WC_NO_END;
    }
    /* the value from the card picks where the next STEP lands */
    state->first = sum % (state->word[WC_CORES] - state->word[WC_COUNT] + 1);
    return WC_AGAIN;
}
