// The Tensix dispatcher's "in" loop on its own processor (td_in.hpp): from out's start
// to its end, then the final words.
#include <stdint.h>

#include "td_in.hpp"

void kernel_main() {
    if (!td::block_placed(td::arg(kTdBlock))) {
        return;  // out ends too, before it writes the block
    }
    td::In in;
    in.init();
    td::Hand& hand = in.hand;
    const uint32_t tag = td::arg(kTdTag);  // an earlier load's words are not this one's
    for (;;) {  // out's start, or its end before it (a start-up that failed)
        invalidate_l1_cache();
        if (hand.word(kTwGo) == tag || hand.word(kTwEnd) == tag) {
            break;
        }
        td::pace();
    }
    while (hand.word(kTwGo) == tag) {
        invalidate_l1_cache();
        if (hand.word(kTwEnd) == tag) {
            in.pass(km::wall64());  // the events out posted before its end
            break;
        }
        if (!in.pass(km::wall64())) {
            td::pace();
        }
    }
    in.over();
}
