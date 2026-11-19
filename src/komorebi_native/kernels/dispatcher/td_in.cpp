// The Tensix dispatcher's "in" loop on its own processor (td_in.hpp): from out's start
// to its end, then the final words.
#include <stdint.h>

#include "td_in.hpp"

void kernel_main() {
    td::In in;
    in.init();
    td::Hand& hand = in.hand;
    for (;;) {  // out's start, or its end before it (a start-up that failed)
        invalidate_l1_cache();
        if (hand.word(kTwGo) != 0 || hand.word(kTwEnd) != 0) {
            break;
        }
        td::pace();
    }
    while (hand.word(kTwGo) != 0) {
        invalidate_l1_cache();
        if (hand.word(kTwEnd) != 0) {
            in.pass(km::wall64());  // the events out posted before its end
            break;
        }
        if (!in.pass(km::wall64())) {
            td::pace();
        }
    }
    in.over();
}
