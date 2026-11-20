// The tree from the command's words, for the movers.
#pragma once

#include <stdint.h>

#include "tree.h"

namespace tree {

    struct Word {
        uint32_t home, phase, senders, wait;
    };

    inline Word word_of(uint32_t a0) {
        return {a0 & 31u, a0 >> 5 & 127u, a0 >> 12 & 31u, 0};
    }

    template <typename Command>
    inline uint32_t parent(const Command& command, uint32_t island) {
        const uint32_t words[6] = {
            command.a1, command.a2, command.a3, command.a4, command.a5, command.a6
        };
        return words[island / 6] >> (5 * (island % 6)) & 31u;
    }

    // The children of `here` in island order (at most `most`); their count.
    template <typename Command>
    inline uint32_t children(
        const Command& command, uint32_t here, uint32_t* kids, uint32_t most
    ) {
        const uint32_t home = command.a0 & 31u;
        uint32_t count = 0;
        for (uint32_t i = 0; i < 32 && count < most; i++) {
            if (i != home && i != here && parent(command, i) == here) {
                kids[count++] = i;
            }
        }
        return count;
    }

    // `here`'s place among its parent's children.
    template <typename Command>
    inline uint32_t place(const Command& command, uint32_t here) {
        uint32_t kids[32];
        const uint32_t count = children(command, parent(command, here), kids, 32);
        for (uint32_t j = 0; j < count; j++) {
            if (kids[j] == here) {
                return j;
            }
        }
        return 0;
    }

}  // namespace tree
