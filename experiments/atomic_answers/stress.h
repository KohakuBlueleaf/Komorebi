// The probe's per-core L1 block, in words.
#pragma once
#include <stdint.h>

namespace hs {
    constexpr uint32_t kTail = 0;     // 16-byte line: the word other cores add to
    constexpr uint32_t kBar = 4;      // start barrier counter (core 0)
    constexpr uint32_t kBarRet = 8;   // the barrier atomic's return line
    constexpr uint32_t kPeek = 12;    // barrier read slot
    constexpr uint32_t kStat = 16;    // 16 words, below
    constexpr uint32_t kSent = 112;   // per destination: added by answered atomics
    constexpr uint32_t kLostN = 232;  // per destination: added by unanswered atomics
    constexpr uint32_t kRing = 352;   // 32 return lines of 4 words (to 480)
    constexpr uint32_t kRingN = 32;
    constexpr uint32_t kTable = 480;  // core table: NoC x | y << 16 (to 599)
    constexpr uint32_t kTailArr = 640;  // tails held for others (counter tiles), 16 B each
    constexpr uint32_t kLastEnd = 1120;  // per destination: p + n of the last answer
    constexpr uint32_t kWords = 1280;

    // Stat words. sFail: 1, 2 start barrier; 3 an atomic's count; 4 every return line
    // lost.
    enum : uint32_t {
        sDone,
        sFail,
        sResv,
        sLost,
        sLatMax,
        sLatLo,
        sLatHi,
        sViol,
        sCycLo,
        sCycHi
    };

    constexpr uint32_t kSentinel = 0xFFFFFFFFu;
}  // namespace hs
