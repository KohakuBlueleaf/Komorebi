// The probe's CPU code, the same on a data-movement CPU and a compute CPU: the table of
// pages (4-way sets found by a hash), the 1,024-pixel lookup loop, the integer colour.
#pragma once

#include <stdint.h>

#include "komorebi/device/km_core.hpp"
#include "lookup.h"

namespace texture {

    constexpr uint32_t kNone = 0xFFFFFFFFu;

    // The core's L1 tensors (the arguments kArgPages to kArgReplay).
    struct Tensors {
        uint32_t pages, slots, streams, planes, out, results, replay;
    };

    inline uint32_t* words(uint32_t address) {
        return reinterpret_cast<uint32_t*>(address);
    }

    inline void fence() {
        asm volatile("fence" ::: "memory");
    }

    // Sets of kWays slots; a slot's tag is its page's key + 1 (0: empty), its use the
    // table's clock at its last hit (least recently used is replaced).
    struct Table {
        uint32_t tag[kSlotsMax];
        uint32_t use[kSlotsMax];
        uint32_t shift = 31;  // 32 - log2(sets)
        uint32_t clock = 0;

        // An empty table of `slots` slots (8, 16 or 32); false for another count.
        bool clear(uint32_t slots) {
            if (slots != 8 && slots != 16 && slots != 32) {
                return false;
            }
            shift = slots == 8 ? 31 : (slots == 16 ? 30 : 29);
            for (uint32_t slot = 0; slot < kSlotsMax; slot++) {
                tag[slot] = 0;
                use[slot] = 0;
            }
            clock = 0;
            return true;
        }

        uint32_t set(uint32_t key) const {
            return ((key * 0x9E3779B1u) >> shift) * kWays;
        }

        // The slot holding `key`, or kNone.
        uint32_t find(uint32_t key) const {
            const uint32_t first = set(key), want = key + 1;
            const uint32_t* ways = tag + first;
            if (ways[0] == want) {
                return first;
            }
            if (ways[1] == want) {
                return first + 1;
            }
            if (ways[2] == want) {
                return first + 2;
            }
            if (ways[3] == want) {
                return first + 3;
            }
            return kNone;
        }

        // The slot `key` replaces in its set: an empty one, else the least recently
        // used. The caller fills it, then sets its tag.
        uint32_t victim(uint32_t key) const {
            const uint32_t first = set(key);
            uint32_t best = first;
            for (uint32_t way = 0; way < kWays; way++) {
                if (tag[first + way] == 0) {
                    return first + way;
                }
                if (use[first + way] < use[best]) {
                    best = first + way;
                }
            }
            return best;
        }
    };

    // The 1,024 pixels of `stream`, four at a time: address(word) is the texel's L1
    // address; the texels go to `out`. Returns their sum.
    template <typename Address>
    inline uint32_t lookup(const uint32_t* stream, uint32_t* out, Address& address) {
        uint32_t sum = 0;
        for (uint32_t pixel = 0; pixel < kPixels; pixel += 4) {
            const uint32_t w0 = stream[pixel], w1 = stream[pixel + 1];
            const uint32_t w2 = stream[pixel + 2], w3 = stream[pixel + 3];
            const uint32_t a0 = address(w0), a1 = address(w1);
            const uint32_t a2 = address(w2), a3 = address(w3);
            const uint32_t t0 = *words(a0), t1 = *words(a1);
            const uint32_t t2 = *words(a2), t3 = *words(a3);
            out[pixel] = t0;
            out[pixel + 1] = t1;
            out[pixel + 2] = t2;
            out[pixel + 3] = t3;
            sum += (t0 + t1) + (t2 + t3);
        }
        return sum;
    }

    // The table of the loop's hash forms: the kSlots keys of its key list in order.
    inline bool table_of(Table& table, const Tensors& at, uint32_t slots) {
        if (!table.clear(slots)) {
            return false;
        }
        const uint32_t list = slots == 8 ? 0 : (slots == 16 ? 1 : 2);
        const uint32_t* keys = words(at.streams) + kKeysAt + list * kSlotsMax;
        for (uint32_t index = 0; index < slots; index++) {
            const uint32_t slot = table.victim(keys[index]);
            table.tag[slot] = keys[index] + 1;
        }
        return true;
    }

    // LOOP on this CPU: `reps` timed runs of the lookup over stream a2 in form a1.
    inline void loop(const Tensors& at, const km_command_t& command) {
        uint32_t* results = words(at.results);
        const uint32_t form = command.a1, stream_index = command.a2;
        const uint32_t reps = command.a4 < kRepsMax ? command.a4 : kRepsMax;
        uint32_t error = stream_index < kStreams ? 0 : kErrorArgs;
        Table table;
        if (form != kFormDirect && !table_of(table, at, command.a3)) {
            error |= kErrorArgs;
        }
        const uint32_t* stream = (
            words(at.streams) + (stream_index % kStreams) * kPixels
        );
        uint32_t* out = words(at.out);
        uint32_t sum = 0, missing = 0;
        const uint32_t pages = at.pages;
        uint32_t last_key = kNone, last_base = pages;
        auto direct = [&](uint32_t word) {
            return pages + (word << 2);  // page << 12 | offset << 2
        };
        auto hashed = [&](uint32_t word) {
            uint32_t slot = table.find(word >> kOffsetBits);
            if (slot == kNone) {
                missing++;
                slot = 0;
            }
            return pages + (slot << kPageShift) + ((word & kOffsetMask) << 2);
        };
        auto hashed_last = [&](uint32_t word) {
            const uint32_t key = word >> kOffsetBits;
            if (key != last_key) {
                uint32_t slot = table.find(key);
                if (slot == kNone) {
                    missing++;
                    slot = 0;
                }
                last_key = key;
                last_base = pages + (slot << kPageShift);
            }
            return last_base + ((word & kOffsetMask) << 2);
        };
        for (uint32_t rep = 0; error == 0 && rep < reps; rep++) {
            last_key = kNone;
            const uint32_t start = km::wall();
            if (form == kFormDirect) {
                sum = lookup(stream, out, direct);
            } else if (form == kFormHash) {
                sum = lookup(stream, out, hashed);
            } else {
                sum = lookup(stream, out, hashed_last);
            }
            results[kResultReps + rep] = km::wall() - start;
        }
        results[kResultSum] = sum;
        results[kResultMisses] = missing;
        results[kResultError] = error;
        results[kResultRun] = command.a9;
        fence();
    }

    // A light term in [0, 1] as float32 bits to 0.16 fixed point (1.0 is 65536).
    inline uint32_t fixed16(uint32_t bits) {
        const uint32_t exponent = bits >> 23;
        const uint32_t mantissa = (bits & 0x7FFFFFu) | 0x800000u;
        const uint32_t right = exponent >= 103 ? 134 - exponent : 31;
        return mantissa >> right;
    }

    // The colour of a pixel: each of r, g, b times the light, rounded; alpha kept.
    inline uint32_t shade(uint32_t texel, uint32_t light) {
        const uint32_t scale = fixed16(light);
        const uint32_t r = ((texel & 0xFFu) * scale + 0x8000u) >> 16;
        const uint32_t g = (((texel >> 8) & 0xFFu) * scale + 0x8000u) >> 16;
        const uint32_t b = (((texel >> 16) & 0xFFu) * scale + 0x8000u) >> 16;
        return r | g << 8 | b << 16 | (texel & 0xFF000000u);
    }

    // COLOUR in integers on this CPU: `reps` timed runs over the texel and light
    // planes, the colour plane into `out`.
    inline void colour(const Tensors& at, const km_command_t& command) {
        uint32_t* results = words(at.results);
        const uint32_t reps = command.a2 < kRepsMax ? command.a2 : kRepsMax;
        const uint32_t* texel = words(at.planes);
        const uint32_t* light = texel + kPixels;
        uint32_t* out = words(at.out);
        for (uint32_t rep = 0; rep < reps; rep++) {
            const uint32_t start = km::wall();
            for (uint32_t pixel = 0; pixel < kPixels; pixel += 2) {
                const uint32_t c0 = shade(texel[pixel], light[pixel]);
                const uint32_t c1 = shade(texel[pixel + 1], light[pixel + 1]);
                out[pixel] = c0;
                out[pixel + 1] = c1;
            }
            results[kResultReps + rep] = km::wall() - start;
        }
        results[kResultError] = 0;
        results[kResultRun] = command.a9;
        fence();
    }

}  // namespace texture
