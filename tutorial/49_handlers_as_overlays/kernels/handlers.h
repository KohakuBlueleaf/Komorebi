// What the parts and Python agree on: the two dynamic handlers, the words of a message
// and the words each core keeps in L1.
#pragma once

#include <stdint.h>

constexpr uint32_t kHandlerClimb = 1;
constexpr uint32_t kHandlerLand = 2;

// A message is 16 words of 4 bytes; word 0 belongs to Komorebi.
constexpr uint32_t kMessageWords = 16;
constexpr uint32_t kHopsLeft = 1;
constexpr uint32_t kValue = 2;

// The notes of a core, at the address of the program's compute argument 0.
constexpr uint32_t kNoteClimbs = 0;
constexpr uint32_t kNoteLanded = 1;
constexpr uint32_t kNoteTotal = 2;
