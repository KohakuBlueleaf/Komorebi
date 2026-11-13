/* The map's vector loop (vmap.c): destination[i] = mix(source[i]) for i < words. */
#pragma once

#include <stdint.h>

void vmap(uint32_t* destination, const uint32_t* source, uint64_t words);
