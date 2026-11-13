/* The map's vector loop (vmap.c): d[i] = f(s[i]) for i < n. */
#pragma once

#include <stdint.h>

void vmap(uint32_t* d, const uint32_t* s, uint64_t n);
