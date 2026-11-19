// An overlay part's own memcpy and memset (overlay.part_memory): a copy or a clear the
// compiler turns into a call lands here, in the part, not in the persistent kernel.
#include <cstddef>
#include <cstdint>

extern "C" {

#if KM_PART_MEMCPY
[[gnu::visibility("hidden"), gnu::optimize("no-tree-loop-distribute-patterns")]] void*
__wrap_memcpy(void* to, const void* from, size_t bytes) {
    const uintptr_t all = (
        reinterpret_cast<uintptr_t>(to) | reinterpret_cast<uintptr_t>(from)
    );
    if (((all | bytes) & 3) == 0) {  // whole words
        auto* out = static_cast<uint32_t*>(to);
        const auto* in = static_cast<const uint32_t*>(from);
        for (size_t i = 0; i < bytes / 4; i++) {
            out[i] = in[i];
        }
        return to;
    }
    auto* out = static_cast<uint8_t*>(to);
    const auto* in = static_cast<const uint8_t*>(from);
    for (size_t i = 0; i < bytes; i++) {
        out[i] = in[i];
    }
    return to;
}
#endif

#if KM_PART_MEMSET
[[gnu::visibility("hidden"), gnu::optimize("no-tree-loop-distribute-patterns")]] void*
__wrap_memset(void* to, int value, size_t bytes) {
    auto* out = static_cast<uint8_t*>(to);
    for (size_t i = 0; i < bytes; i++) {
        out[i] = static_cast<uint8_t>(value);
    }
    return to;
}
#endif
}
