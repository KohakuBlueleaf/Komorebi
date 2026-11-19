// km_overlay_records.hpp on the CPU (L1 an array): a check of the service's own slot
// marks its own record alone. Run by test_overlay_records.py; prints "ok".
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define tt_l1_ptr

static uint8_t memory_[1 << 16];

namespace km {
    inline volatile uint32_t* l1(uint32_t address) {
        return reinterpret_cast<volatile uint32_t*>(memory_ + address);
    }
}  // namespace km

#include "komorebi/device/overlays/km_overlay_records.hpp"

#define CHECK(condition)                               \
    if (!(condition)) {                                \
        printf("line %d: %s\n", __LINE__, #condition); \
        return 1;                                      \
    }

using namespace km;

static const uint32_t kControl = 0x100, kEpoch = 42;
static uint8_t before_[sizeof memory_];

// The area of `slots` slots a part, every record's words set to a pattern of its own.
static void lay_out(uint32_t slots) {
    for (uint32_t byte = 0; byte < sizeof memory_; byte++) {
        memory_[byte] = static_cast<uint8_t>(byte * 7 + 1);
    }
    volatile uint32_t* area = overlay_area(kControl);
    for (uint32_t part = 0; part < kOverlayParts; part++) {
        area[kOverlaySlots + part] = slots;
    }
    memcpy(before_, memory_, sizeof memory_);
}

// The bytes that differ from the layout, but for the word at `allowed`.
static uint32_t changed_but(volatile uint32_t* allowed) {
    const uint32_t at = (
        static_cast<uint32_t>(reinterpret_cast<volatile uint8_t*>(allowed) - memory_)
    );
    uint32_t changed = 0;
    for (uint32_t byte = 0; byte < sizeof memory_; byte++) {
        changed += (byte < at || byte >= at + 4) && memory_[byte] != before_[byte];
    }
    return changed;
}

int main() {
    static_assert(kOverlayOwnSlot == kOverlaySlotsMax + 1, "the own slot's number");
    static_assert(
        kControl + kOverlayArea + kOverlayAreaBytes + 16 * kOverlayParts * 8 <
            sizeof memory_,
        "the area fits the array"
    );
    for (uint32_t slots = 1; slots <= 2; slots++) {  // the two layouts
        lay_out(slots);
        volatile uint32_t* own = overlay_own_record(kControl);
        CHECK(
            overlay_slot_record(kControl, kOverlayPartReader, kOverlayOwnSlot) == own
        );
        CHECK(
            overlay_slot_address(kControl, kOverlayPartReader, kOverlayOwnSlot) ==
            own[kOverlayAddress]
        );
        overlay_check(kControl, kOverlayPartReader, kOverlayOwnSlot, kEpoch);
        CHECK(own[kOverlayChecked] == (kEpoch | kOverlayCheckedMark));
        CHECK(changed_but(own + kOverlayChecked) == 0);  // no other record touched
        own[kOverlayHeld] = kOverlaysMax - 1;
        *overlay_epoch(kControl) = kEpoch;
        CHECK(overlay_own_checked(kControl, kOverlaysMax - 1) == own);  // the fast path
        // a slot of its own still marks that slot's record: the writer's last
        lay_out(slots);
        volatile uint32_t* mine = (
            overlay_record(kControl, kOverlayPartWriter, slots - 1)
        );
        overlay_check(kControl, kOverlayPartWriter, slots - 1, kEpoch);
        CHECK(mine[kOverlayChecked] == (kEpoch | kOverlayCheckedMark));
        CHECK(changed_but(mine + kOverlayChecked) == 0);
    }
    printf("ok\n");
    return 0;
}
