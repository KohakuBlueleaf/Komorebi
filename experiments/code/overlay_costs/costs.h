// The words a tile's five persistent kernels share with the host and with overlays:
// control words, operations, the overlay's call and the runtime's tables of calls.
#pragma once

#include <stdint.h>

namespace cs {

    // Control words of a core: the host writes 4..31, then the sequence word 0.
    constexpr uint32_t kSequence = 0, kOp = 4, kFlags = 5, kArg = 6, kCount = 7;
    constexpr uint32_t kAt = 8;        // a broadcast's start (wall clock, low word)
    constexpr uint32_t kProgram = 9;   // the overlay's or the program's id
    constexpr uint32_t kSlot = 10;     // OP_READ: the slot it runs in
    constexpr uint32_t kFirst = 11;    // OP_READ: the first page
    constexpr uint32_t kLayoutA = 11;  // OP_LAYOUTS: two configurations and masks
    constexpr uint32_t kMaskA = 12, kLayoutB = 13, kMaskB = 14, kTiles = 15;
    constexpr uint32_t kParts = 16;  // 5 x (bank, address, bytes) of the parts
    constexpr uint32_t kResp = 32;   // the response line, 16 words
    constexpr uint32_t kGo = 48;     // the parked cores' go word
    constexpr uint32_t kDone = 49;   // done words: data-movement 1, compute 0, 1, 2
    constexpr uint32_t kStop = 53;   // set by the host: every wait gives up
    constexpr uint32_t kLeave = 0xFFFFFFFFu;  // the go word that ends the parked cores
    constexpr uint32_t kParkedOp = 56;
    constexpr uint32_t kParkedLayout = 57, kParkedMask = 58;  // a configuration
    constexpr uint32_t kOut = 64;  // 8 words per core (as kDone): status, result,
    // the wall clock when it started, the cycles it took, bad buffer pointers
    constexpr uint32_t kOutStatus = 0, kOutResult = 1, kOutStart = 2, kOutCycles = 3;
    constexpr uint32_t kOutBad = 4;   // a buffer pointer outside the layout
    constexpr uint32_t kExpect = 96;  // 16 expected output values in 8 words
    constexpr uint32_t kControlWords = 128;

    constexpr uint32_t kOpClock = 1, kOpExit = 2, kOpRead = 3, kOpCompute = 4;
    constexpr uint32_t kOpSwitch = 5, kOpLayouts = 6;
    constexpr uint32_t kFlagAt = 1;       // start at the wall clock in kAt
    constexpr uint32_t kFlagOverlay = 2;  // run the loop in the slot, not inline

    // What the parked cores run on a go.
    constexpr uint32_t kParkedNone = 0, kParkedDrain = 1, kParkedCompute = 2;
    constexpr uint32_t kParkedComputeSlot = 3, kParkedPart = 4, kParkedSetup = 5;
    constexpr uint32_t kParkedCheck = 6;  // drain, every value of every tile checked

    constexpr uint32_t kOk = 0, kBadOp = 1, kReadTimeout = 2, kTooBig = 3, kIdle = 4;
    constexpr uint32_t kNotParked = 5, kWaitTimeout = 6;

    // Slots of a core: data-movement 0, data-movement 1, compute 0, 1, 2, and a
    // second slot of data-movement 0.
    constexpr uint32_t kSlots = 6, kSlotSecond = 5;
    constexpr uint32_t kCbIn = 0, kCbOut = 16, kCbRead = 2;
    constexpr uint32_t kTileValues = 1024, kTileBytes = 2048;
    constexpr uint32_t kLayoutWords = 4 * 32;  // a configuration: 4 words a buffer

    // The runtime's calls a reader loop makes, as a data-movement core's table.
    struct MoverApi {
        bool (*reserve)(uint32_t cb, uint32_t pages, volatile uint32_t* stop);
        uint32_t (*write_ptr)(uint32_t cb);
        void (*read_page)(uint32_t tensor, uint32_t page, uint32_t bytes, uint32_t l1);
        bool (*read_barrier)();
        void (*push)(uint32_t cb, uint32_t pages);
    };

    // The runtime's calls a compute loop makes, as one compute core's table.
    struct ComputeApi {
        bool (*wait_front)(uint32_t cb, uint32_t pages, volatile uint32_t* stop);
        bool (*reserve_back)(uint32_t cb, uint32_t pages, volatile uint32_t* stop);
        void (*copy_init)(uint32_t cb);
        void (*scalar_init)();
        void (*acquire)();
        void (*copy)(uint32_t cb, uint32_t tile, uint32_t dst);
        void (*mul)(uint32_t dst, uint32_t scalar);
        void (*commit)();
        void (*wait)();
        void (*pack)(uint32_t dst, uint32_t cb);
        void (*release)();
        void (*pop)(uint32_t cb, uint32_t pages);
        void (*push)(uint32_t cb, uint32_t pages);
    };

    // An overlay's entry takes one Call and returns a status or a result word.
    struct Call {
        uint32_t count;  // blocks or tiles
        uint32_t arg;    // the first page (reader), the scalar (compute)
        uint32_t tensor;
        uint32_t page_bytes;
        volatile uint32_t* stop;
        const void* api;  // a MoverApi or a ComputeApi: the table form
        uint32_t (*mix)(uint32_t);
        volatile uint32_t* calls;  // a part's call counter
    };

    using Entry = uint32_t (*)(const Call*);

    // The persistent kernels' mix(), mirrored by the host's reference.
    inline uint32_t mix(uint32_t x) {
        return ((x << 7) | (x >> 25)) ^ 0x5A5A5A5Au;
    }

}  // namespace cs
