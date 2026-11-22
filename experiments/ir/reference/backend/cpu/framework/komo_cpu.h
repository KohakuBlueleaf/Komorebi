// The C interface of a built CPU module: workers, entries by name, invocations.
// Formats, scalar types and failures are numbered as in docs/ir/bytecode.md.
#ifndef KOMO_CPU_H
#define KOMO_CPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KOMO_CPU_ABI 1

// What an argument is: a tensor, a scalar's bits, a boolean (0 or 1).
enum { KOMO_ARG_TENSOR = 0, KOMO_ARG_SCALAR = 1, KOMO_ARG_BOOL = 2 };

// How an invocation ended.
enum { KOMO_COMPLETED = 0, KOMO_FAILED = 1, KOMO_INVALID = 2 };

// A tensor: `rows` by `cols` elements of `format`, row after row, each little-endian.
typedef struct komo_tensor {
    void* data;
    uint32_t format;
    uint64_t rows;
    uint64_t cols;
} komo_tensor;

// One argument of an invocation: `tensor` for a tensor, `bits` for the rest.
typedef struct komo_arg {
    uint32_t kind;
    komo_tensor* tensor;
    uint64_t bits;
} komo_arg;

// An invocation's end: its state, the failure's number when it failed, and the leader's
// result when the entry has one.
typedef struct komo_end {
    uint32_t state;
    uint32_t failure;
    uint32_t has_result;
    uint64_t result;
} komo_end;

typedef struct komo_pool komo_pool;

// A pool of `workers` resident workers; NULL when they cannot be made.
komo_pool* komo_open(uint32_t workers);

// Ends the workers and frees the pool.
void komo_close(komo_pool* pool);

// The entry named `Program.handler`, or -1.
int32_t komo_entry(const char* name);

// Invokes `entry` on the workers `first` to `first + count - 1` and waits for its end.
komo_end komo_invoke(
    komo_pool* pool,
    int32_t entry,
    const komo_arg* args,
    uint32_t count_args,
    uint32_t first,
    uint32_t count
);

#ifdef __cplusplus
}
#endif

#endif
