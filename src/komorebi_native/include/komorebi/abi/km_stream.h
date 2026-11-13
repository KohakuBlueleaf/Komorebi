/* Komorebi ABI: the micro-op stream a host uploads to the dispatcher. Plain C,
   usable from C and C++; builds on km_abi.h. */
#ifndef KOMOREBI_ABI_KM_STREAM_H
#define KOMOREBI_ABI_KM_STREAM_H

#include <stddef.h>

#include "komorebi/abi/km_abi.h"

#ifndef KM_STATIC_ASSERT
#if defined(__cplusplus)
#define KM_STATIC_ASSERT(c, msg) static_assert(c, msg)
#else
#define KM_STATIC_ASSERT(c, msg) _Static_assert(c, msg)
#endif
#endif

#define KM_STREAM_MAGIC 0x4B4D5354u /* "KMST" */
#define KM_STREAM_VERSION 1u
#define KM_STREAM_REGISTERS 16u     /* registers, unsigned 32-bit, wrapping */
#define KM_STREAM_SEED_REGISTER 15u /* holds the xorshift state; starts as the seed */
#define KM_STREAM_MAX_OPS 65535u
#define KM_STREAM_MAX_ACTIVE 4u /* a fifth KM_HANDLER_RUN is rejected */
#define KM_STREAM_LINE 64u      /* uploads are whole lines */
#define KM_TEMPLATE_NONE 0xFFu  /* first / count register: take it from the template */
#define KM_TEMPLATE_PATCHES 4u
#define KM_TEMPLATE_ARGS 10u
#define KM_STEP_OPS 64u       /* micro-ops per stream per kernel pass */
#define KM_QUEUE_JOBS 64u     /* waiting jobs the queue holds */
#define KM_QUEUE_LOOK 16u     /* jobs examined per pass */
#define KM_QUEUE_PASSES 1024u /* times a job may be passed over */

/* Micro-ops. Unused operand fields are ignored. */
#define KM_OP_INVALID 0u
#define KM_OP_SET 1u      /* a = immediate */
#define KM_OP_MOV 2u      /* a = b */
#define KM_OP_ADD 3u      /* a = b + c */
#define KM_OP_ADDI 4u     /* a = b + immediate */
#define KM_OP_SUB 5u      /* a = b - c */
#define KM_OP_MUL 6u      /* a = low 32 bits of b * c */
#define KM_OP_AND 7u      /* a = b & c */
#define KM_OP_SHR 8u      /* a = b >> (immediate & 31), logical */
#define KM_OP_MOD 9u      /* a = b % c; c = 0 is KM_STREAM_ERROR_MOD_ZERO */
#define KM_OP_RAND 10u    /* xorshift32 step on the seed register; a = new state */
#define KM_OP_JMP 16u     /* pc = immediate */
#define KM_OP_JLT 17u     /* if a < b (unsigned) pc = immediate */
#define KM_OP_JEQ 18u     /* if a == b pc = immediate */
#define KM_OP_JNZ 19u     /* if a != 0 pc = immediate */
#define KM_OP_HALT 20u    /* wait for every command and job, then end; status = a */
#define KM_OP_SUBMIT 32u  /* submit template immediate */
#define KM_OP_ENQUEUE 33u /* queue template immediate as a job of a cores */
/* go on when at most immediate commands and jobs are pending */
#define KM_OP_WAIT 40u
/* a = result of the latest completed KM_FLAG_RESULT command */
#define KM_OP_RESULT 41u
#define KM_OP_EVENT 48u /* post KM_EVENT_STREAM with status = a */
#define KM_OP_TIME 49u  /* a = low 32 bits of the tick counter */

/* Events of a stream; `req` is the header's req. For KM_EVENT_STREAM and
   KM_EVENT_STREAM_END auxiliary is the low 24 bits of a register. */
#define KM_EVENT_STREAM 4u
#define KM_EVENT_STREAM_END 5u
#define KM_EVENT_STREAM_ERROR 6u /* auxiliary = reason | pc << 8 */

/* Error reasons. */
#define KM_STREAM_ERROR_HEADER 1u
#define KM_STREAM_ERROR_OPCODE 2u
/* also: the program counter ran past the last micro-op */
#define KM_STREAM_ERROR_JUMP 3u
#define KM_STREAM_ERROR_REGISTER 4u
#define KM_STREAM_ERROR_TEMPLATE 5u
#define KM_STREAM_ERROR_STEPS 6u
#define KM_STREAM_ERROR_MOD_ZERO 7u
#define KM_STREAM_ERROR_DEADLINE 8u
#define KM_STREAM_ERROR_NEVER_FITS 9u /* a job whose count is 0 or above the grid */
#define KM_STREAM_ERROR_CANCELLED 10u
#define KM_STREAM_ERROR_GROUP 11u /* an SUBMIT whose group is outside the grid */

typedef struct {
    uint32_t magic, version;
    uint32_t op_count,
        command_count;  /* then op_count micro-ops, then command_count templates */
    uint32_t max_steps; /* ended with KM_STREAM_ERROR_STEPS after this many micro-ops */
    uint32_t deadline_microseconds; /* for each command the stream submits */
    uint32_t req;                   /* request id of the stream's events */
    uint32_t seed;                  /* register 15 at start; 0 is rejected */
    uint32_t pad[8];
} km_stream_t;

typedef struct {
    uint8_t op, a, b, c; /* opcode and up to three register numbers */
    uint32_t immediate;
    uint32_t imm2;
    uint32_t pad;
} km_uop_t;

typedef struct {
    km_command_t rec;     /* sequence and the landing words are overwritten at submit */
    uint8_t arg[4];       /* argument number 0 to 9 of each patch */
    uint8_t registers[4]; /* register patched into that argument */
    uint8_t first_register, count_register, patch_count, pad[5];
} km_template_t;

KM_STATIC_ASSERT(sizeof(km_stream_t) == 64, "the header is one 64-byte line");
KM_STATIC_ASSERT(sizeof(km_uop_t) == 16, "a micro-op is 16 bytes");
KM_STATIC_ASSERT(sizeof(km_template_t) == 80, "a template is 80 bytes");
KM_STATIC_ASSERT(offsetof(km_stream_t, seed) == 28, "the seed follows req");
KM_STATIC_ASSERT(
    offsetof(km_uop_t, immediate) == 4, "immediate follows the operand bytes"
);
KM_STATIC_ASSERT(
    offsetof(km_template_t, arg) == 64, "the patch list follows the command"
);
KM_STATIC_ASSERT(
    KM_STREAM_REGISTERS == 16 && KM_STREAM_SEED_REGISTER < KM_STREAM_REGISTERS,
    "registers"
);

#endif /* KOMOREBI_ABI_KM_STREAM_H */
