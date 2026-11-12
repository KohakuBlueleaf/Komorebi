/* Komorebi ABI: the micro-op stream a host uploads to the dispatch kernel. Plain C,
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
#define KM_STREAM_REGS 16u     /* registers, unsigned 32-bit, wrapping */
#define KM_STREAM_SEED_REG 15u /* holds the xorshift state; starts as the seed */
#define KM_STREAM_MAX_OPS 65535u
#define KM_STREAM_MAX_ACTIVE 4u /* a fifth KM_SEG_RUN is refused */
#define KM_STREAM_LINE 64u      /* uploads are whole lines */
#define KM_TPL_NONE 0xFFu       /* first_reg / count_reg: take it from the template */
#define KM_TPL_PATCHES 4u
#define KM_TPL_ARGS 10u
#define KM_STEP_OPS 64u       /* micro-ops per stream per kernel pass */
#define KM_QUEUE_JOBS 64u     /* waiting jobs the queue holds */
#define KM_QUEUE_LOOK 16u     /* jobs examined per pass */
#define KM_QUEUE_PASSES 1024u /* times a job may be passed over */

/* Micro-ops. Unused operand fields are ignored. */
#define KM_OP_INVALID 0u
#define KM_OP_SET 1u     /* a = imm */
#define KM_OP_MOV 2u     /* a = b */
#define KM_OP_ADD 3u     /* a = b + c */
#define KM_OP_ADDI 4u    /* a = b + imm */
#define KM_OP_SUB 5u     /* a = b - c */
#define KM_OP_MUL 6u     /* a = low 32 bits of b * c */
#define KM_OP_AND 7u     /* a = b & c */
#define KM_OP_SHR 8u     /* a = b >> (imm & 31), logical */
#define KM_OP_MOD 9u     /* a = b % c; c = 0 is KM_SERR_MOD_ZERO */
#define KM_OP_RAND 10u   /* xorshift32 step on the seed register; a = new state */
#define KM_OP_JMP 16u    /* pc = imm */
#define KM_OP_JLT 17u    /* if a < b (unsigned) pc = imm */
#define KM_OP_JEQ 18u    /* if a == b pc = imm */
#define KM_OP_JNZ 19u    /* if a != 0 pc = imm */
#define KM_OP_HALT 20u   /* wait for every record and job, then end; status = a */
#define KM_OP_ISSUE 32u  /* issue template imm */
#define KM_OP_SUBMIT 33u /* queue template imm as a job of a cores */
#define KM_OP_WAIT 40u   /* go on when at most imm records and jobs are pending */
#define KM_OP_RESULT 41u /* a = result of the latest completed KM_F_RESULT record */
#define KM_OP_EVENT 48u  /* post KM_EV_STREAM with status = a */
#define KM_OP_TIME 49u   /* a = low 32 bits of the tick counter */

/* Events of a stream; `req` is the header's req. aux is the low 24 bits of a register
   (KM_EV_STREAM, KM_EV_STREAM_END) or reason | pc << 8 (KM_EV_STREAM_ERROR). */
#define KM_EV_STREAM 4u
#define KM_EV_STREAM_END 5u
#define KM_EV_STREAM_ERROR 6u

/* Error reasons. */
#define KM_SERR_HEADER 1u
#define KM_SERR_OPCODE 2u
#define KM_SERR_JUMP 3u /* also: the program counter ran past the last micro-op */
#define KM_SERR_REGISTER 4u
#define KM_SERR_TEMPLATE 5u
#define KM_SERR_STEPS 6u
#define KM_SERR_MOD_ZERO 7u
#define KM_SERR_DEADLINE 8u
#define KM_SERR_NEVER_FITS 9u /* a job whose count is 0 or above the grid */
#define KM_SERR_CANCELLED 10u
#define KM_SERR_TEAM 11u /* an ISSUE whose team is outside the grid */

typedef struct {
    uint32_t magic, version;
    uint32_t n_ops, n_records; /* then n_ops micro-ops, then n_records templates */
    uint32_t max_steps;        /* ended with KM_SERR_STEPS after this many micro-ops */
    uint32_t deadline_us;      /* for each record the stream issues */
    uint32_t req;              /* request id of the stream's events */
    uint32_t seed;             /* register 15 at start; 0 is refused */
    uint32_t pad[8];
} km_stream_t;

typedef struct {
    uint8_t op, a, b, c; /* opcode and up to three register numbers */
    uint32_t imm;
    uint32_t imm2;
    uint32_t pad;
} km_uop_t;

typedef struct {
    km_record_t rec; /* seq and the landing words are overwritten at issue */
    uint8_t arg[4];  /* argument number 0 to 9 of each patch */
    uint8_t reg[4];  /* register patched into that argument */
    uint8_t first_reg, count_reg, n_patch, pad[5];
} km_template_t;

KM_STATIC_ASSERT(sizeof(km_stream_t) == 64, "the header is one 64-byte line");
KM_STATIC_ASSERT(sizeof(km_uop_t) == 16, "a micro-op is 16 bytes");
KM_STATIC_ASSERT(sizeof(km_template_t) == 80, "a template is 80 bytes");
KM_STATIC_ASSERT(offsetof(km_stream_t, seed) == 28, "the seed follows req");
KM_STATIC_ASSERT(offsetof(km_uop_t, imm) == 4, "imm follows the operand bytes");
KM_STATIC_ASSERT(
    offsetof(km_template_t, arg) == 64, "the patch list follows the record"
);
KM_STATIC_ASSERT(
    KM_STREAM_REGS == 16 && KM_STREAM_SEED_REG < KM_STREAM_REGS, "registers"
);

#endif /* KOMOREBI_ABI_KM_STREAM_H */
