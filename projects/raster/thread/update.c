/* UPDATE: the field after a sun change, the reference's arithmetic (indirect.py
 * element_light to pack_field); README "The field's update in the thread". */

/* float32 as numpy: a product rounded before its sum (no fused multiply-add); zeroing
 * loops kept as loops (a memset call does not link) */
#pragma GCC optimize("fp-contract=off", "no-tree-loop-distribute-patterns")

#include "update.h"

#include <stdint.h>

#include "frame.h"
#include "kmx.h"

_Static_assert(RF_U_WORDS <= RF_U_TIMES, "the inputs before the times rows");
_Static_assert(
    4 * (RF_U_TIMES + RF_U_TIMES_ROWS * RF_UT_WORDS) <= RF_UPDATE_BYTES,
    "the times rows in the update region"
);

#define COPY_TICKS (4u * KMX_TICKS_PER_SECOND) /* a fetch's or a store's deadline */
#define APPLY_TICKS (4u * KMX_TICKS_PER_SECOND)
#define X_PAGE_BYTES 2048u /* a bf16 tile */
#define Y_PAGE_BYTES 4096u /* a float32 tile */
#define PAGE_BYTES 4096u   /* a light page */
#define TILE 32u

/* What the steps of one update share (in .data: the image links no .bss). */
typedef struct {
    uint32_t state;   /* 0 nothing, 1 the sources done (x of pass 1 stored) */
    uint32_t bounces; /* the passes of this update (step 1's word) */
    uint32_t passes;  /* passes taken */
    uint32_t zeroed;  /* the pages and x's tiles zeroed (the first update) */
    uint32_t number;  /* updates finished */
    uint32_t times[RF_UT_WORDS];
} update_t;

static update_t g_update __attribute__((section(".data"))) = {0, 0, 0, 0, 0, {1}};

/* One step's view of the regions and words, for the parts on the four cores. */
typedef struct {
    const volatile uint32_t* header;
    const uint8_t* elements;
    const float* shadow;
    uint8_t* work;
    uint32_t* pages;
    uint32_t n, tiles, rows, weights, pages_count, side;
    uint32_t more; /* step 2: another pass follows (x made) */
    float texel, bias, sun[3], irradiance[3], camera[16], sky[RF_SKY_TERMS * 3];
    float scale;
    float top[4];    /* each core's largest value */
    uint32_t bad[4]; /* each core's columns, starts or places out of range */
} job_t;

static job_t g_job __attribute__((section(".data"))) = {0};

static uint32_t elapsed(uint64_t* mark) {
    const uint64_t now = kmx_time();
    const uint32_t ticks = (uint32_t)(now - *mark);
    *mark = now;
    return ticks;
}

static float word_float(uint32_t word) {
    union {
        uint32_t u;
        float f;
    } both;

    both.u = word;
    return both.f;
}

static uint32_t float_word(float value) {
    union {
        uint32_t u;
        float f;
    } both;

    both.f = value;
    return both.u;
}

/* bfloat16 bits of a float32: nearest, ties to even (komorebi.formats.bf16_bits) */
static uint16_t bf16(float value) {
    const uint32_t bits = float_word(value);
    return (uint16_t)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16);
}

/* A tile's element (row, column) of 32 x 32 in faces of 16 x 16 */
static uint32_t tile_at(uint32_t row, uint32_t column) {
    return (
        (row >> 4) * 512u + (column >> 4) * 256u + (row & 15u) * 16u + (column & 15u)
    );
}

static double square_root(double value) {
    double root;
    __asm__("fsqrt.d %0, %1" : "=f"(root) : "f"(value));
    return root;
}

/* round to nearest, ties to even (np.rint) */
static int64_t nearest(double value) {
    int64_t out;
    __asm__("fcvt.l.d %0, %1, rne" : "=r"(out) : "f"(value));
    return out;
}

/* np.floor of a float32 within +-2^31 as float32 */
static float floor_of(float value) {
    const int64_t whole = (int64_t)value;
    const float down = (float)whole;
    return down > value ? down - 1.0f : down;
}

/* 1 lit, 0 shadowed at texel (ix, iy) for 1/z `w` (reference/lights.py sun_shadow's
 * lit_at: outside the map lit) */
static float lit_at(const job_t* job, int64_t ix, int64_t iy, float w) {
    const int64_t side = job->side;
    if (ix < 0 || ix >= side || iy < 0 || iy >= side) {
        return 1.0f;
    }
    const uint32_t x = (uint32_t)ix, y = (uint32_t)iy;
    const uint32_t page = (y >> 5) * (job->side >> 5) + (x >> 5);
    const float stored = (
        job->shadow[page * RF_PAGE_TEXELS + (y & 31u) * 32u + (x & 31u)]
    );
    return w < stored - job->bias ? 0.0f : 1.0f;
}

/* The sun's light at element i: sun_shadow (pcf) x max(n . s, 0). */
static float sun_light(const job_t* job, const float* position, const float* normal) {
    const float* camera = job->camera;
    const float q0 = position[0] + normal[0] * job->texel;
    const float q1 = position[1] + normal[1] * job->texel;
    const float q2 = position[2] + normal[2] * job->texel;
    const float d0 = q0 - camera[0], d1 = q1 - camera[1], d2 = q2 - camera[2];
    const float xc = (d0 * camera[3] + d1 * camera[4]) + d2 * camera[5];
    const float yc = (d0 * camera[6] + d1 * camera[7]) + d2 * camera[8];
    const float zc = (d0 * camera[9] + d1 * camera[10]) + d2 * camera[11];
    const float inv = 1.0f / zc;
    const float sx = camera[13] + camera[12] * (xc * inv);
    const float sy = camera[14] - camera[12] * (yc * inv);
    const float gx = sx - 0.5f, gy = sy - 0.5f;
    float shadow = 1.0f;
    /* far outside (or not a number): every texel outside the map, lit */
    if (gx > -2.0e9f && gx < 2.0e9f && gy > -2.0e9f && gy < 2.0e9f) {
        const float x0 = floor_of(gx), y0 = floor_of(gy);
        const float fx = gx - x0, fy = gy - y0;
        const int64_t ix = (int64_t)x0, iy = (int64_t)y0;
        const float top = (
            lit_at(job, ix, iy, inv) * (1.0f - fx) + lit_at(job, ix + 1, iy, inv) * fx
        );
        const float bottom = (
            lit_at(job, ix, iy + 1, inv) * (1.0f - fx) +
            lit_at(job, ix + 1, iy + 1, inv) * fx
        );
        shadow = top * (1.0f - fy) + bottom * fy;
    }
    const float facing = (
        (normal[0] * job->sun[0] + normal[1] * job->sun[1]) + normal[2] * job->sun[2]
    );
    return shadow * (facing > 0.0f ? facing : 0.0f);
}

static const float* element_array(const job_t* job, uint32_t word) {
    return (const float*)(job->elements + job->header[word]);
}

static float* work_array(const job_t* job, uint32_t word) {
    return (float*)(job->work + job->header[word]);
}

/* Step 1's part: sky, sun, L0 of a quarter of the tiles; x of pass 1 (L0, bf16) into
 * x's tiles; the total 0. */
static void sources_part(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    const float* position = element_array(job, RF_U_E_POSITION);
    const float* normal = element_array(job, RF_U_E_NORMAL);
    const float* base = element_array(job, RF_U_E_BASE);
    const float* albedo = element_array(job, RF_U_E_ALBEDO);
    const float* transfer = element_array(job, RF_U_E_TRANSFER);
    float* sky = work_array(job, RF_U_W_SKY);
    float* l0 = work_array(job, RF_U_W_L0);
    float* total = work_array(job, RF_U_W_TOTAL);
    uint16_t* x = (uint16_t*)(job->work + job->header[RF_U_W_X]);
    const uint32_t quarter = (uint32_t)core;
    const uint32_t first = job->tiles * quarter / 4 * TILE;
    const uint32_t end = job->tiles * (quarter + 1) / 4 * TILE;
    for (uint32_t i = first; i < end; i++) {
        const float sun = sun_light(job, position + 3 * i, normal + 3 * i);
        const float* row = transfer + RF_SKY_TERMS * i;
        uint16_t* tile = x + (i / TILE) * (X_PAGE_BYTES / 2);
        double sums[3] = {0.0, 0.0, 0.0}; /* each colour's in term order */
        for (uint32_t k = 0; k < RF_SKY_TERMS; k++) {
            sums[0] += (double)(row[k] * job->sky[3 * k]);
            sums[1] += (double)(row[k] * job->sky[3 * k + 1]);
            sums[2] += (double)(row[k] * job->sky[3 * k + 2]);
        }
        for (uint32_t c = 0; c < 3; c++) {
            const float s = (float)sums[c];
            const float clear = s > 0.0f ? s : 0.0f;
            const float light = (
                base[3 * i + c] + albedo[3 * i + c] * (sun * job->irradiance[c] + clear)
            );
            sky[3 * i + c] = clear;
            l0[3 * i + c] = light;
            total[3 * i + c] = 0.0f;
            tile[tile_at(c, i % TILE)] = bf16(light);
        }
    }
}

/* Step 2's part: E of the pass from y's tiles added to the total; with another pass,
 * its x = albedo / pi x E (bf16) into x's tiles. */
static void pass_part(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    const float* albedo = element_array(job, RF_U_E_ALBEDO);
    float* total = work_array(job, RF_U_W_TOTAL);
    const float* y = (const float*)(job->work + job->header[RF_U_W_Y]);
    uint16_t* x = (uint16_t*)(job->work + job->header[RF_U_W_X]);
    const uint32_t quarter = (uint32_t)core;
    const uint32_t first = job->tiles * quarter / 4 * TILE;
    const uint32_t end = job->tiles * (quarter + 1) / 4 * TILE;
    for (uint32_t i = first; i < end; i++) {
        const float* tile = y + (i / TILE) * (Y_PAGE_BYTES / 4);
        uint16_t* out = x + (i / TILE) * (X_PAGE_BYTES / 2);
        for (uint32_t c = 0; c < 3; c++) {
            const float e = tile[tile_at(c, i % TILE)];
            total[3 * i + c] = total[3 * i + c] + e;
            if (job->more) {
                out[tile_at(c, i % TILE)] = bf16(albedo[3 * i + c] * e);
            }
        }
    }
}

/* Step 3's first part: e = total + sky, in the total's place. */
static void source_part(int core, void* arg) {
    const job_t* job = (const job_t*)arg;
    const float* sky = work_array(job, RF_U_W_SKY);
    float* total = work_array(job, RF_U_W_TOTAL);
    const uint32_t quarter = (uint32_t)core;
    const uint32_t first = 3 * (job->n * quarter / 4);
    const uint32_t end = 3 * (job->n * (quarter + 1) / 4);
    for (uint32_t i = first; i < end; i++) {
        total[i] = total[i] + sky[i];
    }
}

/* Step 3's rows: v[r, c] = the row's sum of w x e[column, c] in its order (float32
 * products, a double sum), and the core's largest value. */
static void rows_part(int core, void* arg) {
    job_t* job = (job_t*)arg;
    const float* e = work_array(job, RF_U_W_TOTAL);
    float* values = work_array(job, RF_U_W_VALUES);
    const uint32_t* starts = (
        (const uint32_t*)(job->elements + job->header[RF_U_E_STARTS])
    );
    const uint32_t* pairs = (
        (const uint32_t*)(job->elements + job->header[RF_U_E_PAIRS])
    );
    const uint32_t quarter = (uint32_t)core;
    const uint32_t first = job->rows * quarter / 4;
    const uint32_t end = job->rows * (quarter + 1) / 4;
    float top = 0.0f;
    uint32_t bad = 0;
    for (uint32_t r = first; r < end; r++) {
        const uint32_t from = starts[r];
        const uint32_t to = starts[r + 1] < job->weights ? starts[r + 1] : job->weights;
        double sum0 = 0.0, sum1 = 0.0, sum2 = 0.0;
        for (uint32_t j = from; j < to; j++) {
            const uint32_t column = pairs[2 * j];
            const float w = word_float(pairs[2 * j + 1]);
            if (column >= job->n) {
                bad++;
                continue;
            }
            sum0 += (double)(w * e[3 * column]);
            sum1 += (double)(w * e[3 * column + 1]);
            sum2 += (double)(w * e[3 * column + 2]);
        }
        bad += from > to;
        const float v0 = (float)sum0, v1 = (float)sum1, v2 = (float)sum2;
        values[3 * r] = v0;
        values[3 * r + 1] = v1;
        values[3 * r + 2] = v2;
        top = v0 > top ? v0 : top;
        top = v1 > top ? v1 : top;
        top = v2 > top ? v2 : top;
    }
    job->top[quarter] = top;
    job->bad[quarter] = bad;
}

/* The pages' quarter zeroed (the first update): no voxel's texel is ever written in
 * the rest. */
static void zero_part(int core, void* arg) {
    job_t* job = (job_t*)arg;
    const uint32_t quarter = (uint32_t)core;
    const uint64_t words = (uint64_t)job->pages_count * (PAGE_BYTES / 8);
    uint64_t* pages = (uint64_t*)job->pages;
    for (uint64_t i = words * quarter / 4; i < words * (quarter + 1) / 4; i++) {
        pages[i] = 0;
    }
    uint64_t* x = (uint64_t*)(job->work + job->header[RF_U_W_X]);
    const uint64_t x_words = (uint64_t)job->tiles * (X_PAGE_BYTES / 8);
    for (uint64_t i = x_words * quarter / 4; i < x_words * (quarter + 1) / 4; i++) {
        x[i] = 0;
    }
}

/* Step 3's texels (indirect.pack_field): bit 31 where the row has data | each
 * channel's round(sqrt(v / scale) (2^bits - 1)) at its shift, into the row's place. */
static void pack_part(int core, void* arg) {
    job_t* job = (job_t*)arg;
    const float* values = work_array(job, RF_U_W_VALUES);
    const uint32_t* places = (
        (const uint32_t*)(job->elements + job->header[RF_U_E_PLACES])
    );
    const uint32_t quarter = (uint32_t)core;
    const uint32_t first = job->rows * quarter / 4;
    const uint32_t end = job->rows * (quarter + 1) / 4;
    const uint32_t texels = job->pages_count * RF_PAGE_TEXELS;
    const double scale = (double)job->scale;
    static const uint32_t most[3] = {(1u << 11) - 1u, (1u << 11) - 1u, (1u << 9) - 1u};
    static const uint32_t shift[3] = {20u, 9u, 0u};
    uint32_t bad = 0;
    for (uint32_t r = first; r < end; r++) {
        const uint32_t place = places[r];
        const uint32_t at = place & 0x7FFFFFFFu;
        if (at >= texels) {
            bad++;
            continue;
        }
        uint32_t texel = 0;
        if (place >> 31) {
            texel = 0x80000000u;
            for (uint32_t c = 0; c < 3; c++) {
                const double v = (double)values[3 * r + c];
                int64_t code = 0;
                if (v > 0.0) {
                    code = nearest(square_root(v / scale) * (double)most[c]);
                }
                code = code < (int64_t)most[c] ? code : (int64_t)most[c];
                texel |= (uint32_t)code << shift[c];
            }
        }
        job->pages[at] = texel;
    }
    job->bad[quarter] += bad;
}

/* The pass's apply E = M x on the cores: APPLY with the update region's words 0 to 10
 * (a0 to a9, the group: 0 none), submitted and waited for. 0, or the code to report. */
static uint32_t apply_pass(const volatile uint32_t* header) {
#ifndef RF_APPLY
    (void)header;
    return RF_NO_APPLY;
#else
    const uint32_t group = header[RF_U_APPLY_GROUP];
    if (group == 0) {
        return RF_NO_APPLY;
    }
    km_command_t command;
    kmx_command(&command, 0, RF_HANDLER_APPLY, group & 0xFFFFu, group >> 16);
    command.a0 = header[RF_U_APPLY_A0 + 0];
    command.a1 = header[RF_U_APPLY_A0 + 1];
    command.a2 = header[RF_U_APPLY_A0 + 2];
    command.a3 = header[RF_U_APPLY_A0 + 3];
    command.a4 = header[RF_U_APPLY_A0 + 4];
    command.a5 = header[RF_U_APPLY_A0 + 5];
    command.a6 = header[RF_U_APPLY_A0 + 6];
    command.a7 = header[RF_U_APPLY_A0 + 7];
    command.a8 = header[RF_U_APPLY_A0 + 8];
    command.a9 = header[RF_U_APPLY_A0 + 9];
    const uint64_t until = kmx_time() + APPLY_TICKS;
    uint32_t handle;
    int error = kmx_submit(&command, &handle);
    while (error == KMX_FULL && kmx_time() < until && !kmx_stop_requested()) {
        error = kmx_submit(&command, &handle);
    }
    if (error != KMX_OK) {
        return RF_RING;
    }
    kmx_event event;
    if (kmx_wait(handle, until, &event) != KMX_OK) {
        return RF_RING;
    }
    return event.kind == KM_EVENT_DONE ? 0u : RF_COMMAND;
#endif
}

/* The regions and words of the update into `job`: 0, or RF_BAD_UPDATE when a region
 * does not hold what the words say. */
static uint32_t take_job(job_t* job) {
    uint64_t bytes[RF_UPDATE_REGIONS];
    void* regions[RF_UPDATE_REGIONS];
    for (uint32_t i = RF_R_UPDATE; i < RF_UPDATE_REGIONS; i++) {
        regions[i] = kmx_region((int)i, &bytes[i]);
        if (regions[i] == 0) {
            return RF_BAD_UPDATE;
        }
    }
    if (bytes[RF_R_UPDATE] < RF_UPDATE_BYTES) {
        return RF_BAD_UPDATE;
    }
    kmx_invalidate_at(RF_R_UPDATE, 0, 4u * RF_U_WORDS);
    const volatile uint32_t* header = (const volatile uint32_t*)regions[RF_R_UPDATE];
    job->header = header;
    job->elements = (const uint8_t*)regions[RF_R_ELEMENTS];
    job->shadow = (const float*)regions[RF_R_SHADOW];
    job->work = (uint8_t*)regions[RF_R_WORK_U];
    job->pages = (uint32_t*)regions[RF_R_FIELDPAGES];
    job->n = header[RF_U_N];
    job->tiles = header[RF_U_TILES];
    job->rows = header[RF_U_ROWS];
    job->weights = header[RF_U_WEIGHTS];
    job->pages_count = header[RF_U_PAGES];
    job->side = header[RF_U_SIDE];
    job->texel = word_float(header[RF_U_TEXEL]);
    job->bias = word_float(header[RF_U_BIAS]);
    for (uint32_t i = 0; i < 3; i++) {
        job->sun[i] = word_float(header[RF_U_SUN + i]);
        job->irradiance[i] = word_float(header[RF_U_IRRADIANCE + i]);
    }
    for (uint32_t i = 0; i < 16; i++) {
        job->camera[i] = word_float(header[RF_U_CAMERA + i]);
    }
    for (uint32_t i = 0; i < RF_SKY_TERMS * 3; i++) {
        job->sky[i] = word_float(header[RF_U_SKY + i]);
    }
    const uint64_t n = job->n, rows = job->rows;
    /* each array's end: (offset word, bytes) of the elements and work regions */
    const uint64_t elements_end[8][2] = {
        {header[RF_U_E_POSITION], 12 * n},
        {header[RF_U_E_NORMAL], 12 * n},
        {header[RF_U_E_BASE], 12 * n},
        {header[RF_U_E_ALBEDO], 12 * n},
        {header[RF_U_E_TRANSFER], 4 * RF_SKY_TERMS * n},
        {header[RF_U_E_STARTS], 4 * (rows + 1)},
        {header[RF_U_E_PAIRS], 8 * (uint64_t)job->weights},
        {header[RF_U_E_PLACES], 4 * rows},
    };
    const uint64_t work_end[6][2] = {
        {header[RF_U_W_SKY], 12 * n},
        {header[RF_U_W_L0], 12 * n},
        {header[RF_U_W_TOTAL], 12 * n},
        {header[RF_U_W_X], (uint64_t)X_PAGE_BYTES * job->tiles},
        {header[RF_U_W_Y], (uint64_t)Y_PAGE_BYTES * job->tiles},
        {header[RF_U_W_VALUES], 12 * rows},
    };
    uint32_t bad = (
        n == 0 || n != (uint64_t)TILE * job->tiles || rows == 0 ||
        header[RF_U_BOUNCES] > RF_U_BOUNCES_MAX ||
        header[RF_U_X_PAGE] != X_PAGE_BYTES || header[RF_U_Y_PAGE] != Y_PAGE_BYTES ||
        header[RF_U_CORES] == 0 || job->side == 0 || job->side % 32u ||
        (uint64_t)job->side * job->side !=
            (uint64_t)RF_PAGE_TEXELS * header[RF_U_SHADOW_PAGES] ||
        bytes[RF_R_SHADOW] < (uint64_t)PAGE_BYTES * header[RF_U_SHADOW_PAGES] ||
        bytes[RF_R_FIELDPAGES] < (uint64_t)PAGE_BYTES * job->pages_count ||
        job->pages_count == 0 || job->pages_count >= (1u << 21)
    );
    for (uint32_t i = 0; i < 8; i++) {
        bad |= (
            elements_end[i][0] % 4 != 0 ||
            elements_end[i][0] + elements_end[i][1] > bytes[RF_R_ELEMENTS]
        );
    }
    for (uint32_t i = 0; i < 6; i++) {
        bad |= (
            work_end[i][0] % 64 != 0 ||
            work_end[i][0] + work_end[i][1] > bytes[RF_R_WORK_U]
        );
    }
    return bad ? RF_BAD_UPDATE : 0u;
}

/* Step 1: the shadow map's pages fetched, the sources, x of pass 1 stored. */
static uint32_t sources(job_t* job, uint64_t* mark) {
    const volatile uint32_t* header = job->header;
    const uint32_t cores = header[RF_U_CORES];
    update_t* update = &g_update;
    kmx_event event;
    if (!update->zeroed) {
        /* lines an earlier image left of the elements' arrays */
        kmx_invalidate(RF_R_ELEMENTS);
    }
    const kmx_dram map = {header[RF_U_LIGHT], KM_COPY_INTERLEAVED, PAGE_BYTES, 0u};
    if (kmx_fetch(
        RF_R_SHADOW,
        0,
        &map,
        (uint64_t)PAGE_BYTES * header[RF_U_SHADOW_PAGES],
        0u,
        cores,
        kmx_time() + COPY_TICKS,
        &event
    ) != KMX_OK) {
        return RF_COPY;
    }
    update->times[RF_UT_FETCH] += elapsed(mark);
    if (!update->zeroed) {
        if (kmx_parallel(zero_part, job) != KMX_OK) {
            return RF_PARALLEL;
        }
        update->zeroed = 1;
        update->times[RF_UT_ZERO] += elapsed(mark);
    }
    if (kmx_parallel(sources_part, job) != KMX_OK) {
        return RF_PARALLEL;
    }
    /* sky and L0 for the host's read after the stop */
    kmx_flush_at(RF_R_WORK_U, header[RF_U_W_SKY], 12ull * job->n);
    kmx_flush_at(RF_R_WORK_U, header[RF_U_W_L0], 12ull * job->n);
    update->times[RF_UT_SOURCES] += elapsed(mark);
    update->state = 1;
    update->passes = 0;
    update->bounces = header[RF_U_BOUNCES];
    if (update->bounces == 0) {
        return 0;
    }
    const kmx_dram x = {header[RF_U_X], KM_COPY_INTERLEAVED, X_PAGE_BYTES, 0u};
    if (kmx_store(
        RF_R_WORK_U,
        header[RF_U_W_X],
        &x,
        (uint64_t)X_PAGE_BYTES * job->tiles,
        0u,
        cores,
        kmx_time() + COPY_TICKS,
        &event
    ) != KMX_OK) {
        return RF_COPY;
    }
    update->times[RF_UT_PASS] += elapsed(mark);
    return 0;
}

/* Step 2: y fetched, added to the total; while passes remain, the next x stored. */
static uint32_t take_pass(job_t* job, uint64_t* mark) {
    const volatile uint32_t* header = job->header;
    const uint32_t cores = header[RF_U_CORES];
    update_t* update = &g_update;
    if (update->state != 1 || update->passes >= update->bounces) {
        return RF_BAD_UPDATE;
    }
    uint32_t* times = update->times + RF_UT_PASS + RF_UT_PASS_WORDS * update->passes;
    kmx_event event;
    const kmx_dram y = {header[RF_U_Y], KM_COPY_INTERLEAVED, Y_PAGE_BYTES, 0u};
    if (kmx_fetch(
        RF_R_WORK_U,
        header[RF_U_W_Y],
        &y,
        (uint64_t)Y_PAGE_BYTES * job->tiles,
        0u,
        cores,
        kmx_time() + COPY_TICKS,
        &event
    ) != KMX_OK) {
        return RF_COPY;
    }
    times[2] += elapsed(mark);
    job->more = update->passes + 1 < update->bounces;
    if (kmx_parallel(pass_part, job) != KMX_OK) {
        return RF_PARALLEL;
    }
    times[3] += elapsed(mark);
    update->passes++;
    if (!job->more) {
        return 0;
    }
    const kmx_dram x = {header[RF_U_X], KM_COPY_INTERLEAVED, X_PAGE_BYTES, 0u};
    if (kmx_store(
        RF_R_WORK_U,
        header[RF_U_W_X],
        &x,
        (uint64_t)X_PAGE_BYTES * job->tiles,
        0u,
        cores,
        kmx_time() + COPY_TICKS,
        &event
    ) != KMX_OK) {
        return RF_COPY;
    }
    times[RF_UT_PASS_WORDS] += elapsed(mark); /* the next pass's store */
    return 0;
}

/* Step 3: e, the rows, the scale, the texels, the pages stored, the table words. */
static uint32_t finish_update(job_t* job, uint64_t* mark) {
    const volatile uint32_t* header = job->header;
    const uint32_t cores = header[RF_U_CORES];
    update_t* update = &g_update;
    if (update->state != 1 || update->passes != update->bounces) {
        return RF_BAD_UPDATE;
    }
    if (
        kmx_parallel(source_part, job) != KMX_OK ||
        kmx_parallel(rows_part, job) != KMX_OK
    ) {
        return RF_PARALLEL;
    }
    float top = 0.0f;
    uint32_t bad = 0;
    for (uint32_t i = 0; i < 4; i++) {
        top = job->top[i] > top ? job->top[i] : top;
        bad += job->bad[i];
    }
    job->scale = top > 0.0f ? top : 1.0f;
    update->times[RF_UT_ROWS] += elapsed(mark);
    if (kmx_parallel(pack_part, job) != KMX_OK) {
        return RF_PARALLEL;
    }
    for (uint32_t i = 0; i < 4; i++) {
        bad += job->bad[i];
    }
    update->times[RF_UT_PACK] += elapsed(mark);
    if (bad) {
        return RF_BAD_UPDATE;
    }
    kmx_event event;
    const kmx_dram to = {
        header[RF_U_LIGHT], KM_COPY_INTERLEAVED, PAGE_BYTES, header[RF_U_FIELD_FIRST]
    };
    if (kmx_store(
        RF_R_FIELDPAGES,
        0,
        &to,
        (uint64_t)PAGE_BYTES * job->pages_count,
        0u,
        cores,
        kmx_time() + COPY_TICKS,
        &event
    ) != KMX_OK) {
        return RF_COPY;
    }
    update->times[RF_UT_STORE] += elapsed(mark);
    /* LIGHT's unit a channel, (1 / max code)^2 x scale, and the scale
     * (light.field_words) */
    const uint32_t at = RF_SUN_TABLE_AT + 4u * RF_LT_FIELD_UNIT;
    const uint32_t line = at & ~63u;
    kmx_invalidate_at(RF_R_SUN, line, 64);
    uint64_t sun_bytes;
    volatile uint32_t* table = (volatile uint32_t*)kmx_region(RF_R_SUN, &sun_bytes);
    static const uint32_t most[3] = {(1u << 11) - 1u, (1u << 11) - 1u, (1u << 9) - 1u};
    for (uint32_t c = 0; c < 3; c++) {
        const float unit = 1.0f / (float)most[c];
        table[at / 4 + c] = float_word((unit * unit) * job->scale);
    }
    table[(RF_SUN_TABLE_AT + 4u * RF_LT_FIELD_SCALE) / 4] = float_word(job->scale);
    kmx_flush_at(RF_R_SUN, line, 64);
    update->times[RF_UT_TABLE] += elapsed(mark);
    update->times[RF_UT_SCALE] = float_word(job->scale);
    update->state = 0;
    update->number++;
    return 0;
}

uint32_t rf_update(const km_command_t* command) {
    const uint64_t started = kmx_time();
    uint64_t mark = started;
    update_t* update = &g_update;
    job_t* job = &g_job;
    const uint32_t step = command->a0;
    const uint32_t number = update->number; /* the update's row */
    uint32_t code = take_job(job);
    if (code == 0 && (step == RF_STEP_ALL || step == RF_STEP_SOURCES)) {
        for (uint32_t i = 0; i < RF_UT_WORDS; i++) {
            update->times[i] = 0;
        }
    }
    if (code == 0) {
        switch (step) {
            case RF_STEP_ALL:
                code = sources(job, &mark);
                while (code == 0 && update->passes < update->bounces) {
                    code = apply_pass(job->header);
                    update->times
                        [
                            RF_UT_PASS + RF_UT_PASS_WORDS * update->passes + 1
                        ] += (elapsed(&mark));
                    if (code == 0) {
                        code = take_pass(job, &mark);
                    }
                }
                if (code == 0) {
                    code = finish_update(job, &mark);
                }
                break;
            case RF_STEP_SOURCES:
                code = sources(job, &mark);
                break;
            case RF_STEP_PASS:
                code = take_pass(job, &mark);
                break;
            case RF_STEP_FINISH:
                code = finish_update(job, &mark);
                break;
            default: code = RF_BAD_UPDATE;
        }
    }
    if (code) {
        update->state = 0;
    }
    update->times[RF_UT_TOTAL] += (uint32_t)(kmx_time() - started);
    update->times[RF_UT_NUMBER] = number;
    update->times[RF_UT_CODE] = code;
    update->times[RF_UT_STEP] = step;
    update->times[RF_UT_PASSES] = update->passes;
    uint64_t bytes;
    volatile uint32_t* header = (volatile uint32_t*)kmx_region(RF_R_UPDATE, &bytes);
    if (header != 0 && bytes >= RF_UPDATE_BYTES) {
        const uint32_t at = RF_U_TIMES + (number % RF_U_TIMES_ROWS) * RF_UT_WORDS;
        for (uint32_t i = 0; i < RF_UT_WORDS; i++) {
            header[at + i] = update->times[i];
        }
        kmx_flush_at(RF_R_UPDATE, 4u * at, 4u * RF_UT_WORDS);
    }
    return code;
}
