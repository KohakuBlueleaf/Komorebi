/* The graph executor: check, start, step, complete. Freestanding: no library calls, no
   floating point, no static data; every loop is capped by a limit of km_graph.h. */
#include "km_graph_engine.h"

#define FAULT(reason, node) ((uint32_t)(reason) | (uint32_t)(node) << 8)
#define REGISTER(r) ((r) & (KM_STREAM_REGISTERS - 1u))
#define RING(i) ((i) & (KM_GRAPH_MAX_NODES - 1u))

/* Register fields a CPU micro-op reads or writes (bit 0 a, 1 b, 2 c), bit 3 for a
   jump; 0xFF for an opcode a CPU slice may not hold. */
static uint32_t cpu_op_use(uint32_t op) {
    switch (op) {
        case KM_OP_SET:
        case KM_OP_RAND:
            return 1u;
        case KM_OP_MOV:
        case KM_OP_ADDI:
        case KM_OP_SHR:
            return 3u;
        case KM_OP_ADD:
        case KM_OP_SUB:
        case KM_OP_MUL:
        case KM_OP_AND:
        case KM_OP_MOD:
            return 7u;
        case KM_OP_JMP:
            return 8u;
        case KM_OP_JNZ:
            return 9u;
        case KM_OP_JLT:
        case KM_OP_JEQ:
            return 11u;
        default:
            return 0xFFu;
    }
}

static uint32_t line_up(uint32_t bytes) {
    return (bytes + KM_STREAM_LINE - 1u) & ~(KM_STREAM_LINE - 1u);
}

static uint32_t bad_register(uint32_t r) {
    return r != KM_GRAPH_NONE && r >= KM_STREAM_REGISTERS;
}

static uint32_t bad_template(const km_template_t* t) {
    uint32_t handler = t->rec.hdr & 0xFFu;
    uint32_t flags = t->rec.hdr >> 8 & 0xFFu;
    if (handler < 1u || handler > KM_HANDLER_USER_MAX) {
        return 1u;
    }
    if (flags & ~(uint32_t)(KM_FLAG_EXCLUSIVE | KM_FLAG_RESULT | KM_FLAG_QUEUE)) {
        return 1u;
    }
    if (t->patch_count > KM_TEMPLATE_PATCHES) {
        return 1u;
    }
    for (uint32_t k = 0; k < t->patch_count; k++) {
        if (t->arg[k] >= KM_TEMPLATE_ARGS || t->registers[k] >= KM_STREAM_REGISTERS) {
            return 1u;
        }
    }
    return bad_register(t->first_register) || bad_register(t->count_register);
}

/* The micro-ops of one CPU slice [first, end): opcode, registers, jump targets. */
static uint32_t check_slice(const km_uop_t* ops, uint32_t first, uint32_t end) {
    for (uint32_t pc = first; pc < end; pc++) {
        const km_uop_t* o = &ops[pc];
        uint32_t use = cpu_op_use(o->op);
        if (use == 0xFFu) {
            return KM_STREAM_ERROR_OPCODE;
        }
        if (
            ((use & 1u) && o->a >= KM_STREAM_REGISTERS) ||
            ((use & 2u) && o->b >= KM_STREAM_REGISTERS) ||
            ((use & 4u) && o->c >= KM_STREAM_REGISTERS)
        ) {
            return KM_STREAM_ERROR_REGISTER;
        }
        if ((use & 8u) && (o->immediate < first || o->immediate > end)) {
            return KM_STREAM_ERROR_JUMP;
        }
    }
    return 0;
}

/* One node's own fields, its template or slice; `cpu_next` is where its slice starts.
 */
static uint32_t check_node(
    const km_graph_t* h,
    const km_node_t* n,
    const km_uop_t* ops,
    const km_template_t* template,
    uint32_t cpu_next
) {
    if (n->kind < KM_NODE_COMMAND || n->kind > KM_NODE_JOIN || n->flags != 0) {
        return KM_STREAM_ERROR_NODE;
    }
    if (n->predecessor_count > KM_GRAPH_MAX_PREDECESSORS) {
        return KM_STREAM_ERROR_EDGE;
    }
    if (bad_register(n->skip_register) || bad_register(n->result_register)) {
        return KM_STREAM_ERROR_REGISTER;
    }
    if (n->kind == KM_NODE_COMMAND) {
        if (n->arg >= h->command_count || bad_template(&template[n->arg])) {
            return KM_STREAM_ERROR_TEMPLATE;
        }
        uint32_t flags = template[n->arg].rec.hdr >> 8 & 0xFFu;
        if (
            n->op_count != 0 ||
            (n->result_register != KM_GRAPH_NONE && !(flags & KM_FLAG_RESULT))
        ) {
            return KM_STREAM_ERROR_NODE;
        }
        return 0;
    }
    if (n->result_register != KM_GRAPH_NONE) {
        return KM_STREAM_ERROR_NODE;
    }
    if (n->kind == KM_NODE_JOIN) {
        return n->arg != 0 || n->op_count != 0 ? KM_STREAM_ERROR_NODE : 0;
    }
    if (
        n->arg != cpu_next || n->op_count == 0 || n->op_count > h->op_count - cpu_next
    ) {
        return KM_STREAM_ERROR_NODE;
    }
    return check_slice(ops, n->arg, n->arg + (uint32_t)n->op_count);
}

uint32_t kmg_check(const void* upload, uint32_t bytes) {
    const km_graph_t* h = (const km_graph_t*)upload;
    const uint32_t none = KM_GRAPH_NO_NODE;
    if (bytes < sizeof(km_graph_t) || bytes % KM_STREAM_LINE != 0) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    if (h->magic != KM_GRAPH_MAGIC || h->version != KM_GRAPH_VERSION) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    if (
        h->node_count < 1u || h->node_count > KM_GRAPH_MAX_NODES ||
        h->edge_count > KM_GRAPH_MAX_EDGES || h->op_count > KM_GRAPH_MAX_OPS ||
        h->command_count > KM_GRAPH_MAX_TEMPLATES
    ) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    if (
        bad_register(h->again_register) || h->max_runs == 0 || h->step_limit == 0 ||
        h->step_limit > KM_GRAPH_MAX_STEPS || (h->flags & ~KM_GRAPH_KEEP) != 0
    ) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    uint32_t at_edges = sizeof(km_graph_t) + line_up(h->node_count * sizeof(km_node_t));
    uint32_t at_ops = at_edges + line_up(h->edge_count * 2u);
    uint32_t at_template = at_ops + line_up(h->op_count * sizeof(km_uop_t));
    if (bytes != at_template + line_up(h->command_count * sizeof(km_template_t))) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    const uint8_t* base = (const uint8_t*)upload;
    const km_node_t* nodes = (const km_node_t*)(base + sizeof(km_graph_t));
    const uint16_t* edges = (const uint16_t*)(base + at_edges);
    const km_uop_t* ops = (const km_uop_t*)(base + at_ops);
    const km_template_t* template = (const km_template_t*)(base + at_template);
    /* Predecessors seen per node; zeroed through a volatile pointer, as a plain loop
       may become a memset call. */
    uint8_t seen[KM_GRAPH_MAX_NODES];
    volatile uint8_t* zero = seen;
    for (uint32_t i = 0; i < KM_GRAPH_MAX_NODES; i++) {
        zero[i] = 0;
    }
    uint32_t cpu_next = 0, edge_next = 0;
    for (uint32_t i = 0; i < h->node_count; i++) {
        const km_node_t* n = &nodes[i];
        uint32_t why = check_node(h, n, ops, template, cpu_next);
        if (why != 0) {
            return FAULT(why, i);
        }
        if (n->kind == KM_NODE_CPU) {
            cpu_next += n->op_count;
        }
        if (
            n->successor_offset != edge_next ||
            n->successor_count > h->edge_count - edge_next
        ) {
            return FAULT(KM_STREAM_ERROR_EDGE, i);
        }
        uint32_t prev = i;
        for (uint32_t k = 0; k < n->successor_count; k++) {
            uint32_t s = edges[edge_next + k];
            if (
                s <= prev || s >= h->node_count || seen[s] >= KM_GRAPH_MAX_PREDECESSORS
            ) {
                return FAULT(KM_STREAM_ERROR_EDGE, i);
            }
            seen[s]++;
            prev = s;
        }
        edge_next += n->successor_count;
    }
    if (edge_next != h->edge_count) {
        return FAULT(KM_STREAM_ERROR_EDGE, none);
    }
    if (cpu_next != h->op_count) {
        return FAULT(KM_STREAM_ERROR_NODE, none);
    }
    for (uint32_t i = 0; i < h->node_count; i++) {
        if (seen[i] != nodes[i].predecessor_count) {
            return FAULT(KM_STREAM_ERROR_EDGE, i);
        }
    }
    for (uint32_t t = 0; t < h->command_count; t++) {
        if (bad_template(&template[t])) {
            return FAULT(KM_STREAM_ERROR_TEMPLATE, none);
        }
    }
    return 0;
}

uint32_t kmg_state_bytes(void) {
    return sizeof(kmg_state_t);
}

static void push(kmg_state_t* s, uint32_t node) {
    s->ready[RING(s->head + s->ready_n)] = (uint16_t)node;
    s->ready_n++;
    s->status[node] = KMG_READY;
}

static uint32_t pop(kmg_state_t* s) {
    uint32_t node = s->ready[RING(s->head)];
    s->head = RING(s->head + 1u);
    s->ready_n--;
    return node;
}

/* Every node waits for its predecessors again; the roots are ready, in node order. */
static void reset(kmg_state_t* s) {
    s->done = s->head = s->ready_n = 0;
    for (uint32_t i = 0; i < s->hdr->node_count; i++) {
        uint32_t predecessor_count = s->nodes[i].predecessor_count;
        s->left[i] = (uint8_t)predecessor_count;
        s->status[i] = predecessor_count == 0 ? KMG_READY : KMG_WAITING;
        if (predecessor_count == 0) {
            push(s, i);
        }
    }
}

static void fail(kmg_state_t* s, uint32_t reason, uint32_t node) {
    s->phase = KMG_ERROR;
    s->error_reason = reason;
    s->error_node = node;
}

static void finish(kmg_state_t* s, uint32_t node) {
    const km_node_t* n = &s->nodes[node];
    s->status[node] = KMG_DONE;
    s->done++;
    for (uint32_t k = 0; k < n->successor_count; k++) {
        uint32_t succ = s->edges[n->successor_offset + k];
        if (--s->left[succ] == 0) {
            push(s, succ);
        }
    }
}

uint32_t kmg_start(
    kmg_state_t* s,
    const void* upload,
    const uint32_t* bindings,
    uint32_t n_bindings,
    uint32_t core_count
) {
    if (n_bindings > KM_GRAPH_BINDINGS) {
        return KM_STREAM_ERROR_HEADER;
    }
    const km_graph_t* h = (const km_graph_t*)upload;
    const uint8_t* base = (const uint8_t*)upload;
    uint32_t at_edges = sizeof(km_graph_t) + line_up(h->node_count * sizeof(km_node_t));
    uint32_t at_ops = at_edges + line_up(h->edge_count * 2u);
    uint32_t at_template = at_ops + line_up(h->op_count * sizeof(km_uop_t));
    s->hdr = h;
    s->nodes = (const km_node_t*)(base + sizeof(km_graph_t));
    s->edges = (const uint16_t*)(base + at_edges);
    s->ops = (const km_uop_t*)(base + at_ops);
    s->template = (const km_template_t*)(base + at_template);
    for (uint32_t r = 0; r < KM_STREAM_REGISTERS; r++) {
        s->registers[r] = r < n_bindings ? bindings[r] : 0u;
    }
    s->runs = 1;
    s->phase = KMG_RUNNING;
    s->error_reason = s->error_node = 0;
    s->in_flight = 0;
    s->core_count = core_count;
    reset(s);
    return 0;
}

/* Runs one CPU slice to its end; returns the micro-ops run (the phase says if it
   failed). At most step_limit micro-ops. */
static uint32_t run_cpu(kmg_state_t* s, uint32_t node) {
    const km_node_t* n = &s->nodes[node];
    uint32_t* r = s->registers;
    uint32_t pc = n->arg, end = n->arg + (uint32_t)n->op_count, steps = 0;
    while (pc != end) {
        if (pc > end) {
            fail(s, KM_STREAM_ERROR_JUMP, node);
            return steps;
        }
        if (steps >= s->hdr->step_limit) {
            fail(s, KM_STREAM_ERROR_STEPS, node);
            return steps;
        }
        const km_uop_t* o = &s->ops[pc];
        uint32_t a = REGISTER(o->a);
        uint32_t b = REGISTER(o->b);
        uint32_t c = REGISTER(o->c);
        uint32_t next = pc + 1u;
        switch (o->op) {
            case KM_OP_SET:
                r[a] = o->immediate;
                break;
            case KM_OP_MOV:
                r[a] = r[b];
                break;
            case KM_OP_ADD:
                r[a] = r[b] + r[c];
                break;
            case KM_OP_ADDI:
                r[a] = r[b] + o->immediate;
                break;
            case KM_OP_SUB:
                r[a] = r[b] - r[c];
                break;
            case KM_OP_MUL:
                r[a] = r[b] * r[c];
                break;
            case KM_OP_AND:
                r[a] = r[b] & r[c];
                break;
            case KM_OP_SHR:
                r[a] = r[b] >> (o->immediate & 31u);
                break;
            case KM_OP_MOD:
                if (r[c] == 0) {
                    fail(s, KM_STREAM_ERROR_MOD_ZERO, node);
                    return steps;
                }
                r[a] = r[b] % r[c];
                break;
            case KM_OP_RAND: {
                uint32_t x = r[KM_STREAM_SEED_REGISTER];
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                r[KM_STREAM_SEED_REGISTER] = x;
                r[a] = x;
                break;
            }
            case KM_OP_JMP:
                next = o->immediate;
                break;
            case KM_OP_JLT:
                next = r[a] < r[b] ? o->immediate : next;
                break;
            case KM_OP_JEQ:
                next = r[a] == r[b] ? o->immediate : next;
                break;
            case KM_OP_JNZ:
                next = r[a] != 0 ? o->immediate : next;
                break;
            default:
                fail(s, KM_STREAM_ERROR_OPCODE, node);
                return steps;
        }
        steps++;
        pc = next;
    }
    return steps;
}

static void set_arg(km_command_t* rec, uint32_t i, uint32_t v) {
    switch (i) {
        case 0:
            rec->a0 = v;
            break;
        case 1:
            rec->a1 = v;
            break;
        case 2:
            rec->a2 = v;
            break;
        case 3:
            rec->a3 = v;
            break;
        case 4:
            rec->a4 = v;
            break;
        case 5:
            rec->a5 = v;
            break;
        case 6:
            rec->a6 = v;
            break;
        case 7:
            rec->a7 = v;
            break;
        case 8:
            rec->a8 = v;
            break;
        default:
            rec->a9 = v;
            break;
    }
}

/* Builds and submits the command of `node`: KMG_SUBMITTED, KMG_BUSY (put back at the
   tail) or KMG_ERROR (group outside the grid, a job that can never fit). */
static uint32_t submit_command(
    kmg_state_t* s, uint32_t node, kmg_submit_function submit, void* ctx
) {
    const km_node_t* n = &s->nodes[node];
    const km_template_t* t = &s->template[n->arg < s->hdr->command_count ? n->arg : 0u];
    const km_command_t* src = &t->rec;
    km_command_t rec;
    rec.sequence = rec.land1 = rec.land2 = rec.land3 = 0;
    rec.hdr = src->hdr & 0xFFFFu;
    rec.a0 = src->a0;
    rec.a1 = src->a1;
    rec.a2 = src->a2;
    rec.a3 = src->a3;
    rec.a4 = src->a4;
    rec.a5 = src->a5;
    rec.a6 = src->a6;
    rec.a7 = src->a7;
    rec.a8 = src->a8;
    rec.a9 = src->a9;
    for (uint32_t k = 0; k < t->patch_count && k < KM_TEMPLATE_PATCHES; k++) {
        set_arg(&rec, t->arg[k], s->registers[REGISTER(t->registers[k])]);
    }
    uint32_t first = src->group & 0xFFFFu, count = src->group >> 16;
    if (t->first_register != KM_GRAPH_NONE) {
        first = s->registers[REGISTER(t->first_register)];
    }
    if (t->count_register != KM_GRAPH_NONE) {
        count = s->registers[REGISTER(t->count_register)];
    }
    if (rec.hdr >> 8 & KM_FLAG_QUEUE) {
        first = 0;
        if (count == 0 || count > s->core_count) {
            fail(s, KM_STREAM_ERROR_NEVER_FITS, node);
            return KMG_ERROR;
        }
    } else if (count == 0 || first > s->core_count || count > s->core_count - first) {
        fail(s, KM_STREAM_ERROR_GROUP, node);
        return KMG_ERROR;
    }
    rec.group = first | count << 16;
    if (submit(ctx, node, &rec) != KMG_SUBMITTED) {
        push(s, node);
        return KMG_BUSY;
    }
    s->status[node] = KMG_FLIGHT;
    s->in_flight++;
    return KMG_SUBMITTED;
}

/* All nodes are done: go round again while `again` is set, else end. */
static uint32_t round_end(kmg_state_t* s) {
    uint32_t again = s->hdr->again_register;
    if (again == KM_GRAPH_NONE || s->registers[REGISTER(again)] == 0) {
        s->phase = KMG_END;
        return KMG_END;
    }
    if (s->runs >= s->hdr->max_runs) {
        fail(s, KM_STREAM_ERROR_RUNS, KM_GRAPH_NO_NODE);
        return KMG_ERROR;
    }
    s->runs++;
    reset(s);
    return KMG_ROUND;
}

uint32_t kmg_step(kmg_state_t* s, kmg_submit_function submit, void* ctx) {
    if (s->phase != KMG_RUNNING) {
        return s->phase;
    }
    uint32_t taken = 0, busy = 0, ops = 0;
    while (
        s->done < s->hdr->node_count && taken < KM_GRAPH_STEP && s->ready_n != 0 &&
        busy < s->ready_n
    ) {
        uint32_t node = pop(s);
        const km_node_t* n = &s->nodes[node];
        taken++;
        if (
            n->skip_register != KM_GRAPH_NONE &&
            s->registers[REGISTER(n->skip_register)] == 0
        ) {
            finish(s, node);
            busy = 0;
            continue;
        }
        if (n->kind == KM_NODE_COMMAND) {
            uint32_t got = submit_command(s, node, submit, ctx);
            if (got == KMG_ERROR) {
                return KMG_ERROR;
            }
            busy = got == KMG_BUSY ? busy + 1u : 0u;
            continue;
        }
        busy = 0;
        if (n->kind == KM_NODE_CPU) {
            ops += run_cpu(s, node);
            if (s->phase != KMG_RUNNING) {
                return KMG_ERROR;
            }
        }
        finish(s, node);
        if (ops >= KM_GRAPH_STEP_OPS) {
            break;
        }
    }
    return s->done == s->hdr->node_count ? round_end(s) : KMG_RUNNING;
}

uint32_t kmg_complete(kmg_state_t* s, uint32_t node, uint32_t result) {
    if (
        s->phase != KMG_RUNNING || node >= s->hdr->node_count ||
        s->status[node] != KMG_FLIGHT
    ) {
        return 1u;
    }
    uint32_t reg = s->nodes[node].result_register;
    if (reg != KM_GRAPH_NONE) {
        s->registers[REGISTER(reg)] = result;
    }
    s->in_flight--;
    finish(s, node);
    return 0;
}

uint32_t kmg_retry(kmg_state_t* s, uint32_t node) {
    if (
        s->phase != KMG_RUNNING || node >= s->hdr->node_count ||
        s->status[node] != KMG_FLIGHT
    ) {
        return 1u;
    }
    s->in_flight--;
    push(s, node);
    return 0;
}

void kmg_stop(kmg_state_t* s, uint32_t reason, uint32_t node) {
    if (s->phase == KMG_RUNNING) {
        fail(s, reason, node);
    }
}
