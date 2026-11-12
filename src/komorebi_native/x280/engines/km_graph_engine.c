/* The graph executor: check, start, step, complete. Freestanding: no library calls, no
   floating point, no static data; every loop is bounded by a limit of km_graph.h. */
#include "km_graph_engine.h"

#define FAULT(reason, node) ((uint32_t)(reason) | (uint32_t)(node) << 8)
#define REG(r) ((r) & (KM_STREAM_REGS - 1u))
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

static uint32_t bad_reg(uint32_t r) {
    return r != KM_GRAPH_NONE && r >= KM_STREAM_REGS;
}

static uint32_t bad_template(const km_template_t* t) {
    uint32_t seg = t->rec.hdr & 0xFFu;
    uint32_t flags = t->rec.hdr >> 8 & 0xFFu;
    if (seg < 1u || seg > KM_SEG_USER_MAX) {
        return 1u;
    }
    if (flags & ~(uint32_t)(KM_F_EXCL | KM_F_RESULT | KM_F_QUEUE)) {
        return 1u;
    }
    if (t->n_patch > KM_TPL_PATCHES) {
        return 1u;
    }
    for (uint32_t k = 0; k < t->n_patch; k++) {
        if (t->arg[k] >= KM_TPL_ARGS || t->reg[k] >= KM_STREAM_REGS) {
            return 1u;
        }
    }
    return bad_reg(t->first_reg) || bad_reg(t->count_reg);
}

/* The micro-ops of one CPU slice [first, end): opcode, registers, jump targets. */
static uint32_t check_slice(const km_uop_t* ops, uint32_t first, uint32_t end) {
    for (uint32_t pc = first; pc < end; pc++) {
        const km_uop_t* o = &ops[pc];
        uint32_t use = cpu_op_use(o->op);
        if (use == 0xFFu) {
            return KM_SERR_OPCODE;
        }
        if (((use & 1u) && o->a >= KM_STREAM_REGS) ||
            ((use & 2u) && o->b >= KM_STREAM_REGS) ||
            ((use & 4u) && o->c >= KM_STREAM_REGS)) {
            return KM_SERR_REGISTER;
        }
        if ((use & 8u) && (o->imm < first || o->imm > end)) {
            return KM_SERR_JUMP;
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
    const km_template_t* tpl,
    uint32_t cpu_next
) {
    if (n->kind < KM_NODE_RECORD || n->kind > KM_NODE_JOIN || n->flags != 0) {
        return KM_SERR_NODE;
    }
    if (n->n_pred > KM_GRAPH_MAX_PRED) {
        return KM_SERR_EDGE;
    }
    if (bad_reg(n->skip_reg) || bad_reg(n->result_reg)) {
        return KM_SERR_REGISTER;
    }
    if (n->kind == KM_NODE_RECORD) {
        if (n->arg >= h->n_records || bad_template(&tpl[n->arg])) {
            return KM_SERR_TEMPLATE;
        }
        uint32_t flags = tpl[n->arg].rec.hdr >> 8 & 0xFFu;
        if (n->n_ops != 0 ||
            (n->result_reg != KM_GRAPH_NONE && !(flags & KM_F_RESULT))) {
            return KM_SERR_NODE;
        }
        return 0;
    }
    if (n->result_reg != KM_GRAPH_NONE) {
        return KM_SERR_NODE;
    }
    if (n->kind == KM_NODE_JOIN) {
        return n->arg != 0 || n->n_ops != 0 ? KM_SERR_NODE : 0;
    }
    if (n->arg != cpu_next || n->n_ops == 0 || n->n_ops > h->n_ops - cpu_next) {
        return KM_SERR_NODE;
    }
    return check_slice(ops, n->arg, n->arg + (uint32_t)n->n_ops);
}

uint32_t kmg_check(const void* upload, uint32_t bytes) {
    const km_graph_t* h = (const km_graph_t*)upload;
    const uint32_t none = KM_GRAPH_NO_NODE;
    if (bytes < sizeof(km_graph_t) || bytes % KM_STREAM_LINE != 0) {
        return FAULT(KM_SERR_HEADER, none);
    }
    if (h->magic != KM_GRAPH_MAGIC || h->version != KM_GRAPH_VERSION) {
        return FAULT(KM_SERR_HEADER, none);
    }
    if (h->n_nodes < 1u || h->n_nodes > KM_GRAPH_MAX_NODES ||
        h->n_edges > KM_GRAPH_MAX_EDGES || h->n_ops > KM_GRAPH_MAX_OPS ||
        h->n_records > KM_GRAPH_MAX_TEMPLATES) {
        return FAULT(KM_SERR_HEADER, none);
    }
    if (bad_reg(h->again_reg) || h->max_runs == 0 || h->step_limit == 0 ||
        h->step_limit > KM_GRAPH_MAX_STEPS || (h->flags & ~KM_GRAPH_KEEP) != 0) {
        return FAULT(KM_SERR_HEADER, none);
    }
    uint32_t at_edges = sizeof(km_graph_t) + line_up(h->n_nodes * sizeof(km_node_t));
    uint32_t at_ops = at_edges + line_up(h->n_edges * 2u);
    uint32_t at_tpl = at_ops + line_up(h->n_ops * sizeof(km_uop_t));
    if (bytes != at_tpl + line_up(h->n_records * sizeof(km_template_t))) {
        return FAULT(KM_SERR_HEADER, none);
    }
    const uint8_t* base = (const uint8_t*)upload;
    const km_node_t* nodes = (const km_node_t*)(base + sizeof(km_graph_t));
    const uint16_t* edges = (const uint16_t*)(base + at_edges);
    const km_uop_t* ops = (const km_uop_t*)(base + at_ops);
    const km_template_t* tpl = (const km_template_t*)(base + at_tpl);
    /* Predecessors seen per node; zeroed through a volatile pointer, as a plain loop
       may become a memset call. */
    uint8_t seen[KM_GRAPH_MAX_NODES];
    volatile uint8_t* zero = seen;
    for (uint32_t i = 0; i < KM_GRAPH_MAX_NODES; i++) {
        zero[i] = 0;
    }
    uint32_t cpu_next = 0, edge_next = 0;
    for (uint32_t i = 0; i < h->n_nodes; i++) {
        const km_node_t* n = &nodes[i];
        uint32_t why = check_node(h, n, ops, tpl, cpu_next);
        if (why != 0) {
            return FAULT(why, i);
        }
        if (n->kind == KM_NODE_CPU) {
            cpu_next += n->n_ops;
        }
        if (n->succ_off != edge_next || n->succ_n > h->n_edges - edge_next) {
            return FAULT(KM_SERR_EDGE, i);
        }
        uint32_t prev = i;
        for (uint32_t k = 0; k < n->succ_n; k++) {
            uint32_t s = edges[edge_next + k];
            if (s <= prev || s >= h->n_nodes || seen[s] >= KM_GRAPH_MAX_PRED) {
                return FAULT(KM_SERR_EDGE, i);
            }
            seen[s]++;
            prev = s;
        }
        edge_next += n->succ_n;
    }
    if (edge_next != h->n_edges) {
        return FAULT(KM_SERR_EDGE, none);
    }
    if (cpu_next != h->n_ops) {
        return FAULT(KM_SERR_NODE, none);
    }
    for (uint32_t i = 0; i < h->n_nodes; i++) {
        if (seen[i] != nodes[i].n_pred) {
            return FAULT(KM_SERR_EDGE, i);
        }
    }
    for (uint32_t t = 0; t < h->n_records; t++) {
        if (bad_template(&tpl[t])) {
            return FAULT(KM_SERR_TEMPLATE, none);
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
    for (uint32_t i = 0; i < s->hdr->n_nodes; i++) {
        uint32_t n_pred = s->nodes[i].n_pred;
        s->left[i] = (uint8_t)n_pred;
        s->status[i] = n_pred == 0 ? KMG_READY : KMG_WAITING;
        if (n_pred == 0) {
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
    for (uint32_t k = 0; k < n->succ_n; k++) {
        uint32_t succ = s->edges[n->succ_off + k];
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
    uint32_t n_cores
) {
    if (n_bindings > KM_GRAPH_BINDINGS) {
        return KM_SERR_HEADER;
    }
    const km_graph_t* h = (const km_graph_t*)upload;
    const uint8_t* base = (const uint8_t*)upload;
    uint32_t at_edges = sizeof(km_graph_t) + line_up(h->n_nodes * sizeof(km_node_t));
    uint32_t at_ops = at_edges + line_up(h->n_edges * 2u);
    uint32_t at_tpl = at_ops + line_up(h->n_ops * sizeof(km_uop_t));
    s->hdr = h;
    s->nodes = (const km_node_t*)(base + sizeof(km_graph_t));
    s->edges = (const uint16_t*)(base + at_edges);
    s->ops = (const km_uop_t*)(base + at_ops);
    s->tpl = (const km_template_t*)(base + at_tpl);
    for (uint32_t r = 0; r < KM_STREAM_REGS; r++) {
        s->regs[r] = r < n_bindings ? bindings[r] : 0u;
    }
    s->runs = 1;
    s->phase = KMG_RUNNING;
    s->error_reason = s->error_node = 0;
    s->in_flight = 0;
    s->n_cores = n_cores;
    reset(s);
    return 0;
}

/* Runs one CPU slice to its end; returns the micro-ops run (the phase says if it
   failed). At most step_limit micro-ops. */
static uint32_t run_cpu(kmg_state_t* s, uint32_t node) {
    const km_node_t* n = &s->nodes[node];
    uint32_t* r = s->regs;
    uint32_t pc = n->arg, end = n->arg + (uint32_t)n->n_ops, steps = 0;
    while (pc != end) {
        if (pc > end) {
            fail(s, KM_SERR_JUMP, node);
            return steps;
        }
        if (steps >= s->hdr->step_limit) {
            fail(s, KM_SERR_STEPS, node);
            return steps;
        }
        const km_uop_t* o = &s->ops[pc];
        uint32_t a = REG(o->a), b = REG(o->b), c = REG(o->c), next = pc + 1u;
        switch (o->op) {
            case KM_OP_SET:
                r[a] = o->imm;
                break;
            case KM_OP_MOV:
                r[a] = r[b];
                break;
            case KM_OP_ADD:
                r[a] = r[b] + r[c];
                break;
            case KM_OP_ADDI:
                r[a] = r[b] + o->imm;
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
                r[a] = r[b] >> (o->imm & 31u);
                break;
            case KM_OP_MOD:
                if (r[c] == 0) {
                    fail(s, KM_SERR_MOD_ZERO, node);
                    return steps;
                }
                r[a] = r[b] % r[c];
                break;
            case KM_OP_RAND: {
                uint32_t x = r[KM_STREAM_SEED_REG];
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                r[KM_STREAM_SEED_REG] = x;
                r[a] = x;
                break;
            }
            case KM_OP_JMP:
                next = o->imm;
                break;
            case KM_OP_JLT:
                next = r[a] < r[b] ? o->imm : next;
                break;
            case KM_OP_JEQ:
                next = r[a] == r[b] ? o->imm : next;
                break;
            case KM_OP_JNZ:
                next = r[a] != 0 ? o->imm : next;
                break;
            default:
                fail(s, KM_SERR_OPCODE, node);
                return steps;
        }
        steps++;
        pc = next;
    }
    return steps;
}

static void set_arg(km_record_t* rec, uint32_t i, uint32_t v) {
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

/* Builds and issues the record of `node`: KMG_ISSUED, KMG_BUSY (put back at the tail)
   or KMG_ERROR (team outside the grid, a job that can never fit). */
static uint32_t issue_record(
    kmg_state_t* s, uint32_t node, kmg_issue_fn issue, void* ctx
) {
    const km_node_t* n = &s->nodes[node];
    const km_template_t* t = &s->tpl[n->arg < s->hdr->n_records ? n->arg : 0u];
    const km_record_t* src = &t->rec;
    km_record_t rec;
    rec.seq = rec.land1 = rec.land2 = rec.land3 = 0;
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
    for (uint32_t k = 0; k < t->n_patch && k < KM_TPL_PATCHES; k++) {
        set_arg(&rec, t->arg[k], s->regs[REG(t->reg[k])]);
    }
    uint32_t first = src->team & 0xFFFFu, count = src->team >> 16;
    if (t->first_reg != KM_GRAPH_NONE) {
        first = s->regs[REG(t->first_reg)];
    }
    if (t->count_reg != KM_GRAPH_NONE) {
        count = s->regs[REG(t->count_reg)];
    }
    if (rec.hdr >> 8 & KM_F_QUEUE) {
        first = 0;
        if (count == 0 || count > s->n_cores) {
            fail(s, KM_SERR_NEVER_FITS, node);
            return KMG_ERROR;
        }
    } else if (count == 0 || first > s->n_cores || count > s->n_cores - first) {
        fail(s, KM_SERR_TEAM, node);
        return KMG_ERROR;
    }
    rec.team = first | count << 16;
    if (issue(ctx, node, &rec) != KMG_ISSUED) {
        push(s, node);
        return KMG_BUSY;
    }
    s->status[node] = KMG_FLIGHT;
    s->in_flight++;
    return KMG_ISSUED;
}

/* All nodes are done: go round again while `again` is set, else end. */
static uint32_t round_end(kmg_state_t* s) {
    uint32_t again = s->hdr->again_reg;
    if (again == KM_GRAPH_NONE || s->regs[REG(again)] == 0) {
        s->phase = KMG_END;
        return KMG_END;
    }
    if (s->runs >= s->hdr->max_runs) {
        fail(s, KM_SERR_RUNS, KM_GRAPH_NO_NODE);
        return KMG_ERROR;
    }
    s->runs++;
    reset(s);
    return KMG_ROUND;
}

uint32_t kmg_step(kmg_state_t* s, kmg_issue_fn issue, void* ctx) {
    if (s->phase != KMG_RUNNING) {
        return s->phase;
    }
    uint32_t taken = 0, busy = 0, ops = 0;
    while (
        s->done < s->hdr->n_nodes && taken < KM_GRAPH_STEP && s->ready_n != 0 &&
        busy < s->ready_n
    ) {
        uint32_t node = pop(s);
        const km_node_t* n = &s->nodes[node];
        taken++;
        if (n->skip_reg != KM_GRAPH_NONE && s->regs[REG(n->skip_reg)] == 0) {
            finish(s, node);
            busy = 0;
            continue;
        }
        if (n->kind == KM_NODE_RECORD) {
            uint32_t got = issue_record(s, node, issue, ctx);
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
    return s->done == s->hdr->n_nodes ? round_end(s) : KMG_RUNNING;
}

uint32_t kmg_complete(kmg_state_t* s, uint32_t node, uint32_t result) {
    if (s->phase != KMG_RUNNING || node >= s->hdr->n_nodes ||
        s->status[node] != KMG_FLIGHT) {
        return 1u;
    }
    uint32_t reg = s->nodes[node].result_reg;
    if (reg != KM_GRAPH_NONE) {
        s->regs[REG(reg)] = result;
    }
    s->in_flight--;
    finish(s, node);
    return 0;
}

void kmg_stop(kmg_state_t* s, uint32_t reason, uint32_t node) {
    if (s->phase == KMG_RUNNING) {
        fail(s, reason, node);
    }
}
