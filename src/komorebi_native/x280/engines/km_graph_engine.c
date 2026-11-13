/* The graph executor: check, start, step, complete. Freestanding: no library calls, no
   floating point, no static data; every loop is capped by a limit of km_graph.h. */
#include "km_graph_engine.h"

#define FAULT(reason, node) ((uint32_t)(reason) | (uint32_t)(node) << 8)
#define REGISTER(register_index) ((register_index) & (KM_STREAM_REGISTERS - 1u))
#define RING(position) ((position) & (KM_GRAPH_MAX_NODES - 1u))

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

static uint32_t bad_register(uint32_t register_index) {
    return register_index != KM_GRAPH_NONE && register_index >= KM_STREAM_REGISTERS;
}

static uint32_t bad_template(const km_template_t* template_entry) {
    uint32_t handler = template_entry->rec.hdr & 0xFFu;
    uint32_t flags = template_entry->rec.hdr >> 8 & 0xFFu;
    if (handler < 1u || handler > KM_HANDLER_USER_MAX) {
        return 1u;
    }
    if (flags & ~(uint32_t)(KM_FLAG_EXCLUSIVE | KM_FLAG_RESULT | KM_FLAG_QUEUE)) {
        return 1u;
    }
    if (template_entry->patch_count > KM_TEMPLATE_PATCHES) {
        return 1u;
    }
    for (uint32_t patch = 0; patch < template_entry->patch_count; patch++) {
        if (
            template_entry->arg[patch] >= KM_TEMPLATE_ARGS ||
            template_entry->registers[patch] >= KM_STREAM_REGISTERS
        ) {
            return 1u;
        }
    }
    return (
        bad_register(template_entry->first_register) ||
        bad_register(template_entry->count_register)
    );
}

/* The micro-ops of one CPU slice [first, end): opcode, registers, jump targets. */
static uint32_t check_slice(const km_uop_t* ops, uint32_t first, uint32_t end) {
    for (uint32_t pc = first; pc < end; pc++) {
        const km_uop_t* micro_op = &ops[pc];
        uint32_t use = cpu_op_use(micro_op->op);
        if (use == 0xFFu) {
            return KM_STREAM_ERROR_OPCODE;
        }
        if (
            ((use & 1u) && micro_op->a >= KM_STREAM_REGISTERS) ||
            ((use & 2u) && micro_op->b >= KM_STREAM_REGISTERS) ||
            ((use & 4u) && micro_op->c >= KM_STREAM_REGISTERS)
        ) {
            return KM_STREAM_ERROR_REGISTER;
        }
        if ((use & 8u) && (micro_op->immediate < first || micro_op->immediate > end)) {
            return KM_STREAM_ERROR_JUMP;
        }
    }
    return 0;
}

/* One node's own fields, its template or slice; `cpu_next` is where its slice starts.
 */
static uint32_t check_node(
    const km_graph_t* graph,
    const km_node_t* node_entry,
    const km_uop_t* ops,
    const km_template_t* templates,
    uint32_t cpu_next
) {
    if (
        node_entry->kind < KM_NODE_COMMAND || node_entry->kind > KM_NODE_JOIN ||
        node_entry->flags != 0
    ) {
        return KM_STREAM_ERROR_NODE;
    }
    if (node_entry->predecessor_count > KM_GRAPH_MAX_PREDECESSORS) {
        return KM_STREAM_ERROR_EDGE;
    }
    if (
        bad_register(node_entry->skip_register) ||
        bad_register(node_entry->result_register)
    ) {
        return KM_STREAM_ERROR_REGISTER;
    }
    if (node_entry->kind == KM_NODE_COMMAND) {
        if (
            node_entry->arg >= graph->command_count ||
            bad_template(&templates[node_entry->arg])
        ) {
            return KM_STREAM_ERROR_TEMPLATE;
        }
        uint32_t flags = templates[node_entry->arg].rec.hdr >> 8 & 0xFFu;
        if (
            node_entry->op_count != 0 ||
            (node_entry->result_register != KM_GRAPH_NONE && !(flags & KM_FLAG_RESULT))
        ) {
            return KM_STREAM_ERROR_NODE;
        }
        return 0;
    }
    if (node_entry->result_register != KM_GRAPH_NONE) {
        return KM_STREAM_ERROR_NODE;
    }
    if (node_entry->kind == KM_NODE_JOIN) {
        return (
            node_entry->arg != 0 || node_entry->op_count != 0 ? KM_STREAM_ERROR_NODE : 0
        );
    }
    if (
        node_entry->arg != cpu_next || node_entry->op_count == 0 ||
        node_entry->op_count > graph->op_count - cpu_next
    ) {
        return KM_STREAM_ERROR_NODE;
    }
    return (
        check_slice(
            ops, node_entry->arg, node_entry->arg + (uint32_t)node_entry->op_count
        )
    );
}

uint32_t kmg_check(const void* upload, uint32_t bytes) {
    const km_graph_t* graph = (const km_graph_t*)upload;
    const uint32_t none = KM_GRAPH_NO_NODE;
    if (bytes < sizeof(km_graph_t) || bytes % KM_STREAM_LINE != 0) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    if (graph->magic != KM_GRAPH_MAGIC || graph->version != KM_GRAPH_VERSION) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    if (
        graph->node_count < 1u || graph->node_count > KM_GRAPH_MAX_NODES ||
        graph->edge_count > KM_GRAPH_MAX_EDGES || graph->op_count > KM_GRAPH_MAX_OPS ||
        graph->command_count > KM_GRAPH_MAX_TEMPLATES
    ) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    if (
        bad_register(graph->again_register) || graph->max_runs == 0 ||
        graph->step_limit == 0 || graph->step_limit > KM_GRAPH_MAX_STEPS ||
        (graph->flags & ~KM_GRAPH_KEEP) != 0
    ) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    uint32_t at_edges = (
        sizeof(km_graph_t) + line_up(graph->node_count * sizeof(km_node_t))
    );
    uint32_t at_ops = at_edges + line_up(graph->edge_count * 2u);
    uint32_t at_template = at_ops + line_up(graph->op_count * sizeof(km_uop_t));
    if (bytes != at_template + line_up(graph->command_count * sizeof(km_template_t))) {
        return FAULT(KM_STREAM_ERROR_HEADER, none);
    }
    const uint8_t* base = (const uint8_t*)upload;
    const km_node_t* nodes = (const km_node_t*)(base + sizeof(km_graph_t));
    const uint16_t* edges = (const uint16_t*)(base + at_edges);
    const km_uop_t* ops = (const km_uop_t*)(base + at_ops);
    const km_template_t* templates = (const km_template_t*)(base + at_template);
    /* Predecessors seen per node; zeroed through a volatile pointer, as a plain loop
       may become a memset call. */
    uint8_t seen[KM_GRAPH_MAX_NODES];
    volatile uint8_t* zero = seen;
    for (uint32_t i = 0; i < KM_GRAPH_MAX_NODES; i++) {
        zero[i] = 0;
    }
    uint32_t cpu_next = 0, edge_next = 0;
    for (uint32_t i = 0; i < graph->node_count; i++) {
        const km_node_t* node_entry = &nodes[i];
        uint32_t fault = check_node(graph, node_entry, ops, templates, cpu_next);
        if (fault != 0) {
            return FAULT(fault, i);
        }
        if (node_entry->kind == KM_NODE_CPU) {
            cpu_next += node_entry->op_count;
        }
        if (
            node_entry->successor_offset != edge_next ||
            node_entry->successor_count > graph->edge_count - edge_next
        ) {
            return FAULT(KM_STREAM_ERROR_EDGE, i);
        }
        uint32_t previous = i;
        for (
            uint32_t successor_position = 0;
            successor_position < node_entry->successor_count;
            successor_position++
        ) {
            uint32_t successor = edges[edge_next + successor_position];
            if (
                successor <= previous || successor >= graph->node_count ||
                seen[successor] >= KM_GRAPH_MAX_PREDECESSORS
            ) {
                return FAULT(KM_STREAM_ERROR_EDGE, i);
            }
            seen[successor]++;
            previous = successor;
        }
        edge_next += node_entry->successor_count;
    }
    if (edge_next != graph->edge_count) {
        return FAULT(KM_STREAM_ERROR_EDGE, none);
    }
    if (cpu_next != graph->op_count) {
        return FAULT(KM_STREAM_ERROR_NODE, none);
    }
    for (uint32_t i = 0; i < graph->node_count; i++) {
        if (seen[i] != nodes[i].predecessor_count) {
            return FAULT(KM_STREAM_ERROR_EDGE, i);
        }
    }
    for (
        uint32_t template_index = 0; template_index < graph->command_count;
        template_index++
    ) {
        if (bad_template(&templates[template_index])) {
            return FAULT(KM_STREAM_ERROR_TEMPLATE, none);
        }
    }
    return 0;
}

uint32_t kmg_state_bytes(void) {
    return sizeof(kmg_state_t);
}

static void push(kmg_state_t* state, uint32_t node) {
    state->ready[RING(state->head + state->ready_n)] = (uint16_t)node;
    state->ready_n++;
    state->status[node] = KMG_READY;
}

static uint32_t pop(kmg_state_t* state) {
    uint32_t node = state->ready[RING(state->head)];
    state->head = RING(state->head + 1u);
    state->ready_n--;
    return node;
}

/* Every node waits for its predecessors again; the roots are ready, in node order. */
static void reset(kmg_state_t* state) {
    state->done = state->head = state->ready_n = 0;
    for (uint32_t i = 0; i < state->hdr->node_count; i++) {
        uint32_t predecessor_count = state->nodes[i].predecessor_count;
        state->left[i] = (uint8_t)predecessor_count;
        state->status[i] = predecessor_count == 0 ? KMG_READY : KMG_WAITING;
        if (predecessor_count == 0) {
            push(state, i);
        }
    }
}

static void fail(kmg_state_t* state, uint32_t reason, uint32_t node) {
    state->phase = KMG_ERROR;
    state->error_reason = reason;
    state->error_node = node;
}

static void finish(kmg_state_t* state, uint32_t node) {
    const km_node_t* node_entry = &state->nodes[node];
    state->status[node] = KMG_DONE;
    state->done++;
    for (
        uint32_t successor_position = 0;
        successor_position < node_entry->successor_count;
        successor_position++
    ) {
        uint32_t successor = (
            state->edges [node_entry->successor_offset + successor_position]
        );
        if (--state->left[successor] == 0) {
            push(state, successor);
        }
    }
}

uint32_t kmg_start(
    kmg_state_t* state,
    const void* upload,
    const uint32_t* bindings,
    uint32_t n_bindings,
    uint32_t core_count
) {
    if (n_bindings > KM_GRAPH_BINDINGS) {
        return KM_STREAM_ERROR_HEADER;
    }
    const km_graph_t* graph = (const km_graph_t*)upload;
    const uint8_t* base = (const uint8_t*)upload;
    uint32_t at_edges = (
        sizeof(km_graph_t) + line_up(graph->node_count * sizeof(km_node_t))
    );
    uint32_t at_ops = at_edges + line_up(graph->edge_count * 2u);
    uint32_t at_template = at_ops + line_up(graph->op_count * sizeof(km_uop_t));
    state->hdr = graph;
    state->nodes = (const km_node_t*)(base + sizeof(km_graph_t));
    state->edges = (const uint16_t*)(base + at_edges);
    state->ops = (const km_uop_t*)(base + at_ops);
    state->template = (const km_template_t*)(base + at_template);
    for (
        uint32_t register_index = 0; register_index < KM_STREAM_REGISTERS;
        register_index++
    ) {
        state->registers[register_index] = (
            register_index < n_bindings ? bindings[register_index] : 0u
        );
    }
    state->runs = 1;
    state->phase = KMG_RUNNING;
    state->error_reason = state->error_node = 0;
    state->in_flight = 0;
    state->core_count = core_count;
    reset(state);
    return 0;
}

/* Runs one CPU slice to its end; returns the micro-ops run (the phase says if it
   failed). At most step_limit micro-ops. */
static uint32_t run_cpu(kmg_state_t* state, uint32_t node) {
    const km_node_t* node_entry = &state->nodes[node];
    uint32_t* registers = state->registers;
    uint32_t pc = node_entry->arg;
    uint32_t end = node_entry->arg + (uint32_t)node_entry->op_count;
    uint32_t steps = 0;
    while (pc != end) {
        if (pc > end) {
            fail(state, KM_STREAM_ERROR_JUMP, node);
            return steps;
        }
        if (steps >= state->hdr->step_limit) {
            fail(state, KM_STREAM_ERROR_STEPS, node);
            return steps;
        }
        const km_uop_t* micro_op = &state->ops[pc];
        uint32_t index_a = REGISTER(micro_op->a);
        uint32_t index_b = REGISTER(micro_op->b);
        uint32_t index_c = REGISTER(micro_op->c);
        uint32_t next = pc + 1u;
        switch (micro_op->op) {
            case KM_OP_SET:
                registers[index_a] = micro_op->immediate;
                break;
            case KM_OP_MOV:
                registers[index_a] = registers[index_b];
                break;
            case KM_OP_ADD:
                registers[index_a] = registers[index_b] + registers[index_c];
                break;
            case KM_OP_ADDI:
                registers[index_a] = registers[index_b] + micro_op->immediate;
                break;
            case KM_OP_SUB:
                registers[index_a] = registers[index_b] - registers[index_c];
                break;
            case KM_OP_MUL:
                registers[index_a] = registers[index_b] * registers[index_c];
                break;
            case KM_OP_AND:
                registers[index_a] = registers[index_b] & registers[index_c];
                break;
            case KM_OP_SHR:
                registers[index_a] = registers[index_b] >> (micro_op->immediate & 31u);
                break;
            case KM_OP_MOD:
                if (registers[index_c] == 0) {
                    fail(state, KM_STREAM_ERROR_MOD_ZERO, node);
                    return steps;
                }
                registers[index_a] = registers[index_b] % registers[index_c];
                break;
            case KM_OP_RAND: {
                uint32_t random = registers[KM_STREAM_SEED_REGISTER];
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                registers[KM_STREAM_SEED_REGISTER] = random;
                registers[index_a] = random;
                break;
            }
            case KM_OP_JMP:
                next = micro_op->immediate;
                break;
            case KM_OP_JLT:
                next = (
                    registers[index_a] < registers[index_b] ? micro_op->immediate : next
                );
                break;
            case KM_OP_JEQ:
                next = (
                    registers[index_a] == registers[index_b]
                        ? micro_op->immediate
                        : next
                );
                break;
            case KM_OP_JNZ:
                next = registers[index_a] != 0 ? micro_op->immediate : next;
                break;
            default:
                fail(state, KM_STREAM_ERROR_OPCODE, node);
                return steps;
        }
        steps++;
        pc = next;
    }
    return steps;
}

static void set_arg(km_command_t* command, uint32_t index, uint32_t value) {
    switch (index) {
        case 0:
            command->a0 = value;
            break;
        case 1:
            command->a1 = value;
            break;
        case 2:
            command->a2 = value;
            break;
        case 3:
            command->a3 = value;
            break;
        case 4:
            command->a4 = value;
            break;
        case 5:
            command->a5 = value;
            break;
        case 6:
            command->a6 = value;
            break;
        case 7:
            command->a7 = value;
            break;
        case 8:
            command->a8 = value;
            break;
        default:
            command->a9 = value;
            break;
    }
}

/* Builds and submits the command of `node`: KMG_SUBMITTED, KMG_BUSY (put back at the
   tail) or KMG_ERROR (group outside the grid, a job that can never fit). */
static uint32_t submit_command(
    kmg_state_t* state, uint32_t node, kmg_submit_function submit, void* context
) {
    const km_node_t* node_entry = &state->nodes[node];
    const km_template_t* template_entry = (
        &state->template
             [node_entry->arg < state->hdr->command_count ? node_entry->arg : 0u]
    );
    const km_command_t* template_command = &template_entry->rec;
    km_command_t command;
    command.sequence = command.land1 = command.land2 = command.land3 = 0;
    command.hdr = template_command->hdr & 0xFFFFu;
    command.a0 = template_command->a0;
    command.a1 = template_command->a1;
    command.a2 = template_command->a2;
    command.a3 = template_command->a3;
    command.a4 = template_command->a4;
    command.a5 = template_command->a5;
    command.a6 = template_command->a6;
    command.a7 = template_command->a7;
    command.a8 = template_command->a8;
    command.a9 = template_command->a9;
    for (
        uint32_t patch = 0;
        patch < template_entry->patch_count && patch < KM_TEMPLATE_PATCHES;
        patch++
    ) {
        set_arg(
            &command,
            template_entry->arg[patch],
            state->registers[REGISTER(template_entry->registers[patch])]
        );
    }
    uint32_t first = template_command->group & 0xFFFFu;
    uint32_t count = template_command->group >> 16;
    if (template_entry->first_register != KM_GRAPH_NONE) {
        first = state->registers[REGISTER(template_entry->first_register)];
    }
    if (template_entry->count_register != KM_GRAPH_NONE) {
        count = state->registers[REGISTER(template_entry->count_register)];
    }
    if (command.hdr >> 8 & KM_FLAG_QUEUE) {
        first = 0;
        if (count == 0 || count > state->core_count) {
            fail(state, KM_STREAM_ERROR_NEVER_FITS, node);
            return KMG_ERROR;
        }
    } else if (
        KM_GROUP_IS_CLUSTER(first, count)
            ? (command.hdr >> 8 & KM_FLAG_EXCLUSIVE) != 0
            : count == 0 || first > state->core_count ||
                count > state->core_count - first
    ) {
        fail(state, KM_STREAM_ERROR_GROUP, node);
        return KMG_ERROR;
    }
    command.group = first | count << 16;
    if (submit(context, node, &command) != KMG_SUBMITTED) {
        push(state, node);
        return KMG_BUSY;
    }
    state->status[node] = KMG_FLIGHT;
    state->in_flight++;
    return KMG_SUBMITTED;
}

/* All nodes are done: go round again while `again` is set, else end. */
static uint32_t round_end(kmg_state_t* state) {
    uint32_t again = state->hdr->again_register;
    if (again == KM_GRAPH_NONE || state->registers[REGISTER(again)] == 0) {
        state->phase = KMG_END;
        return KMG_END;
    }
    if (state->runs >= state->hdr->max_runs) {
        fail(state, KM_STREAM_ERROR_RUNS, KM_GRAPH_NO_NODE);
        return KMG_ERROR;
    }
    state->runs++;
    reset(state);
    return KMG_ROUND;
}

uint32_t kmg_step(kmg_state_t* state, kmg_submit_function submit, void* context) {
    if (state->phase != KMG_RUNNING) {
        return state->phase;
    }
    uint32_t taken = 0, busy = 0, micro_ops_run = 0;
    while (
        state->done < state->hdr->node_count && taken < KM_GRAPH_STEP &&
        state->ready_n != 0 && busy < state->ready_n
    ) {
        uint32_t node = pop(state);
        const km_node_t* node_entry = &state->nodes[node];
        taken++;
        if (
            node_entry->skip_register != KM_GRAPH_NONE &&
            state->registers[REGISTER(node_entry->skip_register)] == 0
        ) {
            finish(state, node);
            busy = 0;
            continue;
        }
        if (node_entry->kind == KM_NODE_COMMAND) {
            uint32_t outcome = submit_command(state, node, submit, context);
            if (outcome == KMG_ERROR) {
                return KMG_ERROR;
            }
            busy = outcome == KMG_BUSY ? busy + 1u : 0u;
            continue;
        }
        busy = 0;
        if (node_entry->kind == KM_NODE_CPU) {
            micro_ops_run += run_cpu(state, node);
            if (state->phase != KMG_RUNNING) {
                return KMG_ERROR;
            }
        }
        finish(state, node);
        if (micro_ops_run >= KM_GRAPH_STEP_OPS) {
            break;
        }
    }
    return state->done == state->hdr->node_count ? round_end(state) : KMG_RUNNING;
}

uint32_t kmg_complete(kmg_state_t* state, uint32_t node, uint32_t result) {
    if (
        state->phase != KMG_RUNNING || node >= state->hdr->node_count ||
        state->status[node] != KMG_FLIGHT
    ) {
        return 1u;
    }
    uint32_t register_index = state->nodes[node].result_register;
    if (register_index != KM_GRAPH_NONE) {
        state->registers[REGISTER(register_index)] = result;
    }
    state->in_flight--;
    finish(state, node);
    return 0;
}

uint32_t kmg_retry(kmg_state_t* state, uint32_t node) {
    if (
        state->phase != KMG_RUNNING || node >= state->hdr->node_count ||
        state->status[node] != KMG_FLIGHT
    ) {
        return 1u;
    }
    state->in_flight--;
    push(state, node);
    return 0;
}

void kmg_stop(kmg_state_t* state, uint32_t reason, uint32_t node) {
    if (state->phase == KMG_RUNNING) {
        fail(state, reason, node);
    }
}
