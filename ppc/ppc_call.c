// SPDX-License-Identifier: MIT
// ppc_call.c
// Backend C: call-threaded dispatch — one small function per primary opcode
// (like the tail backend) but invoked by plain call_indirect from a central
// loop (like the switch backend's loop).  Separates the "small functions
// register-allocate better" effect from the "tail-call chaining" effect.

#include "ppc_ops.h"

// Handler returns the next-instruction address; pc on entry is CIA+4
typedef uint32_t ppc_call_fn(ppc_t *c, uint32_t insn, uint32_t pc);

#define CT_PROTO(key, name, stmt) static ppc_call_fn f_##name;
PPC_PRIMARY(CT_PROTO)
#undef CT_PROTO
static ppc_call_fn f_bad;

// Primary dispatch table
static ppc_call_fn *const g_call_dispatch[64] = {
    [0 ... 63] = f_bad,
#define CT_SLOT(key, name, stmt) [key] = f_##name,
    PPC_PRIMARY(CT_SLOT)
#undef CT_SLOT
};

// Stamp one handler per primary opcode from the shared list
#define CT_HANDLER(key, name, stmt) \
    static uint32_t f_##name(ppc_t *c, uint32_t insn, uint32_t pc) { \
        { stmt; } \
        return pc; \
    }
PPC_PRIMARY(CT_HANDLER)
#undef CT_HANDLER

// Terminal handler for opcodes outside the benchmark subset
static uint32_t f_bad(ppc_t *c, uint32_t insn, uint32_t pc) {
    ppc_illegal(c, insn);
    return pc;
}

// Execute exactly `budget` instructions starting at c->pc
void ppc_run_call(ppc_t *c, uint32_t budget) {
    uint32_t pc = c->pc;
    uint32_t n = budget;
    while (n > 0) {
        uint32_t insn = memory_read_uint32(pc);
        n--;
        pc = g_call_dispatch[insn >> 26](c, insn, pc + 4);
    }
    c->pc = pc;
}
