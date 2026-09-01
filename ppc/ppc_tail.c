// SPDX-License-Identifier: MIT
// ppc_tail.c
// Backend B: musttail tail-call dispatch (wasm3/CPython-3.14 style).  Each
// primary opcode is its own small function; dispatch is replicated at the
// end of every handler as a musttail call through a const function-pointer
// table, which emcc/clang lowers to return_call_indirect with -mtail-call.
// Hot state (pc, remaining budget) rides in argument registers; register
// state stays in the ppc_t like the switch backend.

#include "ppc_ops.h"

// NO_TAIL probe build: alias the tail backend to the switch backend so a
// module can be produced with zero tail-call instructions (feature off)
#ifdef NO_TAIL
void ppc_run_tail(ppc_t *c, uint32_t budget) { ppc_run_switch(c, budget); }
#else

// Common handler signature: pc = CIA+4 of the current instruction, n =
// instructions remaining AFTER the current one
typedef void ppc_handler_fn(ppc_t *c, uint32_t insn, uint32_t pc, uint32_t n);

// Forward declarations so the table can be a const initializer
#define TC_PROTO(key, name, stmt) static ppc_handler_fn h_##name;
PPC_PRIMARY(TC_PROTO)
#undef TC_PROTO
static ppc_handler_fn h_bad;

// Primary dispatch table; const so the compiler knows it never changes
static ppc_handler_fn *const g_dispatch[64] = {
    [0 ... 63] = h_bad, // GNU range initializer; specific slots override below
#define TC_SLOT(key, name, stmt) [key] = h_##name,
    PPC_PRIMARY(TC_SLOT)
#undef TC_SLOT
};

// Replicated dispatch tail: stop when the budget is spent, else fetch the
// next instruction and musttail-chain to its handler
#define NEXT() \
    do { \
        if (__builtin_expect(n == 0, 0)) { \
            c->pc = pc; \
            return; \
        } \
        uint32_t _insn = memory_read_uint32(pc); \
        MUSTTAIL return g_dispatch[_insn >> 26](c, _insn, pc + 4, n - 1); \
    } while (0)

// Stamp one handler function per primary opcode from the shared list
#define TC_HANDLER(key, name, stmt) \
    static void h_##name(ppc_t *c, uint32_t insn, uint32_t pc, uint32_t n) { \
        { stmt; } \
        NEXT(); \
    }
PPC_PRIMARY(TC_HANDLER)
#undef TC_HANDLER

// Terminal handler for opcodes outside the benchmark subset
static void h_bad(ppc_t *c, uint32_t insn, uint32_t pc, uint32_t n) {
    (void)pc;
    (void)n;
    ppc_illegal(c, insn);
}

// Execute exactly `budget` instructions starting at c->pc
void ppc_run_tail(ppc_t *c, uint32_t budget) {
    if (budget == 0)
        return;
    uint32_t pc = c->pc;
    uint32_t insn = memory_read_uint32(pc);
    g_dispatch[insn >> 26](c, insn, pc + 4, budget - 1);
}

#endif // NO_TAIL
