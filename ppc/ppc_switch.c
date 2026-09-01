// SPDX-License-Identifier: MIT
// ppc_switch.c
// Backend A: the repo-conventional big-switch interpreter loop (cf.
// cpu_decode.h / cpu_68030.c).  One function, one while loop, primary
// dispatch as a switch on the top 6 opcode bits (one br_table in wasm).

#include "ppc_ops.h"

// Execute exactly `budget` instructions starting at c->pc
void ppc_run_switch(ppc_t *c, uint32_t budget) {
    uint32_t pc = c->pc;
    uint32_t n = budget;
    while (n > 0) {
        uint32_t insn = memory_read_uint32(pc);
        pc += 4; // handlers see pc = CIA+4
        n--;
        switch (insn >> 26) {
#define SW_CASE(key, name, stmt) \
    case key: { \
        stmt; \
    } break;
            PPC_PRIMARY(SW_CASE)
#undef SW_CASE
        default:
            ppc_illegal(c, insn);
        }
    }
    c->pc = pc;
}
