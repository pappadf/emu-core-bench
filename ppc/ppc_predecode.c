// SPDX-License-Identifier: MIT
// ppc_predecode.c
// Backend D: predecoded threaded-code interpreter.  The code region is
// decoded once per run into a flat array of (handler id, raw instruction)
// entries; the execution loop then indexes that array by pc instead of
// fetching/byte-swapping from guest RAM, and dispatches through ONE flat
// br_table (opcode-31/19 secondary decode is resolved at predecode time).
// This measures the "cheap predecode" ceiling: operand fields are still
// extracted at runtime, so a real design pre-extracting rd/ra/rb/imm would
// do somewhat better still.

#include "ppc_ops.h"

// Flat handler-id space: every leaf from both X-macro lists plus the two
// opcode-19 forms.  (The grp19/grp31 ids exist but are never produced by
// pd_decode, which resolves to leaves.)
enum {
#define PD_ID(key, name, stmt) PD_##name,
    PPC_PRIMARY(PD_ID)
    PPC_OP31(PD_ID)
#undef PD_ID
    PD_bclr_,
    PD_bcctr_,
    PD_bad_,
};

// One predecoded entry per instruction word: flat id + raw big-endian-
// decoded instruction (8 bytes; a production design would also pre-extract
// operand fields here)
typedef struct {
    uint16_t id; // flat handler id
    uint16_t pad;
    uint32_t insn; // raw instruction word
} pd_ent_t;

#define PD_REGION_BYTES (256u * 1024) // guest [0, 256 KB): covers all kernels
#define PD_ENTRIES      (PD_REGION_BYTES / 4)
static pd_ent_t g_pd[PD_ENTRIES];

// Resolve one instruction word to its flat handler id (predecode-time only)
static uint16_t pd_decode(uint32_t insn) {
    uint32_t prim = insn >> 26;
    if (prim == 31) {
        switch (insn & 0x7FF) {
#define PD31(key, name, stmt) \
    case key: \
        return PD_##name;
            PPC_OP31(PD31)
#undef PD31
        default:
            return PD_bad_;
        }
    }
    if (prim == 19) {
        switch ((insn >> 1) & 0x3FF) {
        case 16:
            return PD_bclr_;
        case 528:
            return PD_bcctr_;
        default:
            return PD_bad_;
        }
    }
    switch (prim) {
#define PDP(key, name, stmt) \
    case key: \
        return PD_##name;
        PPC_PRIMARY(PDP)
#undef PDP
    default:
        return PD_bad_;
    }
}

// Execute exactly `budget` instructions starting at c->pc.  The whole code
// region is (re)decoded up front; at 64 K entries that is <0.1% of a 100 M-
// instruction sprint and is charged to this backend's own time.
void ppc_run_pd(ppc_t *c, uint32_t budget) {
    for (uint32_t a = 0; a < PD_REGION_BYTES; a += 4) {
        uint32_t insn = memory_read_uint32(a);
        g_pd[a >> 2].id = pd_decode(insn);
        g_pd[a >> 2].insn = insn;
    }
    uint32_t pc = c->pc;
    uint32_t n = budget;
    while (n > 0) {
        pd_ent_t e = g_pd[pc >> 2]; // kernels never leave the region
        uint32_t insn = e.insn;
        pc += 4;
        n--;
        switch (e.id) {
#define PDX(key, name, stmt) \
    case PD_##name: { \
        stmt; \
    } break;
            PPC_PRIMARY(PDX)
            PPC_OP31(PDX)
#undef PDX
        case PD_bclr_: {
            bool taken = bc_taken(c, insn, true);
            uint32_t target = c->lr & ~3u;
            if (insn & 1)
                c->lr = pc;
            pc = taken ? target : pc;
        } break;
        case PD_bcctr_: {
            bool taken = bc_taken(c, insn, false);
            if (insn & 1)
                c->lr = pc;
            pc = taken ? (c->ctr & ~3u) : pc;
        } break;
        default:
            ppc_illegal(c, insn);
        }
    }
    c->pc = pc;
}
