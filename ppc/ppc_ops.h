// SPDX-License-Identifier: MIT
// ppc_ops.h
// Shared PPC opcode header for the dispatch benchmark: generic static-inline
// instruction bodies (Layer 1) plus X-macro instruction lists (Layer 2) that
// each backend stamps into its own dispatch shape.  Rc on opcode-31 forms is
// a compile-time constant because the secondary key is (XO << 1) | Rc.
// Intended for multiple inclusion by backend .c files; no include guard on
// the X-macro section is needed since each backend is its own TU.

#ifndef PPC_OPS_H
#define PPC_OPS_H

#include "ppc_bench.h"

// === Instruction field extractors ===
#define P_RD(i)   (((i) >> 21) & 31) // rD / rS / BO field
#define P_RA(i)   (((i) >> 16) & 31) // rA / BI field
#define P_RB(i)   (((i) >> 11) & 31) // rB / SH field
#define P_SIMM(i) ((int32_t)(int16_t)(i)) // sign-extended D field
#define P_UIMM(i) ((i) & 0xFFFFu) // zero-extended D field

// Operation selectors for the generic arithmetic/logic bodies
enum {
    K_ADD,
    K_SUBF,
    K_MULLW,
    K_AND,
    K_OR,
    K_XOR,
    K_SLW,
    K_SRW,
};

// Replace CR field f (0..7) with the 4-bit value `bits` (LT=8 GT=4 EQ=2 SO=1)
static inline void set_crf(ppc_t *c, int f, uint32_t bits) {
    int sh = 28 - 4 * f;
    c->cr = (c->cr & ~(0xFu << sh)) | (bits << sh);
}

// Record CR0 from a result value (Rc=1 semantics; XER.SO not modelled)
static inline void record0(ppc_t *c, uint32_t r) {
    uint32_t b = ((int32_t)r < 0) ? 8 : ((int32_t)r > 0) ? 4 : 2;
    set_crf(c, 0, b);
}

// Rotate left, well-defined for sh==0
static inline uint32_t rotl32(uint32_t v, uint32_t sh) {
    return (v << sh) | (sh ? (v >> (32 - sh)) : 0);
}

// PPC MB..ME wrapping bit mask (bit 0 = MSB)
static inline uint32_t ppc_mask(uint32_t mb, uint32_t me) {
    uint32_t x = 0xFFFFFFFFu >> mb;
    uint32_t y = (me == 31) ? 0xFFFFFFFFu : ~(0xFFFFFFFFu >> (me + 1));
    return (mb <= me) ? (x & y) : (x | y);
}

// === Layer 1: generic instruction bodies ===
// Flag parameters are compile-time constants at every stamped call site, so
// the optimizer folds each stamp to straight-line code (verify at -O2).

// addi/addis: rD = (rA|0) + imm, imm optionally shifted
static inline void gen_addi(ppc_t *c, uint32_t insn, bool shifted) {
    uint32_t ra = P_RA(insn);
    uint32_t imm = shifted ? ((uint32_t)P_SIMM(insn) << 16) : (uint32_t)P_SIMM(insn);
    c->gpr[P_RD(insn)] = (ra ? c->gpr[ra] : 0) + imm;
}

// mulli: rD = rA * simm (low 32 bits)
static inline void gen_mulli(ppc_t *c, uint32_t insn) {
    c->gpr[P_RD(insn)] = (uint32_t)((int32_t)c->gpr[P_RA(insn)] * P_SIMM(insn));
}

// D-form logical: rA = rS op imm (ori/oris/xori/andi.)
static inline void gen_logic_d(ppc_t *c, uint32_t insn, int kind, bool shifted, bool rc) {
    uint32_t imm = P_UIMM(insn);
    if (shifted)
        imm <<= 16;
    uint32_t s = c->gpr[P_RD(insn)];
    uint32_t r = (kind == K_AND) ? (s & imm) : (kind == K_OR) ? (s | imm) : (s ^ imm);
    c->gpr[P_RA(insn)] = r;
    if (rc)
        record0(c, r);
}

// XO-form arithmetic: rD = rA op rB (add/subf/mullw), optional CR0 record
static inline void gen_xo_arith(ppc_t *c, uint32_t insn, int kind, bool rc) {
    uint32_t a = c->gpr[P_RA(insn)], b = c->gpr[P_RB(insn)];
    uint32_t r = (kind == K_ADD) ? (a + b) : (kind == K_SUBF) ? (b - a) : (a * b);
    c->gpr[P_RD(insn)] = r;
    if (rc)
        record0(c, r);
}

// X-form logical/shift: rA = rS op rB (and/or/xor/slw/srw), optional CR0 record
static inline void gen_x_logic(ppc_t *c, uint32_t insn, int kind, bool rc) {
    uint32_t s = c->gpr[P_RD(insn)], b = c->gpr[P_RB(insn)];
    uint32_t r;
    switch (kind) {
    case K_AND: r = s & b; break;
    case K_OR:  r = s | b; break;
    case K_XOR: r = s ^ b; break;
    case K_SLW: r = (b & 0x20) ? 0 : (s << (b & 31)); break;
    default:    r = (b & 0x20) ? 0 : (s >> (b & 31)); break; // K_SRW
    }
    c->gpr[P_RA(insn)] = r;
    if (rc)
        record0(c, r);
}

// srawi: rA = rS >> sh (arithmetic), sets XER.CA, optional CR0 record
static inline void gen_srawi(ppc_t *c, uint32_t insn, bool rc) {
    uint32_t sh = P_RB(insn);
    int32_t s = (int32_t)c->gpr[P_RD(insn)];
    uint32_t r = (uint32_t)(s >> sh);
    uint32_t lost = sh ? ((uint32_t)s & ((1u << sh) - 1)) : 0;
    c->xer_ca = (s < 0) && lost != 0;
    c->gpr[P_RA(insn)] = r;
    if (rc)
        record0(c, r);
}

// rlwinm: rA = rotl(rS, SH) & mask(MB, ME); Rc is a runtime bit here (primary-
// opcode dispatch cannot fold it) — one predictable branch, noted as a caveat
static inline void gen_rlwinm(ppc_t *c, uint32_t insn) {
    uint32_t sh = P_RB(insn), mb = (insn >> 6) & 31, me = (insn >> 1) & 31;
    uint32_t r = rotl32(c->gpr[P_RD(insn)], sh) & ppc_mask(mb, me);
    c->gpr[P_RA(insn)] = r;
    if (insn & 1)
        record0(c, r);
}

// cmp family: set CR field crfD from signed/unsigned compare of rA with rB/imm
static inline void gen_cmp(ppc_t *c, uint32_t insn, bool immediate, bool is_unsigned) {
    int crf = (insn >> 23) & 7;
    uint32_t a = c->gpr[P_RA(insn)];
    uint32_t b = immediate ? (is_unsigned ? P_UIMM(insn) : (uint32_t)P_SIMM(insn)) : c->gpr[P_RB(insn)];
    uint32_t bits;
    if (is_unsigned)
        bits = (a < b) ? 8 : (a > b) ? 4 : 2;
    else
        bits = ((int32_t)a < (int32_t)b) ? 8 : ((int32_t)a > (int32_t)b) ? 4 : 2;
    set_crf(c, crf, bits);
}

// D-form load: rD = mem[(rA|0) + simm]; update forms write EA back to rA
static inline void gen_load_d(ppc_t *c, uint32_t insn, int width, bool update) {
    uint32_t ra = P_RA(insn);
    uint32_t ea = (ra ? c->gpr[ra] : 0) + (uint32_t)P_SIMM(insn);
    uint32_t v = (width == 4) ? memory_read_uint32(ea) : (width == 2) ? memory_read_uint16(ea) : memory_read_uint8(ea);
    c->gpr[P_RD(insn)] = v;
    if (update)
        c->gpr[ra] = ea;
}

// D-form store: mem[(rA|0) + simm] = rS; update forms write EA back to rA
static inline void gen_store_d(ppc_t *c, uint32_t insn, int width, bool update) {
    uint32_t ra = P_RA(insn);
    uint32_t ea = (ra ? c->gpr[ra] : 0) + (uint32_t)P_SIMM(insn);
    uint32_t v = c->gpr[P_RD(insn)];
    if (width == 4)
        memory_write_uint32(ea, v);
    else if (width == 2)
        memory_write_uint16(ea, (uint16_t)v);
    else
        memory_write_uint8(ea, (uint8_t)v);
    if (update)
        c->gpr[ra] = ea;
}

// X-form indexed load: rD = mem[(rA|0) + rB]
static inline void gen_load_x(ppc_t *c, uint32_t insn, int width) {
    uint32_t ra = P_RA(insn);
    uint32_t ea = (ra ? c->gpr[ra] : 0) + c->gpr[P_RB(insn)];
    uint32_t v = (width == 4) ? memory_read_uint32(ea) : (width == 2) ? memory_read_uint16(ea) : memory_read_uint8(ea);
    c->gpr[P_RD(insn)] = v;
}

// X-form indexed store: mem[(rA|0) + rB] = rS
static inline void gen_store_x(ppc_t *c, uint32_t insn, int width) {
    uint32_t ra = P_RA(insn);
    uint32_t ea = (ra ? c->gpr[ra] : 0) + c->gpr[P_RB(insn)];
    uint32_t v = c->gpr[P_RD(insn)];
    if (width == 4)
        memory_write_uint32(ea, v);
    else if (width == 2)
        memory_write_uint16(ea, (uint16_t)v);
    else
        memory_write_uint8(ea, (uint8_t)v);
}

// b/bl/ba: unconditional branch; returns the new next-instruction address.
// `pc` on entry is CIA+4 by decoder convention.
static inline uint32_t gen_b(ppc_t *c, uint32_t insn, uint32_t pc) {
    int32_t li = ((int32_t)(insn << 6) >> 6) & ~3; // sign-extended 24-bit LI << 2
    if (insn & 1)
        c->lr = pc; // LK
    return (insn & 2) ? (uint32_t)li : (pc - 4) + (uint32_t)li; // AA
}

// bc family condition evaluation shared by bc/bclr/bcctr: true = branch taken
static inline bool bc_taken(ppc_t *c, uint32_t insn, bool allow_ctr) {
    uint32_t bo = P_RD(insn), bi = P_RA(insn);
    bool ctr_ok = true;
    if (allow_ctr && !(bo & 4)) { // decrement CTR unless BO[2]
        c->ctr--;
        ctr_ok = (c->ctr != 0) ^ ((bo >> 1) & 1); // BO[3]: branch on CTR==0
    }
    bool cond_ok = (bo & 16) || (((c->cr >> (31 - bi)) & 1) == ((bo >> 3) & 1));
    return ctr_ok && cond_ok;
}

// bc: conditional relative branch; returns the new next-instruction address
static inline uint32_t gen_bc(ppc_t *c, uint32_t insn, uint32_t pc) {
    bool taken = bc_taken(c, insn, true);
    if (insn & 1)
        c->lr = pc;
    if (taken) {
        int32_t bd = (int32_t)(int16_t)(insn & 0xFFFC);
        return (insn & 2) ? (uint32_t)bd : (pc - 4) + (uint32_t)bd;
    }
    return pc;
}

// mfspr/mtspr for LR and CTR only (benchmark subset)
static inline void gen_mspr(ppc_t *c, uint32_t insn, bool to_spr) {
    uint32_t spr = ((insn >> 16) & 31) | (((insn >> 11) & 31) << 5);
    uint32_t *reg = (spr == 8) ? &c->lr : (spr == 9) ? &c->ctr : NULL;
    if (!reg) {
        ppc_illegal(c, insn);
        return;
    }
    if (to_spr)
        *reg = c->gpr[P_RD(insn)];
    else
        c->gpr[P_RD(insn)] = *reg;
}

// === Layer 2: X-macro instruction lists ===
// Each entry: X(key, name, statement).  Statements execute with `c` (ppc_t *),
// `insn` (uint32_t) and `pc` (uint32_t, CIA+4, assignable) in scope.

// Secondary decode for primary opcode 31, keyed on (XO << 1) | Rc so Rc is a
// constant at every stamp.  XO-form arithmetic entries assume OE=0.
#define PPC_OP31(X) \
    X(0,    cmpw,     gen_cmp(c, insn, false, false)) \
    X(64,   cmplw,    gen_cmp(c, insn, false, true)) \
    X(532,  add,      gen_xo_arith(c, insn, K_ADD, false)) \
    X(533,  add_rc,   gen_xo_arith(c, insn, K_ADD, true)) \
    X(80,   subf,     gen_xo_arith(c, insn, K_SUBF, false)) \
    X(81,   subf_rc,  gen_xo_arith(c, insn, K_SUBF, true)) \
    X(470,  mullw,    gen_xo_arith(c, insn, K_MULLW, false)) \
    X(471,  mullw_rc, gen_xo_arith(c, insn, K_MULLW, true)) \
    X(56,   and,      gen_x_logic(c, insn, K_AND, false)) \
    X(57,   and_rc,   gen_x_logic(c, insn, K_AND, true)) \
    X(888,  or,       gen_x_logic(c, insn, K_OR, false)) \
    X(889,  or_rc,    gen_x_logic(c, insn, K_OR, true)) \
    X(632,  xor,      gen_x_logic(c, insn, K_XOR, false)) \
    X(633,  xor_rc,   gen_x_logic(c, insn, K_XOR, true)) \
    X(48,   slw,      gen_x_logic(c, insn, K_SLW, false)) \
    X(1072, srw,      gen_x_logic(c, insn, K_SRW, false)) \
    X(1648, srawi,    gen_srawi(c, insn, false)) \
    X(1649, srawi_rc, gen_srawi(c, insn, true)) \
    X(46,   lwzx,     gen_load_x(c, insn, 4)) \
    X(558,  lhzx,     gen_load_x(c, insn, 2)) \
    X(174,  lbzx,     gen_load_x(c, insn, 1)) \
    X(302,  stwx,     gen_store_x(c, insn, 4)) \
    X(814,  sthx,     gen_store_x(c, insn, 2)) \
    X(430,  stbx,     gen_store_x(c, insn, 1)) \
    X(678,  mfspr,    gen_mspr(c, insn, false)) \
    X(934,  mtspr,    gen_mspr(c, insn, true))

// Secondary dispatch for opcode 31 as a nested switch in BOTH backends: the
// XO space is sparse, so a real core would use a switch (or packed table)
// here either way; keeping it identical isolates the primary-dispatch
// architecture as the only variable under test.
static inline void do_op31(ppc_t *c, uint32_t insn) {
    switch (insn & 0x7FF) {
#define X31_CASE(key, name, stmt) \
    case key: { \
        stmt; \
    } break;
        PPC_OP31(X31_CASE)
#undef X31_CASE
    default:
        ppc_illegal(c, insn);
    }
}

// Secondary dispatch for opcode 19 (branch-to-register forms)
static inline uint32_t do_op19(ppc_t *c, uint32_t insn, uint32_t pc) {
    switch ((insn >> 1) & 0x3FF) {
    case 16: { // bclr
        bool taken = bc_taken(c, insn, true);
        uint32_t target = c->lr & ~3u;
        if (insn & 1)
            c->lr = pc;
        return taken ? target : pc;
    }
    case 528: { // bcctr (CTR decrement forms are invalid; condition only)
        bool taken = bc_taken(c, insn, false);
        if (insn & 1)
            c->lr = pc;
        return taken ? (c->ctr & ~3u) : pc;
    }
    default:
        ppc_illegal(c, insn);
        return pc;
    }
}

// Primary-opcode instruction list (the single source of truth both backends
// and any future disassembler stamp from)
#define PPC_PRIMARY(X) \
    X(7,  mulli,   gen_mulli(c, insn)) \
    X(10, cmplwi,  gen_cmp(c, insn, true, true)) \
    X(11, cmpwi,   gen_cmp(c, insn, true, false)) \
    X(14, addi,    gen_addi(c, insn, false)) \
    X(15, addis,   gen_addi(c, insn, true)) \
    X(16, bc,      pc = gen_bc(c, insn, pc)) \
    X(18, b,       pc = gen_b(c, insn, pc)) \
    X(19, grp19,   pc = do_op19(c, insn, pc)) \
    X(21, rlwinm,  gen_rlwinm(c, insn)) \
    X(24, ori,     gen_logic_d(c, insn, K_OR, false, false)) \
    X(25, oris,    gen_logic_d(c, insn, K_OR, true, false)) \
    X(26, xori,    gen_logic_d(c, insn, K_XOR, false, false)) \
    X(28, andi_rc, gen_logic_d(c, insn, K_AND, false, true)) \
    X(31, grp31,   do_op31(c, insn)) \
    X(32, lwz,     gen_load_d(c, insn, 4, false)) \
    X(33, lwzu,    gen_load_d(c, insn, 4, true)) \
    X(34, lbz,     gen_load_d(c, insn, 1, false)) \
    X(36, stw,     gen_store_d(c, insn, 4, false)) \
    X(37, stwu,    gen_store_d(c, insn, 4, true)) \
    X(38, stb,     gen_store_d(c, insn, 1, false)) \
    X(40, lhz,     gen_load_d(c, insn, 2, false)) \
    X(44, sth,     gen_store_d(c, insn, 2, false))

#endif // PPC_OPS_H
