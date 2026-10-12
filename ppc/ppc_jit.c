// SPDX-License-Identifier: MIT
// ppc_jit.c
// Backend G: a WebAssembly JIT.  Each guest code page (1024 instructions)
// is translated at run time into one wasm function, instantiated
// synchronously against this module's own memory and function table, and
// called through an ordinary C function pointer.
//
// Inside a translated page:
//   - every guest register lives in a wasm local, loaded at entry and
//     stored at exit, so the engine register-allocates them;
//   - the page's basic blocks are laid out in address order inside one
//     loop, entered through a br_table on the block index; fallthrough is
//     the natural nesting order and a branch to a block in the same page is
//     `idx = j; br $loop` -- no exit, no call;
//   - each block checks the remaining budget against its (known) length
//     first and leaves with the block's start pc when it would overrun,
//     so a run still executes exactly `budget` instructions (the switch
//     interpreter finishes the tail);
//   - memory accesses are inlined with the mem_fastpath.h structure: the
//     masked address, the SoA entry, the page-crossing test, the byte
//     swap (as rotates: wasm has no bswap), the value-trap test on stores,
//     and call_indirect into the C slow paths;
//   - a branch out of the page, an indirect branch (bclr/bcctr) or an
//     instruction outside the subset leaves with the pc; the C dispatcher
//     then enters the target page.
//
// Translated pages are cached across runs and revalidated against a copy of
// their bytes when a run first enters them (the benchmark's stand-in for
// write invalidation).  Compile time is charged to the run that compiles.
//
// wasm builds only: native builds have no runtime code generation here.

#include "ppc_ops.h"

#include "mem_env.h"

#ifdef __EMSCRIPTEN__

#include <emscripten.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// === Instantiation (JS side) ===
// Compile and instantiate synchronously (allowed for any size off the
// browser main thread, and in Node), importing this module's memory and
// function table; append the export to the table and return its index,
// which is the C function pointer value.
// clang-format off
EM_JS(uint32_t, jit_instantiate, (const uint8_t *p, uint32_t n, uint32_t slot), {
    var T = globalThis.__jitT || (globalThis.__jitT = { mod: 0, inst: 0, n: 0, dump: 0 });
    var table = wasmExports['__indirect_function_table'];
    var bytes = HEAPU8.slice(p, p + n);
    // JIT_DUMP=<dir> (Node): keep each module, for the compile measurements
    if (typeof process == 'object' && process.env && process.env.JIT_DUMP)
        require('fs').writeFileSync(process.env.JIT_DUMP + '/page' + (T.dump++) + '.wasm', bytes);
    var t0 = performance.now();
    var mod = new WebAssembly.Module(bytes);
    var t1 = performance.now();
    var inst = new WebAssembly.Instance(mod, { e: { m: wasmMemory, t: table } });
    var t2 = performance.now();
    T.mod += t1 - t0;
    T.inst += t2 - t1;
    T.n++;
    var idx = slot ? slot : table.grow(1); // a freed slot, or a new one
    table.set(idx, inst.exports.f);
    return idx;
});
// Free a translation's slot: the table no longer holds its function, so the
// engine can collect the module once nothing else does.
EM_JS(void, jit_release, (uint32_t slot), {
    wasmExports['__indirect_function_table'].set(slot, null);
});
EM_JS(double, jit_t_module, (void), { return globalThis.__jitT ? globalThis.__jitT.mod : 0; });
EM_JS(double, jit_t_instance, (void), { return globalThis.__jitT ? globalThis.__jitT.inst : 0; });
// clang-format on

// === Byte-code writer ===
static uint8_t *g_buf;
static uint32_t g_len, g_cap;

static void put(uint8_t b) {
    if (g_len == g_cap) {
        g_cap = g_cap ? g_cap * 2 : (256u << 10);
        g_buf = realloc(g_buf, g_cap);
        if (!g_buf) {
            fprintf(stderr, "FATAL: jit buffer allocation failed\n");
            exit(1);
        }
    }
    g_buf[g_len++] = b;
}
static void uleb(uint32_t v) {
    do {
        uint8_t b = v & 0x7F;
        v >>= 7;
        put(v ? (b | 0x80) : b);
    } while (v);
}
static void sleb(int32_t v) {
    for (;;) {
        uint8_t b = v & 0x7F;
        v >>= 7;
        if ((v == 0 && !(b & 0x40)) || (v == -1 && (b & 0x40))) {
            put(b);
            return;
        }
        put(b | 0x80);
    }
}

// wasm opcodes used here
enum {
    W_BLOCK = 0x02, W_LOOP = 0x03, W_IF = 0x04, W_ELSE = 0x05, W_END = 0x0B, W_BR = 0x0C, W_BR_TABLE = 0x0E,
    W_CALL_INDIRECT = 0x11, W_SELECT = 0x1B, W_LGET = 0x20, W_LSET = 0x21, W_LTEE = 0x22,
    W_LOAD = 0x28, W_LOAD8U = 0x2D, W_LOAD16U = 0x2F, W_STORE = 0x36, W_STORE8 = 0x3A, W_STORE16 = 0x3B,
    W_CONST = 0x41, W_EQZ = 0x45, W_NE = 0x47, W_LT_S = 0x48, W_LT_U = 0x49, W_GT_S = 0x4A, W_GT_U = 0x4B,
    W_ADD = 0x6A, W_SUB = 0x6B, W_MUL = 0x6C, W_AND = 0x71, W_OR = 0x72, W_XOR = 0x73, W_SHL = 0x74,
    W_SHR_S = 0x75, W_SHR_U = 0x76, W_ROTL = 0x77, W_ROTR = 0x78,
};
#define BT_VOID 0x40
#define BT_I32  0x7F

static void op(uint8_t o) {
    put(o);
}
static void iconst(uint32_t v) {
    put(W_CONST);
    sleb((int32_t)v);
}
static void lget(uint32_t l) {
    put(W_LGET);
    uleb(l);
}
static void lset(uint32_t l) {
    put(W_LSET);
    uleb(l);
}
static void ltee(uint32_t l) {
    put(W_LTEE);
    uleb(l);
}
static void mem(uint8_t o, uint32_t align, uint32_t off) {
    put(o);
    uleb(align);
    uleb(off);
}

// Function types of the module (index order of the type section)
enum { TY_PAGE, TY_RD, TY_WR, TY_TRAP }; // (i32,i32,i32)->i32, (i32)->i32, (i32,i32)->(), (i32,i32,i32)->()

// Locals: params 0..2, then the guest state and temporaries
enum {
    L_C = 0, L_BUDGET = 1, L_ENTRY = 2,
    L_GPR = 3, // 32 of them
    L_CR = L_GPR + 32, L_CTR, L_LR, L_CA, L_IDX, L_NPC, L_RTAB, L_WTAB, L_AMASK, L_T0, L_T1, L_T2, L_T3, L_EA, L_EPOCH,
    L_COUNT
};
#define NLOCALS (L_COUNT - 3)
#define GPR(r) (L_GPR + (r))

// === Translation state for one page ===
#define PAGE_INSNS 1024
#define NONE16     0xFFFFu

typedef struct jit_page {
    uint32_t fn; // table index of the translation (0: none)
    uint32_t gen; // run generation it was last validated in
    uint16_t blk_of[PAGE_INSNS]; // block index of a leader, NONE16 otherwise
    uint8_t extra[PAGE_INSNS]; // leaders added for entries seen at run time
    uint8_t bytes[MEM_PAGE_SIZE]; // the page as translated (validation copy)
    uint32_t cooling; // entries left to run interpreted after a code write (0: translate on entry)
    uint32_t pending; // chaos mode: entries left before the translation "arrives" (0: none pending)
} jit_page_t;

// Chaos mode (seed != 0): a translation becomes available only after a
// pseudo-random number of page entries (1..64), which run interpreted
// meanwhile — the order in which an asynchronous pipeline would deliver
// translations is outside the guest's control, and the guest state must not
// depend on it.  The verification runs every kernel under several seeds.
uint32_t g_jit_chaos_seed;
static uint32_t g_chaos_state;

// After a write invalidates a page's translation, the page is entered this
// many times in the interpreter (JIT_COOL_SLICE instructions each) before
// it is translated again; another write restarts the count.  A page that is
// being rewritten is not recompiled for every write (proposal §4.2).
#define JIT_COOL_ENTRIES 64
#define JIT_COOL_SLICE   256

// Bumped by every invalidation: a translation that stores through the C
// slow path (JIT_SLOW_CALL) compares it after the call and leaves if the
// store invalidated code (possibly its own page).
uint32_t g_code_epoch;

static jit_page_t **g_pages; // per guest page (lazily allocated), of the current slow-path mode
static jit_page_t **g_pages_by_mode[2]; // each slow-path mode keeps its own translations
static uint32_t g_npages;
static uint32_t g_gen;

static uint32_t g_page_base; // guest address of the page being translated
static uint32_t g_depth; // structured-control depth of the current point, relative to the block's code
static uint32_t g_nblocks; // blocks in the page
static uint32_t g_cur_block; // block being emitted
static uint32_t g_cur_addr; // guest address of the instruction being emitted
static uint32_t g_blk_left; // instructions of the block not yet executed, the current one included

// Why the page function left (written by the function, read by the
// dispatcher): JIT_EXIT_PC = continue at the stored pc; JIT_EXIT_DEOPT = the
// instruction at the stored pc missed the memory fast path and must run in
// the interpreter, with everything before it committed.
enum { JIT_EXIT_PC = 0, JIT_EXIT_DEOPT = 1 };
uint32_t g_jit_reason;

// How a translated access that misses the fast path is handled
// (translation-time switch):
//   JIT_SLOW_DEOPT: leave at that instruction (the proposal's Phase B);
//   JIT_SLOW_CALL:  write every register back to the state, then call the
//                   C slow path from the translation (Phase D's variant).
// Both leave the state exact at the access.
enum { JIT_SLOW_DEOPT = 0, JIT_SLOW_CALL = 1 };
int g_jit_slow_mode = JIT_SLOW_DEOPT;

// Depths of the two branch targets from the code of block i (no `if` open):
// the still-open blocks $b(i+1)..$b(n-1), then $loop, then $out.
static uint32_t depth_loop(void) {
    return g_nblocks - 1 - g_cur_block + g_depth;
}
static uint32_t depth_out(void) {
    return depth_loop() + 1;
}

// Leave the page function with next pc `npc` (on the stack: nothing)
static void emit_exit_const(uint32_t npc) {
    iconst(npc);
    lset(L_NPC);
    op(W_BR);
    uleb(depth_out());
}
// Leave with the next pc taken from the stack
static void emit_exit_stack(void) {
    lset(L_NPC);
    op(W_BR);
    uleb(depth_out());
}

static uint16_t *g_blk_of; // the page's leader -> block map, during translation

// Go to guest address `target`: a block of this page, or leave
static void emit_goto(uint32_t target) {
    if ((target & ~(uint32_t)PAGE_MASK) == g_page_base && !(target & 3)) {
        uint16_t b = g_blk_of[(target & PAGE_MASK) >> 2];
        if (b != NONE16) {
            iconst(b);
            lset(L_IDX);
            op(W_BR);
            uleb(depth_loop());
            return;
        }
    }
    emit_exit_const(target);
}

// Leave at the current instruction for the interpreter: give back the budget
// of the block's instructions not executed (the current one included) and
// say why.  The locals hold exactly the state before this instruction.
static void emit_deopt(void) {
    lget(L_BUDGET);
    iconst(g_blk_left);
    op(W_ADD);
    lset(L_BUDGET);
    iconst((uint32_t)(uintptr_t)&g_jit_reason);
    iconst(JIT_EXIT_DEOPT);
    mem(W_STORE, 2, 0);
    emit_exit_const(g_cur_addr);
}

// Write every register local back to the state (before a call into C that
// may observe it)
static void emit_spill(void) {
    for (int r = 0; r < 32; r++) {
        lget(L_C);
        lget(GPR(r));
        mem(W_STORE, 2, (uint32_t)offsetof(ppc_t, gpr) + 4u * r);
    }
    lget(L_C); lget(L_CR); mem(W_STORE, 2, offsetof(ppc_t, cr));
    lget(L_C); lget(L_CTR); mem(W_STORE, 2, offsetof(ppc_t, ctr));
    lget(L_C); lget(L_LR); mem(W_STORE, 2, offsetof(ppc_t, lr));
    lget(L_C); lget(L_CA); mem(W_STORE, 2, offsetof(ppc_t, xer_ca));
}

// Big-endian fixups of a value on the stack (wasm memory is little-endian)
static void emit_bswap32(void) {
    ltee(L_T2);
    iconst(8);
    op(W_ROTL);
    iconst(0x00FF00FFu);
    op(W_AND);
    lget(L_T2);
    iconst(8);
    op(W_ROTR);
    iconst(0xFF00FF00u);
    op(W_AND);
    op(W_OR);
}
static void emit_bswap16(void) { // value in the low 16 bits, upper bits zero
    ltee(L_T2);
    iconst(8);
    op(W_SHL);
    lget(L_T2);
    iconst(8);
    op(W_SHR_U);
    op(W_OR);
    iconst(0xFFFF);
    op(W_AND);
}

// The slow-path functions, as table indices (their C function pointers)
static uint32_t fp(void *f) {
    return (uint32_t)(uintptr_t)f;
}

// EA on the stack -> loaded value on the stack (mem_fastpath.h structure)
static void emit_load(int width) {
    lget(L_AMASK);
    op(W_AND);
    lset(L_T0); // masked
    lget(L_RTAB);
    lget(L_T0);
    iconst(PAGE_SHIFT);
    op(W_SHR_U);
    iconst(2);
    op(W_SHL);
    op(W_ADD);
    mem(W_LOAD, 2, 0);
    ltee(L_T1); // base
    op(W_EQZ);
    if (width > 1) {
        lget(L_T0);
        iconst(PAGE_MASK);
        op(W_AND);
        iconst(MEM_PAGE_SIZE - width);
        op(W_GT_U);
        op(W_OR);
    }
    op(W_IF);
    put(BT_I32);
    g_depth++;
    if (g_jit_slow_mode == JIT_SLOW_DEOPT) {
        emit_deopt(); // (stack-polymorphic after the br: the i32 result is satisfied)
    } else {
        emit_spill();
        lget(L_T0);
        iconst(width == 4   ? fp(memory_read_uint32_slow)
               : width == 2 ? fp(memory_read_uint16_slow)
                            : fp(memory_read_uint8_slow));
        op(W_CALL_INDIRECT);
        uleb(TY_RD);
        put(0);
    }
    op(W_ELSE);
    lget(L_T1);
    lget(L_T0);
    op(W_ADD);
    if (width == 4) {
        mem(W_LOAD, 0, 0);
        emit_bswap32();
    } else if (width == 2) {
        mem(W_LOAD16U, 0, 0);
        emit_bswap16();
    } else {
        mem(W_LOAD8U, 0, 0);
    }
    op(W_END);
    g_depth--;
}

// EA in L_T0 (unmasked), value in L_T3 -> store
static void emit_store(int width) {
    lget(L_T0);
    lget(L_AMASK);
    op(W_AND);
    lset(L_T0);
    // The value trap: one load and a branch on the fast path, as in C
    iconst(fp(&g_value_trap_active));
    mem(W_LOAD, 2, 0);
    op(W_IF);
    put(BT_VOID);
    g_depth++;
    lget(L_T0);
    lget(L_T3);
    iconst((uint32_t)width);
    iconst(fp(value_trap_check));
    op(W_CALL_INDIRECT);
    uleb(TY_TRAP);
    put(0);
    op(W_END);
    g_depth--;
    lget(L_WTAB);
    lget(L_T0);
    iconst(PAGE_SHIFT);
    op(W_SHR_U);
    iconst(2);
    op(W_SHL);
    op(W_ADD);
    mem(W_LOAD, 2, 0);
    ltee(L_T1);
    op(W_EQZ);
    if (width > 1) {
        lget(L_T0);
        iconst(PAGE_MASK);
        op(W_AND);
        iconst(MEM_PAGE_SIZE - width);
        op(W_GT_U);
        op(W_OR);
    }
    op(W_IF);
    put(BT_VOID);
    g_depth++;
    if (g_jit_slow_mode == JIT_SLOW_DEOPT) {
        emit_deopt();
        op(W_ELSE);
        goto fast_store;
    }
    emit_spill();
    lget(L_T0);
    lget(L_T3);
    if (width == 2) {
        iconst(0xFFFF);
        op(W_AND);
    } else if (width == 1) {
        iconst(0xFF);
        op(W_AND);
    }
    iconst(width == 4   ? fp(memory_write_uint32_slow)
           : width == 2 ? fp(memory_write_uint16_slow)
                        : fp(memory_write_uint8_slow));
    op(W_CALL_INDIRECT);
    uleb(TY_WR);
    put(0);
    // The store may have invalidated code, this page's included: leave
    // after it (it completed: its slot is spent) rather than run on.
    iconst(fp(&g_code_epoch));
    mem(W_LOAD, 2, 0);
    lget(L_EPOCH);
    op(W_NE);
    op(W_IF);
    put(BT_VOID);
    g_depth++;
    lget(L_BUDGET);
    iconst(g_blk_left - 1);
    op(W_ADD);
    lset(L_BUDGET);
    emit_exit_const(g_cur_addr + 4);
    op(W_END);
    g_depth--;
    op(W_ELSE);
fast_store:
    lget(L_T1);
    lget(L_T0);
    op(W_ADD);
    lget(L_T3);
    if (width == 4) {
        emit_bswap32();
        mem(W_STORE, 0, 0);
    } else if (width == 2) {
        iconst(0xFFFF);
        op(W_AND);
        emit_bswap16();
        mem(W_STORE16, 0, 0);
    } else {
        mem(W_STORE8, 0, 0);
    }
    op(W_END);
    g_depth--;
}

// (rA|0) on the stack
static void emit_ra0(uint32_t ra) {
    if (ra)
        lget(GPR(ra));
    else
        iconst(0);
}

// CR field `crf` = LT/GT/EQ of the signed (or unsigned) compare of L_T0, L_T1
static void emit_set_crf_cmp(uint32_t crf, bool is_unsigned) {
    uint32_t sh = 28 - 4 * crf;
    lget(L_CR);
    iconst(~(0xFu << sh));
    op(W_AND);
    iconst(8);
    iconst(4);
    iconst(2);
    lget(L_T0);
    lget(L_T1);
    op(is_unsigned ? W_GT_U : W_GT_S);
    op(W_SELECT);
    lget(L_T0);
    lget(L_T1);
    op(is_unsigned ? W_LT_U : W_LT_S);
    op(W_SELECT);
    iconst(sh);
    op(W_SHL);
    op(W_OR);
    lset(L_CR);
}

// CR0 from the signed sign of local `l` (record0)
static void emit_record0(uint32_t l) {
    lget(l);
    lset(L_T0);
    iconst(0);
    lset(L_T1);
    emit_set_crf_cmp(0, false);
}

// The bc condition (bc_taken) on the stack: 1 = taken.  CTR is decremented
// here when BO says so.
static void emit_bc_taken(uint32_t insn, bool allow_ctr) {
    uint32_t bo = P_RD(insn), bi = P_RA(insn);
    bool have = false;
    if (allow_ctr && !(bo & 4)) {
        lget(L_CTR);
        iconst(1);
        op(W_SUB);
        ltee(L_CTR);
        if (bo & 2)
            op(W_EQZ); // branch on CTR == 0
        else {
            iconst(0);
            op(W_NE);
        }
        have = true;
    }
    if (!(bo & 16)) {
        lget(L_CR);
        iconst(31 - bi);
        op(W_SHR_U);
        iconst(1);
        op(W_AND);
        if (!((bo >> 3) & 1))
            op(W_EQZ); // branch if the bit is clear
        if (have)
            op(W_AND);
        have = true;
    }
    if (!have)
        iconst(1);
}

// Translate one instruction at guest address `addr`.  Returns false when it
// is outside the subset (the caller has made it its own empty block).
static void emit_insn(uint32_t addr, uint32_t insn) {
    uint32_t rd = P_RD(insn), ra = P_RA(insn), rb = P_RB(insn);
    uint32_t simm = (uint32_t)P_SIMM(insn), uimm = P_UIMM(insn);
    switch (insn >> 26) {
    case 7: // mulli
        lget(GPR(ra));
        iconst(simm);
        op(W_MUL);
        lset(GPR(rd));
        return;
    case 10: // cmplwi
        lget(GPR(ra));
        lset(L_T0);
        iconst(uimm);
        lset(L_T1);
        emit_set_crf_cmp((insn >> 23) & 7, true);
        return;
    case 11: // cmpwi
        lget(GPR(ra));
        lset(L_T0);
        iconst(simm);
        lset(L_T1);
        emit_set_crf_cmp((insn >> 23) & 7, false);
        return;
    case 14: // addi
        emit_ra0(ra);
        iconst(simm);
        op(W_ADD);
        lset(GPR(rd));
        return;
    case 15: // addis
        emit_ra0(ra);
        iconst(simm << 16);
        op(W_ADD);
        lset(GPR(rd));
        return;
    case 16: { // bc
        uint32_t target = (insn & 2) ? (uint32_t)(int32_t)(int16_t)(insn & 0xFFFC)
                                     : addr + (uint32_t)(int32_t)(int16_t)(insn & 0xFFFC);
        emit_bc_taken(insn, true);
        if (insn & 1) {
            iconst(addr + 4);
            lset(L_LR);
        }
        op(W_IF);
        put(BT_VOID);
        g_depth++;
        emit_goto(target);
        op(W_END);
        g_depth--;
        return; // not taken: falls through to the next block
    }
    case 18: { // b
        int32_t li = ((int32_t)(insn << 6) >> 6) & ~3;
        uint32_t target = (insn & 2) ? (uint32_t)li : addr + (uint32_t)li;
        if (insn & 1) {
            iconst(addr + 4);
            lset(L_LR);
        }
        emit_goto(target);
        return;
    }
    case 19: { // bclr / bcctr: indirect, leave with the target
        uint32_t xo = (insn >> 1) & 0x3FF;
        bool lr_form = (xo == 16);
        lget(lr_form ? L_LR : L_CTR);
        iconst(~3u);
        op(W_AND);
        lset(L_T3); // target, read before LK updates LR
        emit_bc_taken(insn, lr_form);
        if (insn & 1) {
            iconst(addr + 4);
            lset(L_LR);
        }
        op(W_IF);
        put(BT_VOID);
        g_depth++;
        lget(L_T3);
        emit_exit_stack();
        op(W_END);
        g_depth--;
        return;
    }
    case 21: { // rlwinm
        uint32_t mb = (insn >> 6) & 31, me = (insn >> 1) & 31;
        lget(GPR(rd));
        iconst(rb);
        op(W_ROTL);
        iconst(ppc_mask(mb, me));
        op(W_AND);
        lset(GPR(ra));
        if (insn & 1)
            emit_record0(GPR(ra));
        return;
    }
    case 24: // ori
    case 25: // oris
    case 26: // xori
    case 28: { // andi.
        uint32_t imm = (insn >> 26) == 25 ? uimm << 16 : uimm;
        lget(GPR(rd));
        iconst(imm);
        op((insn >> 26) == 26 ? W_XOR : (insn >> 26) == 28 ? W_AND : W_OR);
        lset(GPR(ra));
        if ((insn >> 26) == 28)
            emit_record0(GPR(ra));
        return;
    }
    case 31: {
        uint32_t key = insn & 0x7FF;
        bool rc = key & 1;
        switch (key & ~1u) {
        case 0: // cmpw
        case 64: // cmplw
            if (rc)
                break;
            lget(GPR(ra));
            lset(L_T0);
            lget(GPR(rb));
            lset(L_T1);
            emit_set_crf_cmp((insn >> 23) & 7, key == 64);
            return;
        case 532: // add
        case 80: // subf
        case 470: // mullw
            if ((key & ~1u) == 80) {
                lget(GPR(rb));
                lget(GPR(ra));
                op(W_SUB);
            } else {
                lget(GPR(ra));
                lget(GPR(rb));
                op((key & ~1u) == 532 ? W_ADD : W_MUL);
            }
            lset(GPR(rd));
            if (rc)
                emit_record0(GPR(rd));
            return;
        case 56: // and
        case 888: // or
        case 632: // xor
            lget(GPR(rd));
            lget(GPR(rb));
            op((key & ~1u) == 56 ? W_AND : (key & ~1u) == 888 ? W_OR : W_XOR);
            lset(GPR(ra));
            if (rc)
                emit_record0(GPR(ra));
            return;
        case 48: // slw
        case 1072: // srw
            if (rc)
                break;
            iconst(0);
            lget(GPR(rd));
            lget(GPR(rb));
            op(key == 48 ? W_SHL : W_SHR_U); // wasm masks the count to 5 bits
            lget(GPR(rb));
            iconst(0x20);
            op(W_AND);
            op(W_SELECT); // b & 0x20 ? 0 : shifted
            lset(GPR(ra));
            return;
        case 1648: { // srawi
            uint32_t sh = rb;
            lget(GPR(rd));
            lset(L_T3);
            if (sh) {
                lget(L_T3);
                iconst(0);
                op(W_LT_S);
                lget(L_T3);
                iconst((1u << sh) - 1);
                op(W_AND);
                iconst(0);
                op(W_NE);
                op(W_AND);
            } else {
                iconst(0);
            }
            lset(L_CA);
            lget(L_T3);
            iconst(sh);
            op(W_SHR_S);
            lset(GPR(ra));
            if (rc)
                emit_record0(GPR(ra));
            return;
        }
        case 46: // lwzx
        case 558: // lhzx
        case 174: // lbzx
            if (rc)
                break;
            emit_ra0(ra);
            lget(GPR(rb));
            op(W_ADD);
            emit_load(key == 46 ? 4 : key == 558 ? 2 : 1);
            lset(GPR(rd));
            return;
        case 302: // stwx
        case 814: // sthx
        case 430: // stbx
            if (rc)
                break;
            emit_ra0(ra);
            lget(GPR(rb));
            op(W_ADD);
            lset(L_T0);
            lget(GPR(rd));
            lset(L_T3);
            emit_store(key == 302 ? 4 : key == 814 ? 2 : 1);
            return;
        case 678: // mfspr
        case 934: { // mtspr
            uint32_t spr = ((insn >> 16) & 31) | (((insn >> 11) & 31) << 5);
            if (rc || (spr != 8 && spr != 9))
                break;
            uint32_t reg = spr == 8 ? L_LR : L_CTR;
            if (key == 678) {
                lget(reg);
                lset(GPR(rd));
            } else {
                lget(GPR(rd));
                lset(reg);
            }
            return;
        }
        default:
            break;
        }
        break;
    }
    case 32: // lwz
    case 33: // lwzu
    case 34: // lbz
    case 40: // lhz
    {
        uint32_t o = insn >> 26;
        bool update = (o == 33);
        if (update)
            lget(GPR(ra));
        else
            emit_ra0(ra);
        iconst(simm);
        op(W_ADD);
        if (update) {
            ltee(L_T3);
            emit_load(4);
            lset(GPR(rd));
            lget(L_T3);
            lset(GPR(ra)); // update after the load
        } else {
            emit_load(o == 32 ? 4 : o == 40 ? 2 : 1);
            lset(GPR(rd));
        }
        return;
    }
    case 36: // stw
    case 37: // stwu
    case 38: // stb
    case 44: { // sth
        uint32_t o = insn >> 26;
        if (o == 37)
            lget(GPR(ra));
        else
            emit_ra0(ra);
        iconst(simm);
        op(W_ADD);
        ltee(L_T0);
        lset(L_EA); // the unmasked EA, for the update form (emit_store masks L_T0 in place)
        lget(GPR(rd));
        lset(L_T3);
        emit_store(o == 37 || o == 36 ? 4 : o == 44 ? 2 : 1);
        if (o == 37) { // update after the store: rS == rA stores the old value
            lget(L_EA);
            lset(GPR(ra));
        }
        return;
    }
    default:
        break;
    }
    // Outside the subset: never reached (translatable() filters these)
    fprintf(stderr, "FATAL: jit emit of untranslatable insn %08x at %08x\n", insn, addr);
    exit(1);
}

// Is this word one emit_insn handles?
static bool translatable(uint32_t insn) {
    uint32_t key = insn & 0x7FF;
    switch (insn >> 26) {
    case 7: case 10: case 11: case 14: case 15: case 16: case 18: case 21:
    case 24: case 25: case 26: case 28: case 32: case 33: case 34: case 36: case 37: case 38: case 40: case 44:
        return true;
    case 19: {
        uint32_t xo = (insn >> 1) & 0x3FF;
        return xo == 16 || xo == 528;
    }
    case 31:
        switch (key) {
        case 0: case 64: case 532: case 533: case 80: case 81: case 470: case 471: case 56: case 57:
        case 888: case 889: case 632: case 633: case 48: case 1072: case 1648: case 1649:
        case 46: case 558: case 174: case 302: case 814: case 430:
            return true;
        case 678: case 934: {
            uint32_t spr = ((insn >> 16) & 31) | (((insn >> 11) & 31) << 5);
            return spr == 8 || spr == 9;
        }
        default:
            return false;
        }
    default:
        return false;
    }
}

static bool is_branch(uint32_t insn) {
    uint32_t o = insn >> 26;
    return o == 16 || o == 18 || o == 19;
}

// Static in-page target of a direct branch, or ~0
static uint32_t direct_target(uint32_t addr, uint32_t insn) {
    uint32_t o = insn >> 26;
    if (o == 16)
        return (insn & 2) ? (uint32_t)(int32_t)(int16_t)(insn & 0xFFFC)
                          : addr + (uint32_t)(int32_t)(int16_t)(insn & 0xFFFC);
    if (o == 18) {
        int32_t li = ((int32_t)(insn << 6) >> 6) & ~3;
        return (insn & 2) ? (uint32_t)li : addr + (uint32_t)li;
    }
    return ~0u;
}

static void section(uint8_t id, uint32_t start_len) {
    // Patch helper: sections are emitted into the main buffer with a fixed
    // 5-byte LEB size placeholder, filled here.
    uint32_t size = g_len - start_len - 5;
    g_buf[start_len - 1] = id; // id byte precedes the placeholder
    for (int i = 0; i < 5; i++) {
        uint8_t b = (size >> (7 * i)) & 0x7F;
        g_buf[start_len + i] = (i < 4) ? (b | 0x80) : b;
    }
}
static uint32_t open_section(void) {
    put(0); // id, patched
    uint32_t at = g_len;
    for (int i = 0; i < 5; i++)
        put(0);
    return at;
}

static void put_name(const char *s) {
    uleb((uint32_t)strlen(s));
    while (*s)
        put((uint8_t)*s++);
}

// Translate the page at `base` (its words in `words`) with the leaders in
// `leader`; fills blk_of.  Returns the module length in g_buf.
static void translate(uint32_t base, const uint32_t *words, const uint8_t *leader, uint16_t *blk_of) {
    g_len = 0;
    g_page_base = base;
    g_blk_of = blk_of;
    // Blocks: [start, end) instruction ranges.  An untranslatable word is a
    // block of its own with no executable instructions: it only leaves.
    static uint16_t bstart[PAGE_INSNS + 1];
    uint32_t nb = 0;
    for (uint32_t i = 0; i < PAGE_INSNS; i++) {
        blk_of[i] = NONE16;
        if (leader[i])
            bstart[nb++] = (uint16_t)i;
    }
    for (uint32_t b = 0; b < nb; b++)
        blk_of[bstart[b]] = (uint16_t)b;
    bstart[nb] = PAGE_INSNS;
    g_nblocks = nb;

    // Header
    static const uint8_t hdr[8] = {0x00, 0x61, 0x73, 0x6D, 0x01, 0x00, 0x00, 0x00};
    for (int i = 0; i < 8; i++)
        put(hdr[i]);
    // Types
    uint32_t s = open_section();
    uleb(4);
    put(0x60); uleb(3); put(0x7F); put(0x7F); put(0x7F); uleb(1); put(0x7F); // TY_PAGE
    put(0x60); uleb(1); put(0x7F); uleb(1); put(0x7F); // TY_RD
    put(0x60); uleb(2); put(0x7F); put(0x7F); uleb(0); // TY_WR
    put(0x60); uleb(3); put(0x7F); put(0x7F); put(0x7F); uleb(0); // TY_TRAP
    section(1, s);
    // Imports: memory e.m, table e.t
    s = open_section();
    uleb(2);
    put_name("e"); put_name("m"); put(0x02); put(0x00); uleb(1);
    put_name("e"); put_name("t"); put(0x01); put(0x70); put(0x00); uleb(1);
    section(2, s);
    // Functions
    s = open_section();
    uleb(1);
    uleb(TY_PAGE);
    section(3, s);
    // Exports
    s = open_section();
    uleb(1);
    put_name("f");
    put(0x00);
    uleb(0);
    section(7, s);
    // Code
    s = open_section();
    uleb(1); // one function body
    uint32_t body = g_len; // its 5-byte size placeholder
    for (int i = 0; i < 5; i++)
        put(0);
    uleb(1);
    uleb(NLOCALS);
    put(0x7F);

    // Prologue: the guest state into locals, the memory-system pointers
    for (int r = 0; r < 32; r++) {
        lget(L_C);
        mem(W_LOAD, 2, (uint32_t)offsetof(ppc_t, gpr) + 4u * r);
        lset(GPR(r));
    }
    lget(L_C); mem(W_LOAD, 2, offsetof(ppc_t, cr)); lset(L_CR);
    lget(L_C); mem(W_LOAD, 2, offsetof(ppc_t, ctr)); lset(L_CTR);
    lget(L_C); mem(W_LOAD, 2, offsetof(ppc_t, lr)); lset(L_LR);
    lget(L_C); mem(W_LOAD, 2, offsetof(ppc_t, xer_ca)); lset(L_CA);
    iconst(fp(&g_active_read)); mem(W_LOAD, 2, 0); lset(L_RTAB);
    iconst(fp(&g_active_write)); mem(W_LOAD, 2, 0); lset(L_WTAB);
    iconst(fp(&g_address_mask)); mem(W_LOAD, 2, 0); lset(L_AMASK);
    iconst(fp(&g_code_epoch)); mem(W_LOAD, 2, 0); lset(L_EPOCH);
    lget(L_ENTRY);
    lset(L_IDX);

    // block $out; loop $loop; block $b(n-1) ... block $b0; br_table
    op(W_BLOCK); put(BT_VOID);
    op(W_LOOP); put(BT_VOID);
    for (uint32_t b = 0; b < nb; b++) {
        op(W_BLOCK);
        put(BT_VOID);
    }
    lget(L_IDX);
    op(W_BR_TABLE);
    uleb(nb);
    for (uint32_t b = 0; b < nb; b++)
        uleb(b);
    uleb(nb + 1); // default: $out (never taken)
    for (uint32_t b = 0; b < nb; b++) {
        op(W_END); // closes $b(b): what follows is block b's code
        g_cur_block = b;
        g_depth = 0;
        uint32_t i0 = bstart[b], i1 = bstart[b + 1];
        uint32_t addr0 = base + 4 * i0;
        if (!translatable(words[i0])) {
            emit_exit_const(addr0); // the C side runs it
            continue;
        }
        // The block's length: up to the next leader, its first branch, or
        // the first untranslatable word (which then must be a leader).
        uint32_t n = 0;
        for (uint32_t i = i0; i < i1; i++) {
            n++;
            if (is_branch(words[i]))
                break;
        }
        // Budget: leave at the block's start if it would overrun
        lget(L_BUDGET);
        iconst(n);
        op(W_LT_U);
        op(W_IF);
        put(BT_VOID);
        g_depth++;
        emit_exit_const(addr0);
        op(W_END);
        g_depth--;
        lget(L_BUDGET);
        iconst(n);
        op(W_SUB);
        lset(L_BUDGET);
        for (uint32_t i = i0; i < i0 + n; i++) {
            g_cur_addr = base + 4 * i;
            g_blk_left = i0 + n - i;
            emit_insn(base + 4 * i, words[i]);
        }
        // Fallthrough: into the next block's code; past the page: leave
        if (b == nb - 1)
            emit_exit_const(base + 4 * (i0 + n));
        else if (i0 + n != i1) // (cannot happen: a branch ends at a leader)
            emit_exit_const(base + 4 * (i0 + n));
    }
    op(W_END); // $loop (not reached by fallthrough: every path branches)
    op(W_END); // $out
    // Epilogue: the state back, pc, remaining budget
    for (int r = 0; r < 32; r++) {
        lget(L_C);
        lget(GPR(r));
        mem(W_STORE, 2, (uint32_t)offsetof(ppc_t, gpr) + 4u * r);
    }
    lget(L_C); lget(L_CR); mem(W_STORE, 2, offsetof(ppc_t, cr));
    lget(L_C); lget(L_CTR); mem(W_STORE, 2, offsetof(ppc_t, ctr));
    lget(L_C); lget(L_LR); mem(W_STORE, 2, offsetof(ppc_t, lr));
    lget(L_C); lget(L_CA); mem(W_STORE, 2, offsetof(ppc_t, xer_ca));
    lget(L_C); lget(L_NPC); mem(W_STORE, 2, offsetof(ppc_t, pc));
    lget(L_BUDGET);
    op(W_END); // function body
    // Patch the body size
    {
        uint32_t size = g_len - body - 5;
        for (int i = 0; i < 5; i++) {
            uint8_t bb = (size >> (7 * i)) & 0x7F;
            g_buf[body + i] = (i < 4) ? (bb | 0x80) : bb;
        }
    }
    section(10, s);
}

// === The dispatcher ===

static uint32_t g_compiles, g_compile_bytes;
static uint32_t g_deopts; // exits at an access that missed the fast path, all runs
static uint32_t g_invalidations; // translations dropped by code writes, all runs
static double g_compile_ms; // translation + synchronous compile + instantiate, all runs
static double g_translate_ms, g_first_call_ms, g_second_call_ms; // the split (Phase 0)

// Translate (or retranslate) a page from its current bytes
static uint32_t take_slot(void);

static void compile_page(jit_page_t *jp, uint32_t base) {
    static uint32_t words[PAGE_INSNS];
    static uint8_t leader[PAGE_INSNS];
    memcpy(jp->bytes, mem_env_ram() + base, MEM_PAGE_SIZE);
    memset(leader, 0, sizeof leader);
    leader[0] = 1;
    for (uint32_t i = 0; i < PAGE_INSNS; i++) {
        uint32_t w = LOAD_BE32(jp->bytes + 4 * i);
        words[i] = w;
        if (jp->extra[i])
            leader[i] = 1;
        if (!translatable(w)) {
            leader[i] = 1;
            if (i + 1 < PAGE_INSNS)
                leader[i + 1] = 1;
            continue;
        }
        if (is_branch(w)) {
            if (i + 1 < PAGE_INSNS)
                leader[i + 1] = 1;
            uint32_t t = direct_target(base + 4 * i, w);
            if (t != ~0u && (t & ~(uint32_t)PAGE_MASK) == base && !(t & 3))
                leader[(t & PAGE_MASK) >> 2] = 1;
        }
    }
    double t0 = emscripten_get_now();
    translate(base, words, leader, jp->blk_of);
    double t1 = emscripten_get_now();
    // The page function's table slot (an old translation's slot is leaked:
    // a prototype)
    jp->fn = jit_instantiate(g_buf, g_len, take_slot());
    mem_env_code_mark(base >> PAGE_SHIFT); // stores into it now reach jit_code_written
    g_compiles++;
    g_compile_bytes += g_len;
    g_compile_ms += emscripten_get_now() - t0;
    g_translate_ms += t1 - t0;
    // First and second call with a zero budget, on a scratch state: the
    // first block's guard leaves at once, so this times the call itself and
    // any compilation the engine deferred to the first call.
    {
        typedef int32_t (*fn_t)(ppc_t *, int32_t, int32_t);
        static ppc_t scratch;
        fn_t f = (fn_t)(uintptr_t)jp->fn;
        double a = emscripten_get_now();
        f(&scratch, 0, 0);
        double b = emscripten_get_now();
        f(&scratch, 0, 0);
        double d = emscripten_get_now();
        g_first_call_ms += b - a;
        g_second_call_ms += d - b;
    }
}

void ppc_jit_split(double *translate, double *module, double *instance, double *first_call, double *second_call) {
    *translate = g_translate_ms;
    *module = jit_t_module();
    *instance = jit_t_instance();
    *first_call = g_first_call_ms;
    *second_call = g_second_call_ms;
}

typedef int32_t (*page_fn)(ppc_t *c, int32_t budget, int32_t entry);

// Free function-table slots, reused by the next translations
static uint32_t *g_free_slots;
static uint32_t g_free_count, g_free_cap;
static void release_fn(uint32_t fn) {
    if (!fn)
        return;
    jit_release(fn);
    if (g_free_count == g_free_cap) {
        g_free_cap = g_free_cap ? 2 * g_free_cap : 256;
        g_free_slots = realloc(g_free_slots, g_free_cap * sizeof *g_free_slots);
    }
    g_free_slots[g_free_count++] = fn;
}
static uint32_t take_slot(void) {
    return g_free_count ? g_free_slots[--g_free_count] : 0;
}

// A store into a marked page (mem_env.h): drop its translations (both
// slow-path modes keep their own) and start its cooling.
static void jit_code_written(uint32_t page) {
    g_code_epoch++;
    for (int m = 0; m < 2; m++) {
        jit_page_t *jp = g_pages_by_mode[m] ? g_pages_by_mode[m][page] : NULL;
        if (jp && jp->fn) {
            release_fn(jp->fn);
            jp->fn = 0;
            g_invalidations++;
        }
        if (jp)
            jp->cooling = JIT_COOL_ENTRIES;
    }
    mem_env_code_unmark(page); // stores are fast again until the next translation
}

static void run_jit(ppc_t *c, uint32_t budget, int mode) {
    g_jit_slow_mode = mode;
    if (!g_pages_by_mode[mode]) {
        g_npages = (g_address_mask + 1) >> PAGE_SHIFT;
        g_pages_by_mode[mode] = calloc(g_npages, sizeof *g_pages);
    }
    g_pages = g_pages_by_mode[mode];
    g_code_write_hook = jit_code_written;
    g_chaos_state = g_jit_chaos_seed;
    g_gen++; // a new run: every page revalidates on first entry
    uint32_t n = budget;
    while (n > 0) {
        uint32_t pc = c->pc & g_address_mask;
        uint32_t page = pc >> PAGE_SHIFT, base = pc & ~(uint32_t)PAGE_MASK;
        jit_page_t *jp = g_pages[page];
        if (!jp) {
            jp = calloc(1, sizeof *jp);
            g_pages[page] = jp;
        }
        if (jp->gen != g_gen) {
            jp->gen = g_gen;
            if (jp->fn && memcmp(jp->bytes, mem_env_ram() + base, MEM_PAGE_SIZE) != 0) {
                release_fn(jp->fn);
                jp->fn = 0;
                memset(jp->extra, 0, sizeof jp->extra);
            }
            if (jp->fn)
                mem_env_code_mark(page); // the marks do not outlive a run (below)
        }
        if (!jp->fn && !jp->cooling && g_jit_chaos_seed) {
            if (!jp->pending) {
                g_chaos_state = g_chaos_state * 1664525u + 1013904223u;
                jp->pending = 1 + ((g_chaos_state >> 16) & 63);
            }
            if (--jp->pending) {
                uint32_t k = n < JIT_COOL_SLICE ? n : JIT_COOL_SLICE;
                ppc_run_switch(c, k);
                n -= k;
                continue;
            }
            // the translation "arrives" now: compiled below
        }
        if (!jp->fn && jp->cooling) {
            // Recently written: interpret a slice, count the entry
            uint32_t k = n < JIT_COOL_SLICE ? n : JIT_COOL_SLICE;
            ppc_run_switch(c, k);
            n -= k;
            jp->cooling--;
            continue;
        }
        uint32_t idx = (pc & PAGE_MASK) >> 2;
        if (!jp->fn || (pc & 3) || jp->blk_of[idx] == NONE16) {
            if (pc & 3) { // never in the kernels: let the interpreter decide
                ppc_run_switch(c, 1);
                n--;
                continue;
            }
            jp->extra[idx] = 1; // a run-time entry point: make it a leader
            compile_page(jp, base);
        }
        g_jit_reason = JIT_EXIT_PC;
        int32_t left = ((page_fn)(uintptr_t)jp->fn)(c, (int32_t)n, (int32_t)jp->blk_of[idx]);
        uint32_t rem = (uint32_t)left;
        if (g_jit_reason == JIT_EXIT_DEOPT) {
            // The instruction at c->pc missed the fast path: the interpreter
            // runs it, with the state committed up to it.
            g_deopts++;
            ppc_run_switch(c, 1);
            n = rem - 1;
            continue;
        }
        if (rem == n) {
            // No progress: the entry block is longer than the budget left
            // (the run's tail), or the instruction there is untranslatable.
            // The interpreter finishes the run exactly.
            ppc_run_switch(c, rem);
            rem = 0;
        }
        n = rem;
    }
    // Other backends run on the same RAM: give every write entry back
    mem_env_code_unmark_all();
    g_code_write_hook = NULL;
}

// Backend G: misses of the memory fast path leave the translation
void ppc_run_jit(ppc_t *c, uint32_t budget) {
    run_jit(c, budget, JIT_SLOW_DEOPT);
}
// Backend G': misses write the registers back and call the C slow path
void ppc_run_jit_call(ppc_t *c, uint32_t budget) {
    run_jit(c, budget, JIT_SLOW_CALL);
}

uint32_t ppc_jit_compiles(void) {
    return g_compiles;
}
uint32_t ppc_jit_compile_bytes(void) {
    return g_compile_bytes;
}
uint32_t ppc_jit_deopts(void) {
    return g_deopts;
}
void ppc_jit_reset(void) {
    for (int m = 0; m < 2; m++)
        if (g_pages_by_mode[m])
            for (uint32_t i = 0; i < g_npages; i++)
                if (g_pages_by_mode[m][i]) {
                    release_fn(g_pages_by_mode[m][i]->fn);
                    g_pages_by_mode[m][i]->fn = 0;
                    g_pages_by_mode[m][i]->cooling = 0;
                    g_pages_by_mode[m][i]->pending = 0;
                }
}
uint32_t ppc_jit_invalidations(void) {
    return g_invalidations;
}
double ppc_jit_compile_ms(void) {
    return g_compile_ms;
}

#endif // __EMSCRIPTEN__
