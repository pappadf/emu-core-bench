// SPDX-License-Identifier: MIT
// ppc_pdx_impl.h
// Backends E/F: predecode with operand pre-extraction and id-space
// specialization — the "rich intermediate format" experiments.
//
// Format ideas under test (vs backend D's raw {id, insn} entries):
//   - registers stored as PRE-SCALED byte offsets into the gpr file
//     (no shift/mask extraction at run time),
//   - the handler-id space is the specialization axis: ra=0 forms, Rc
//     record forms, addi/addis (immediate pre-shifted) and the common
//     bc shapes (bdnz, branch-on-bit-true/false with a precomputed CR
//     shift) each get their own leaf id, eliminating runtime flag tests,
//   - branch targets stored as PRECOMPUTED ENTRY INDICES; the hot loop
//     walks an entry pointer and never materializes the guest pc
//     (reconstructed as (cur - tab) << 2 only at sprint exit / rare ops),
//   - rare/awkward shapes keep the raw instruction word in the entry and
//     fall back to the generic semantics from ppc_ops.h — the optimal
//     format is allowed to be per-id.
//
// Included twice: ppc_pdx8.c (8-byte entries; rlwinm keeps the raw word
// and re-derives sh/mask at run time) and ppc_pdx16.c (-DPDX_WIDE,
// 16-byte entries; rlwinm mask and shift fully pre-extracted).  The pair
// prices the classic width trade-off: extraction work vs cache footprint.

#include "ppc_ops.h"

#include <stddef.h>

// One predecoded entry.  Field meaning is per-id; see the decoder.
typedef struct {
    uint16_t id; // specialized handler id
    uint8_t a; // usually: destination register byte offset / CR bit shift
    uint8_t b; // usually: source register byte offset / shift amount
    uint32_t c; // imm (sign-extended/pre-shifted) / reg offset / target idx / raw insn
#ifdef PDX_WIDE
    uint32_t d; // rlwinm: rotate amount; else unused
    uint32_t pad;
#endif
} pdx_ent_t;

// Pre-scaled register byte offset; offset 128 is ppc_t.zero_base (always 0),
// used as the base register for ra=0 addressing forms
#define RO(r) ((uint8_t)((r) * 4))
#define ZERO_OFF 128
_Static_assert(offsetof(ppc_t, zero_base) == ZERO_OFF, "zero_base offset");
#define GPRO(cpu, off) (*(uint32_t *)((uint8_t *)(cpu)->gpr + (off)))

// Specialized handler ids (shared by both entry widths)
enum {
    PDX_ILLEGAL = 0,
    // X/XO-form register-register: a=dst off, b=src1 off, c=src2 off
    PDX_ADD, PDX_ADD_RC, PDX_SUBF, PDX_SUBF_RC, PDX_MULLW, PDX_MULLW_RC,
    PDX_AND, PDX_AND_RC, PDX_OR, PDX_OR_RC, PDX_XOR, PDX_XOR_RC,
    PDX_SLW, PDX_SRW,
    PDX_SRAWI, PDX_SRAWI_RC, // a=dst off, b=shift, c=src off
    // D-form immediates: a=dst off, b=src off, c=imm (pre-shifted for *is forms)
    PDX_ADDI, PDX_LI, // LI: ra=0 folded away (a=dst, c=imm)
    PDX_MULLI, PDX_ORI, PDX_XORI, PDX_ANDI_RC,
    // rotate: a=dst off, b=src off; narrow: c=raw insn; wide: c=mask, d=sh
    PDX_RLWINM, PDX_RLWINM_RC,
    // compares: a=src1 off, b=CR field shift (28-4*crf), c=imm or src2 off
    PDX_CMPWI, PDX_CMPLWI, PDX_CMPW, PDX_CMPLW,
    // D-form memory: a=data reg off, b=base reg off (ZERO_OFF for ra=0), c=imm
    PDX_LWZ, PDX_LHZ, PDX_LBZ, PDX_LWZU,
    PDX_STW, PDX_STH, PDX_STB, PDX_STWU,
    // X-form memory: a=data reg off, b=base reg off, c=index reg off
    PDX_LWZX, PDX_LHZX, PDX_LBZX, PDX_STWX, PDX_STHX, PDX_STBX,
    // branches: c=target entry index (or raw insn for _GEN fallbacks)
    PDX_B, PDX_BDNZ,
    PDX_BC_T, PDX_BC_F, // a=CR bit shift (31-BI)
    PDX_B_GEN, PDX_BC_GEN, PDX_OP19_GEN,
    // SPR moves: a=gpr off
    PDX_MTCTR, PDX_MFCTR, PDX_MTLR, PDX_MFLR,
};

#define PDX_REGION_BYTES (256u * 1024)
#define PDX_ENTRIES      (PDX_REGION_BYTES / 4)
static pdx_ent_t PDX_TAB[PDX_ENTRIES];

// Decode one instruction word at guest address `addr` into an entry
static void pdx_decode(uint32_t addr, uint32_t insn, pdx_ent_t *e) {
    uint32_t rd = (insn >> 21) & 31, ra = (insn >> 16) & 31, rb = (insn >> 11) & 31;
    uint32_t simm = (uint32_t)(int32_t)(int16_t)insn;
    e->id = PDX_ILLEGAL;
    e->a = e->b = 0;
    e->c = insn; // fallbacks keep the raw word
#ifdef PDX_WIDE
    e->d = 0;
    e->pad = 0;
#endif
    switch (insn >> 26) {
    case 7: // mulli
        e->id = PDX_MULLI; e->a = RO(rd); e->b = RO(ra); e->c = simm;
        break;
    case 10: // cmplwi
        e->id = PDX_CMPLWI; e->a = RO(ra); e->b = (uint8_t)(28 - 4 * ((insn >> 23) & 7)); e->c = insn & 0xFFFF;
        break;
    case 11: // cmpwi
        e->id = PDX_CMPWI; e->a = RO(ra); e->b = (uint8_t)(28 - 4 * ((insn >> 23) & 7)); e->c = simm;
        break;
    case 14: // addi / li
        if (ra) { e->id = PDX_ADDI; e->a = RO(rd); e->b = RO(ra); e->c = simm; }
        else    { e->id = PDX_LI;   e->a = RO(rd); e->c = simm; }
        break;
    case 15: // addis / lis: merge into addi/li with the immediate pre-shifted
        if (ra) { e->id = PDX_ADDI; e->a = RO(rd); e->b = RO(ra); e->c = simm << 16; }
        else    { e->id = PDX_LI;   e->a = RO(rd); e->c = simm << 16; }
        break;
    case 16: { // bc: specialize the common BO shapes, precompute the target index
        uint32_t bo = rd, bi = ra;
        uint32_t target = addr + (uint32_t)(int32_t)(int16_t)(insn & 0xFFFC);
        if (insn & 3) { e->id = PDX_BC_GEN; break; } // AA/LK: generic
        if (bo == 16 && bi == 0) { e->id = PDX_BDNZ; e->c = target >> 2; }
        else if (bo == 12) { e->id = PDX_BC_T; e->a = (uint8_t)(31 - bi); e->c = target >> 2; }
        else if (bo == 4)  { e->id = PDX_BC_F; e->a = (uint8_t)(31 - bi); e->c = target >> 2; }
        else e->id = PDX_BC_GEN;
        break;
    }
    case 18: { // b: precompute target index; bl/ba fall back to generic
        if (insn & 3) { e->id = PDX_B_GEN; break; }
        int32_t li = ((int32_t)(insn << 6) >> 6) & ~3;
        e->id = PDX_B; e->c = (addr + (uint32_t)li) >> 2;
        break;
    }
    case 19:
        e->id = PDX_OP19_GEN;
        break;
    case 21: // rlwinm: wide pre-extracts mask+sh; narrow keeps the raw word
        e->id = (insn & 1) ? PDX_RLWINM_RC : PDX_RLWINM;
        e->a = RO(ra); e->b = RO(rd);
#ifdef PDX_WIDE
        e->c = ppc_mask((insn >> 6) & 31, (insn >> 1) & 31);
        e->d = rb; // SH field lives in the rb slot
#endif
        break;
    case 24: e->id = PDX_ORI;  e->a = RO(ra); e->b = RO(rd); e->c = insn & 0xFFFF; break;
    case 25: e->id = PDX_ORI;  e->a = RO(ra); e->b = RO(rd); e->c = (insn & 0xFFFF) << 16; break;
    case 26: e->id = PDX_XORI; e->a = RO(ra); e->b = RO(rd); e->c = insn & 0xFFFF; break;
    case 27: e->id = PDX_XORI; e->a = RO(ra); e->b = RO(rd); e->c = (insn & 0xFFFF) << 16; break;
    case 28: e->id = PDX_ANDI_RC; e->a = RO(ra); e->b = RO(rd); e->c = insn & 0xFFFF; break;
    case 31: {
        uint8_t base = ra ? RO(ra) : ZERO_OFF;
        switch (insn & 0x7FF) {
        case 0:    e->id = PDX_CMPW;  e->a = RO(ra); e->b = (uint8_t)(28 - 4 * ((insn >> 23) & 7)); e->c = RO(rb); break;
        case 64:   e->id = PDX_CMPLW; e->a = RO(ra); e->b = (uint8_t)(28 - 4 * ((insn >> 23) & 7)); e->c = RO(rb); break;
        case 532:  e->id = PDX_ADD;      e->a = RO(rd); e->b = RO(ra); e->c = RO(rb); break;
        case 533:  e->id = PDX_ADD_RC;   e->a = RO(rd); e->b = RO(ra); e->c = RO(rb); break;
        case 80:   e->id = PDX_SUBF;     e->a = RO(rd); e->b = RO(ra); e->c = RO(rb); break;
        case 81:   e->id = PDX_SUBF_RC;  e->a = RO(rd); e->b = RO(ra); e->c = RO(rb); break;
        case 470:  e->id = PDX_MULLW;    e->a = RO(rd); e->b = RO(ra); e->c = RO(rb); break;
        case 471:  e->id = PDX_MULLW_RC; e->a = RO(rd); e->b = RO(ra); e->c = RO(rb); break;
        case 56:   e->id = PDX_AND;    e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 57:   e->id = PDX_AND_RC; e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 888:  e->id = PDX_OR;     e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 889:  e->id = PDX_OR_RC;  e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 632:  e->id = PDX_XOR;    e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 633:  e->id = PDX_XOR_RC; e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 48:   e->id = PDX_SLW; e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 1072: e->id = PDX_SRW; e->a = RO(ra); e->b = RO(rd); e->c = RO(rb); break;
        case 1648: e->id = PDX_SRAWI;    e->a = RO(ra); e->b = (uint8_t)rb; e->c = RO(rd); break;
        case 1649: e->id = PDX_SRAWI_RC; e->a = RO(ra); e->b = (uint8_t)rb; e->c = RO(rd); break;
        case 46:   e->id = PDX_LWZX; e->a = RO(rd); e->b = base; e->c = RO(rb); break;
        case 558:  e->id = PDX_LHZX; e->a = RO(rd); e->b = base; e->c = RO(rb); break;
        case 174:  e->id = PDX_LBZX; e->a = RO(rd); e->b = base; e->c = RO(rb); break;
        case 302:  e->id = PDX_STWX; e->a = RO(rd); e->b = base; e->c = RO(rb); break;
        case 814:  e->id = PDX_STHX; e->a = RO(rd); e->b = base; e->c = RO(rb); break;
        case 430:  e->id = PDX_STBX; e->a = RO(rd); e->b = base; e->c = RO(rb); break;
        case 678: { // mfspr LR/CTR
            uint32_t spr = ((insn >> 16) & 31) | (((insn >> 11) & 31) << 5);
            if (spr == 8) { e->id = PDX_MFLR; e->a = RO(rd); }
            else if (spr == 9) { e->id = PDX_MFCTR; e->a = RO(rd); }
            break;
        }
        case 934: { // mtspr LR/CTR
            uint32_t spr = ((insn >> 16) & 31) | (((insn >> 11) & 31) << 5);
            if (spr == 8) { e->id = PDX_MTLR; e->a = RO(rd); }
            else if (spr == 9) { e->id = PDX_MTCTR; e->a = RO(rd); }
            break;
        }
        default: break; // stays PDX_ILLEGAL
        }
        break;
    }
    case 32: e->id = PDX_LWZ;  e->a = RO(rd); e->b = ra ? RO(ra) : ZERO_OFF; e->c = simm; break;
    case 33: e->id = PDX_LWZU; e->a = RO(rd); e->b = RO(ra); e->c = simm; break;
    case 34: e->id = PDX_LBZ;  e->a = RO(rd); e->b = ra ? RO(ra) : ZERO_OFF; e->c = simm; break;
    case 36: e->id = PDX_STW;  e->a = RO(rd); e->b = ra ? RO(ra) : ZERO_OFF; e->c = simm; break;
    case 37: e->id = PDX_STWU; e->a = RO(rd); e->b = RO(ra); e->c = simm; break;
    case 38: e->id = PDX_STB;  e->a = RO(rd); e->b = ra ? RO(ra) : ZERO_OFF; e->c = simm; break;
    case 40: e->id = PDX_LHZ;  e->a = RO(rd); e->b = ra ? RO(ra) : ZERO_OFF; e->c = simm; break;
    case 44: e->id = PDX_STH;  e->a = RO(rd); e->b = ra ? RO(ra) : ZERO_OFF; e->c = simm; break;
    default:
        break; // stays PDX_ILLEGAL
    }
}

// Execute exactly `budget` instructions starting at c->pc.  The hot loop
// walks entry pointers; the guest pc exists only at the boundaries.
void PDX_RUN_NAME(ppc_t *c, uint32_t budget) {
    for (uint32_t addr = 0; addr < PDX_REGION_BYTES; addr += 4)
        pdx_decode(addr, memory_read_uint32(addr), &PDX_TAB[addr >> 2]);
    const pdx_ent_t *tab = PDX_TAB;
    const pdx_ent_t *cur = tab + (c->pc >> 2);
    uint32_t n = budget;
    while (n > 0) {
        pdx_ent_t e = *cur;
        n--;
        switch (e.id) {
        case PDX_ADD:      GPRO(c, e.a) = GPRO(c, e.b) + GPRO(c, e.c); cur++; break;
        case PDX_ADD_RC: { uint32_t r = GPRO(c, e.b) + GPRO(c, e.c); GPRO(c, e.a) = r; record0(c, r); cur++; break; }
        case PDX_SUBF:     GPRO(c, e.a) = GPRO(c, e.c) - GPRO(c, e.b); cur++; break;
        case PDX_SUBF_RC: { uint32_t r = GPRO(c, e.c) - GPRO(c, e.b); GPRO(c, e.a) = r; record0(c, r); cur++; break; }
        case PDX_MULLW:    GPRO(c, e.a) = GPRO(c, e.b) * GPRO(c, e.c); cur++; break;
        case PDX_MULLW_RC: { uint32_t r = GPRO(c, e.b) * GPRO(c, e.c); GPRO(c, e.a) = r; record0(c, r); cur++; break; }
        case PDX_AND:      GPRO(c, e.a) = GPRO(c, e.b) & GPRO(c, e.c); cur++; break;
        case PDX_AND_RC: { uint32_t r = GPRO(c, e.b) & GPRO(c, e.c); GPRO(c, e.a) = r; record0(c, r); cur++; break; }
        case PDX_OR:       GPRO(c, e.a) = GPRO(c, e.b) | GPRO(c, e.c); cur++; break;
        case PDX_OR_RC: { uint32_t r = GPRO(c, e.b) | GPRO(c, e.c); GPRO(c, e.a) = r; record0(c, r); cur++; break; }
        case PDX_XOR:      GPRO(c, e.a) = GPRO(c, e.b) ^ GPRO(c, e.c); cur++; break;
        case PDX_XOR_RC: { uint32_t r = GPRO(c, e.b) ^ GPRO(c, e.c); GPRO(c, e.a) = r; record0(c, r); cur++; break; }
        case PDX_SLW: { uint32_t bv = GPRO(c, e.c); GPRO(c, e.a) = (bv & 0x20) ? 0 : GPRO(c, e.b) << (bv & 31); cur++; break; }
        case PDX_SRW: { uint32_t bv = GPRO(c, e.c); GPRO(c, e.a) = (bv & 0x20) ? 0 : GPRO(c, e.b) >> (bv & 31); cur++; break; }
        case PDX_SRAWI: case PDX_SRAWI_RC: {
            uint32_t sh = e.b;
            int32_t s = (int32_t)GPRO(c, e.c);
            uint32_t r = (uint32_t)(s >> sh);
            uint32_t lost = sh ? ((uint32_t)s & ((1u << sh) - 1)) : 0;
            c->xer_ca = (s < 0) && lost != 0;
            GPRO(c, e.a) = r;
            if (e.id == PDX_SRAWI_RC)
                record0(c, r);
            cur++;
            break;
        }
        case PDX_ADDI: GPRO(c, e.a) = GPRO(c, e.b) + e.c; cur++; break;
        case PDX_LI:   GPRO(c, e.a) = e.c; cur++; break;
        case PDX_MULLI: GPRO(c, e.a) = (uint32_t)((int32_t)GPRO(c, e.b) * (int32_t)e.c); cur++; break;
        case PDX_ORI:  GPRO(c, e.a) = GPRO(c, e.b) | e.c; cur++; break;
        case PDX_XORI: GPRO(c, e.a) = GPRO(c, e.b) ^ e.c; cur++; break;
        case PDX_ANDI_RC: { uint32_t r = GPRO(c, e.b) & e.c; GPRO(c, e.a) = r; record0(c, r); cur++; break; }
        case PDX_RLWINM: case PDX_RLWINM_RC: {
#ifdef PDX_WIDE
            uint32_t r = rotl32(GPRO(c, e.b), e.d) & e.c;
#else
            uint32_t sh = (e.c >> 11) & 31, mb = (e.c >> 6) & 31, me = (e.c >> 1) & 31;
            uint32_t r = rotl32(GPRO(c, e.b), sh) & ppc_mask(mb, me);
#endif
            GPRO(c, e.a) = r;
            if (e.id == PDX_RLWINM_RC)
                record0(c, r);
            cur++;
            break;
        }
        case PDX_CMPWI: case PDX_CMPW: {
            uint32_t bv = (e.id == PDX_CMPW) ? GPRO(c, e.c) : e.c;
            int32_t av = (int32_t)GPRO(c, e.a);
            uint32_t bits = (av < (int32_t)bv) ? 8 : (av > (int32_t)bv) ? 4 : 2;
            c->cr = (c->cr & ~(0xFu << e.b)) | (bits << e.b);
            cur++;
            break;
        }
        case PDX_CMPLWI: case PDX_CMPLW: {
            uint32_t bv = (e.id == PDX_CMPLW) ? GPRO(c, e.c) : e.c;
            uint32_t av = GPRO(c, e.a);
            uint32_t bits = (av < bv) ? 8 : (av > bv) ? 4 : 2;
            c->cr = (c->cr & ~(0xFu << e.b)) | (bits << e.b);
            cur++;
            break;
        }
        case PDX_LWZ:  GPRO(c, e.a) = memory_read_uint32(GPRO(c, e.b) + e.c); cur++; break;
        case PDX_LHZ:  GPRO(c, e.a) = memory_read_uint16(GPRO(c, e.b) + e.c); cur++; break;
        case PDX_LBZ:  GPRO(c, e.a) = memory_read_uint8(GPRO(c, e.b) + e.c); cur++; break;
        case PDX_LWZU: { uint32_t ea = GPRO(c, e.b) + e.c; GPRO(c, e.a) = memory_read_uint32(ea); GPRO(c, e.b) = ea; cur++; break; }
        case PDX_STW:  memory_write_uint32(GPRO(c, e.b) + e.c, GPRO(c, e.a)); cur++; break;
        case PDX_STH:  memory_write_uint16(GPRO(c, e.b) + e.c, (uint16_t)GPRO(c, e.a)); cur++; break;
        case PDX_STB:  memory_write_uint8(GPRO(c, e.b) + e.c, (uint8_t)GPRO(c, e.a)); cur++; break;
        case PDX_STWU: { uint32_t ea = GPRO(c, e.b) + e.c; memory_write_uint32(ea, GPRO(c, e.a)); GPRO(c, e.b) = ea; cur++; break; }
        case PDX_LWZX: GPRO(c, e.a) = memory_read_uint32(GPRO(c, e.b) + GPRO(c, e.c)); cur++; break;
        case PDX_LHZX: GPRO(c, e.a) = memory_read_uint16(GPRO(c, e.b) + GPRO(c, e.c)); cur++; break;
        case PDX_LBZX: GPRO(c, e.a) = memory_read_uint8(GPRO(c, e.b) + GPRO(c, e.c)); cur++; break;
        case PDX_STWX: memory_write_uint32(GPRO(c, e.b) + GPRO(c, e.c), GPRO(c, e.a)); cur++; break;
        case PDX_STHX: memory_write_uint16(GPRO(c, e.b) + GPRO(c, e.c), (uint16_t)GPRO(c, e.a)); cur++; break;
        case PDX_STBX: memory_write_uint8(GPRO(c, e.b) + GPRO(c, e.c), (uint8_t)GPRO(c, e.a)); cur++; break;
        case PDX_B:    cur = tab + e.c; break;
        case PDX_BDNZ: cur = (--c->ctr != 0) ? tab + e.c : cur + 1; break;
        case PDX_BC_T: cur = ((c->cr >> e.a) & 1) ? tab + e.c : cur + 1; break;
        case PDX_BC_F: cur = ((c->cr >> e.a) & 1) ? cur + 1 : tab + e.c; break;
        case PDX_B_GEN: {
            uint32_t pc = ((uint32_t)(cur - tab) << 2) + 4;
            cur = tab + (gen_b(c, e.c, pc) >> 2);
            break;
        }
        case PDX_BC_GEN: {
            uint32_t pc = ((uint32_t)(cur - tab) << 2) + 4;
            cur = tab + (gen_bc(c, e.c, pc) >> 2);
            break;
        }
        case PDX_OP19_GEN: {
            uint32_t pc = ((uint32_t)(cur - tab) << 2) + 4;
            cur = tab + (do_op19(c, e.c, pc) >> 2);
            break;
        }
        case PDX_MTCTR: c->ctr = GPRO(c, e.a); cur++; break;
        case PDX_MFCTR: GPRO(c, e.a) = c->ctr; cur++; break;
        case PDX_MTLR:  c->lr = GPRO(c, e.a); cur++; break;
        case PDX_MFLR:  GPRO(c, e.a) = c->lr; cur++; break;
        default:
            c->pc = (uint32_t)(cur - tab) << 2;
            ppc_illegal(c, e.c);
        }
    }
    c->pc = (uint32_t)(cur - tab) << 2;
}
