// SPDX-License-Identifier: MIT
// m68k_pd_impl.h
// M68K backends B/C: predecoded threaded-code interpreter with the format
// tuned to the 68K's shape:
//   - ONE ENTRY PER 16-BIT WORD (index = guest pc >> 1); multi-word
//     instructions leave filler entries after their head, so guest pc maps
//     to entry index by shift alone and instruction LENGTH IS FOLDED INTO
//     THE SPECIALIZED ID (each handler advances by its own constant),
//   - extension words (displacements, immediates) PRE-EXTRACTED into the
//     entry: the hot loop never re-fetches or sign-extends them,
//   - EA modes and operation sizes resolved into the id space (one leaf id
//     per op x size x EA combination the subset supports),
//   - branch targets stored as precomputed entry indices; conditions baked
//     into per-condition ids,
//   - optionally (-DM68K_ELIDE, backend C) a FLAG-LIVENESS PASS at
//     predecode time: an instruction whose eagerly-computed NZVC would be
//     overwritten by the next sequential instruction before any reader is
//     retargeted to a no-flags twin handler (compare/test instructions
//     whose only effect is flags become sized NOPs).  Sound by 1-lookahead:
//     between adjacent instructions nothing reads CCR, and the property is
//     transitive across elided successors.  X is conservatively preserved.
//
// Included by m68k_pd.c (plain) and m68k_pd_elide.c (-DM68K_ELIDE).

#include "m68k_ops.h"

// One predecoded entry per 16-bit guest word
typedef struct {
    uint16_t id; // specialized handler id
    uint8_t a; // dst reg byte offset / branch fall-through length (words)
    uint8_t b; // src reg byte offset / quick value / shift count
    uint32_t c; // imm32 / sign-extended d16 / target entry index / raw opcode
} m68k_ent_t;

// Elidable ids come in adjacent pairs: id, then its no-flags twin (id+1)
#define M68K_ELIDABLE(X) \
    X(MOVEQ) X(MV_L_IMM_DN) X(MV_W_IMM_DN) \
    X(MV_L_DD) X(MV_L_ANP_DN) X(MV_L_DN_ANP) \
    X(MV_L_D16_DN) X(MV_L_DN_D16) \
    X(MV_W_D16_DN) X(MV_W_DN_D16) X(MV_B_D16_DN) X(MV_B_DN_D16) \
    X(ADD_L_DD) X(SUB_L_DD) X(ADDQ_L_DN) X(SUBQ_L_DN) \
    X(AND_L_DD) X(OR_L_DD) X(EOR_L_DD) X(LSL_L_IMM) X(LSR_L_IMM)

// Pure-flag ids (effect is condition codes only): elide to a sized NOP
#define M68K_PUREFLAG(X) X(CMP_L_DD) X(CMPI_L_DN) X(TST_L_DN) X(TST_W_DN) X(TST_B_DN)

#define M68K_BCC(X) X(CC) X(CS) X(NE) X(EQ) X(PL) X(MI) X(GE) X(LT) X(GT) X(LE)

enum {
    MI_ILLEGAL = 0,
    MI_FILLER, // trailing word of a multi-word instruction (never executed)
    MI_NOP1, MI_NOP2, MI_NOP3, // sized NOPs (elided pure-flag ops keep length)
#define PAIR(nm) MI_##nm, MI_##nm##_NF,
    M68K_ELIDABLE(PAIR)
#undef PAIR
#define ONE(nm) MI_##nm,
    M68K_PUREFLAG(ONE)
#undef ONE
    MI_MOVEA_L_IMM, MI_ADDQ_L_AN, MI_LEA_D16, MI_NOP,
    MI_BRA, MI_DBRA,
#define BC(nm) MI_B##nm,
    M68K_BCC(BC)
#undef BC
    MI_COUNT
};

#define MPD_REGION_BYTES (256u * 1024)
#define MPD_ENTRIES      (MPD_REGION_BYTES / 2)
static m68k_ent_t MPD_TAB[MPD_ENTRIES];
#ifdef M68K_ELIDE
static uint8_t MPD_LEN[MPD_ENTRIES]; // instruction length in words (elide pass walk)
#endif

// Decode the instruction at byte address `addr`; returns length in words
static unsigned mpd_decode(uint32_t addr, m68k_ent_t *e) {
    uint16_t op = memory_read_uint16(addr);
    uint32_t dr = (op >> 9) & 7, sr = op & 7;
    e->id = MI_ILLEGAL;
    e->a = e->b = 0;
    e->c = op;
    switch (op >> 12) {
    case 0x0:
        if ((op & 0xFFF8) == 0x0C80) {
            e->id = MI_CMPI_L_DN;
            e->a = DR_OFF(sr);
            e->c = memory_read_uint32(addr + 2);
            return 3;
        }
        return 1;
    case 0x1: case 0x2: case 0x3: { // MOVE family
        unsigned szc = op >> 12; // 1=B, 2=L, 3=W
        unsigned sm = (op >> 3) & 7, dm = (op >> 6) & 7;
        if (szc == 2 && dm == 1 && sm == 7 && sr == 4) { // MOVEA.L #imm,An
            e->id = MI_MOVEA_L_IMM;
            e->a = AR_OFF(dr);
            e->c = memory_read_uint32(addr + 2);
            return 3;
        }
        if (szc == 2 && sm == 7 && sr == 4 && dm == 0) { // MOVE.L #imm,Dn
            e->id = MI_MV_L_IMM_DN;
            e->a = DR_OFF(dr);
            e->c = memory_read_uint32(addr + 2);
            return 3;
        }
        if (szc == 3 && sm == 7 && sr == 4 && dm == 0) { // MOVE.W #imm,Dn
            e->id = MI_MV_W_IMM_DN;
            e->a = DR_OFF(dr);
            e->c = memory_read_uint16(addr + 2);
            return 2;
        }
        if (szc == 2 && sm == 0 && dm == 0) { // MOVE.L Dm,Dn
            e->id = MI_MV_L_DD;
            e->a = DR_OFF(dr);
            e->b = DR_OFF(sr);
            return 1;
        }
        if (szc == 2 && sm == 3 && dm == 0) { // MOVE.L (Am)+,Dn
            e->id = MI_MV_L_ANP_DN;
            e->a = DR_OFF(dr);
            e->b = AR_OFF(sr);
            return 1;
        }
        if (szc == 2 && sm == 0 && dm == 3) { // MOVE.L Dm,(An)+
            e->id = MI_MV_L_DN_ANP;
            e->a = DR_OFF(sr);
            e->b = AR_OFF(dr);
            return 1;
        }
        if (sm == 5 && dm == 0) { // MOVE.sz d16(Am),Dn
            e->id = szc == 2 ? MI_MV_L_D16_DN : szc == 3 ? MI_MV_W_D16_DN : MI_MV_B_D16_DN;
            e->a = DR_OFF(dr);
            e->b = AR_OFF(sr);
            e->c = (uint32_t)(int32_t)(int16_t)memory_read_uint16(addr + 2);
            return 2;
        }
        if (sm == 0 && dm == 5) { // MOVE.sz Dm,d16(An)
            e->id = szc == 2 ? MI_MV_L_DN_D16 : szc == 3 ? MI_MV_W_DN_D16 : MI_MV_B_DN_D16;
            e->a = DR_OFF(sr);
            e->b = AR_OFF(dr);
            e->c = (uint32_t)(int32_t)(int16_t)memory_read_uint16(addr + 2);
            return 2;
        }
        return 1;
    }
    case 0x4:
        if ((op & 0xFFC0) == 0x4A80 && ((op >> 3) & 7) == 0) { e->id = MI_TST_L_DN; e->a = DR_OFF(sr); return 1; }
        if ((op & 0xFFC0) == 0x4A40 && ((op >> 3) & 7) == 0) { e->id = MI_TST_W_DN; e->a = DR_OFF(sr); return 1; }
        if ((op & 0xFFC0) == 0x4A00 && ((op >> 3) & 7) == 0) { e->id = MI_TST_B_DN; e->a = DR_OFF(sr); return 1; }
        if ((op & 0xF1F8) == 0x41E8) { // LEA d16(Am),An
            e->id = MI_LEA_D16;
            e->a = AR_OFF(dr);
            e->b = AR_OFF(sr);
            e->c = (uint32_t)(int32_t)(int16_t)memory_read_uint16(addr + 2);
            return 2;
        }
        if (op == 0x4E71) { e->id = MI_NOP; return 1; }
        return 1;
    case 0x5:
        if ((op & 0xF0F8) == 0x50C8) { // DBcc (subset: DBRA/DBF only)
            if (((op >> 8) & 0xF) != 1)
                return 1; // other DBcc left unsupported -> illegal
            e->id = MI_DBRA;
            e->b = DR_OFF(sr);
            e->c = (addr + 2 + (uint32_t)(int32_t)(int16_t)memory_read_uint16(addr + 2)) >> 1;
            return 2;
        }
        if (((op >> 6) & 3) == 2) {
            uint32_t q = dr ? dr : 8;
            unsigned mode = (op >> 3) & 7;
            if (mode == 0) {
                e->id = (op & 0x0100) ? MI_SUBQ_L_DN : MI_ADDQ_L_DN;
                e->a = DR_OFF(sr);
                e->b = (uint8_t)q;
                return 1;
            }
            if (mode == 1 && !(op & 0x0100)) {
                e->id = MI_ADDQ_L_AN;
                e->a = AR_OFF(sr);
                e->b = (uint8_t)q;
                return 1;
            }
        }
        return 1;
    case 0x6: { // BRA / Bcc (BSR unsupported)
        unsigned cond = (op >> 8) & 0xF;
        int32_t d8 = (int32_t)(int8_t)(op & 0xFF);
        unsigned len = d8 == 0 ? 2 : 1;
        uint32_t target = d8 == 0 ? addr + 2 + (uint32_t)(int32_t)(int16_t)memory_read_uint16(addr + 2)
                                  : addr + 2 + (uint32_t)d8;
        if (cond == 1)
            return 1; // BSR -> illegal
        static const uint16_t bcc_id[16] = {
            [0] = MI_BRA,
            [4] = MI_BCC, [5] = MI_BCS, [6] = MI_BNE, [7] = MI_BEQ,
            [10] = MI_BPL, [11] = MI_BMI, [12] = MI_BGE, [13] = MI_BLT,
            [14] = MI_BGT, [15] = MI_BLE,
        };
        if (bcc_id[cond] == 0 && cond != 0)
            return 1;
        e->id = bcc_id[cond];
        e->a = (uint8_t)len; // fall-through advance
        e->c = target >> 1; // target entry index
        return len;
    }
    case 0x7:
        e->id = MI_MOVEQ;
        e->a = DR_OFF(dr);
        e->c = (uint32_t)(int32_t)(int8_t)(op & 0xFF);
        return 1;
    case 0x8:
        if ((op & 0x01F8) == 0x0080) { e->id = MI_OR_L_DD; e->a = DR_OFF(dr); e->b = DR_OFF(sr); }
        return 1;
    case 0x9:
        if ((op & 0x01F8) == 0x0080) { e->id = MI_SUB_L_DD; e->a = DR_OFF(dr); e->b = DR_OFF(sr); }
        return 1;
    case 0xB:
        if ((op & 0x01F8) == 0x0080) { e->id = MI_CMP_L_DD; e->a = DR_OFF(dr); e->b = DR_OFF(sr); }
        else if ((op & 0x01F8) == 0x0180) { e->id = MI_EOR_L_DD; e->a = DR_OFF(sr); e->b = DR_OFF(dr); }
        return 1;
    case 0xC:
        if ((op & 0x01F8) == 0x0080) { e->id = MI_AND_L_DD; e->a = DR_OFF(dr); e->b = DR_OFF(sr); }
        return 1;
    case 0xD:
        if ((op & 0x01F8) == 0x0080) { e->id = MI_ADD_L_DD; e->a = DR_OFF(dr); e->b = DR_OFF(sr); }
        return 1;
    case 0xE:
        if (((op >> 6) & 3) == 2 && ((op >> 3) & 7) == 1) {
            e->id = (op & 0x0100) ? MI_LSL_L_IMM : MI_LSR_L_IMM;
            e->a = DR_OFF(sr);
            e->b = (uint8_t)(dr ? dr : 8);
            return 1;
        }
        return 1;
    default:
        return 1;
    }
}

#ifdef M68K_ELIDE
// True if this (original) id unconditionally writes all of N,Z,V,C
static bool mpd_writes_all_nzvc(uint16_t id) {
    switch (id) {
#define PAIR(nm) case MI_##nm:
        M68K_ELIDABLE(PAIR)
#undef PAIR
#define ONE(nm) case MI_##nm:
        M68K_PUREFLAG(ONE)
#undef ONE
        return true;
    default:
        return false;
    }
}

// True if this id has a no-flags twin at id+1
static bool mpd_is_elidable(uint16_t id) {
    switch (id) {
#define PAIR(nm) case MI_##nm:
        M68K_ELIDABLE(PAIR)
#undef PAIR
        return true;
    default:
        return false;
    }
}

// Flag-liveness pass: walk instructions in address order; if the NEXT
// sequential instruction overwrites all NZVC (by its original id — sound
// transitively even if it too gets elided), retarget the current one to
// its no-flags twin, or NOP it out entirely if flags were its only effect.
static void mpd_elide_pass(void) {
    uint32_t idx = 0;
    while (idx < MPD_ENTRIES) {
        unsigned len = MPD_LEN[idx];
        uint32_t next = idx + len;
        if (next >= MPD_ENTRIES)
            break;
        uint16_t id = MPD_TAB[idx].id;
        // Original id of the successor: NF twins are id+1, so the stored id
        // is still original here (we transform current before advancing)
        if (mpd_writes_all_nzvc(MPD_TAB[next].id)) {
            if (mpd_is_elidable(id))
                MPD_TAB[idx].id = id + 1; // no-flags twin
            else
                switch (id) { // pure-flag ops become sized NOPs
                case MI_CMP_L_DD: case MI_TST_L_DN: case MI_TST_W_DN: case MI_TST_B_DN:
                    MPD_TAB[idx].id = MI_NOP1;
                    break;
                case MI_CMPI_L_DN:
                    MPD_TAB[idx].id = MI_NOP3;
                    break;
                default:
                    break;
                }
        }
        idx = next;
    }
}
#endif // M68K_ELIDE

// Execute exactly `budget` instructions starting at m->pc
void MPD_RUN_NAME(m68k_t *m, uint32_t budget) {
    for (uint32_t addr = 0; addr < MPD_REGION_BYTES;) {
        unsigned len = mpd_decode(addr, &MPD_TAB[addr >> 1]);
#ifdef M68K_ELIDE
        MPD_LEN[addr >> 1] = (uint8_t)len;
#endif
        for (unsigned w = 1; w < len; w++)
            MPD_TAB[(addr >> 1) + w].id = MI_FILLER;
        addr += len * 2;
    }
#ifdef M68K_ELIDE
    mpd_elide_pass();
#endif
    const m68k_ent_t *tab = MPD_TAB;
    const m68k_ent_t *cur = tab + (m->pc >> 1);
    uint32_t n = budget;
    while (n > 0) {
        m68k_ent_t e = *cur;
        n--;
        switch (e.id) {
        // Elidable pairs: the _NF twin (FL=0) skips condition-code updates.
        // Variadic so bodies may contain top-level commas; FL is a scoped
        // enum constant, so the compiler folds the flag branches away.
#define HPAIR(nm, LEN, ...) \
    case MI_##nm: { \
        enum { FL = 1 }; \
        __VA_ARGS__; \
        cur += LEN; \
        break; \
    } \
    case MI_##nm##_NF: { \
        enum { FL = 0 }; \
        __VA_ARGS__; \
        cur += LEN; \
        break; \
    }

        HPAIR(MOVEQ, 1, {
            uint32_t r = e.c;
            REGO(m, e.a) = r;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(MV_L_IMM_DN, 3, {
            REGO(m, e.a) = e.c;
            if (FL) cc_nz_l(m, e.c);
        })
        HPAIR(MV_W_IMM_DN, 2, {
            REGO(m, e.a) = (REGO(m, e.a) & 0xFFFF0000u) | (e.c & 0xFFFF);
            if (FL) cc_nz_w(m, (uint16_t)e.c);
        })
        HPAIR(MV_L_DD, 1, {
            uint32_t r = REGO(m, e.b);
            REGO(m, e.a) = r;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(MV_L_ANP_DN, 1, {
            uint32_t a = REGO(m, e.b);
            uint32_t r = memory_read_uint32(a);
            REGO(m, e.b) = a + 4;
            REGO(m, e.a) = r;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(MV_L_DN_ANP, 1, {
            uint32_t a = REGO(m, e.b);
            uint32_t r = REGO(m, e.a);
            memory_write_uint32(a, r);
            REGO(m, e.b) = a + 4;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(MV_L_D16_DN, 2, {
            uint32_t r = memory_read_uint32(REGO(m, e.b) + e.c);
            REGO(m, e.a) = r;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(MV_L_DN_D16, 2, {
            uint32_t r = REGO(m, e.a);
            memory_write_uint32(REGO(m, e.b) + e.c, r);
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(MV_W_D16_DN, 2, {
            uint16_t r = memory_read_uint16(REGO(m, e.b) + e.c);
            REGO(m, e.a) = (REGO(m, e.a) & 0xFFFF0000u) | r;
            if (FL) cc_nz_w(m, r);
        })
        HPAIR(MV_W_DN_D16, 2, {
            uint16_t r = (uint16_t)REGO(m, e.a);
            memory_write_uint16(REGO(m, e.b) + e.c, r);
            if (FL) cc_nz_w(m, r);
        })
        HPAIR(MV_B_D16_DN, 2, {
            uint8_t r = memory_read_uint8(REGO(m, e.b) + e.c);
            REGO(m, e.a) = (REGO(m, e.a) & 0xFFFFFF00u) | r;
            if (FL) cc_nz_b(m, r);
        })
        HPAIR(MV_B_DN_D16, 2, {
            uint8_t r = (uint8_t)REGO(m, e.a);
            memory_write_uint8(REGO(m, e.b) + e.c, r);
            if (FL) cc_nz_b(m, r);
        })
        HPAIR(ADD_L_DD, 1, {
            uint32_t s = REGO(m, e.b), d = REGO(m, e.a), r = d + s;
            REGO(m, e.a) = r;
            if (FL) cc_add_l(m, s, d, r);
            else m->x = r < s; // X is always architecturally live
        })
        HPAIR(SUB_L_DD, 1, {
            uint32_t s = REGO(m, e.b), d = REGO(m, e.a), r = d - s;
            REGO(m, e.a) = r;
            if (FL) cc_sub_l(m, s, d, r);
            else m->x = s > d;
        })
        HPAIR(ADDQ_L_DN, 1, {
            uint32_t s = e.b, d = REGO(m, e.a), r = d + s;
            REGO(m, e.a) = r;
            if (FL) cc_add_l(m, s, d, r);
            else m->x = r < s;
        })
        HPAIR(SUBQ_L_DN, 1, {
            uint32_t s = e.b, d = REGO(m, e.a), r = d - s;
            REGO(m, e.a) = r;
            if (FL) cc_sub_l(m, s, d, r);
            else m->x = s > d;
        })
        HPAIR(AND_L_DD, 1, {
            uint32_t r = REGO(m, e.a) & REGO(m, e.b);
            REGO(m, e.a) = r;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(OR_L_DD, 1, {
            uint32_t r = REGO(m, e.a) | REGO(m, e.b);
            REGO(m, e.a) = r;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(EOR_L_DD, 1, {
            uint32_t r = REGO(m, e.a) ^ REGO(m, e.b);
            REGO(m, e.a) = r;
            if (FL) cc_nz_l(m, r);
        })
        HPAIR(LSL_L_IMM, 1, {
            uint32_t val = REGO(m, e.a), q = e.b, r = val << q;
            REGO(m, e.a) = r;
            m->x = (val >> (32 - q)) & 1; // X always live
            if (FL) {
                m->cf = m->x;
                m->n = r >> 31;
                m->z = (r == 0);
                m->v = 0;
            }
        })
        HPAIR(LSR_L_IMM, 1, {
            uint32_t val = REGO(m, e.a), q = e.b, r = val >> q;
            REGO(m, e.a) = r;
            m->x = (val >> (q - 1)) & 1;
            if (FL) {
                m->cf = m->x;
                m->n = r >> 31;
                m->z = (r == 0);
                m->v = 0;
            }
        })
#undef HPAIR
        case MI_CMP_L_DD: {
            uint32_t s = REGO(m, e.b), d = REGO(m, e.a);
            cc_cmp_l(m, s, d, d - s);
            cur += 1;
            break;
        }
        case MI_CMPI_L_DN: {
            uint32_t d = REGO(m, e.a);
            cc_cmp_l(m, e.c, d, d - e.c);
            cur += 3;
            break;
        }
        case MI_TST_L_DN: cc_nz_l(m, REGO(m, e.a)); cur += 1; break;
        case MI_TST_W_DN: cc_nz_w(m, (uint16_t)REGO(m, e.a)); cur += 1; break;
        case MI_TST_B_DN: cc_nz_b(m, (uint8_t)REGO(m, e.a)); cur += 1; break;
        case MI_MOVEA_L_IMM: REGO(m, e.a) = e.c; cur += 3; break;
        case MI_ADDQ_L_AN: REGO(m, e.a) += e.b; cur += 1; break;
        case MI_LEA_D16: REGO(m, e.a) = REGO(m, e.b) + e.c; cur += 2; break;
        case MI_NOP:  cur += 1; break;
        case MI_NOP1: cur += 1; break;
        case MI_NOP2: cur += 2; break;
        case MI_NOP3: cur += 3; break;
        case MI_BRA: cur = tab + e.c; break;
        case MI_DBRA: {
            uint32_t r = REGO(m, e.b);
            uint16_t w = (uint16_t)(r - 1);
            REGO(m, e.b) = (r & 0xFFFF0000u) | w;
            cur = (w != 0xFFFF) ? tab + e.c : cur + 2;
            break;
        }
#define BC(nm) \
    case MI_B##nm: \
        cur = (M68K_COND_##nm) ? tab + e.c : cur + e.a; \
        break;
#define M68K_COND_CC (!m->cf)
#define M68K_COND_CS (m->cf)
#define M68K_COND_NE (!m->z)
#define M68K_COND_EQ (m->z)
#define M68K_COND_PL (!m->n)
#define M68K_COND_MI (m->n)
#define M68K_COND_GE (m->n == m->v)
#define M68K_COND_LT (m->n != m->v)
#define M68K_COND_GT (!m->z && m->n == m->v)
#define M68K_COND_LE (m->z || m->n != m->v)
        M68K_BCC(BC)
#undef BC
        default:
            m->pc = (uint32_t)(cur - tab) << 1;
            m68k_illegal(m, e.c);
        }
    }
    m->pc = (uint32_t)(cur - tab) << 1;
}
