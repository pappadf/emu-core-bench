// SPDX-License-Identifier: MIT
// m68k_switch.c
// M68K backend A: conventional big-switch interpreter with runtime decode —
// 16-bit opcode fetch, nested switches on opcode groups, runtime EA-mode
// switches with extension-word fetches, eager condition codes.  This is the
// realistic per-instruction work profile of a classic 68000 interpreter.

#include "m68k_ops.h"

// Fetch helpers over the shared big-endian fast path
static inline uint16_t f16(uint32_t a) { return memory_read_uint16(a); }
static inline uint32_t f32(uint32_t a) { return memory_read_uint32(a); }

// Execute exactly `budget` instructions starting at m->pc
void m68k_run_switch(m68k_t *m, uint32_t budget) {
    uint32_t pc = m->pc;
    uint32_t n = budget;
    while (n > 0) {
        n--;
        uint16_t op = f16(pc);
        pc += 2;
        switch (op >> 12) {
        case 0x0: // CMPI.L #imm,Dn
            if ((op & 0xFFF8) == 0x0C80) {
                uint32_t s = f32(pc);
                pc += 4;
                uint32_t d = m->regs[op & 7];
                cc_cmp_l(m, s, d, d - s);
            } else
                m68k_illegal(m, op);
            break;
        case 0x1: // MOVE.B
        case 0x2: // MOVE.L / MOVEA.L
        case 0x3: { // MOVE.W
            unsigned size = (op >> 12) == 2 ? 4 : (op >> 12) == 3 ? 2 : 1;
            unsigned sm = (op >> 3) & 7, sr = op & 7;
            unsigned dm = (op >> 6) & 7, dr = (op >> 9) & 7;
            uint32_t val;
            switch (sm) { // source EA
            case 0:
                val = m->regs[sr];
                break;
            case 3: { // (An)+
                uint32_t a = m->regs[8 + sr];
                val = size == 4 ? f32(a) : size == 2 ? f16(a) : memory_read_uint8(a);
                m->regs[8 + sr] = a + size;
                break;
            }
            case 5: { // d16(An)
                uint32_t a = m->regs[8 + sr] + (uint32_t)(int32_t)(int16_t)f16(pc);
                pc += 2;
                val = size == 4 ? f32(a) : size == 2 ? f16(a) : memory_read_uint8(a);
                break;
            }
            case 7: // #imm (reg field 4)
                if (sr == 4) {
                    if (size == 4) {
                        val = f32(pc);
                        pc += 4;
                    } else {
                        val = f16(pc);
                        pc += 2;
                    }
                    break;
                }
                // fall through
            default:
                m68k_illegal(m, op);
                val = 0;
            }
            switch (dm) { // destination EA
            case 0: // Dn (partial write below word/byte)
                if (size == 4)
                    m->regs[dr] = val;
                else if (size == 2)
                    m->regs[dr] = (m->regs[dr] & 0xFFFF0000u) | (val & 0xFFFF);
                else
                    m->regs[dr] = (m->regs[dr] & 0xFFFFFF00u) | (val & 0xFF);
                break;
            case 1: // MOVEA.L: whole An, NO condition codes
                m->regs[8 + dr] = val;
                goto move_done;
            case 3: { // (An)+
                uint32_t a = m->regs[8 + dr];
                if (size == 4)
                    memory_write_uint32(a, val);
                else if (size == 2)
                    memory_write_uint16(a, (uint16_t)val);
                else
                    memory_write_uint8(a, (uint8_t)val);
                m->regs[8 + dr] = a + size;
                break;
            }
            case 5: { // d16(An)
                uint32_t a = m->regs[8 + dr] + (uint32_t)(int32_t)(int16_t)f16(pc);
                pc += 2;
                if (size == 4)
                    memory_write_uint32(a, val);
                else if (size == 2)
                    memory_write_uint16(a, (uint16_t)val);
                else
                    memory_write_uint8(a, (uint8_t)val);
                break;
            }
            default:
                m68k_illegal(m, op);
            }
            if (size == 4)
                cc_nz_l(m, val);
            else if (size == 2)
                cc_nz_w(m, (uint16_t)val);
            else
                cc_nz_b(m, (uint8_t)val);
        move_done:
            break;
        }
        case 0x4: // TST / LEA / NOP
            if ((op & 0xFFC0) == 0x4A80 && ((op >> 3) & 7) == 0) // TST.L Dn
                cc_nz_l(m, m->regs[op & 7]);
            else if ((op & 0xFFC0) == 0x4A40 && ((op >> 3) & 7) == 0) // TST.W Dn
                cc_nz_w(m, (uint16_t)m->regs[op & 7]);
            else if ((op & 0xFFC0) == 0x4A00 && ((op >> 3) & 7) == 0) // TST.B Dn
                cc_nz_b(m, (uint8_t)m->regs[op & 7]);
            else if ((op & 0xF1F8) == 0x41E8) { // LEA d16(Am),An
                m->regs[8 + ((op >> 9) & 7)] = m->regs[8 + (op & 7)] + (uint32_t)(int32_t)(int16_t)f16(pc);
                pc += 2;
            } else if (op == 0x4E71) { // NOP
            } else
                m68k_illegal(m, op);
            break;
        case 0x5: // ADDQ / SUBQ / DBcc
            if ((op & 0xF0F8) == 0x50C8) { // DBcc Dn,d16
                unsigned cond = (op >> 8) & 0xF, rn = op & 7;
                int16_t d16 = (int16_t)f16(pc);
                uint32_t target = pc + (uint32_t)(int32_t)d16;
                pc += 2;
                bool cnd = cond == 0 ? true : cond == 1 ? false : m68k_cond(m, cond);
                if (!cnd) {
                    uint16_t w = (uint16_t)(m->regs[rn] - 1);
                    m->regs[rn] = (m->regs[rn] & 0xFFFF0000u) | w;
                    if (w != 0xFFFF)
                        pc = target;
                }
            } else if (((op >> 6) & 3) == 2) { // .L
                uint32_t q = (op >> 9) & 7;
                q = q ? q : 8;
                unsigned mode = (op >> 3) & 7, rn = op & 7;
                if (mode == 0) { // Dn, full flags
                    uint32_t d = m->regs[rn];
                    if (op & 0x0100) { // SUBQ
                        uint32_t r = d - q;
                        m->regs[rn] = r;
                        cc_sub_l(m, q, d, r);
                    } else {
                        uint32_t r = d + q;
                        m->regs[rn] = r;
                        cc_add_l(m, q, d, r);
                    }
                } else if (mode == 1 && !(op & 0x0100)) { // ADDQ.L #q,An: no flags
                    m->regs[8 + rn] += q;
                } else
                    m68k_illegal(m, op);
            } else
                m68k_illegal(m, op);
            break;
        case 0x6: { // Bcc / BRA
            unsigned cond = (op >> 8) & 0xF;
            int32_t d8 = (int32_t)(int8_t)(op & 0xFF);
            uint32_t target;
            if (d8 == 0) {
                target = pc + (uint32_t)(int32_t)(int16_t)f16(pc);
                pc += 2;
            } else
                target = pc + (uint32_t)d8;
            if (cond == 1)
                m68k_illegal(m, op); // BSR unsupported
            if (cond == 0 || m68k_cond(m, cond))
                pc = target;
            break;
        }
        case 0x7: { // MOVEQ
            uint32_t r = (uint32_t)(int32_t)(int8_t)(op & 0xFF);
            m->regs[(op >> 9) & 7] = r;
            cc_nz_l(m, r);
            break;
        }
        case 0x8: // OR.L Dm,Dn
            if ((op & 0x01F8) == 0x0080) {
                uint32_t r = m->regs[(op >> 9) & 7] | m->regs[op & 7];
                m->regs[(op >> 9) & 7] = r;
                cc_nz_l(m, r);
            } else
                m68k_illegal(m, op);
            break;
        case 0x9: // SUB.L Dm,Dn
            if ((op & 0x01F8) == 0x0080) {
                uint32_t s = m->regs[op & 7], d = m->regs[(op >> 9) & 7], r = d - s;
                m->regs[(op >> 9) & 7] = r;
                cc_sub_l(m, s, d, r);
            } else
                m68k_illegal(m, op);
            break;
        case 0xB: // CMP.L / EOR.L
            if ((op & 0x01F8) == 0x0080) { // CMP.L Dm,Dn
                uint32_t s = m->regs[op & 7], d = m->regs[(op >> 9) & 7];
                cc_cmp_l(m, s, d, d - s);
            } else if ((op & 0x01F8) == 0x0180) { // EOR.L Dn,Dm
                uint32_t r = m->regs[op & 7] ^ m->regs[(op >> 9) & 7];
                m->regs[op & 7] = r;
                cc_nz_l(m, r);
            } else
                m68k_illegal(m, op);
            break;
        case 0xC: // AND.L Dm,Dn
            if ((op & 0x01F8) == 0x0080) {
                uint32_t r = m->regs[(op >> 9) & 7] & m->regs[op & 7];
                m->regs[(op >> 9) & 7] = r;
                cc_nz_l(m, r);
            } else
                m68k_illegal(m, op);
            break;
        case 0xD: // ADD.L Dm,Dn
            if ((op & 0x01F8) == 0x0080) {
                uint32_t s = m->regs[op & 7], d = m->regs[(op >> 9) & 7], r = d + s;
                m->regs[(op >> 9) & 7] = r;
                cc_add_l(m, s, d, r);
            } else
                m68k_illegal(m, op);
            break;
        case 0xE: // LSL.L/LSR.L #q,Dn
            if (((op >> 6) & 3) == 2 && ((op >> 3) & 7) == 1) { // size L, imm count, LS type
                uint32_t q = (op >> 9) & 7;
                q = q ? q : 8;
                unsigned rn = op & 7;
                uint32_t val = m->regs[rn], r;
                if (op & 0x0100) { // LSL
                    r = val << q;
                    m->cf = (val >> (32 - q)) & 1;
                } else { // LSR
                    r = val >> q;
                    m->cf = (val >> (q - 1)) & 1;
                }
                m->x = m->cf;
                m->regs[rn] = r;
                m->n = r >> 31;
                m->z = (r == 0);
                m->v = 0;
            } else
                m68k_illegal(m, op);
            break;
        default:
            m68k_illegal(m, op);
        }
    }
    m->pc = pc;
}
