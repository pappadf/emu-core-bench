// SPDX-License-Identifier: MIT
// mem_env.h
// Benchmark memory environment: allocates a contiguous guest RAM at guest
// address 0 and populates the SoA fast-path tables over it.  Shared by all
// target architectures (ppc/, future m68k/).

#ifndef MEM_ENV_H
#define MEM_ENV_H

#include "mem_fastpath.h"

// Allocate guest RAM (power-of-two size) and install SoA entries; sets
// g_address_mask = ram_bytes - 1.  Callable once per process.
void mem_env_init(uint32_t ram_bytes);

// Host pointer to the guest RAM backing store
uint8_t *mem_env_ram(void);

// A device window: pages whose SoA entries are zero, so every access takes
// the slow path, where the device serves it.  Each access mixes the address,
// the value and a hash of the complete CPU state at that moment
// (g_dev_cpu_snapshot) into the device's state; a read returns a value
// derived from that state.  The device state is part of the verification
// hash, so a backend whose registers are not exact at a slow-path access
// (an uncommitted register, a wrong budget) diverges.
void mem_env_device(uint32_t base, uint32_t bytes);
void mem_env_device_reset(void);
uint64_t mem_env_device_state(void);
uint64_t mem_env_device_accesses(void);
extern uint64_t (*g_dev_cpu_snapshot)(void);

// Code-page marks (the JIT's coherence): a marked page keeps its read entry
// and loses its write entry, so every store into it takes the slow path,
// which performs the store and calls g_code_write_hook(page) — the owner
// drops what it derived from the page.  Unmarking restores the write entry.
void mem_env_code_mark(uint32_t page);
void mem_env_code_unmark(uint32_t page);
void mem_env_code_unmark_all(void);
bool mem_env_code_marked(uint32_t page);
extern void (*g_code_write_hook)(uint32_t page);

#endif // MEM_ENV_H
