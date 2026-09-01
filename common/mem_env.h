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

#endif // MEM_ENV_H
