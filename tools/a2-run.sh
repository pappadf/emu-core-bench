#!/bin/sh
# SPDX-License-Identifier: MIT
# a2-run.sh — Phase A2: every (kernel, backend) pair in a process of its own,
# with the switch interpreter as the reference.  Translations of one kernel
# or variant must not load the engine while another is measured (in one
# process the region variant ran at less than half its own speed).
#   tools/a2-run.sh [node|<runner>] [budget] [reps]
RUNNER=${1:-node}
BUDGET=${2:-100000000}
REPS=${3:-5}
for K in 0x1 0x2 0x4 0x8 0x10 0x20 0x40; do
    for MASK in 0x10 0x40 0x80 0x100 0x200 0x400 0x800 0x1000; do
        $RUNNER build/ppc-bench.js "$BUDGET" "$REPS" "$K" "$MASK" | grep -v "^ppc dispatch\|benchmark done\|^wasm-jit"
    done
done
