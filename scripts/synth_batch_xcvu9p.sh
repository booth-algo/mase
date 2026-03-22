#!/bin/bash
# Batch synthesis on beholder0 for xcvu9p paper results
# Runs: JSC paper-scope + DiffLogic (MNIST, NID, JSC)
set -e

VIVADO=/mnt/applications/Xilinx/24.2/Vivado/2024.2/bin/vivado
BASE=~/dwn_synth
TCL=$BASE/synth_paper_match.tcl
PART=xcvu9p-flgb2104-2-i

# 1. JSC paper-scope (DWN n=6, 1.15ns = 869 MHz target, matching MNIST best)
echo "=== JSC paper-scope ==="
$VIVADO -mode batch -source $TCL \
    -tclargs $BASE/jsc_learnable_100ep_rtl $BASE/jsc_paper_scope_results \
    $PART dwn_top_paper_scope 1.15 \
    2>&1 | tee $BASE/jsc_paper_scope.log

# 2. DiffLogic MNIST on xcvu9p (4ns clock, synth only — no pipeline regs)
echo "=== DiffLogic MNIST ==="
$VIVADO -mode batch -source $TCL \
    -tclargs $BASE/difflogic_mnist_rtl $BASE/difflogic_mnist_results \
    $PART difflogic_top 4.0 \
    2>&1 | tee $BASE/difflogic_mnist.log

# 3. DiffLogic NID on xcvu9p
echo "=== DiffLogic NID ==="
$VIVADO -mode batch -source $TCL \
    -tclargs $BASE/difflogic_nid_rtl $BASE/difflogic_nid_results \
    $PART difflogic_top 4.0 \
    2>&1 | tee $BASE/difflogic_nid.log

# 4. DiffLogic JSC on xcvu9p
echo "=== DiffLogic JSC ==="
$VIVADO -mode batch -source $TCL \
    -tclargs $BASE/difflogic_jsc_rtl $BASE/difflogic_jsc_results \
    $PART difflogic_top 4.0 \
    2>&1 | tee $BASE/difflogic_jsc.log

echo "=== ALL DONE ==="
echo "Results:"
for d in jsc_paper_scope difflogic_mnist difflogic_nid difflogic_jsc; do
    echo "--- $d ---"
    grep -E "CLB LUTs|Slice LUTs|WNS:|Fmax:" $BASE/${d}.log 2>/dev/null | tail -5
done
