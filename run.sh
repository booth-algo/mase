#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES=1

# ---------------------------------------------------------------------------
# Config 1: Paper baseline — DWN uniform n=6 (Bacellar et al., ICML 2024)
# Targets ~98.3% on MNIST
# ---------------------------------------------------------------------------
python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --real-mnist \
    --epochs 30 \
    --lut-n 6 \
    --hidden-sizes 2000 1000 \
    --num-bits 3 \
    --mapping-first learnable \
    --batch-size 32 \
    --lr 0.01 \
    --lr-step 14 \
    --ckpt-name baseline_n6

python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --eval --real-mnist --ckpt-name baseline_n6

python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6 --output-dir mase_output/dwn/baseline_n6_rtl

# ---------------------------------------------------------------------------
# Config 2: Mixed fan-in (n=[6,2]) — high expressivity early, cheap later
# ---------------------------------------------------------------------------
# python test/passes/graph/transforms/dwn/run_dwn_training.py \
#     --real-mnist \
#     --epochs 30 \
#     --lut-n 6,2 \
#     --hidden-sizes 2000 1000 \
#     --num-bits 3 \
#     --mapping-first learnable \
#     --batch-size 32 \
#     --lr 0.01 \
#     --lr-step 14 \
#     --ckpt-name mixed_n6_2

# python test/passes/graph/transforms/dwn/run_dwn_training.py \
#     --eval --real-mnist --ckpt-name mixed_n6_2

# python scripts/emit_dwn_rtl.py --ckpt-name mixed_n6_2 --output-dir mase_output/dwn/mixed_n6_2_rtl

# ---------------------------------------------------------------------------
# Config 3: Mixed fan-in (n=[6,4,2]) — 3-layer funnel fan-in
# ---------------------------------------------------------------------------
# python test/passes/graph/transforms/dwn/run_dwn_training.py \
#     --real-mnist \
#     --epochs 30 \
#     --lut-n 6,4,2 \
#     --hidden-sizes 2000 1000 500 \
#     --num-bits 3 \
#     --mapping-first learnable \
#     --batch-size 32 \
#     --lr 0.01 \
#     --lr-step 14 \
#     --ckpt-name mixed_n6_4_2

# python test/passes/graph/transforms/dwn/run_dwn_training.py \
#     --eval --real-mnist --ckpt-name mixed_n6_4_2

# python scripts/emit_dwn_rtl.py --ckpt-name mixed_n6_4_2 --output-dir mase_output/dwn/mixed_n6_4_2_rtl

# ---------------------------------------------------------------------------
# Vivado synthesis — run on Vivado server after copying mase_output/dwn/*_rtl/
# Requires: source $VHLS/Vivado/$XLNX_VERSION/settings64.sh
# ---------------------------------------------------------------------------
# LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 \
#     vivado -mode batch -source scripts/synth_dwn.tcl \
#     -tclargs mase_output/dwn/baseline_n6_rtl/hardware/rtl \
#              mase_output/dwn/baseline_n6_rtl/synth_results

# LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 \
#     vivado -mode batch -source scripts/synth_dwn.tcl \
#     -tclargs mase_output/dwn/mixed_n6_2_rtl/hardware/rtl \
#              mase_output/dwn/mixed_n6_2_rtl/synth_results

# LD_PRELOAD=/lib/x86_64-linux-gnu/libudev.so.1 \
#     vivado -mode batch -source scripts/synth_dwn.tcl \
#     -tclargs mase_output/dwn/mixed_n6_4_2_rtl/hardware/rtl \
#              mase_output/dwn/mixed_n6_4_2_rtl/synth_results
