#!/usr/bin/env bash
set -euo pipefail

# DWN Results Replication Script
# Requires: conda activate plena2
# Run from: ~/mase/mase-fork

export CUDA_VISIBLE_DEVICES=1

# ============================================================================
# 1. TRAINING — DWN
# ============================================================================

# --- MNIST n=2 / n=4 / n=6 ---
for n in 2 4 6; do
    python test/passes/graph/transforms/dwn/run_dwn_training.py \
        --dataset mnist --epochs 30 --hidden-sizes 2000 1000 --lut-n $n \
        --num-bits 3 --tau 33.333 --lr 0.01 --batch-size 32 \
        --ckpt-name mnist_n${n}
done

# --- MNIST mixed fan-in ---
python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --dataset mnist --epochs 30 --hidden-sizes 2000 1000 --lut-n 6,2 \
    --num-bits 3 --tau 33.333 --lr 0.01 --batch-size 32 \
    --mapping-first learnable --ckpt-name mixed_n6_2

python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --dataset mnist --epochs 30 --hidden-sizes 2000 1000 500 --lut-n 6,4,2 \
    --num-bits 3 --tau 33.333 --lr 0.01 --batch-size 32 \
    --mapping-first learnable --ckpt-name mixed_n6_4_2

# --- CIFAR-10 (needs ~47GB VRAM on A6000) ---
python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --dataset cifar10 --epochs 100 --hidden-sizes 8000 --lut-n 6 \
    --num-bits 10 --tau 33.333 --lr 0.01 --lr-milestones 30 60 90 --batch-size 100

# --- JSC n=2 / n=4 / n=6 ---
for n in 2 4 6; do
    python test/passes/graph/transforms/dwn/run_dwn_training.py \
        --dataset jsc --epochs 100 --hidden-sizes 3000 --lut-n $n \
        --num-bits 200 --tau 33.333 --lr 0.01 --lr-milestones 30 60 90 --batch-size 100 \
        --ckpt-name jsc_n${n}
done

# --- NID n=2 / n=4 / n=6 ---
for n in 2 4 6; do
    python test/passes/graph/transforms/dwn/run_dwn_training.py \
        --dataset nid --epochs 30 --hidden-sizes 256 252 --lut-n $n \
        --num-bits 3 --tau 33.333 --lr 0.01 --batch-size 32 \
        --ckpt-name nid_n${n}
done

# --- KWS ---
python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --dataset kws --epochs 100 --hidden-sizes 1608 --lut-n 6 \
    --num-bits 8 --tau 33.333 --lr 0.01 --lr-milestones 30 60 90 --batch-size 100 \
    --ckpt-name kws_100ep

# --- Mixed fan-in sweep (CIFAR-10 2-layer + 3-layer, all n combos) ---
python test/passes/graph/transforms/dwn/run_mixed_n_search.py

# ============================================================================
# 2. TRAINING — DiffLogic (for Pareto comparison)
# ============================================================================

python test/passes/graph/transforms/difflogic/run_difflogic_training.py \
    --dataset mnist --epochs 30 --hidden-sizes 2000 2000 --batch-size 32

python test/passes/graph/transforms/difflogic/run_difflogic_training.py \
    --dataset nid --epochs 30 --hidden-sizes 258 258 --batch-size 32

python test/passes/graph/transforms/difflogic/run_difflogic_training.py \
    --dataset jsc --epochs 30 --hidden-sizes 3000 --batch-size 32

# ============================================================================
# 3. RTL EMISSION
# ============================================================================

# DWN: standard + BLIF + pipelined + full pipeline
for ckpt in baseline_n6 mnist_n2 mnist_n4 mixed_n6_2 mixed_n6_4_2 \
            nid_n2 nid_n4 nid_n6 jsc_n2 jsc_n4 jsc_learnable_100ep kws_100ep; do
    python scripts/emit_dwn_rtl.py --ckpt-name "$ckpt" --emit-blif
    python scripts/emit_dwn_rtl.py --ckpt-name "$ckpt" --pipelined
    python scripts/emit_full_pipeline_rtl.py --ckpt-name "$ckpt"
done

# DiffLogic
for ckpt in difflogic_mnist difflogic_nid difflogic_jsc; do
    python scripts/emit_difflogic_rtl.py --ckpt-name "$ckpt"
done

# ============================================================================
# 4. RTL VERIFICATION (cocotb)
# ============================================================================

cd src/mase_components/dwn_layers/test
python -m pytest test_rtl_equiv_dwn_top.py -v
python -m pytest test_dwn_mnist_fullsim.py -v
python -m pytest test_dwn_paper_scope_fullsim.py -v
cd -

# ============================================================================
# 5. ABC BOOLEAN MINIMIZATION
# ============================================================================

for config in baseline_n6 mnist_n2 mnist_n4 nid_n2 nid_n4 nid_n6 \
              jsc_n2 jsc_learnable_100ep kws_100ep; do
    blif="mase_output/dwn/${config}_rtl/network.blif"
    if [ -f "$blif" ]; then
        echo "=== ABC: $config ==="
        ~/.local/bin/abc -c "read $blif; strash; dc2; print_stats"
        ~/.local/bin/abc -c "read $blif; strash; dc2; if -K 6; write_verilog mase_output/dwn/${config}_rtl/abc_mapped.v"
    fi
done

# ============================================================================
# 6. VIVADO SYNTHESIS (run on beholder0, not kraken)
# ============================================================================
#
# Transfer RTL from kraken → local Mac → beholder0:
#
#   ssh beholder0 "mkdir -p ~/dwn_synth"
#   for config in baseline_n6 mnist_n2 mnist_n4 mixed_n6_2 mixed_n6_4_2 \
#                 nid_n2 nid_n4 nid_n6 jsc_n2 jsc_learnable_100ep kws_100ep; do
#       mkdir -p /tmp/${config}_rtl
#       rsync -av kraken:~/mase/mase-fork/mase_output/dwn/${config}_rtl/hardware/rtl/ /tmp/${config}_rtl/
#       scp /tmp/${config}_rtl/*.sv beholder0:~/dwn_synth/${config}_rtl/
#       rm -rf /tmp/${config}_rtl
#   done
#   scp kraken:~/mase/mase-fork/scripts/synth_dwn.tcl beholder0:~/dwn_synth/
#   scp kraken:~/mase/mase-fork/scripts/synth_paper_match.tcl beholder0:~/dwn_synth/
#
# On beholder0 (source ~/.bashrc for Vivado):
#
#   # LUT-stack-only
#   for config in baseline_n6 mnist_n2 mnist_n4 nid_n2 nid_n4 nid_n6 \
#                 jsc_n2 jsc_learnable_100ep kws_100ep; do
#       mkdir -p ~/dwn_synth/${config}_results
#       vivado -mode batch -source ~/dwn_synth/synth_dwn.tcl \
#           -tclargs ~/dwn_synth/${config}_rtl ~/dwn_synth/${config}_results \
#           xcvu9p-flgb2104-2-i dwn_top 4.0 \
#           2>&1 | tee ~/dwn_synth/${config}_synth.log
#   done
#
#   # Paper-scope (Flow_PerfOptimized_high)
#   for config in baseline_n6 mnist_n2 mnist_n4 mixed_n6_2 mixed_n6_4_2 \
#                 nid_n2 nid_n4 nid_n6 jsc_n2 jsc_learnable_100ep kws_100ep; do
#       mkdir -p ~/dwn_synth/${config}_paper_results
#       vivado -mode batch -source ~/dwn_synth/synth_paper_match.tcl \
#           -tclargs ~/dwn_synth/${config}_rtl ~/dwn_synth/${config}_paper_results \
#           xcvu9p-flgb2104-2-i dwn_top_paper_scope 1.15 \
#           2>&1 | tee ~/dwn_synth/${config}_paper_synth.log
#   done
#
#   # DiffLogic
#   for config in difflogic_mnist difflogic_nid difflogic_jsc; do
#       mkdir -p ~/dwn_synth/${config}_results
#       vivado -mode batch -source ~/dwn_synth/synth_dwn.tcl \
#           -tclargs ~/dwn_synth/${config}_rtl ~/dwn_synth/${config}_results \
#           xcvu9p-flgb2104-2-i difflogic_top 4.0 \
#           2>&1 | tee ~/dwn_synth/${config}_synth.log
#   done
#
# Results: <results_dir>/utilization.rpt (LUTs), timing_summary.rpt (Fmax)

echo "Done."
