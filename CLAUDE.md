# MASE Project Instructions

## Python Environment

**Always use the `plena2` conda environment** for all Python commands:

```bash
conda run -n plena2 python ...
# or activate first:
conda activate plena2
```

When running any `python`, `pytest`, `pip`, or Python-related bash commands, prefix with `conda run -n plena2` or ensure the environment is active.

---

# DWN Implementation Status

## Completed

### Core RTL Emission Pipeline (fixed)
- `src/chop/nn/dwn/thermometer.py`: torch.fx-compatible `binarize()` using `flatten(-2)`
- `src/chop/nn/dwn/group_sum.py`: torch.fx-compatible forward (no Proxy control flow)
- `scripts/emit_dwn_rtl.py`: traces only `DWNHardwareCore` (LUT layers only)
- `src/mase_components/dwn_layers/passes.py`: `dwn_hardware_metadata_pass` now injects correct Verilog params (`INPUT_SIZE`, `OUTPUT_SIZE`, `LUT_N`, `INPUT_INDICES`, `LUT_CONTENTS`) from trained weights

### RTL Verification
- `src/mase_components/dwn_layers/test/dwn_top_equiv_tb.py`: cocotb testbench for full `dwn_top` network
- `src/mase_components/dwn_layers/test/test_rtl_equiv_dwn_top.py`: pytest runner — emits tiny DWN, simulates, verifies RTL matches SW golden model

### Multi-Dataset Training Support
- `test/passes/graph/transforms/dwn/run_dwn_training.py`: added `--dataset` flag
  - Supports: `mnist`, `fashion_mnist` (torchvision)
  - Supports: 11 tabular datasets from paper via OpenML/sklearn
  - Auto-detects `input_features` and `num_classes` from dataset

### Hardware-Aware Training
- `test/passes/graph/transforms/dwn/run_dwn_training.py`: added `--area-lambda` flag
  - Differentiable mapping entropy regularization for LUT layers with LearnableMapping
  - Area metric logged each epoch: `sum_l(output_size_l × 2^n_l)` total LUT storage

### Novel Findings (2026-03-08)

**Paper Gap Explained & Verified**:
- All 51% CIFAR-10 runs accidentally used `--mapping-first random`, disabling `LearnableMapping`
- With default `--mapping-first learnable` (paper config): **57.03% at epoch 8** on 8k neurons
- Paper reports 57.42% at 100 epochs — we match before any LR decay fires
- 16× scaling insight: random mapping needed 128k neurons to reach 57%; learnable needs just 8k

**LearnableMapping OOM Fix** (enables 8k CIFAR-10 on A6000 48GB):
- `src/chop/nn/dwn/mapping.py`: chunked backward (24 × 2000-col chunks, avoids 5.9GB softmax)
- `run_dwn_training.py`: epoch-end area logging uses direct LUT count (no softmax needed)

**Mixed-N Fan-In Dominance** (CIFAR-10, num_bits=10):
- 2-layer: `6-4` beats `6-6` and `4-6` — CIFAR-10 needs high fan-in in L1
- 3-layer: `6-2-4` (172k LUTs, 40.44%) dominates `6-6-6` (393k LUTs, 38.28%) — 2.3× fewer LUTs

## RTL Configs Available
- `mase_output/dwn/baseline_n6_rtl/` — 2352→2000→1000 bits, LUT_N=6
- `mase_output/dwn/mixed_n6_2_rtl/` — 2352→2000→1000 bits, LUT_N=6,2
- `mase_output/dwn/mixed_n6_4_2_rtl/` — 2352→2000→1000→500 bits, LUT_N=6,4,2

## Example Commands

```bash
# Train (learnable mapping, paper config)
conda run -n plena2 python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --dataset cifar10 --epochs 100 --hidden-sizes 8000 --lut-n 6 \
    --num-bits 10 --tau 33.333 --lr 0.01 --lr-milestones 30 60 90 --batch-size 100

# Emit RTL from checkpoint
conda run -n plena2 python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6

# Run RTL equivalence test
cd src/mase_components/dwn_layers/test && conda run -n plena2 python -m pytest test_rtl_equiv_dwn_top.py -v

# Monitor training (stdout is buffered — use checkpoint mtime instead)
conda run -n plena2 python -c "import torch; ckpt=torch.load('mase_output/dwn/best.pt',map_location='cpu'); print('Epoch:', ckpt['epoch'], 'Acc:', ckpt['acc'])"
```

## Vivado Synthesis (kraken → beholder0)

RTL lives on kraken at `mase_output/dwn/*_rtl/`. Transfer to beholder0 via local Mac:

```bash
# On local Mac — rsync kraken→local, scp local→beholder0
# (rsync remote-to-remote doesn't work; beholder0's shell breaks rsync protocol)
ssh beholder0 "mkdir -p ~/dwn_synth"
for config in baseline_n6 mixed_n6_2 mixed_n6_4_2 cifar10_n6_4 cifar10_n6_2_4; do
    mkdir -p /tmp/${config}_rtl
    rsync -av kraken:/home/khl22/mase-fork/mase_output/dwn/${config}_rtl/hardware/rtl/ /tmp/${config}_rtl/
    scp /tmp/${config}_rtl/*.sv beholder0:~/dwn_synth/${config}_rtl/
    rm -rf /tmp/${config}_rtl
done
scp kraken:/home/khl22/mase-fork/scripts/synth_dwn.tcl beholder0:~/dwn_synth/
```

Then on beholder0:

```bash
cd ~/dwn_synth
for config in baseline_n6 mixed_n6_2 mixed_n6_4_2 cifar10_n6_4 cifar10_n6_2_4; do
    mkdir -p ${config}_results
    vivado -mode batch -source synth_dwn.tcl \
           -tclargs ~/dwn_synth/${config}_rtl ~/dwn_synth/${config}_results \
           2>&1 | tee ${config}_synth.log
done
```

Key report files: `<config>_results/utilization.rpt` (LUT count), `<config>_results/timing_summary.rpt` (Fmax).

### Pipelined (Clocked) Synthesis

To emit and synthesise the clocked DWN variant targeting 500 MHz (comparable to paper's 200 MHz Zynq results):

```bash
# On kraken: emit pipelined RTL
conda run -n plena2 python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6 --pipelined

# On beholder0: run synthesis with 4th tclarg specifying clocked variant
vivado -mode batch -source synth_dwn.tcl \
       -tclargs ~/dwn_synth/baseline_n6_rtl ~/dwn_synth/baseline_n6_results xcvc1902-viva1596-3HP-e-S dwn_top_clocked \
       2>&1 | tee baseline_n6_clocked_synth.log

# Extract Fmax from timing report
grep -E "clk.*MHz|Fmax" ~/dwn_synth/baseline_n6_results/timing.rpt
```

The `--pipelined` flag registers layer outputs and inputs with proper pipeline handshaking. The clocked variant uses a 500 MHz target clock and reports Fmax and FF (flip-flop) counts alongside LUT metrics for direct comparison with the original DWN paper.

---

# Research Logbook — DWN FPGA Project

## 2026-02-24

### Context
Working on integrating Differentiable Weightless Networks (DWN) into the MASE
hardware generation framework. MASE's `emit_verilog` pipeline was extended to
support DWN's `LUTLayer`, with bit-exact Verilator co-simulation verification.

### Related Papers
- **DWN** (Bacellar et al., ICML 2024, arXiv:2410.11112): Original paper. 2522× energy-delay vs FINN.
- **DiffLogic + MASE** (Jino et al., Imperial College): fan-in=2 only, cites DWN as future work.
- **NeuraLUT-Assemble** (arXiv:2504.00592, 2025): mixed fan-in *within* neuron trees, not across layers.
- **SparseLUT** (arXiv:2503.12829, 2025): fixed fan-in, optimises *which* inputs each neuron reads.
- **PolyLUT-Add** (arXiv:2406.04910, 2024): extends PolyLUT to wider inputs.

---

## Candidate Novel Contributions

### #1 — Controlled Fan-in Pareto: DWN vs. DiffLogic (LOW effort, HIGH value)
Train DWN with n={2,4,6} and DiffLogic (n=2) on MNIST/JSC/NID under unified MASE toolflow.
Emit RTL, synthesise with Vivado, plot Pareto frontier. **Status**: Not done.

### #2 — Mixed Fan-in Architecture (LOW effort, MEDIUM-HIGH novelty)
Per-layer fan-in for DWN (e.g., n=[6,4,2]). **Status**: DONE — see CLAUDE_NOVEL.md.
Key result: `6-2-4` (172k LUTs) beats `6-6-6` (393k LUTs) on CIFAR-10.

### #3 — Post-Training Boolean Minimisation via ABC (MEDIUM effort, HIGH novelty)
Export trained LUT truth tables → ABC (`strash; dc2; map`) → fewer physical FPGA LUTs.
**Status**: Not done.

### #4 — Hardware-Aware NAS for DWN (HIGH effort)
Joint optimisation of DWN hyperparameters + FPGA cost model via MASE's Optuna/RL search.
**Status**: Not done.

**Strategic recommendation**: Combine #1 + #2 for best paper. Add #3 if time allows.

---

## Experimental Results (2026-02-24)

### MNIST Mixed Fan-in Sweep

| Config | Fan-in | Hidden sizes | Accuracy |
|--------|--------|--------------|----------|
| baseline | n=[6,6] | [2000,1000] | **0.9851** |
| mixed | n=[6,2] | [2000,1000] | 0.9830 |
| mixed | n=[6,4,2] | [2000,1000,500] | 0.9816 |

All trained with `mapping_first=learnable`, 30 epochs, batch=32, lr=0.01.

### Completed Engineering Work
- Fixed two bugs in MASE `emit_verilog` for DWN (parameter case mismatch, hex literal quoting)
- Added tests: `test_emit_verilog_dwn.py`, `test_rtl_equiv_from_model.py`, `test_dwn_emit_rtl_equiv.py`
- All 6 tests passing.

---

## 2026-03-08

### Paper Gap Resolved: Learnable Mapping Was the Missing Piece

**Root Cause**: All CIFAR-10 runs passed `--mapping-first random`, disabling `LearnableMapping`.
Script default is `--mapping-first learnable` (paper's config). One flag = entire 6.4% gap.

**Paper Table 10**:
| Config | Accuracy |
|--------|----------|
| Random mapping | 48.37% |
| Learnable mapping (LM) | 55.36% |
| LM + EFD (full paper config) | **57.42%** |

**Empirical Verification**:
- MNIST 4k: random 96.11% → learnable 98.06% (+1.95%)
- CIFAR-10 8k: **57.03% at epoch 8** with learnable mapping (paper: 57.42% at 100ep)

**16× Scaling Insight**: Random mapping needed 128k neurons to reach 57%. Learnable needs 8k.

### LearnableMapping OOM Fix (A6000 48GB)

1. `src/chop/nn/dwn/mapping.py` chunked backward: 24 × 2000-col chunks instead of 5.9GB tensor
2. `run_dwn_training.py` line 420: direct LUT count instead of `compute_area_loss()` softmax

Result: 8k learnable mapping trains at ~47GB peak on A6000, ~13 min/epoch.

### Mixed-N Fan-In Findings (CIFAR-10, num_bits=10)

- **2-layer**: `6-4` (40.69%) dominates `4-6` and `6-6` — CIFAR-10 needs high fan-in in L1
- **3-layer**: `6-2-4` (172k LUTs, 40.44%) dominates `6-6-6` (393k LUTs, 38.28%)


---

## Research Notes & Results

Canonical research notes are tracked in `.claude/` (committed to `kev/impl`):

- `.claude/research/benchmark.md` — accuracy vs paper + Vivado LUT counts (all final)
- `.claude/research/novel_findings.md` — mixed-N, Pareto, ABC findings
- `.claude/research/future_tasks.md` — outstanding ideas
- `.claude/results/vivado/` — 12 Vivado utilization reports (xc7a35tcpg236-1, 2026-03-12)
- `.claude/results/abc/abc_summary.md` — AND node counts after ABC minimisation

Local `CLAUDE_*.md` files have been removed; use `.claude/research/` instead.
