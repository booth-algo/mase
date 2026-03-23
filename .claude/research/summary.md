# DWN FPGA Project — Consolidated Summary

**Project**: Behavioral RTL Synthesis for Differentiable Weightless Networks in MASE
**Authors**: Kevin H. Lam, Aaron Jino, Jianyi Cheng (Imperial College London)
**Period**: 2026-02-18 to 2026-03-15
**Target**: xcvu9p-flgb2104-2-i (UltraScale+), Vivado 2023.1

---

## What Was Built

### Core Pipeline
1. **DWN PyTorch modules** (`src/chop/nn/dwn/`) — thermometer encoder, LUT layer with EFD CUDA kernel, GroupSum, learnable mapping with chunked backward (OOM fix for A6000 48GB)
2. **RTL emission** (`scripts/emit_dwn_rtl.py`, `emit_full_pipeline_rtl.py`) — torch.fx tracing → behavioral Verilog via MASE hardware metadata passes
3. **SystemVerilog RTL** (`src/mase_components/dwn_layers/rtl/`) — combinational, clocked, full-pipeline, and paper-scope (2-stage pipelined GroupSum) variants
4. **cocotb verification** — UVM-style 500-transaction bit-exact RTL-vs-software equivalence tests
5. **Training infrastructure** — multi-dataset (MNIST, CIFAR-10, JSC, NID, KWS), mixed per-layer fan-in sweep, DiffLogic comparison
6. **BLIF export** for ABC Boolean minimization
7. **Vivado synthesis** — TCL scripts for OOC synthesis (standard + Flow_PerfOptimized_high)

### Key Bug Fixes
- **INDEX_BITS fix** (2026-03-14): `passes.py` packed indices into 8-bit fields, truncating indices >255 for networks with >256 inputs. Fixed with dynamic `ceil(log2(input_size))` bits. All results after this date are correct.
- **LearnableMapping OOM fix**: Chunked softmax backward (24 × 2000-col chunks) enables 8k-neuron CIFAR-10 on A6000 48GB.
- **EFD CUDA overflow**: `packed_accessor32` → `packed_accessor64` for large input sizes.

---

## Final Results (all xcvu9p, post-INDEX_BITS fix)

### 1. LUT combining (Vivado's optimization that packs pairs of LUT5 functions into LUT6_2 dual-output cells) — Paper-Scope, MNIST n=6 [2000,1000]

| Variant | LUTs | FFs | Fmax |
|---------|------|-----|------|
| MASE behavioral (ours) | **2,655** | 1,752 | 791 MHz |
| Bacellar et al. (structural) | 4,082 | 3,385 | 827 MHz |

**1.54x fewer LUTs**, 95.6% of paper Fmax. LUT combining enables Vivado to pack pairs of LUT5 neurons sharing common thermometer inputs into single LUT6_2 cells.

### 2. DWN vs DiffLogic Pareto (Paper-Scope, synth-only)

| Dataset | Architecture | n | LUTs | Accuracy |
|---------|-------------|---|------|----------|
| MNIST | DiffLogic | 2 | 3,184 | 87.77% |
| MNIST | DWN | 2 | 2,594 | 97.44% |
| MNIST | DWN | 6 | 2,436 | 98.51% |
| JSC | DiffLogic | 2 | 3,834 | 63.45% |
| JSC | DWN | 2 | 2,126 | 75.19% |
| JSC | DWN | 6 | 6,608 | 75.09% |

DWN dominates DiffLogic on MNIST and JSC. JSC n=2: **+11.7pp accuracy at 1.8x fewer LUTs**.

### 3. Mixed Per-Layer Fan-In (CIFAR-10, 3-layer)

| Fan-in | Analytical LUTs | Accuracy | Pareto |
|--------|----------------|----------|--------|
| 6-6-6 | 393,216 | 38.28% | dominated |
| 6-2-4 | 172,032 | **40.44%** | yes |

**2.3x fewer LUTs AND +2.16pp accuracy**. First-layer fan-in dominance: high n in layer 1, low in middle layers, moderate in last.

### 4. ABC Boolean Minimization (LUT-stack-only)

| Config | Behavioral LUTs | ABC LUTs | Change |
|--------|----------------|----------|--------|
| MNIST n=6 | 1,256 | 4,574 | **+264%** |
| MNIST n=2 | 995 | 988 | -0.7% |
| JSC n=6 | 1,178 | 1,179 | +0.1% |

ABC is **counterproductive at n=6** (destroys LUT6_2 packing) and neutral otherwise.

### 5. Accuracy vs Paper

| Dataset | Config | Ours | Paper |
|---------|--------|------|-------|
| MNIST | [2000,1000], n=6, 30ep | **98.51%** | 98.31% |
| CIFAR-10 | [8000], n=6, 100ep | **57.93%** | 57.42% |
| JSC | [3000], n=6, 100ep | 75.03% | **76.30%** |

---

## Key Discoveries

1. **Learnable mapping is critical** — All early CIFAR-10 runs accidentally used `--mapping-first random` (51%). With learnable mapping (paper default): 57.93%. This explains a 6.9pp gap and matches the paper's Table 10 ablation.

2. **16x scaling insight** — Random mapping needs 128k neurons to reach 57%; learnable needs 8k. Learnable mapping eliminates 16x redundant neurons.

3. **LUT combining mechanism** — Behavioral RTL (`LUT_CONTENTS[data_in]`) lets Vivado decompose neurons into LUT5 primitives and pack pairs sharing >=5 inputs into LUT6_2 cells. Structural RTL (explicit LUT6 primitives) prevents this. This accounts for the 1.54x area advantage.

4. **First-layer fan-in dominance** — The first DWN layer benefits most from high fan-in (processes raw thermometer bits). Later layers work on abstract features and need less fan-in. Optimal: high-low-moderate (e.g., 6-2-4).

5. **FloPoCo replication** — We replicated the paper's structural approach using FloPoCo VHDL, achieving 4,836 LUTs / 837.5 MHz (vs paper 4,972 / 827 MHz). Confirms the structural vs behavioral gap is real.

---

## Infrastructure

### Servers
- **kraken** (ee-kraken): Training (4x A6000 48GB), RTL emission, cocotb simulation. No xcvu9p Vivado license.
- **beholder0**: Vivado 2023.1 with xcvu9p license. All synthesis results from here.

### Checkpoints
Hosted at https://huggingface.co/booth-algo/dwn-checkpoints

### Replication
See `run.sh` in repo root — covers training, RTL emission, verification, ABC, and synthesis.

---

## Files Removed During Cleanup (2026-03-22)

Experiment artifacts superseded by final results:
- `structural_dwn_lut_*.sv` — LUT combining experiment structural variants (buggy 8-bit INDEX_BITS, pre-fix)
- `emit_dwn_flopoco_real.py`, `emit_dwn_vhdl_flopoco.py`, `emit_flopoco_thermo.py` — FloPoCo experiment scripts
- `synth_dwn_throughput.tcl`, `synth_wafr_experiment.tcl`, `synth_dwn_vhdl.tcl` — experiment synthesis
- `throughput_matching_task.md`, `wafr_packing_experiment.md`, `plan_paper_scope_synth.md`, `synthesis_tracker.md` — superseded research notes
- Pre-fix Vivado reports (xc7a35t and pre-fix xcvu9p)
