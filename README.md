# DWN Hardware Implementation

Differentiable Weightless Neural Networks (DWN) mapped to synthesizable RTL
through the MASE hardware compilation pipeline.

## Setup

From the repository root, install all dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For GPU-accelerated training (recommended), ensure a CUDA-capable GPU is available.
EFD training has no CPU fallback and will skip automatically when no GPU is detected.

Simulation prerequisites (unit and integration tests only):
- [Verilator](https://verilator.org) ≥ 5.0
- `cocotb`, `cocotb-test`, `cocotbext-axi` (included via `uv sync`)

## Quick Start

### 1. Train a model

From the repository root:

```bash
uv run python scripts/dwn/run_dwn_training.py \
    --dataset mnist \
    --epochs 30 \
    --lut-n 6 \
    --hidden-sizes 2000 1000 \
    --num-bits 3 \
    --batch-size 32 \
    --mapping-first learnable \
    --lr 0.01 \
    --lr-step 14
```

This targets ~98.3% accuracy on MNIST (Bacellar et al., ICML 2024 reference config).
The best checkpoint is saved to `mase_output/dwn/best.pt`.

### 2. Run the integration test

```bash
cd src/mase_components/dwn_layers/test/integration

# LUT-layer core only (combinational)
python run_top_tb.py --model best

# Full pipeline (thermometer + LUT stack + groupsum), pipelined/registered
python run_top_tb.py --model best --full --pipelined
```

The runner emits RTL from the checkpoint, builds a Verilator simulation, then runs
the cocotb testbench comparing RTL outputs bit-for-bit against the PyTorch model
across 100 samples with and without backpressure.

## File Tree

```
src/mase_components/dwn_layers/
├── passes.py                              # MASE hardware metadata passes
├── blif.py                                # BLIF export for ABC minimization
├── emit.py                                # RTL emission library (callable from scripts/tests)
├── hardware_core.py                       # DWNHardwareCore wrapper (LUT-only subgraph)
├── rtl/
│   ├── fixed/                             # Behavioral RTL (portable, simulation-friendly)
│   │   ├── fixed_dwn_lut_neuron.sv        #   Single neuron: LUT_CONTENTS[addr]
│   │   ├── fixed_dwn_lut_layer.sv         #   Parallel neurons, combinational
│   │   ├── fixed_dwn_lut_layer_clocked.sv #   Same + output register (1-cycle latency)
│   │   ├── fixed_dwn_thermometer.sv       #   Feature >= threshold comparators
│   │   ├── fixed_dwn_groupsum.sv          #   Popcount per class group, combinational
│   │   ├── fixed_dwn_groupsum_pipelined.sv#   2-stage pipelined popcount
│   │   └── fixed_dwn_flatten.sv           #   2D unpacked -> 1D packed wiring
│   └── structural/                        # Xilinx LUT6-primitive RTL (synthesis-targeted)
│       ├── sim_lut6.sv                    #   Behavioral stub for Verilator simulation
│       ├── structural_dwn_lut_neuron.sv   #   1 neuron = 1 LUT6 site (DONT_TOUCH)
│       ├── structural_dwn_lut_layer.sv    #   Parallel LUT6 neurons, combinational
│       └── structural_dwn_lut_layer_clocked.sv  # Same + output register
├── test/
│   ├── Makefile                           # cocotb + Verilator runner
│   ├── unit/                              # Per-module cocotb testbenches
│   │   ├── dwn_lut_layer_common.py        #   Shared Config/Tx/SWModel/Scoreboard
│   │   ├── fixed_dwn_groupsum_tb.py
│   │   ├── fixed_dwn_lut_layer_tb.py      #   Unified: tests both fixed & structural
│   │   ├── fixed_dwn_thermometer_tb.py
│   │   ├── structural_dwn_lut_neuron_tb.py
│   │   └── test_rtl_sim.py                #   Pytest runner for all unit tests
│   └── integration/                       # End-to-end RTL verification
│       ├── run_top_tb.py                  #   Runner: emits RTL, builds sim, launches cocotb
│       └── top_tb.py                      #   cocotb testbench: AXI-Stream stimulus + scoring

scripts/
├── dwn/
│   └── run_ckpt_to_rtl.py                 # Checkpoint -> RTL via MASE pipeline
├── emit_dwn_rtl.py                        # Checkpoint -> RTL (standalone, with BLIF/pipeline options)
└── synth_dwn.tcl                          # Vivado synthesis TCL script

test/passes/graph/transforms/dwn/
├── test_dwn_emit_rtl_equiv.py             # Parametrized emit pipeline equivalence
├── test_emit_verilog_dwn.py               # Verilog syntax/parameter validation
└── test_dwn_modules.py                    # Unit tests for DWN PyTorch modules
```

## Fixed vs Structural RTL

Both variants implement identical functionality. The difference is **how** the LUT is expressed:

| | Fixed (Behavioral) | Structural |
|---|---|---|
| **Implementation** | `assign out = LUT_CONTENTS[addr]` | Explicit `LUT6` primitive instantiation |
| **Portability** | Any simulator/synthesizer | Xilinx 7-series / UltraScale only |
| **Synthesis control** | Synthesizer decides mapping | `DONT_TOUCH` forces 1 neuron = 1 LUT6 |
| **Use case** | Simulation, functional verification | Area-accurate synthesis, placement control |
| **Simulation model** | Native Verilog | Requires `sim_lut6.sv` behavioral stub |

The structural variant guarantees each DWN neuron maps to exactly one Xilinx LUT6 site,
giving precise area counts. The fixed variant lets the synthesizer optimize freely.

## RTL Module Summary

| Module | Type | Latency | Purpose |
|--------|------|---------|---------|
| `fixed_dwn_lut_neuron` | Fixed, comb | 0 | `out = LUT_CONTENTS[addr]` |
| `fixed_dwn_lut_layer` | Fixed, comb | 0 | OUTPUT_SIZE neurons in parallel |
| `fixed_dwn_lut_layer_clocked` | Fixed, 1-clk | 1 | Output-registered LUT layer |
| `fixed_dwn_thermometer` | Fixed, comb | 0 | `out[t] = (feature >= threshold[t])` |
| `fixed_dwn_groupsum` | Fixed, comb | 0 | `$countones` per class group |
| `fixed_dwn_groupsum_pipelined` | Fixed, 2-clk | 2 | 2-stage pipelined popcount |
| `fixed_dwn_flatten` | Fixed, comb | 0 | 2D->1D wire reshape |
| `structural_dwn_lut_neuron` | Structural, comb | 0 | 1 LUT6 per neuron |
| `structural_dwn_lut_layer` | Structural, comb | 0 | Parallel LUT6 neurons |
| `structural_dwn_lut_layer_clocked` | Structural, 1-clk | 1 | Output-registered LUT6 layer |
| `sim_lut6` | Sim stub | 0 | Verilator-compatible LUT6 model |

## Full Inference Pipeline

```
              ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
 features ──> │ thermometer  │ ──> │  LUT layer 0 │ ──> │  LUT layer N │ ──> │  groupsum  │ ──> class scores
  (int)       │ (>= thresh)  │     │  (parallel)  │     │  (parallel)  │     │ (popcount)  │     (int)
              └─────────────┘     └──────────────┘     └──────────────┘     └────────────┘
                 comb / 0 clk        comb or 1 clk        comb or 1 clk       comb or 2 clk
```

## Scripts

### `scripts/dwn/run_ckpt_to_rtl.py`
Emit RTL from a trained checkpoint via the MASE pass pipeline.
```bash
python scripts/dwn/run_ckpt_to_rtl.py --ckpt mase_output/dwn/best.pt --top-name dwn_top
```

### `scripts/emit_dwn_rtl.py`
Standalone RTL emission with optional BLIF export, pipelined variant, and full-pipeline
wrappers (thermometer + LUT + groupsum). Delegates to `mase_components.dwn_layers.emit`.
```bash
python scripts/emit_dwn_rtl.py --ckpt-name mnist_n2 --output-dir hw/mnist_n2
python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6 --full-pipeline --pipelined
```

### `scripts/synth_dwn.tcl`
Vivado OOC synthesis. Run after RTL emission:
```bash
vivado -mode batch -source scripts/synth_dwn.tcl \
  -tclargs <rtl_dir> <results_dir> xcvu9p-flgb2104-2-i full_pipeline_top_clocked 4.0
```

## RTL Files Generated by `emit_dwn_rtl()`

`emit_dwn_rtl()` (in `src/mase_components/dwn_layers/emit.py`) generates ROM-based
SystemVerilog files to `<output_dir>/hardware/rtl/`. ROM-based modules embed trained
weights as `case`-statement functions rather than giant parameter literals, avoiding
Verilator's 65536-bit width limit.

The files produced depend on which flags are set:

| File | Condition | Description |
|------|-----------|-------------|
| `fixed_dwn_lut_layer_{i}.sv` | always | Combinational ROM-based LUT layer, one per layer |
| `fixed_dwn_lut_neuron.sv` | always | Single-neuron LUT primitive (copied from static RTL) |
| `dwn_top.sv` | always | Top-level module wiring all LUT layers together |
| `fixed_dwn_lut_layer_clocked_{i}.sv` | `emit_pipelined=True` | Output-registered LUT layer, one per layer |
| `dwn_top_clocked.sv` | `emit_pipelined=True` | Clocked top: FF register between every LUT layer |
| `fixed_dwn_thermometer.sv` | `full_pipeline=True` | ROM-based thermometer (overwrites static RTL) |
| `fixed_dwn_thermometer_clocked.sv` | `full_pipeline=True` | Output-registered ROM thermometer |
| `fixed_dwn_groupsum.sv` | `full_pipeline=True` | Copied from static RTL |
| `fixed_dwn_groupsum_pipelined.sv` | `full_pipeline=True` | Copied from static RTL |
| `full_pipeline_top.sv` | `full_pipeline=True` | Combinational full pipeline wrapper |
| `full_pipeline_top_clocked.sv` | `full_pipeline=True` | Clocked full pipeline; latency = 1 + N LUT layers + 2 groupsum cycles |
| `dwn_top_paper_scope.sv` | `full_pipeline=True` | LUT stack + pipelined GroupSum, no thermometer (matches Table 2 of Bacellar et al.) |

### Top-level module interfaces

All generated top-level modules use AXI-Stream–style handshake signals:

```
data_in_0        [W-1:0]   - packed input
data_in_0_valid            - producer asserts valid
data_in_0_ready            - consumer asserts ready
data_out_0       [W-1:0]   - packed output
data_out_0_valid
data_out_0_ready
clk, rst                   - present on all clocked modules
```

## Tests

### Unit Tests

Run all unit tests:
```bash
cd src/mase_components/dwn_layers/test
python -m pytest unit/test_rtl_sim.py -v -s
```

| Test | DUT | What it checks |
|------|-----|----------------|
| `test_rtl_groupsum` | `fixed_dwn_groupsum` | Popcount matches PyTorch for basic, exhaustive, and handshake patterns |
| `test_rtl_thermometer` | `fixed_dwn_thermometer` | Threshold comparisons match for boundary, random (seed=42), and handshake |
| `test_rtl_lut_layer` | `fixed_dwn_lut_layer` | LUT output matches for basic, all-zeros, all-ones, exhaustive, and handshake |
| `test_structural_rtl_lut_layer` | `structural_dwn_lut_layer` | Same tests as fixed, auto-detected via unified testbench |
| `test_structural_rtl_lut_neuron` | `structural_dwn_lut_neuron` | Exhaustive truth-table sweep of all 2^LUT_N addresses |

Prerequisites: `cocotb`, `cocotb-test`, Verilator, PyTorch.

### Integration Tests

The integration testbench is split into two files:

- **`run_top_tb.py`** - Runner script. Loads the checkpoint, emits RTL via `emit_dwn_rtl()`,
  builds the Verilator simulation, and launches cocotb with the correct environment.
  Supports `--full` (include thermometer + groupsum), `--pipelined` (clocked variant),
  and `--no-emit` (reuse previously generated RTL).
- **`top_tb.py`** - cocotb testbench. Uses AXI-Stream source/sink to drive data through
  the DUT and compare RTL outputs bit-for-bit against the PyTorch software model.
  Runs four tests: `reset_test`, `backpressure_test`, `basic_test`, and
  `continuous_test` (100 samples, with and without random backpressure).

Run from the integration directory:
```bash
cd src/mase_components/dwn_layers/test/integration

# Core-only (LUT layers), combinational
python run_top_tb.py --model <checkpoint_name>

# Full pipeline (thermometer + LUT + groupsum), pipelined
python run_top_tb.py --model <checkpoint_name> --full --pipelined

# Skip RTL emission (reuse previously generated RTL)
python run_top_tb.py --model <checkpoint_name> --full --pipelined --no-emit
```

Environment variables:
- `SIM` - simulator to use (default: `verilator`)
- `MODEL_PATH` - set automatically by `run_top_tb.py`; path to the `.pt` checkpoint
- `MODEL_MODE` - set automatically; `core` (LUT-only) or `full` (with thermometer + groupsum)
