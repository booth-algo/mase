# DWN Hardware Implementation

Differentiable Weightless Neural Networks (DWN) mapped to synthesizable RTL
through the MASE hardware compilation pipeline.

## File Tree

```
src/mase_components/dwn_layers/
├── passes.py                              # MASE hardware metadata passes
├── blif.py                                # BLIF export for ABC minimization
├── rtl/
│   ├── fixed/                             # Behavioral RTL (portable, simulation-friendly)
│   │   ├── fixed_dwn_lut_neuron.sv        #   Single neuron: LUT_CONTENTS[addr]
│   │   ├── fixed_dwn_lut_layer.sv         #   Parallel neurons, combinational
│   │   ├── fixed_dwn_lut_layer_clocked.sv #   Same + output register (1-cycle latency)
│   │   ├── fixed_dwn_thermometer.sv       #   Feature >= threshold comparators
│   │   ├── fixed_dwn_groupsum.sv          #   Popcount per class group, combinational
│   │   ├── fixed_dwn_groupsum_pipelined.sv#   2-stage pipelined popcount
│   │   └── fixed_dwn_flatten.sv           #   2D unpacked → 1D packed wiring
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
│   └── integration/                       # Full-pipeline verification
│       ├── dwn_test_utils.py              #   Shared helpers (sw_forward, load_mnist, etc.)
│       ├── dwn_lut_layer_equiv_tb.py      #   Config-driven LUT layer equivalence
│       ├── dwn_top_equiv_tb.py            #   Multi-layer network equivalence
│       ├── dwn_mnist_uvm_tb.py            #   UVM-style MNIST scoreboard (combinational)
│       ├── dwn_paper_scope_uvm_tb.py      #   UVM-style MNIST scoreboard (clocked pipeline)
│       ├── dwn_paper_scope_sim_wrapper.sv  #   Clocked wrapper for paper-scope pipeline
│       ├── dwn_pipeline_depth_probe.py    #   Pipeline latency characterization
│       ├── test_dwn_mnist_fullsim.py      #   Pytest: emit + simulate MNIST (combinational)
│       ├── test_dwn_paper_scope_fullsim.py#   Pytest: emit + simulate MNIST (clocked)
│       ├── test_dwn_pipeline_depth_probe.py
│       └── test_rtl_equiv_from_model.py   #   Pytest: single-layer model→RTL equivalence

scripts/
├── dwn/
│   ├── run_ckpt_to_rtl.py                 # Checkpoint → RTL via MASE pipeline
│   └── hardware_core.py                   # Shared DWNHardwareCore wrapper class
├── emit_dwn_rtl.py                        # Checkpoint → RTL (standalone, with BLIF option)
├── emit_full_pipeline_rtl.py              # Full pipeline wrapper (thermo + LUT + groupsum)
└── synth_dwn.tcl                          # Vivado synthesis TCL script

test/passes/graph/transforms/dwn/
├── test_dwn_emit_rtl_equiv.py             # Parametrized emit pipeline equivalence
└── test_emit_verilog_dwn.py               # Verilog syntax/parameter validation
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
| `fixed_dwn_flatten` | Fixed, comb | 0 | 2D→1D wire reshape |
| `structural_dwn_lut_neuron` | Structural, comb | 0 | 1 LUT6 per neuron |
| `structural_dwn_lut_layer` | Structural, comb | 0 | Parallel LUT6 neurons |
| `structural_dwn_lut_layer_clocked` | Structural, 1-clk | 1 | Output-registered LUT6 layer |
| `sim_lut6` | Sim stub | 0 | Verilator-compatible LUT6 model |

## Parameter Packing

Trained LUT weights are packed into Verilog parameters as hex literals:

- **INPUT_INDICES**: `(i*LUT_N + k) * INDEX_BITS` bits per entry, where `INDEX_BITS = ceil(log2(INPUT_SIZE))`
- **LUT_CONTENTS**: `i * 2^LUT_N` bits per neuron truth table

Example for INPUT_SIZE=4, OUTPUT_SIZE=2, LUT_N=2:
```
INDEX_BITS = 2
INPUT_INDICES = 8'hE4   // neuron0 reads [0,1], neuron1 reads [2,3]
LUT_CONTENTS  = 8'hAC   // neuron0 truth table = 4'b1100, neuron1 = 4'b1010
```

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
Standalone RTL emission with optional BLIF export and pipelined variant.
```bash
python scripts/emit_dwn_rtl.py --ckpt-name mnist_n2 --output-dir hw/mnist_n2
```

### `scripts/emit_full_pipeline_rtl.py`
Generate full-pipeline wrappers (thermometer + LUT stack + groupsum).
```bash
python scripts/emit_full_pipeline_rtl.py --ckpt-name baseline_n6
```

### `scripts/synth_dwn.tcl`
Vivado OOC synthesis. Run after RTL emission:
```bash
vivado -mode batch -source scripts/synth_dwn.tcl \
  -tclargs <rtl_dir> <results_dir> xcvu9p-flgb2104-2-i full_pipeline_top_clocked 4.0
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

| Test | What it does | Prerequisites |
|------|-------------|---------------|
| `test_dwn_mnist_fullsim.py` | Emits RTL from checkpoint, simulates 500 MNIST samples through combinational `dwn_top`, compares bit-for-bit vs SW model | Trained checkpoint, MNIST cache |
| `test_dwn_paper_scope_fullsim.py` | Same but through clocked 4-stage pipeline (`dwn_top_paper_scope`), compares class scores | Trained checkpoint, MNIST cache |
| `test_dwn_pipeline_depth_probe.py` | Characterizes pipeline latency (tries delays 1-8, asserts correct depth found) | Trained checkpoint, MNIST cache |
| `test_rtl_equiv_from_model.py` | Creates a small LUTLayer, packs weights, simulates in RTL, compares exhaustively | cocotb, Verilator |

Run individually:
```bash
cd src/mase_components/dwn_layers/test
python -m pytest integration/test_dwn_mnist_fullsim.py -v -s
```

Environment variables for integration tests:
- `DWN_CKPT` — path to trained `.pt` checkpoint
- `DWN_RTL_DIR` — path to emitted RTL directory
- `DWN_MNIST_CACHE` — path to MNIST feature cache (default: `~/.cache/dwn/mnist/mnist_features.pt`)
- `DWN_UVM_NUM_SAMPLES` — number of MNIST samples to simulate

### Pass-Level Tests

```bash
python -m pytest test/passes/graph/transforms/dwn/ -v -s
```

| Test | What it does |
|------|-------------|
| `test_dwn_emit_rtl_equiv.py` | Parametrized: builds 1-layer and 2-layer models, runs full emit pipeline, simulates RTL, compares vs `sw_forward()` golden model |
| `test_emit_verilog_dwn.py` | Builds model, runs emit pipeline, validates generated Verilog syntax (lowercase names, unquoted hex params) |
