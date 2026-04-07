# DWN & LTCNN in MASE

Differentiable Weightless Neural Networks (DWN) and Look-Up Table Convolutional
Neural Networks (LTCNN) integrated into the MASE framework, providing end-to-end
training, NAS, benchmarking, and hardware generation.

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

## LTCNN Extension

LTCNN extends DWN to convolutions by replacing CNN dot-product kernels with
n-ary LUT trees (ICLR 2026).

```
Input -> Encoder -> [LTConvLayer + MaxPool] x L -> Flatten -> [LTNFeedForward] x M -> GroupSum -> Classes
```

### Quick Start

```python
from chop.nn.ltcnn import LTCNN

model = LTCNN(in_channels=1, num_classes=10, image_size=28, conv_channels=[4, 8])
x = torch.rand(2, 1, 28, 28)
out = model(x)  # (2, 10) log-probabilities
```

| Checkpoint | Dataset | Conv | FF |
|---|---|---|---|
| `ltcnn-mnist` | MNIST (1ch, 28x28) | [4, 8] | [200, 100] |
| `ltcnn-cifar10` | CIFAR-10 (3ch, 32x32) | [8, 16] | [500, 200] |

### Architecture parameters

| Parameter | CLI flag | Paper range (Table 1) | Description |
|---|---|---|---|
| LUT arity | `--lut-n` | [2, 6] | Inputs per LUT node. Higher = more expressive but larger area (2^n entries) |
| Conv channels | `--conv-channels` | Nk in [1, 16] | Output channels per conv layer |
| Kernel size | `--kernel-size` | [2, 5] | Spatial receptive field |
| FF hidden sizes | `--ff-hidden-sizes` | npl in [10, 5000] | Nodes per feed-forward layer |
| Bit depth | `--bit-depth` | b in [1, 4] | Encoder precision (channels multiplied by b) |
| Q | `--Q` | {None, 1, 2, 4, 8} | Channel subsampling per kernel (None = use all) |
| Encoding | `--encoding` | quantization / thermometer | Input binarization method |

### 1. Train LTCNN

```bash
# LTCNN on MNIST
python scripts/ltcnn/run_ltcnn_training.py \
    --dataset mnist --epochs 30 --lut-n 4 \
    --conv-channels 4 8 --kernel-size 3 \
    --ff-hidden-sizes 200 100 --bit-depth 2 \
    --batch-size 64 --lr 0.02 --ckpt-name ltcnn_mnist

# LTCNN on CIFAR-10
python scripts/ltcnn/run_ltcnn_training.py \
    --dataset cifar10 --epochs 50 --lut-n 4 \
    --conv-channels 8 16 --kernel-size 3 \
    --ff-hidden-sizes 500 200 --bit-depth 2 \
    --batch-size 64 --lr 0.02 --ckpt-name ltcnn_cifar10

# Matched CNN baseline (same topology, standard Conv2d+Linear)
python scripts/ltcnn/run_ltcnn_training.py \
    --dataset mnist --epochs 30 --baseline-cnn \
    --conv-channels 4 8 --kernel-size 3 \
    --ff-hidden-sizes 200 100 --batch-size 64 \
    --lr 0.001 --ckpt-name cnn_mnist_baseline
```

Checkpoints saved to `mase_output/ltcnn/<ckpt-name>.pt`.

On completion, prints best accuracy, parameter count, LUT area, and comparison
against the ICLR 2026 paper's Table 2:

```
# Results
  model         : LTCNN
  dataset       : mnist
  best val acc  : 0.9512
  parameters    : 16,576  (10^4.22)
  training time : 245.3s  (4.1 min)
  LUT area      : 16,576 entries
  checkpoint    : mase_output/ltcnn/ltcnn_mnist.pt

# Paper comparison (ICLR 2026 Table 2, ~1e4.5 params)
  paper max acc : 0.940
  paper mean acc: 0.766
  your acc      : 0.9512
  delta         : +0.0112
```

Delta should be near zero or positive. CIFAR-10 reaches 43-52% (expected).

**Paper reference accuracy (LTCNN, Table 2):**

| Params | MNIST max | FashionMNIST max | CIFAR-10 max |
|---|---|---|---|
| 10^4.0 | 86.5% | 80.3% | 43.2% |
| 10^4.5 | 94.0% | 84.0% | 47.4% |
| 10^5.0 | 95.0% | 85.7% | 50.4% |
| 10^5.5 | 97.2% | 87.4% | 52.2% |
| 10^6.0 | 97.8% | 87.7% | 48.9% |

### 2. Robustness evaluation

Measures how accuracy degrades under salt-and-pepper noise and rectangular
occlusions. Reproduces the paper's Figure 4 and the β decay metric.

```bash
# Evaluate LTCNN
python scripts/ltcnn/run_robustness_eval.py \
    --ckpt mase_output/ltcnn/ltcnn_mnist.pt \
    --dataset mnist --num-trials 5

# Evaluate CNN baseline for comparison
python scripts/ltcnn/run_robustness_eval.py \
    --ckpt mase_output/ltcnn/cnn_mnist_baseline.pt \
    --dataset mnist --num-trials 5
```

Results saved to `mase_output/ltcnn/robustness_<name>.json`.

Output shows accuracy at each noise level and a fitted decay rate β:

```
# Salt & Pepper Robustness
  noise=0.00  acc=0.9534
  noise=0.05  acc=0.9312
  ...
  noise=0.50  acc=0.3456
  β_s&p = 2.3410

# Occlusion Robustness
  size=2x2    acc=0.9498
  size=4x4    acc=0.9312
  ...
  size=12x12  acc=0.7234
  β_occ = 0.078000
```

**What to check:**
- β (beta) is the decay rate: higher = more brittle to noise.
- Compare LTCNN β vs CNN baseline β. DWNNs become more brittle at larger
  parameter counts, while standard CNNs become more robust.
- S&P robustness is typically better than occlusion robustness for LTCNNs.

### 3. LTCNN NAS

Searches over six architecture dimensions using NSGA-II multi-objective optimization,
balancing accuracy against LUT area. Reference config at
`configs/ltcnn/search_ltcnn_mnist.toml`.

| Dimension | Candidates | Config key |
|---|---|---|
| LUT arity (n) | [2, 3, 4, 5, 6] | `n_values` |
| Conv channels | [4,8], [8,16], [16,32] | `conv_channels_options` |
| Kernel size | [3, 5] | `kernel_size_options` |
| FF hidden sizes | [200,100], [500,200] | `ff_hidden_sizes_options` |
| Bit depth | [2, 3, 4] | `bit_depth_options` |
| Channel subsampling Q | [0, 2, 4] | `Q_options` (0 = no subsampling) |

Uses the MASE search framework (`chop.actions.search.search`) with registered
`LTCNNSearchSpace`, `RunnerLTCNNTrain` (NLLLoss, early stopping), and
`RunnerLTCNNArea` (analytical LUT count). tau = sqrt(nf / C) per candidate.

```bash
python scripts/ltcnn/run_ltcnn_nas.py \
    --config configs/ltcnn/search_ltcnn_mnist.toml \
    --save-dir mase_output/ltcnn/nas_mnist
```

Each trial trains an LTCNN for 20 epochs, evaluates accuracy, and computes
LUT area. On completion, the Pareto-optimal trials are printed:

```
Best trial(s):
| number | software_metrics           | hardware_metrics       |
|--------|----------------------------|------------------------|
|     12 | {'accuracy': 0.952, ...}   | {'area_luts': 32768}   |
|      5 | {'accuracy': 0.941, ...}   | {'area_luts': 16576}   |
```

**Output files** in `mase_output/ltcnn/nas_mnist/`:
- `best.json` — Pareto-optimal trials with full configs
- `log.json` — all 30 trials with accuracy, area, and sampled architecture
- `study.pkl` — resumable Optuna study object

Multiple Pareto points with different `n`/channel/depth tradeoffs confirms
NSGA-II is finding a real tradeoff frontier. Extract winning architectures
from `best.json` for full training runs with more epochs.

**Inspect results:**
```python
import json, joblib

# Pareto front
with open("mase_output/ltcnn/nas_mnist/best.json") as f:
    best = json.load(f)
for k, v in best.items():
    cfg = v["sampled_config"]["model_config"]
    print(f"Trial {v['number']:2d}  n={cfg['n']}  conv={cfg['conv_channels']}  "
          f"acc={v['software_metrics']['accuracy']:.4f}  "
          f"area={v['hardware_metrics']['area_luts']:,}")

# Full study (resume or plot)
study = joblib.load("mase_output/ltcnn/nas_mnist/study.pkl")
```

### 4. RTL validation (flat DWN only)

LTCNN RTL emission requires new SystemVerilog modules for tree-structured LUT
kernels, which are not yet implemented. The existing flat DWN RTL pipeline can
be validated independently:

```bash
python scripts/dwn/run_dwn_training.py \
    --dataset mnist --real-mnist --epochs 30 \
    --lut-n 6 --hidden-sizes 2000 1000 --ckpt-name best

cd src/mase_components/dwn_layers/test
python -m pytest unit/test_rtl_sim.py -v -s
cd integration
python run_top_tb.py --model best --full --pipelined
```

All 5 unit tests and 6 integration tests should PASS. The integration testbench
streams 500 samples through the RTL and compares every output bit-for-bit against
the PyTorch reference.

## Mixed Fan-In NAS Search

Searches over per-layer LUT arity (n) for the flat DWN using NSGA-II, balancing
accuracy against hardware area. Each layer can independently use a different
fan-in value.

```bash
python scripts/dwn/run_nas_search.py \
    --config configs/dwn/search_dwn_mixed_n.toml \
    --save-dir mase_output/dwn/nas
```

Results: Pareto-optimal architectures in `mase_output/dwn/nas/best.json`.

## File Tree

```
src/chop/nn/ltcnn/
├── __init__.py
├── encoders.py                            # QuantizationEncoder, ThermometerEncoder
├── lut_nodes.py                           # BatchedLUTNodes (multilinear interpolation)
├── tree_kernel.py                         # LUTTreeKernel (n-ary tree -> scalar)
├── conv_layer.py                          # LTConvLayer (sliding window + tree kernels)
├── ff_layer.py                            # LTNFeedForwardLayer (LUT classifier head)
├── model.py                               # LTCNN (full model)
└── metrics.py                             # compute_area_luts(), compute_parameter_count()

src/chop/models/ltcnn/
├── __init__.py
└── ltcnn.py                               # MASE model registration + checkpoint getters

scripts/ltcnn/
├── run_ltcnn_training.py                  # Train LTCNN / CNN baseline
├── run_robustness_eval.py                 # S&P + occlusion noise robustness
└── run_ltcnn_nas.py                       # LTCNN NAS via MASE search framework

configs/ltcnn/
├── search_ltcnn_mnist.toml                # NSGA-II search config for MNIST
├── search_ltcnn_cifar10.toml              # NSGA-II search config for CIFAR-10
└── search_ltcnn_mnist_quick.toml          # Fast 2-trial integration test

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
│   ├── run_dwn_training.py                # Train flat DWN
│   ├── run_nas_search.py                  # DWN mixed fan-in NAS via MASE search
│   └── run_ckpt_to_rtl.py                 # Checkpoint -> RTL via MASE pipeline
├── emit_dwn_rtl.py                        # Checkpoint -> RTL (standalone, with BLIF/pipeline options)
└── synth_dwn.tcl                          # Vivado synthesis TCL script

test/passes/graph/transforms/dwn/
├── test_dwn_emit_rtl_equiv.py             # Parametrized emit pipeline equivalence
├── test_emit_verilog_dwn.py               # Verilog syntax/parameter validation
└── test_dwn_modules.py                    # Unit tests for DWN PyTorch modules
```

## MASE Framework Integration

Both DWN and LTCNN are registered as first-class models in the MASE search framework.
The integration follows the standard MASE pattern:

```
Script (run_ltcnn_nas.py / run_nas_search.py)
  └── chop.actions.search.search()
        ├── SearchSpace (LTCNNSearchSpace / DWNSearchSpace)
        │     ├── _post_init_setup()    — parse TOML config
        │     ├── build_search_space()  — define choice dimensions
        │     ├── optuna_sampler()      — sample architecture per trial
        │     └── rebuild_model()       — instantiate model from config
        ├── SWRunner (RunnerLTCNNTrain / RunnerDWNTrain)
        │     └── __call__()            — train model, return {accuracy, loss}
        ├── HWRunner (RunnerLTCNNArea / RunnerDWNArea)
        │     └── __call__()            — compute LUT area estimate
        └── SearchStrategyOptuna (NSGA-II multi-objective)
              └── optimize(accuracy↑, area_luts↓)
```

**Registration points** (all in `src/chop/actions/search/`):
- `search_space/__init__.py` — `"ltcnn/architecture"` and `"dwn/architecture"`
- `strategies/runners/software/__init__.py` — `"ltcnn_train"` and `"dwn_train"`
- `strategies/runners/hardware/__init__.py` — `"ltcnn_area"` and `"dwn_area"`

**Model factory** (`src/chop/models/ltcnn/ltcnn.py`):
- `@register_mase_model("ltcnn")` with checkpoints `ltcnn-mnist`, `ltcnn-cifar10`

**Area metric**: Both DWN and LTCNN expose `compute_area_luts()` returning total
LUT entries (sum of num_nodes * 2^n_inputs across all BatchedLUTNodes). This is
directly comparable between architectures on the accuracy-vs-area Pareto plane.

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

All generated top-level modules use AXI-Stream-style handshake signals:

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
