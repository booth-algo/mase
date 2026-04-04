# MASE — DWN & LTCNN Framework

Differentiable Weightless Neural Networks (DWN) and Look-Up Table Convolutional Neural Networks (LTCNN) mapped to trainable PyTorch modules, synthesizable RTL, and neural architecture search through the MASE hardware compilation pipeline.

## Setup

From the repository root, install all dependencies with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For GPU-accelerated training (recommended), ensure a CUDA-capable GPU is available.
EFD training has no CPU fallback and will skip automatically when no GPU is detected.

Simulation prerequisites (unit and integration tests only):
- [Verilator](https://verilator.org) >= 5.0
- `cocotb`, `cocotb-test`, `cocotbext-axi` (included via `uv sync`)

## LTCNN Integration

The LTCNN extends MASE's DWN framework with convolutional capabilities, based on the ICLR 2026 paper "Differentiable Weightless Neural Networks: A Benchmark on Performance and Robustness". Where standard DWN uses flat feed-forward layers of LUT nodes, the LTCNN replaces CNN dot-product kernels with n-ary trees of differentiable LUT nodes.

### Architecture

```
Input -> Encoder -> [LTConvLayer + MaxPool] x L -> Flatten -> [LTNFeedForward] x M -> GroupSum -> Classes
```

Each convolutional kernel is an n-ary tree (default n=4) of LUT nodes with multilinear interpolation, giving tree depth O(log_n(N)) instead of the LGCNN's O(log_2(N)).

### Module layout

```
src/chop/nn/ltcnn/
├── __init__.py
├── encoders.py       # QuantizationEncoder, ThermometerEncoder
├── lut_nodes.py      # BatchedLUTNodes (vectorized multilinear interpolation)
├── tree_kernel.py    # LUTTreeKernel (n-ary tree reducing patch to scalar)
├── conv_layer.py     # LTConvLayer (sliding-window convolution with tree kernels)
├── ff_layer.py       # LTNFeedForwardLayer (classifier head with LUT nodes)
└── model.py          # LTCNN (full model)
```

### Registered MASE models

The LTCNN is registered as a first-class MASE model with two checkpoints:

| Checkpoint | Dataset | Channels | Conv layers | FF layers |
|---|---|---|---|---|
| `ltcnn-mnist` | MNIST (1ch, 28x28) | [4, 8] | 2 | [200, 100] |
| `ltcnn-cifar10` | CIFAR-10 (3ch, 32x32) | [8, 16] | 2 | [500, 200] |

### Quick start

```python
import torch
from chop.nn.ltcnn import LTCNN

model = LTCNN(in_channels=1, num_classes=10, image_size=28, conv_channels=[4, 8])
x = torch.rand(2, 1, 28, 28)
out = model(x)  # (2, 10) log-probabilities
```

Or via the MASE model registry:

```python
from chop.models import get_model
model = get_model("ltcnn-mnist", pretrained=False)
```

## DWN Training

### Train a model

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

## Mixed Fan-In NAS Search

Recent work on branch `kw/testbench_update` integrates DWN mixed fan-in neural architecture search into the MASE search framework.

### What it does

Searches over per-layer LUT arity (n) using NSGA-II multi-objective optimization, balancing accuracy against hardware area (LUT count). Each layer can independently use a different fan-in value (e.g., layer 0 with n=4, layer 1 with n=6).

### Configuration

Reference config at `configs/dwn/search_dwn_mixed_n.toml`:

| Parameter | Value | Description |
|---|---|---|
| `n_values` | [2, 4, 6, 8] | LUT arities to search over per layer |
| `hidden_sizes` | [2000, 1000] | Layer widths |
| `num_bits` | 3 | Thermometer encoding bits |
| `n_trials` | 16 | Number of Optuna trials |
| Metrics | accuracy (max), area_luts (min) | Multi-objective targets |

### Run the search

```bash
python scripts/dwn/run_nas_search.py \
    --config configs/dwn/search_dwn_mixed_n.toml \
    --save-dir mase_output/dwn/nas
```

To change the dataset, edit the `dataset` field in the TOML config (`mnist`, `fashion_mnist`, or `cifar10`).

Results are saved to `--save-dir` and include Pareto-optimal architectures with per-layer fan-in configurations.

### Search components

| Component | File | Role |
|---|---|---|
| Search space | `src/chop/actions/search/search_space/dwn.py` | Defines per-layer n and hidden_size choices |
| SW runner | `src/chop/actions/search/strategies/runners/software/dwn_train.py` | Trains candidate and returns accuracy |
| HW runner | `src/chop/actions/search/strategies/runners/hardware/dwn_area.py` | Computes LUT area metric |
| CLI entry | `scripts/dwn/run_nas_search.py` | Loads config, data, and invokes search |

## DWN Integration Tests (Hardware)

The cocotb-based integration testbench at `src/mase_components/dwn_layers/test/integration/` verifies end-to-end correctness of generated RTL against the PyTorch reference model.

### Test suite

| Test | What it checks |
|---|---|
| `reset_test` | Clean reset behavior, outputs deasserted |
| `backpressure_test` | Data integrity under AXI-Stream sink backpressure |
| `basic_test` | Single inference pass correctness |
| `corner_case_test` | All-zeros and all-max boundary inputs |
| `continuous_test_no_backpressure` | 500-sample batch streaming |
| `continuous_test_random_backpressure` | 500-sample batch with random pauses |

### How they work

`TopEnv` loads a trained DWN checkpoint, creates both a Python reference model (`DWNModel` / `DWNHardwareCore`) and a hardware DUT. Inputs are encoded, sent via AXI-Stream, and outputs are compared bit-for-bit against the Python reference. A `Scoreboard` tracks pass/fail with structured mismatch diagnostics. All RNGs are seeded from `cocotb.RANDOM_SEED` for reproducibility.

### Commands

```bash
cd src/mase_components/dwn_layers/test/integration

# Core-only testbench (LUT layers only, combinational)
python run_top_tb.py --model <checkpoint_name>

# Full pipeline (thermometer + LUT layers + groupsum)
python run_top_tb.py --model <checkpoint_name> --full

# Full pipeline with pipelined/registered RTL
python run_top_tb.py --model <checkpoint_name> --full --pipelined

# Skip RTL emission (reuse previously generated RTL)
python run_top_tb.py --model <checkpoint_name> --full --pipelined --no-emit
```

Model checkpoints are expected at `mase_output/dwn/<checkpoint_name>.pt`. Requires Verilator (default) or another simulator set via the `SIM` environment variable.

### Unit tests

```bash
cd src/mase_components/dwn_layers/test
python -m pytest unit/test_rtl_sim.py -v -s
```

## Full Pipeline RTL

```
              +--------------+     +---------------+     +---------------+     +-------------+
 features --> | thermometer  | --> |  LUT layer 0  | --> |  LUT layer N  | --> |  groupsum   | --> class scores
  (int)       | (>= thresh)  |     |  (parallel)   |     |  (parallel)   |     | (popcount)  |     (int)
              +--------------+     +---------------+     +---------------+     +-------------+
                 comb / 0 clk        comb or 1 clk        comb or 1 clk       comb or 2 clk
```

## RTL Emission

```bash
# Via MASE pass pipeline
python scripts/dwn/run_ckpt_to_rtl.py --ckpt mase_output/dwn/best.pt --top-name dwn_top

# Standalone with BLIF export, pipelined variant, full-pipeline wrappers
python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6 --full-pipeline --pipelined

# Vivado OOC synthesis
vivado -mode batch -source scripts/synth_dwn.tcl \
  -tclargs <rtl_dir> <results_dir> xcvu9p-flgb2104-2-i full_pipeline_top_clocked 4.0
```
