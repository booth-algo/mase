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
  - Supports: 11 tabular datasets from paper via OpenML/sklearn:
    phoneme, skin-seg, higgs, australian, nomao, segment, miniboone, christine, jasmine, sylvine, blood
  - Auto-detects `input_features` and `num_classes` from dataset

### Hardware-Aware Training
- `test/passes/graph/transforms/dwn/run_dwn_training.py`: added `--area-lambda` flag
  - Differentiable mapping entropy regularization for LUT layers with LearnableMapping
  - High entropy = LUT input attends to many features (high routing complexity) — penalized
  - Low entropy = concentrated connections = fewer effective routing resources — encouraged
  - Area metric logged each epoch: `sum_l(output_size_l × 2^n_l)` total LUT storage
  - Usage: `--area-lambda 1e-3` for hardware-aware training

## RTL Configs Available
- `mase_output/dwn/baseline_n6_rtl/` — 2352→2000→1000 bits, LUT_N=6
- `mase_output/dwn/mixed_n6_2_rtl/` — 2352→2000→1000 bits, LUT_N=6,2
- `mase_output/dwn/mixed_n6_4_2_rtl/` — 2352→2000→1000→500 bits, LUT_N=6,4,2

## Example Commands

```bash
# Train on FashionMNIST
python run_dwn_training.py --dataset fashion_mnist --epochs 30 --lut-n 6 \
    --hidden-sizes 2000 1000 --num-bits 3 --ckpt-name fashion_n6

# Train on tabular dataset
python run_dwn_training.py --dataset phoneme --epochs 50 --lut-n 4 \
    --hidden-sizes 500 200 --num-bits 3 --ckpt-name phoneme_n4

# Hardware-aware training (mapping entropy regularization)
python run_dwn_training.py --real-mnist --epochs 30 --lut-n 6 \
    --area-lambda 1e-3 --ckpt-name baseline_n6_hwaware

# Emit RTL from checkpoint
python scripts/emit_dwn_rtl.py --ckpt-name baseline_n6

# Run full RTL equivalence test
cd src/mase_components/dwn_layers/test && python -m pytest test_rtl_equiv_dwn_top.py -v
```
