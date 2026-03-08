# Future Tasks (DWN)

## Pending (GPU required)

### CIFAR-10 longer training
Run with paper-scale config to target ~60%+ accuracy (3 epochs with small model gave 40.4%).
```bash
cd test/passes/graph/transforms/dwn
python run_dwn_training.py \
  --dataset cifar10 --epochs 50 \
  --hidden-sizes 2048 2048 --lut-n 6 \
  --num-bits 2 --tau 3.333 --lr 0.01 --batch 64 \
  --ckpt mase_output/dwn/cifar10_50ep.pt
```

## Ideas / Follow-up

- Run `run_mixed_n_search.py` to get a real Pareto front (varies `lut_n` per layer → varying `area_luts`)
- Test checkpoint-to-RTL pipeline on `cifar10_50ep.pt` once trained
- Vivado synthesis on emitted RTL (friend's machine)
