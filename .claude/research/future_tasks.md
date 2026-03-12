# Future Tasks (DWN)

## In Progress

### CIFAR-10 8k Learnable Mapping — Full 100-Epoch Run
- PID 987098, GPU 0, A6000
- Epoch 8: 57.03% (paper target: 57.42%)
- LR decays at epochs 30, 60, 90 — expected to exceed 57.42%
- Monitor: `python -c "import torch; ckpt=torch.load('mase_output/dwn/best.pt',map_location='cpu'); print(ckpt['epoch'], ckpt['acc'])"`

---

## Vivado Synthesis — DONE (MNIST), PARTIAL (CIFAR-10)

Real CLB LUT counts from Vivado 2023.1 on beholder0 (xcvc1902-viva1596-3HP-e-S):

| Config | LUT_N | CLB LUTs |
|--------|-------|----------|
| MNIST baseline_n6 | [6] | 1,256 |
| MNIST mixed_n6_2 | [6,2] | 889 (−29%) |
| MNIST mixed_n6_4_2 | [6,4,2] | 705 (−44%) |
| CIFAR-10 cifar10_n6_4 | [6,4] | 2,027 (untrained) |
| CIFAR-10 cifar10_n6_2_4 | [6,2,4] | 1,960 (untrained) |

Design is purely combinational — no WNS (no clock domain). Propagation delay TBD.

**Remaining**: Re-synthesise CIFAR-10 configs with trained weights (currently from dummy untrained checkpoint).

---

## Follow-up Ideas

### DWN vs DiffLogic Controlled Pareto (#1 from logbook)
- Train DWN n={2,4,6} + DiffLogic n=2 on MNIST/JSC/NID
- Emit RTL via MASE, synthesise with Vivado
- Plot accuracy vs LUT count frontier
- **Status**: Scripts ready. Run training then synthesise.
  - `test/passes/graph/transforms/difflogic/run_difflogic_training.py` — CLI trainer (MNIST/CIFAR-10)
  - `scripts/emit_difflogic_rtl.py` — checkpoint → SystemVerilog
  - `scripts/synth_difflogic.tcl` — Vivado OOC synthesis

### Post-Training Boolean Minimisation via ABC (#3 from logbook)
- Export trained LUT truth tables → BLIF format → ABC (`strash; dc2; map`)
- Replace fixed LUT RTL with minimised netlist
- Verify bit-exactness via existing cocotb equivalence test
- **Status**: BLIF export implemented. ABC minimisation + verification pending.
  - `src/mase_components/dwn_layers/blif.py` — `emit_network_blif(model, path)`
  - `scripts/emit_dwn_rtl.py --emit-blif` — emit RTL + BLIF together

---

## Completed (moved here from pending)

- ✓ CIFAR-10 training at paper scale (8k neurons, 100ep) — 51% random, 57%+ learnable
- ✓ Mixed-N Pareto sweep: `run_mixed_n_search.py` — 2-layer and 3-layer CIFAR-10 + MNIST
- ✓ Checkpoint-to-RTL: `cifar10_50ep.pt`, `cifar10_quick.pt`, `mixed_n6_4_2.pt` emitted
- ✓ Paper gap explained: `--mapping-first random` was disabling LearnableMapping
- ✓ LearnableMapping OOM fix: chunked backward in `mapping.py` + direct LUT count in training script
