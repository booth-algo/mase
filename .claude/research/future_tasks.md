# Future Tasks (DWN)

## In Progress

### xcvu9p Synthesis Batch (beholder0, 2026-03-14)
- JSC paper-scope (DWN n=6): `dwn_top_paper_scope` at 1.15 ns target
- DiffLogic MNIST/NID/JSC: `difflogic_top` at 4.0 ns target
- Script: `~/dwn_synth/synth_batch_xcvu9p.sh`
- Log: `~/dwn_synth/batch_synth.log`

### ABC xcvu9p Re-synthesis (pending)
- Need to re-synthesize ABC-mapped Verilog on xcvu9p (currently only xc7a35t results)
- ABC BLIF files already exist for all configs

---

## Follow-up Ideas

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
