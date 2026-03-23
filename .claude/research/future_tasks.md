# Future Tasks (DWN)

## Follow-up Ideas

### Post-Training Boolean Minimisation via ABC (#3 from logbook)
- Export trained LUT truth tables → BLIF format → ABC (`strash; dc2; map`)
- Replace fixed LUT RTL with minimised netlist
- Verify bit-exactness via existing cocotb equivalence test
- **Status**: BLIF export implemented. ABC xcvu9p synthesis done (10 configs). Key finding: ABC is counterproductive for large networks (MNIST n=6: +264% LUTs due to lost LUT6_2 packing).
  - `src/mase_components/dwn_layers/blif.py` — `emit_network_blif(model, path)`
  - `scripts/emit_dwn_rtl.py --emit-blif` — emit RTL + BLIF together

---

## Completed (moved here from pending)

- ✓ CIFAR-10 training at paper scale (8k neurons, 100ep) — 51% random, 57%+ learnable
- ✓ Mixed-N Pareto sweep: `run_mixed_n_search.py` — 2-layer and 3-layer CIFAR-10 + MNIST
- ✓ Checkpoint-to-RTL: `cifar10_50ep.pt`, `cifar10_quick.pt`, `mixed_n6_4_2.pt` emitted
- ✓ Paper gap explained: `--mapping-first random` was disabling LearnableMapping
- ✓ LearnableMapping OOM fix: chunked backward in `mapping.py` + direct LUT count in training script
- ✓ xcvu9p synthesis batch (2026-03-15): JSC paper-scope (6,608 LUTs), DiffLogic MNIST/NID/JSC (3,184/293/3,834), all 11 paper-scope configs
- ✓ ABC xcvu9p re-synthesis (2026-03-15): 10 configs in `abc_xcvu9p_results/` on beholder0
- ✓ dwn_top_paper_scope testbench (2026-03-15): driver/monitor fix, 100% RTL==SW verified
