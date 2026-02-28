# Research Logbook — DWN FPGA Project

## 2026-02-24

### Context
Working on integrating Differentiable Weightless Networks (DWN) into the MASE
hardware generation framework. MASE's `emit_verilog` pipeline was extended to
support DWN's `LUTLayer`, with bit-exact Verilator co-simulation verification.

### Related Papers (Landscape)
- **DWN** (Bacellar et al., ICML 2024, arXiv:2410.11112): Original paper.
  Already includes FPGA results on Zynq Z-7045 and xcvu9p. 2522× energy-delay
  improvement vs FINN. Uses custom RTL flow (not MASE).
- **DiffLogic + MASE** (Jino et al., Imperial College): DiffLogic (fan-in=2
  logic gates) through MASE on MNIST/JSC/NID. Explicitly cites DWN as future
  work (ref [31]). Uses Vivado targeting xcvc1902.
- **NeuraLUT-Assemble** (arXiv:2504.00592, 2025): Assembles NeuraLUT neurons
  into tree structures with larger effective fan-in. Uses mixed fan-in *within*
  a single assembled neuron (different tree levels), not across layers. For
  NeuraLUT architecture specifically.
- **SparseLUT** (arXiv:2503.12829, 2025): Optimises connectivity during
  training. Fixed fan-in per layer (not mixed). Improves accuracy by choosing
  *which* inputs each neuron reads rather than random assignment.
- **PolyLUT-Add** (arXiv:2406.04910, 2024): Extends PolyLUT to wider inputs.

---

## Candidate Novel Contributions

### #1 — Controlled Fan-in Pareto: DWN vs. DiffLogic (LOW effort, HIGH value)
**Gap**: Neither the DWN paper (custom RTL) nor the DiffLogic+MASE paper
(fan-in=2 only) compares both architectures under a unified MASE toolflow.

**Idea**: Train DWN with n={2,4,6} and DiffLogic (n=2) on the same datasets
(MNIST, JSC, NID). Emit RTL via MASE, synthesise with Vivado, report
LUT/Fmax/latency/accuracy. Plot Pareto frontier.

**Paper claim**: "First controlled comparison of DWN and DiffLogic on FPGA
using a unified automated toolflow, characterising the fan-in tradeoff."

**Effort**: 1–2 months. Infrastructure exists. Datasets match Jino et al.

**Status**: Not done in any existing paper.

---

### #2 — Mixed Fan-in Architecture: Per-layer Fan-in for DWN (LOW effort, MEDIUM-HIGH novelty)
**Gap**: All existing DWN/DiffLogic papers use uniform fan-in across all layers.
NeuraLUT-Assemble (2025) uses mixed fan-in within a neuron tree structure, but
not across layers, and only for the NeuraLUT architecture.

**Idea**: Allow different fan-in n per DWN layer (e.g., n=[6,4,2] for a
3-layer network). Early layers seeing raw binary inputs may benefit from higher
fan-in; deeper layers may work well with n=2, exponentially reducing LUT cost.

**Implementation**: `DWNModel.__init__` currently takes scalar `lut_n`. Change
to accept a list. RTL already handles per-layer `LUT_N` independently since
each `fixed_dwn_lut_layer` instance has its own `LUT_N` parameter.

**Paper claim**: "Mixed fan-in DWN architectures achieve Pareto-optimal
accuracy-area tradeoffs vs. uniform fan-in designs."

**Novelty check (2026-02-24)**:
- NeuraLUT-Assemble: mixed fan-in *within* neuron trees, not across layers.
  Different architecture (NeuraLUT, not DWN).
- SparseLUT: fixed fan-in per layer, optimises connectivity only.
- **Conclusion**: Per-layer mixed fan-in for DWN is NOT done in existing work.
  This angle remains novel.

**Richer story — combining with SparseLUT insight**:
SparseLUT (arXiv:2503.12829) shows that *which* inputs a neuron reads matters
as much as *how many* inputs it reads — randomly-assigned connections leave
accuracy on the table. Together, these two axes define a richer design space:
  - **How many inputs per layer** (this contribution — mixed fan-in)
  - **Which inputs each neuron reads** (SparseLUT-style connectivity learning)
A combined approach — per-layer fan-in + learnable/optimised connectivity —
would be strictly more expressive than either alone and fills a gap neither
paper addresses. Specifically: use `mapping='learnable'` (already in DWN
codebase) for all layers AND vary `n` per layer. The DWN paper only uses
`mapping='learnable'` for the first layer and random for the rest.

**Effort**: 1–2 months. ~10 line model change + architecture sweep.

---

### #2a — Implementation Steps: Per-layer Fan-in in MASE DWN

#### File 1: `src/chop/nn/dwn/model.py`
Current: `lut_n: int = 6` (scalar, same for all layers, line 45).
Change: Accept `int | list[int]`. Resolve to list in `__init__`.

```python
# Resolve lut_n to per-layer list
if isinstance(lut_n, int):
    lut_ns = [lut_n] * len(hidden_sizes)
else:
    assert len(lut_n) == len(hidden_sizes), "lut_n list length must match hidden_sizes"
    lut_ns = list(lut_n)

# In the layer loop (line 71-74), change n=lut_n → n=lut_ns[i]
for i, out_sz in enumerate(self.hidden_sizes):
    mp = mapping_first if i == 0 else mapping_rest
    lut_layers.append(LUTLayer(input_size=in_sz, output_size=out_sz, n=lut_ns[i], mapping=mp))
    in_sz = out_sz
```

Also consider: accept `mapping` as a list too (one per layer), enabling
`mapping=['learnable', 'learnable', 'random']` etc.

#### File 2: `src/chop/passes/graph/transforms/dwn/quantize.py`
Current: `_graph_iterator_dwn_by_type` applies `n=node_config.get("lut_n", 6)`
uniformly to every Linear node it encounters.
Change: If `lut_n` in config is a list, index it by the order of Linear nodes
encountered (counter variable).

```python
lut_n_cfg = node_config.get("lut_n", 6)
lut_n_list = lut_n_cfg if isinstance(lut_n_cfg, list) else None
linear_counter = 0

for node in graph.fx_graph.nodes:
    ...
    n_for_layer = lut_n_list[linear_counter] if lut_n_list else lut_n_cfg
    new_module = LUTLayer(..., n=n_for_layer, ...)
    linear_counter += 1
```

#### File 3: `test/passes/graph/transforms/dwn/run_dwn_training.py`
Change `--lut-n` argument to accept comma-separated ints.
Parse: `lut_n = [int(x) for x in args.lut_n.split(",")]`
If single value, convert to int for backwards compatibility.
Pass to `DWNModel(lut_n=lut_n, ...)`.

#### File 4: `src/mase_components/dwn_layers/passes.py`
**No changes needed.** Each `LUTLayer` node already has its own `.n` attribute.
`add_verilog_param(node)` reads per-node parameters, so `LUT_N` will be set
correctly per layer automatically.

#### File 5: RTL (`fixed_dwn_lut_layer.sv`)
**No changes needed.** RTL is already parametric in `LUT_N`.
Each emitted module instantiation gets its own `LUT_N` value.

#### File 6: Tests to add/update
- Update `test_emit_verilog_dwn.py`: test a mixed fan-in model
  (e.g., `lut_n=[6, 2]`, 2-layer) — verify both `lut_LUT_N` values appear.
- Update `test_dwn_emit_rtl_equiv.py`: verify end-to-end bit-exact equivalence
  holds for a mixed fan-in model.
- Add `test_mixed_fan_in_training.py`: train a small mixed fan-in model on
  a toy dataset, verify accuracy is close to uniform fan-in baseline.

#### File 7: Sweep script (new)
`scripts/sweep_fan_in.py` — trains all combinations of per-layer fan-in on
MNIST/JSC/NID, emits RTL, runs Vivado synthesis, collects results into CSV.
Combinations to test (3-layer model): all permutations of {2,4,6}^3 = 27 runs.

---

### #3 — Post-Training Boolean Minimisation via ABC (MEDIUM effort, HIGH novelty)
**Gap**: DWN truth tables are frozen after training. If trained LUTs exhibit
sparse Boolean structure (fewer effective inputs than fan-in), ABC/Espresso can
decompose them into smaller functions → fewer physical FPGA LUT primitives with
zero accuracy loss.

**Implementation**: Export truth tables to BLIF/BENCH format (logicnets/bench.py
exists as starting point). Run ABC (`strash; dc2; map`). Replace monolithic
`fixed_dwn_lut_neuron` with minimised netlist. Verify bit-exactness via
existing cocotb equivalence test.

**Paper claim**: "Post-training Boolean minimisation of DWN truth tables reduces
FPGA LUT consumption by X% with guaranteed bit-exact correctness."

**Status**: Not done in any existing paper.

**Effort**: 2–3 months. Highest risk (unknown if trained LUTs have structure).

---

### #4 — Hardware-Aware NAS for DWN (HIGH effort, HIGH novelty)
**Idea**: Use MASE's existing Optuna/RL search infrastructure to jointly
optimise DWN hyperparameters (fan-in per layer, hidden sizes, depth) with a
hardware cost model derived from FPGA synthesis. First NAS applied to weightless
networks.

**Status**: Not done. MASE search infrastructure exists but no DWN search space
defined.

**Effort**: 2–3 months. Risk: small search space may not justify NAS over grid
search.

---

## Strategic Recommendation
**Best paper**: Combine #1 + #2.
- #1 provides baseline story (fan-in comparison).
- #2 provides novel architectural insight (mixed fan-in Pareto dominance).
- Together: "We characterise the fan-in design space for weightless NNs on FPGA
  and show mixed fan-in architectures dominate uniform designs on the Pareto
  frontier."
- Achievable in ~2–3 months using existing MASE codebase.

**If time allows**: Add #3 (logic minimisation) as headline result.

---

---

## Experimental Results

### MNIST Accuracy — Mixed Fan-in Sweep (2026-02-24)

| Config | Fan-in (per layer) | Hidden sizes | Test accuracy | LUTs (Vivado) | Fmax (MHz) |
|---|---|---|---|---|---|
| 1: baseline | n=[6, 6] | [2000, 1000] | **0.9851** | TBD | TBD |
| 2: mixed | n=[6, 2] | [2000, 1000] | 0.9830 | TBD | TBD |
| 3: mixed | n=[6, 4, 2] | [2000, 1000, 500] | 0.9816 | TBD | TBD |

Notes:
- All trained with `mapping_first=learnable`, `mapping_rest=random`, 30 epochs, batch=32, lr=0.01
- Accuracy drops ~0.2–0.35% going from uniform n=6 to mixed fan-in
- Key question: do LUT counts drop proportionally? Expected large savings in later layers
  (n=2 → 4-entry table vs n=6 → 64-entry table per neuron, 16× reduction per neuron)
- Vivado synthesis results pending (target part: xcvc1902-viva1596-3HP-e-S, OOC mode)

---

## Completed Engineering Work
- Fixed two bugs in MASE `emit_verilog` for DWN:
  1. Parameter case mismatch (`LUT_INPUT_SIZE` → `lut_INPUT_SIZE`)
  2. Hex literal quoting (`64'h...` was being wrapped in quotes)
- Added tests:
  - `test_emit_verilog_dwn.py`: structural Verilog check
  - `test_rtl_equiv_from_model.py` + `dwn_lut_layer_equiv_tb.py`:
    component-level RTL/SW equivalence with real model weights
  - `test_dwn_emit_rtl_equiv.py`: full pipeline (train → emit → Verilator →
    bit-exact comparison, 256 random vectors)
- All 6 tests passing.
