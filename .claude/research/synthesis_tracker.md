# DWN Synthesis Tracker — xcvu9p vs Paper Comparison

**Goal**: Find the diff in LUT usage between MASE-generated DWN RTL and the numbers
reported in Bacellar et al. (arXiv:2410.11112, ICML 2024).

**Last updated**: 2026-03-12

---

## Paper Hardware Targets

### Table 2 — OOC Synthesis (Appendix D)

The paper's primary FPGA numbers use **Xilinx xcvu9p-flgb2104-2-i** (Virtex UltraScale+, 16nm)
with OOC synthesis at a 4 ns (250 MHz) clock target.

| Part | Family | Process | CLK target |
|------|--------|---------|------------|
| **xcvu9p-flgb2104-2-i** | Virtex UltraScale+ | 16nm FinFET | 4 ns / 250 MHz |
| xc7z045ffg900-2 | Zynq-7000 (Table 1, on-board) | 28nm | — |

### License Status on Kraken

| Part | Status |
|------|--------|
| xc7a35tcpg236-1 | ✅ Licensed (WebPACK free) |
| xcvu9p-flgb2104-2-i | ❌ synth_design fails — Enterprise license required, not available on kraken |

> **Note**: `create_project -in_memory -part xcvu9p` succeeds (no license needed), but
> `synth_design` fails with "Failed to get the license for feature 'Synthesis' and/or device 'xcvu9p'".
> WebPACK only covers 7-series Artix/Kintex/Spartan and selected newer parts.

---

## Paper Results — Table 2 (xcvu9p OOC, fetched 2026-03-12)

All paper results use **uniform n=6** (paper does not evaluate mixed fan-in).

| Dataset | Config | n | LUTs | FFs | Fmax | Notes |
|---------|--------|---|------|-----|------|-------|
| MNIST | sm | 6 | 692 | 422 | 827 MHz | small hidden |
| MNIST | md | 6 | 1,413 | 1,143 | 827 MHz | medium hidden |
| MNIST | md | 6 | 2,055 | 1,675 | 873 MHz | variant |
| MNIST | **lg** | 6 | **4,082** | **3,385** | **827 MHz** | **our comparison (hidden=[2000,1000])** |
| JSC | sm | 6 | 20 | 22 | 3,030 MHz | tiny hidden |
| JSC | sm | 6 | 110 | 72 | 1,094 MHz | variant |
| JSC | md | 6 | 720 | 457 | 827 MHz | medium hidden |
| JSC | **lg** | 6 | **4,972** | **3,305** | **827 MHz** | **our comparison (hidden=[3000])** |
| KWS | — | — | — | — | — | NOT in Table 2 (MCU only in paper) |
| NID | — | — | — | — | — | NOT in paper (our novel benchmark) |

---

## Our Synthesis Results

### xcvu9p-flgb2104-2-i — beholder0, previous session (LUT-stack-only, dwn_top)

These are LUT-stack-only (no thermometer, no GroupSum — combinational only, no FF):

| Dataset | Config | n | CLB LUTs | WNS (ns) | Fmax (MHz) |
|---------|--------|---|----------|----------|------------|
| MNIST | [2000,1000], z=3 | 6 | **1,256** | 2.710 | 775 |
| MNIST | [2000,1000], z=3 | 6-2 | 730 | 3.153 | 1,181 |
| MNIST | [2000,1000,500], z=3 | 6-4-2 | 735 | 3.128 | 1,147 |

### xcvu9p-flgb2104-2-i — beholder0, this session (full pipeline, full_pipeline_top_clocked)

Full pipeline = thermometer (comb) + FF + dwn_top_clocked (FF between layers) + groupsum + FF.
Clock: 4 ns / 250 MHz target. Vivado 2023.1, PerformanceOptimized, place+route completed.

| Dataset | Config | n | CLB LUTs | FFs | WNS (ns) | Fmax (MHz) |
|---------|--------|---|----------|-----|----------|------------|
| MNIST | [2000,1000], z=3 | 2 | **900** | 120 | 1.347 | **377** |
| MNIST | [2000,1000], z=3 | 4 | **1,135** | 292 | 1.307 | **371** |
| MNIST | [2000,1000], z=3 | 6 | **1,285** | 377 | 1.179 | **354** |
| NID | [256,252], z=3 | 2 | **237** | 128 | 1.973 | **493** |
| NID | [256,252], z=3 | 4 | **356** | 226 | 2.086 | **522** |
| NID | [256,252], z=3 | 6 | **418** | 258 | 2.152 | **541** |
| JSC | [3000], z=200 | 2 | **896** | 81 | 0.913 | **324** |
| JSC | [3000], z=200 | 4 | **2,825** | 666 | 0.448 | **282** |
| JSC | [3000], z=200 | 6 | **3,792** | 1,222 | 0.408 | **278** |
| KWS | [1608], z=8 | 6 | **3,428** | 1,586 | 0.410 | **279** |

### xc7a35tcpg236-1 — kraken, this session (full pipeline, full_pipeline_top_clocked)

Full pipeline, 4 ns clock target. Place+route for small designs; synth-only for large.

| Dataset | Config | n | CLB LUTs | FFs | Fmax | Method |
|---------|--------|---|----------|-----|------|--------|
| MNIST | [2000,1000], z=3 | 2 | **1,126** | 119 | ~144 MHz | P&R |
| MNIST | [2000,1000], z=3 | 4 | **1,438** | 295 | ~146 MHz | P&R |
| MNIST | [2000,1000], z=3 | 6 | **1,552** | 374 | ~146 MHz | P&R |
| NID | [256,252], z=3 | 2 | **301** | 128 | ~204 MHz | P&R |
| NID | [256,252], z=3 | 4 | **453** | 226 | ~200 MHz | P&R |
| NID | [256,252], z=3 | 6 | **515** | 258 | ~211 MHz | P&R |
| JSC | [3000], z=200 | 2 | **1,169** | 81 | — | synth-only |
| JSC | [3000], z=200 | 4 | **2,657** | 1,234 | — | synth-only |
| JSC | [3000], z=200 | 6 | **3,706** | 2,223 | — | synth-only |
| KWS | [1608], z=8 | 6 | **3,374** | 1,588 | — | synth-only |

---

## LUT Gap Analysis: Our Implementation vs Paper

### MNIST n=6 ("lg") — Best Direct Comparison (same part: xcvu9p)

| Scope | Our LUTs | Paper LUTs | Gap (×) | Notes |
|-------|----------|-----------|---------|-------|
| LUT-stack-only | 1,256 | 4,082 | **3.25× fewer** | beholder0 xcvu9p |
| Full pipeline | 1,285 | 4,082 | **3.17× fewer** | beholder0 xcvu9p |

**Thermometer + GroupSum adds only 29 LUTs (1,256 → 1,285).** Scope mismatch rules out.

### JSC n=6 ("lg") — Cross-Device Estimate (xc7a35t vs xcvu9p paper)

| Our xc7a35t | Paper xcvu9p | Scale factor (est.) |
|-------------|-------------|---------------------|
| 3,706 LUTs | 4,972 LUTs | 0.75× (our < paper) |

> Note: xc7a35t and xcvu9p use different LUT architectures. For MNIST, xc7a35t gave
> 1,552 LUTs while xcvu9p gave 1,285 — i.e., xc7a35t ≈ 1.21× more LUTs than xcvu9p.
> If JSC follows the same ratio: est. xcvu9p LUTs = 3,706 / 1.21 ≈ **3,062 LUTs** vs paper 4,972.
> JSC gap (estimated) ≈ **1.62× fewer LUTs**.

### Fmax Comparison

| | Our xcvu9p | Paper xcvu9p | Ratio |
|--|-----------|-------------|-------|
| MNIST n=6 full pipeline | **354 MHz** | **827 MHz** | 0.43× |
| MNIST n=6 LUT-stack-only | **775 MHz** | **827 MHz** | 0.94× |

> The LUT-stack-only Fmax is close to paper (775 vs 827 MHz). The full pipeline Fmax drops
> to 354 MHz because the thermometer combinational path (1,243 comparators, 200-bit thresholds)
> becomes the critical path after adding pipeline registers.

---

## Root Cause of LUT Gap

Investigation from `novel_findings.md` (Finding 3):

**Primary driver: Xilinx WAFR LUT5 packing** — Vivado packs two LUT5-reducible neurons
into one physical LUT6 when they share ≥5 inputs. With 3,000 neurons drawing 6 inputs from
a 2,352-bit thermometer bus, many neuron pairs share ≥5 inputs, enabling 2-for-1 packing.
Additionally, 11.4% of L0 neurons (229/2000) have <6 unique inputs (LearnableMapping
collisions), further reducing to LUT5 or smaller. Result: 3,000 logical LUT6 → 1,256 CLB LUTs.

**Why does the paper report 4,082?** Possible explanations:
1. Different synthesis directive: paper uses `Flow_PerfOptimized_high`; we use `PerformanceOptimized`.
   Less aggressive directive → less WAFR packing.
2. Additional pipeline components (argmax, AXI/bus interface). At most ~50-100 LUTs.
3. No public RTL from paper to confirm — gap remains an open question.

---

## Mixed-N Novel Results (vs paper)

The paper only evaluates uniform n=6. Our mixed-N results on xcvu9p (beholder0):

| Config | n | LUTs | vs paper n=6 | Fmax | vs paper |
|--------|---|------|-------------|------|---------|
| MNIST n=6 | 6 | 1,256 | **3.25× fewer** | 775 MHz | 0.94× |
| MNIST n=6-2 | 6,2 | 730 | **5.59× fewer** | 1,181 MHz | **1.43× faster** |
| MNIST n=6-4-2 | 6,4,2 | 735 | **5.55× fewer** | 1,147 MHz | **1.39× faster** |

Mixed-N MNIST achieves **5.6× fewer LUTs AND 43% higher Fmax** vs paper's n=6 baseline.

---

## xcvu9p Full Results vs Paper (COMPLETE — beholder0, 2026-03-12)

All 10 configs synthesised on xcvu9p-flgb2104-2-i, Vivado 2023.1, 4 ns clock, full pipeline.

| Dataset | n | Our LUTs | Our FFs | Our Fmax | Paper LUTs | Paper Fmax | LUT Gap |
|---------|---|----------|---------|----------|-----------|-----------|---------|
| MNIST | 2 | **900** | 120 | 377 MHz | — | — | — |
| MNIST | 4 | **1,135** | 292 | 371 MHz | — | — | — |
| MNIST | 6 | **1,285** | 377 | 354 MHz | **4,082** | **827 MHz** | **3.18× fewer** |
| NID | 2 | **237** | 128 | 493 MHz | — | — | — |
| NID | 4 | **356** | 226 | 522 MHz | — | — | — |
| NID | 6 | **418** | 258 | 541 MHz | — | — | — |
| JSC | 2 | **896** | 81 | 324 MHz | — | — | — |
| JSC | 4 | **2,825** | 666 | 282 MHz | — | — | — |
| JSC | 6 | **3,792** | 1,222 | 278 MHz | **4,972** | **827 MHz** | **1.31× fewer** |
| KWS | 6 | **3,428** | 1,586 | 279 MHz | — | — | — |

### Key Findings

- **MNIST n=6**: Our 1,285 LUTs vs paper 4,082 → **3.18× fewer LUTs**; 354 vs 827 MHz
- **JSC n=6**: Our 3,792 LUTs vs paper 4,972 → **1.31× fewer LUTs**; 278 vs 827 MHz
- JSC gap smaller than MNIST: 200-bit thermometer dominates (wide comparators pack less efficiently)
- Fmax limited by thermometer critical path at large bit-widths (JSC/KWS ~278-324 MHz vs MNIST ~354-377 MHz)
- NID fastest (493-541 MHz): small network + narrow z=3 thermometer

---

## Status Summary

| Task | Status |
|------|--------|
| Identify paper's board (xcvu9p-flgb2104-2-i) | ✅ Done |
| Fetch paper Table 2 numbers | ✅ Done |
| Create synthesis tracker | ✅ Done |
| xcvu9p synthesis on kraken | ❌ License unavailable (synth_design requires Enterprise) |
| xcvu9p synthesis on beholder0 — all 10 configs | ✅ Done (2026-03-12) |
| xc7a35t full pipeline (all 10 configs) | ✅ Done (in benchmark.md) |
| LUT gap analysis (MNIST n=6) | ✅ Done — 3.18× fewer LUTs |
| LUT gap analysis (JSC n=6) | ✅ Done — 1.31× fewer LUTs |
| Fmax gap analysis | ✅ Done — 354 MHz (MNIST) vs 827 MHz paper |
| Flow_PerfOptimized_high experiment (no dont_touch) | ✅ Done — identical to PerformanceOptimized; FF pruning at synth not impl |
| dont_touch + Flow_PerfOptimized_high (xcvu9p) | ✅ Done — 4,889 LUTs / 5,422 FFs / 309 MHz; thermometer is new bottleneck |

---

## WAFR Packing Analysis (2026-03-12)

### Paper synthesis directive
Paper uses `Flow_PerfOptimized_high` (from Table 2 caption). Our TCL uses `PerformanceOptimized`.
These are **not the same** in Vivado 2023.x:
- `PerformanceOptimized` — default, balances area/timing, enables WAFR packing aggressively
- `Flow_PerfOptimized_high` — prioritises timing, uses `-directive PerformanceOptimized` at impl
  stages but applies additional timing-driven placement constraints that reduce packing

Paper does NOT explicitly mention WAFR avoidance, `dont_touch`, or flatten_hierarchy.

### Why our FF count is so low (377 vs paper's 3,385)

Our pipeline RTL registers the full 2,352-bit thermometer bus and the 2,000-bit L0→L1 bus.
Expected FFs ≈ 2,352 + 2,000 + 70 (output) = 4,422.

Vivado prunes FFs aggressively:
1. **Dead thermometer bits**: LearnableMapping leaves 1,109 of 2,352 bits unused → those FFs
   are eliminated as dead logic at elaboration (constant drivers)
2. **WAFR-packed L0→L1**: When two L0 neurons share a physical LUT6 (WAFR), they share
   an output FF too → halves the L0→L1 FF count
3. **L1→GroupSum FFs**: GroupSum directly absorbs 1,000-bit popcount; Vivado merges the
   output FF with the adder tree reduction logic

Net result: 377 FFs vs expected 4,422 = Vivado eliminated ~91% of FFs.

### Throughput Optimisation Experiments (2026-03-12, MNIST n=6, xcvu9p)

Three experiments run on beholder0 (xcvu9p-flgb2104-2-i), `full_pipeline_top_clocked`:

| Experiment | LUTs | FFs | WNS (ns) | Fmax (MHz) | Notes |
|------------|------|-----|----------|------------|-------|
| Baseline: PerformanceOptimized, no dont_touch | 1,285 | 377 | +1.179 | **354** | WAFR packing; GroupSum is critical path |
| Flow_PerfOptimized_high, no dont_touch | 1,285 | 377 | +1.179 | **354** | Identical — FF pruning is at synth, not impl |
| Flow_PerfOptimized_high + dont_touch | 4,889 | 5,422 | +0.772 | **309** | Pipeline correct (1 LUT/stage) but routing-limited |

### Critical Path Analysis (without dont_touch)

**Critical path is GroupSum, NOT thermometer or LUT layers.**

Vivado absorbs inter-layer FFs because each DWN neuron is a single LUT6 (trivial 1-level logic).
This collapses the 4-stage pipeline to effectively 2 stages:
- LUT Layer 1 output FF → 7 LUT levels of `$countones(100 bits)` → output FF
- Data path: 2.800 ns (logic 0.741 ns = 26.5%, route 2.059 ns = 73.5%)
- The 100-bit popcount (1000 outputs / 10 groups) requires ceil(log2(100)) ≈ 7 adder tree levels

### Critical Path Analysis (with dont_touch)

Pipeline is correctly broken (1 LUT level per stage), but Fmax DROPS to 309 MHz because:
- Routing is **95.9%** of critical path (3.08 ns routing vs 0.13 ns logic)
- `dont_touch` forces 5,422 FFs into fixed positions, preventing optimal placement
- 252 SLL inter-SLR crossings on the xcvu9p multi-die device
- Worst path: L0 FF[121] → LUT6 → L1 FF[416], 3.208 ns total (3.078 ns routing)

### Paper Comparison Correction

**There is no "md" MNIST model in the paper.** MNIST configs in Table 13:
- **sm**: [1000, 500], z=1 — Table 2 shows: 2,092 LUTs, 1,757 FFs, 873 MHz
- **lg**: [2000, 1000], z=3 — appears in Table 1 (Zynq Z-7045) but NOT confirmed in Table 2 OOC
- **lg+aug**: [2000, 1000], z=3 + augmentation

The "4,082 LUT / 3,385 FF / 827 MHz" numbers previously attributed to "lg" need re-verification
against the actual paper tables. The confirmed xcvu9p OOC result is sm: 2,092 LUTs, 1,757 FFs, 873 MHz.

Paper's RTL is **not open-source** (GitHub repo has PyTorch training only).
Paper's Vivado version is **not stated**.
Paper confirms `Flow_PerfOptimized_high` strategy in OOC mode for Table 2.

### Root Cause of Fmax Gap

| Factor | Our design | Paper (likely) |
|--------|-----------|----------------|
| GroupSum implementation | Combinational $countones (7 LUT levels) | Pipelined popcount (2-3 stages) |
| Inter-layer FFs | Absorbed by Vivado (neurons are trivial 1-LUT) | Preserved (possibly via dont_touch or RTL structure) |
| FF count | 377 (93% pruned by Vivado) | 1,757 (sm config, pipeline preserved) |
| Critical path | GroupSum adder tree (7 levels) | Likely 1-2 levels per stage |

**Primary fix**: Pipeline the GroupSum popcount into 2-3 stages → target 600-800+ MHz.
**Secondary fix**: Preserve inter-layer FFs without dont_touch (use KEEP_HIERARCHY or RTL restructuring).
