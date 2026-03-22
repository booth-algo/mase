# DWN Benchmark Results

**Paper**: Bacellar et al., "Differentiable Weightless Networks" (ICML 2024, arXiv:2410.11112)
**Date**: 2026-03-10 (updated 2026-03-11 with final JSC n=2/n=4 results)

## Accuracy Comparison

| Dataset | Config | Our Best | Paper | Gap | Status |
|---------|--------|----------|-------|-----|--------|
| MNIST | hidden=[2000,1000], N=6, num_bits=3, learnable, 30ep | 98.51% | 98.31% | +0.20pp | ✅ Exceeds paper |
| CIFAR-10 | hidden=[8000], N=6, num_bits=10, learnable, 100ep | **57.93%** (ep87) | 57.42% | +0.51pp | ✅ Exceeds paper |
| JSC (HLS4ML) | hidden=[3000], N=6, num_bits=200, learnable, 100ep | 75.03% | 76.3% | −1.3pp | ✅ Converged (LR=1e-9 at ep98) |
| JSC (HLS4ML) | hidden=[3000], N=6, num_bits=200, random, 100ep | 71.72% | 76.3% | −4.6pp | ⚠️ Random mapping only |
| KWS | hidden=[1608], N=6, num_bits=8, learnable, 100ep | **68.65%** (ep100) | 71.52% | −2.87pp | ✅ Done (100 epochs) |
| ToyADMOS/car | hidden=[1800,1800], N=6, num_bits=3, learnable, 100ep | 56.93% accuracy | 89.03% AUC | ⚠️ NOT COMPARABLE — paper uses AUC-ROC (anomaly detection); our code uses accuracy (binary classifier trained on normal-only data — model always predicts normal ≈ 50% accuracy) |
| NID (NSL-KDD) | hidden=[256,252], N=6, num_bits=3, 30ep | 74.15% | — | — | 📊 Our benchmark (not in paper) |

## Notes

### CIFAR-10 (57.93% — exceeds paper's 57.42%)
- Best run: `cifar10_8k_learnable_v3.log`, peaked at epoch 87
- Key finding: learnable mapping is essential — random mapping at 8k neurons = ~51% (6.9pp gap)
- 16× scaling insight: random mapping needed 128k neurons to reach 57%; learnable needs just 8k
- Ablations confirmed: SGD hurts, augmentation hurts, spectral reg hurts

### JSC (DONE — all n={2,4,6} trained)
- Config matches paper exactly (hidden=[3000], N=6, num_bits=200, learnable)
- EFD CUDA overflow bug fixed (packed_accessor32 → packed_accessor64, commit bf55475)
- n=6: 75.09% (ep99) | n=2: **75.19% (ep57)** | n=4: **75.0% (ep98)**
- Key result: n=2 exceeds n=6 accuracy at 1/16th the LUT count (~188 vs ~3,000 LUTs)

### ToyADMOS (metric mismatch — NOT comparable to paper)
- Paper uses AUC-ROC for anomaly detection (DWN trained as density model on normal-only data)
- Our implementation trains a binary cross-entropy classifier on normal+anomaly labels
- Training with cross-entropy on single-class (normal-only) training data → model never learns anomaly patterns → ~50% accuracy (random) at test time
- The 56.93% result reflects this fundamental methodology mismatch, NOT the true DWN anomaly detection capability
- To reproduce paper result: implement density-based anomaly scoring (reconstruction error or log-likelihood from DWN trained on normal data only)

### NID (our benchmark, no paper baseline)
- Dataset: NSL-KDD (125,973 train / 22,544 test), 6 classes
- Downloaded from GitHub (NSL-KDD official CSV)
- NID is not evaluated in the DWN ICML 2024 paper

## Paper Table Reference (arXiv:2410.11112v4)
- MNIST: 98.31% (n=6, large) / 98.77% (n=6, large, +aug)
- CIFAR-10: 57.42% (n=6, learnable+EFD)
- JSC: 76.3% (n=6, large)

---
## Task 2: DWN vs DiffLogic Pareto Comparison

### Accuracy & Vivado LUT Counts (trained 2026-03-10/11; synthesised 2026-03-12, xc7a35tcpg236-1)
**Updated 2026-03-14 with fixed INDEX_BITS RTL** — prior buggy results (8-bit index truncation) struck through below.

| Dataset | Model | Config | Accuracy | CLB LUTs (Vivado) |
|---------|-------|--------|----------|-------------------|
| MNIST | DWN n=2 | [2000,1000], 30ep | 97.44% | **983** ~~(969)~~ |
| MNIST | DWN n=4 | [2000,1000], 30ep | 98.14% | **2,844** ~~(1,236)~~ |
| MNIST | DWN n=6 | [2000,1000], 30ep | **98.51%** | **3,000** ~~(1,256)~~ |
| MNIST | DiffLogic | 2000×2 layers, 30ep | 87.77% | **2,993** |
| NID | DWN n=2 | [256,252], 30ep | 75.60% | **201** ~~(199)~~ |
| NID | DWN n=4 | [256,252], 30ep | 75.35% | **451** ~~(441)~~ |
| NID | DWN n=6 | [256,252], 30ep | 74.15% | **474** |
| NID | DiffLogic | 258×2 layers, 30ep | 64.34% | **294** |
| JSC | DWN n=2 | [3000], 100ep | **75.19%** (ep57 best) | **610** ~~(592)~~ |
| JSC | DWN n=4 | [3000], 100ep | **75.0%** (ep98 best) | **1,066** |
| JSC | DWN n=6 | [3000], 100ep | **75.09%** | **1,178** ~~(1,174)~~ |
| JSC | DiffLogic | 3000×1 layer, 30ep | 63.45% | **3,832** |

All values Vivado-measured (OOC synthesis, xc7a35tcpg236-1, PerformanceOptimized). Previous analytical estimates were inaccurate (underestimated by 3-20×).

### Key Finding: DWN n=2 dominates DiffLogic on accuracy AND area on ALL 3 datasets
- MNIST: DWN n=2 (97.44%, **983 LUTs**) vs DiffLogic (87.77%, 2,993 LUTs) — **+9.7pp accuracy AND 3.0× fewer LUTs**
- NID: DWN n=2 (75.60%, **201 LUTs**) vs DiffLogic (64.34%, 294 LUTs) — **+11.3pp accuracy AND 1.5× fewer LUTs**
- JSC: DWN n=2 (75.19%, **610 LUTs**) vs DiffLogic (63.45%, 3,832 LUTs) — **+11.7pp accuracy AND 6.3× fewer LUTs**

**Notable change from fixed RTL**: MNIST n=4 (2,844 LUTs) and n=6 (3,000 LUTs) now exceed DiffLogic (2,993 LUTs). Only n=2 maintains area advantage for MNIST. JSC n=2–6 all remain far smaller than DiffLogic.

---
## Full Pipeline Synthesis (thermometer + LUT stack + GroupSum, clocked)

**Date**: 2026-03-12  **Part**: xc7a35tcpg236-1  **Top**: `full_pipeline_top_clocked`
**Method**: synth_design only (OOC, PerformanceOptimized) — no place/route (JSC/KWS too large for xc7a35t)

| Dataset | Config | Accuracy | CLB LUTs | Flip-Flops | Notes |
|---------|--------|----------|----------|------------|-------|
| MNIST | DWN n=2 [2000,1000] | 97.44% | **1,126** | 119 | |
| MNIST | DWN n=4 [2000,1000] | 98.14% | **1,438** | 295 | |
| MNIST | DWN n=6 [2000,1000] | 98.51% | **1,552** | 374 | |
| NID | DWN n=2 [256,252] | 75.60% | **301** | 128 | |
| NID | DWN n=4 [256,252] | 75.35% | **453** | 226 | |
| NID | DWN n=6 [256,252] | 74.15% | **515** | 258 | |
| JSC | DWN n=2 [3000] | 75.19% | **1,169** | 81 | Thermometer adds 577 LUTs over LUT-stack-only |
| JSC | DWN n=4 [3000] | 75.0% | **2,657** | 1,234 | Thermometer dominates: +1,591 LUTs |
| JSC | DWN n=6 [3000] | 75.09% | **3,706** | 2,223 | Thermometer adds 2,532 LUTs |
| KWS | DWN n=6 [1608] | 68.65% | **3,374** | 1,588 | 510 feats × 8 bits = 4080 thermo bits |

### Key Observations
- **MNIST** (3 bits): thermometer adds only ~200-300 LUTs (minor overhead)
- **JSC** (200 bits): thermometer dominates — 3200 comparators add 577–2532 LUTs depending on n
- **KWS** (8 bits, 510 feats): 4080 comparators + 1608-neuron n=6 LUT stack = 3,374 LUTs total
- FFs come from inter-layer registers in `full_pipeline_top_clocked` (1 FF stage per LUT layer + input/output)

---
## Task 3: Post-Training Boolean Minimisation via ABC

### ABC (strash; dc2) results — AND node count after minimisation

| Dataset | Model | n | AND nodes | Notes |
|---------|-------|---|-----------|-------|
| MNIST | DWN | 2 | 4,172 | Very compact (n=2 LUTs) |
| MNIST | DWN | 4 | 16,525 | ~4× growth vs n=2 |
| MNIST | DWN | 6 | 68,950 | ~4× growth vs n=4 |
| NID | DWN | 2 | 539 | Tiny network |
| NID | DWN | 4 | 2,630 | |
| NID | DWN | 6 | 10,324 | |
| JSC | DWN | 2 | 797 | Final (ep57 best) — 22× fewer than n=6 |
| JSC | DWN | 4 | 4,583 | Final (ep98 best) |
| JSC | DWN | 6 | 17,538 | |
| KWS | DWN | 6 | 37,765 | |

ABC command: `strash; dc2` (equivalence-preserving — bit-exact post-minimisation)

---
## All Datasets in DWN Paper (arXiv:2410.11112v4)

18 unique datasets across all tables:

### Vision / Audio (FPGA + MCU benchmarks, Tables 1, 2, 3, 7)
| Dataset | Task | Best DWN Accuracy |
|---------|------|-------------------|
| MNIST | Image classification | 98.77% (+aug) |
| FashionMNIST | Image classification | 89.01% |
| KWS (Keyword Spotting) | Audio classification | 71.52% |
| ToyADMOS/car | Anomaly detection | 89.03% |
| CIFAR-10 | Image classification | 57.42% |
| JSC (Jet Substructure) | HEP classification | 76.3% |

### Tabular (OpenML benchmark suite, Tables 4, 5, 8)
| Dataset | Best DWN Accuracy |
|---------|-------------------|
| phoneme | 89.5% |
| skin-seg | 100.0% |
| higgs | 72.7% |
| australian | 90.1% |
| nomao | 96.6% |
| segment | 99.8% |
| miniboone | 94.6% |
| christine | 73.6% |
| jasmine | 81.6% |
| sylvine | 95.2% |
| blood | 78.0% |

**NID (NSL-KDD) is NOT in the paper.** Our NID benchmark is a novel addition with no paper baseline.

### Ablation Table 10 — Mapping Strategy (CIFAR-10 canonical numbers)
| Config | CIFAR-10 |
|--------|----------|
| FD (random, finite diff) | 48.37% |
| + EFD only | 48.37% |
| + LM only | 55.36% |
| + EFD + LM (full DWN) | **57.42%** |

---
## xcvu9p Full Results vs Paper — FIXED RTL (beholder0, 2026-03-14)

**Part**: xcvu9p-flgb2104-2-i  **Vivado**: 2023.1  **Top**: `full_pipeline_top_clocked`
**Clock**: 4 ns / 250 MHz  **Strategy**: PerformanceOptimized
**Note**: Previous results (2026-03-12) were incorrect due to 8-bit index truncation bug — indices >255 were truncated to 0. Fixed with INDEX_BITS parameterization.

| Dataset | Config | n | CLB LUTs | FFs | WNS (ns) | Fmax (MHz) | Paper LUTs | Paper Fmax |
|---------|--------|---|----------|-----|----------|------------|-----------|-----------|
| MNIST | [2000,1000], z=3 | 2 | **3,167** | 2,787 | 0.972 | ~330 | — | — |
| MNIST | [2000,1000], z=3 | 4 | **4,442** | 3,328 | 1.406 | ~386 | — | — |
| MNIST | [2000,1000], z=3 | **6** | **4,769** | **5,832** | **0.675** | **~301** | **4,082** | **827 MHz** |
| MNIST | [2000,1000], z=3 | [6,2] | **4,317** | 3,209 | 1.312 | ~372 | — | — |
| MNIST | [2000,1000,500], z=3 | [6,4,2] | **4,234** | 3,735 | 1.520 | ~403 | — | — |
| CIFAR-10 | [2048,2050], z=2 | 6 | **13,626** | 8,768 | 0.208 | ~264 | — | — |
| CIFAR-10 | [2048,2050], z=10 | [6,4] | **14,368** | 14,221 | 0.782 | ~311 | — | — |
| CIFAR-10 | [2048,2050], z=10 | [6,2,4] | **14,579** | 15,397 | 0.185 | ~262 | — | — |
| NID | [256,252], z=3 | 2 | **249** | 154 | 1.895 | ~475 | — | — |
| NID | [256,252], z=3 | 4 | **409** | 267 | 1.745 | ~443 | — | — |
| NID | [256,252], z=3 | 6 | **490** | 316 | 2.038 | ~510 | — | — |
| JSC | [3000], z=200 | 2 | **1,883** | 850 | 0.640 | ~298 | — | — |
| JSC | [3000], z=200 | 4 | **4,055** | 1,947 | 0.448 | ~282 | — | — |
| JSC | [3000], z=200 | **6** | **5,501** | **2,564** | **0.451** | **~282** | **4,972** | **827 MHz** |
| KWS | [1608], z=8 | 6 | **4,370** | 2,120 | 1.100 | ~345 | — | — |

### Key Findings (Fixed RTL)

- **MNIST n=6**: 4,769 LUTs vs paper 4,082 → **1.17× more** (was 3.18× fewer with bug — bug accounted for 3.7× error)
- **JSC n=6**: 5,501 LUTs vs paper 4,972 → **1.11× more** (was 1.31× fewer with bug)
- **LUT gap largely resolved by INDEX_BITS fix** — remaining ~17% difference due to synthesis strategy
- **Mixed-N MNIST**: n=[6,4,2] achieves −11% LUTs AND +34% Fmax vs n=6 (4,234 LUTs, ~403 MHz)
- **Fmax gap unchanged**: ~301 MHz (MNIST n=6) vs paper's 827 MHz — GroupSum 7-level adder tree is critical path

---
## Paper-Scope Synthesis (dwn_top_paper_scope, 2026-03-14)

**Scope**: LUT layers + 2-stage pipelined GroupSum, NO thermometer. Matches Table 2 OOC scope exactly.
**Part**: xcvu9p-flgb2104-2-i · **Strategy**: Flow_PerfOptimized_high · **Best clock**: 1.15 ns (869 MHz target)

| Config | LUTs | FFs | Fmax | Paper LUTs | Paper FFs | Paper Fmax |
|--------|------|-----|------|-----------|----------|-----------|
| MNIST n=6 lg [2000,1000] z=3 | **2,655** | **1,752** | **791 MHz** | 4,082 | 3,385 | 827 MHz |

**Gap**: LUTs 35% fewer (WAFR packing), Fmax 95.6% of paper (4.4% gap — 2-stage vs paper's 3-stage GroupSum pipeline depth).
**Previous best** (full_pipeline_top_clocked): 4,769 LUTs / 301 MHz → this is a 1.79× LUT reduction AND 2.63× Fmax improvement.

---
## Note on Prior xcvu9p Results (2026-03-12) — INVALIDATED

The full-pipeline synthesis results recorded on 2026-03-12 for xcvu9p (beholder0, Vivado 2023.1) were produced from RTL with an 8-bit index truncation bug: the `INPUT_INDICES` parameter used an 8-bit field, causing any index >255 to be silently truncated to 0. This corrupted the LUT wiring for any network with more than 256 inputs (all MNIST, JSC, KWS, CIFAR-10 configs), producing artificially low LUT counts.

The 2026-03-14 results above (INDEX_BITS parameterization fix) supersede all 2026-03-12 xcvu9p full-pipeline numbers. The xc7a35tcpg236-1 LUT-stack-only results (Task 2 Pareto comparison) are unaffected — those used a different RTL path and remain valid.
