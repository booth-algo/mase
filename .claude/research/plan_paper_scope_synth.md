# Plan: Paper-Scope Synthesis for All Pareto Configs

## Goal
Synthesize `dwn_top_paper_scope` (LUT layers + pipelined GroupSum, NO thermometer) for all DWN Pareto configs on xcvu9p. This gives a consistent scope across all paper tables matching the original DWN paper's Table 2 OOC methodology.

## Configs Needed

### Already have paper-scope on xcvu9p:
- MNIST n=6 (baseline_n6): 2,655 LUTs, 1,752 FFs, 791 MHz ✅
- JSC n=6 (jsc_learnable_100ep): 6,608 LUTs, 3,048 FFs ✅

### Need to create dwn_top_paper_scope.sv + synthesize:
1. MNIST n=2 (mnist_n2): hidden=[2000,1000], 10 classes, thermo=2352
2. MNIST n=4 (mnist_n4): hidden=[2000,1000], 10 classes, thermo=2352
3. MNIST mixed [6,2] (mixed_n6_2): hidden=[2000,1000], 10 classes, thermo=2352
4. MNIST mixed [6,4,2] (mixed_n6_4_2): hidden=[2000,1000,500], 10 classes, thermo=2352
5. NID n=2 (nid_n2): hidden=[256,252], 6 classes, thermo=366
6. NID n=4 (nid_n4): hidden=[256,252], 6 classes, thermo=366
7. NID n=6 (nid_n6): hidden=[256,252], 6 classes, thermo=366
8. JSC n=2 (jsc_n2): hidden=[3000], 5 classes, thermo=3200
9. JSC n=4 (jsc_n4): hidden=[3000], 5 classes, thermo=3200
10. KWS n=6 (kws_100ep): hidden=[1608], 12 classes, thermo=4080

## Per-Config Parameters for dwn_top_paper_scope.sv

Each wrapper needs:
- Input width = thermo_width (data_in_0)
- LUT output width = last hidden layer size (wire from dwn_top_clocked → GroupSum)
- NUM_GROUPS = num_classes
- Score width = $clog2(last_hidden / num_classes) + 1

| Config | thermo | last_hidden | classes | group_size | score_bits |
|--------|--------|-------------|---------|------------|------------|
| MNIST n=2/4/6, mixed | 2352 | 1000 | 10 | 100 | 7 ($clog2(100)=7) |
| MNIST mixed [6,4,2] | 2352 | 500 | 10 | 50 | 6 ($clog2(50)=6) |
| NID n=2/4/6 | 366 | 252 | 6 | 42 | 6 ($clog2(42)=6) |
| JSC n=2/4/6 | 3200 | 3000 | 5 | 600 | 10 ($clog2(600)=10) |
| KWS n=6 | 4080 | 1608 | 12 | 134 | 8 ($clog2(134)=8) |

Note: score_bits should match what fixed_dwn_groupsum_pipelined computes internally: `$clog2(INPUT_SIZE/NUM_GROUPS) + 1`. Let Vivado figure out the width — use a wide enough output port.

Actually, looking at the MNIST paper-scope that works (baseline_n6), it uses `logic [7:0] data_out_0 [0:9]` = 8-bit scores. The groupsum module computes `$clog2(INPUT_SIZE/NUM_GROUPS) + 1` internally. So the output port width needs to match.

For safety, compute: $clog2(group_size) + 1:
- MNIST 2-layer (group_size=100): $clog2(100)+1 = 7+1 = 8 bits → [7:0]
- MNIST 3-layer [6,4,2] (group_size=50): $clog2(50)+1 = 6+1 = 7 bits → [6:0]
- NID (group_size=42): $clog2(42)+1 = 6+1 = 7 bits → [6:0]
- JSC (group_size=600): $clog2(600)+1 = 10+1 = 11 bits → [10:0]
- KWS (group_size=134): $clog2(134)+1 = 8+1 = 9 bits → [8:0]

## Steps

1. Create dwn_top_paper_scope.sv for each config (10 files)
2. Copy fixed_dwn_groupsum_pipelined.sv to each RTL dir (if not present)
3. Transfer to beholder0
4. Batch synthesis with synth_paper_match.tcl (1.15ns clock for consistency)
5. Pull LUT/FF/Fmax results
6. Update paper tables:
   - Pareto table: replace full_pipeline with paper-scope
   - Mixed fan-in table: replace full_pipeline with paper-scope
   - WAFR table: add JSC n=2 row if strong result
7. Update abstract/discussion ratios

## DiffLogic Scope Note
DiffLogic `difflogic_top` = flatten + logic layers + GroupSum (no thermometer).
This is equivalent to our paper-scope (LUT layers + GroupSum, no thermometer).
DiffLogic xcvu9p numbers (3,184 / 293 / 3,834) are already the right scope. ✅
