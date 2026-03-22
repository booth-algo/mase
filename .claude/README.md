# .claude/ — DWN FPGA Research Notes

Tracks research progress, results, and findings for the DWN FPGA project (Bacellar et al., ICML 2024).

## Structure

```
.claude/
├── README.md                    ← this file
├── research/
│   ├── benchmark.md             ← accuracy vs paper + Vivado LUT counts (primary results)
│   ├── novel_findings.md        ← novel contributions: mixed-N, Pareto, ABC, WAFR
│   ├── future_tasks.md          ← outstanding tasks and ideas
│   ├── session_2026_03_14.md    ← INDEX_BITS fix + paper-scope synthesis
│   └── session_2026_03_15.md    ← final paper numbers (all xcvu9p)
├── results/
│   ├── vivado/
│   │   ├── fixed_rtl/           ← post-INDEX_BITS-fix xcvu9p results (15 configs)
│   │   └── paper_scope/         ← paper-scope full-impl (best result)
│   └── abc/
│       └── abc_summary.md       ← AND node counts after strash;dc2 minimisation
└── commands/
```

## Quick Reference

### Key Results (all xcvu9p-flgb2104-2-i, Vivado 2023.1)

| Result | Value |
|--------|-------|
| **WAFR packing** | 2,655 LUTs vs paper 4,082 — **1.54x fewer** (MNIST n=6, paper-scope) |
| **Fmax** | 791 MHz vs paper 827 MHz — **95.6% of paper** |
| **DWN vs DiffLogic (JSC)** | +11.7pp accuracy at 1.8x fewer LUTs |
| **Mixed fan-in** | 6-2-4 beats 6-6-6: 2.3x fewer LUTs, +2.16pp accuracy (CIFAR-10) |
| **ABC minimization** | Counterproductive at n=6 (+264% LUTs); neutral at n<=4 |

### Accuracy vs Paper

| Dataset | Ours | Paper |
|---------|------|-------|
| MNIST | **98.51%** | 98.31% |
| CIFAR-10 | **57.93%** | 57.42% |
| JSC | 75.03% | **76.30%** |

### Replication

See `run.sh` in the repo root for complete training, RTL emission, verification, and synthesis commands.

### Vivado Synthesis (beholder0)

```bash
# Paper-scope (LUT layers + pipelined GroupSum, Flow_PerfOptimized_high)
vivado -mode batch -source scripts/synth_paper_match.tcl \
    -tclargs <rtl_dir> <results_dir> xcvu9p-flgb2104-2-i dwn_top_paper_scope 1.15
```

### Checkpoints

Hosted at: https://huggingface.co/booth-algo/dwn-checkpoints
