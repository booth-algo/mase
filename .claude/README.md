# .claude/ — DWN FPGA Research Notes

Inspired by AReaL's .claude/ template. Tracks research progress, results, and findings for the DWN FPGA project (Bacellar et al., ICML 2024).

## Structure

```
.claude/
├── README.md                    ← this file
├── research/
│   ├── benchmark.md             ← accuracy vs paper + Vivado LUT counts (primary results)
│   ├── novel_findings.md        ← novel contributions: mixed-N, Pareto, ABC
│   └── future_tasks.md          ← outstanding tasks and ideas
├── results/
│   ├── vivado/                  ← utilization.rpt files from Vivado OOC synthesis
│   │   ├── dwn_*_utilization.rpt
│   │   └── difflogic_*_utilization.rpt
│   └── abc/
│       └── abc_summary.md       ← AND node counts after strash;dc2 minimisation
└── commands/                    ← custom Claude slash commands (if any)
```

## Quick Reference

### Key Results
- **CIFAR-10**: 57.93% (ep87) vs paper 57.42% — ✅ exceeds paper
- **MNIST**: 98.51% vs paper 98.31% — ✅ exceeds paper
- **JSC n=2**: 75.19% at 592 LUTs vs DiffLogic 63.45% at 3,832 LUTs — **+11.7pp, 6.5× cheaper**

### Vivado Access
```bash
source ~/.bashrc  # adds Vivado to PATH
which vivado      # /mnt/applications/Xilinx/24.2/Vivado/2024.2/bin/vivado
# Part: xc7a35tcpg236-1 (Artix-7 WebPACK, licensed via IC servers)
# xcvu9p NOT licensed on this server
```

### Synthesis Command
```bash
vivado -mode batch -source scripts/synth_dwn.tcl \
    -tclargs <rtl_dir> <results_dir> xc7a35tcpg236-1 dwn_top 4.0
```

### Training Command
```bash
conda run -n plena2 python test/passes/graph/transforms/dwn/run_dwn_training.py \
    --dataset mnist --epochs 30 --hidden-sizes 2000 1000 --lut-n 6 \
    --num-bits 3 --tau 33.333 --lr 0.01
```

## Related Files (gitignored, local only)
- `CLAUDE_BENCHMARK.md` → mirrored as `.claude/research/benchmark.md`
- `CLAUDE_NOVEL.md` → mirrored as `.claude/research/novel_findings.md`
- `CLAUDE_FUTURE_TASKS.md` → mirrored as `.claude/research/future_tasks.md`
- Training logs: `mase_output/dwn/*.log`
- Checkpoints: `mase_output/dwn/*.pt`
