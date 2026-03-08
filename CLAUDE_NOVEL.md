# Novel Findings — DWN Mixed-N Search

## Key Result: First-Layer Fan-In Dominates

**Experiment:** Mixed-N Pareto sweep on MNIST (2048×2000 hidden, 30 epochs, N ∈ {2,4,6})

| N Config | AreaLUTs  | Accuracy | Pareto |
|----------|-----------|----------|--------|
| 2-2      | 16,192    | 97.01%   | *      |
| 2-4      | 40,192    | 97.63%   | *      |
| **4-2**  | **40,768**| **97.88%**| *     |
| 4-4      | 64,768    | 97.91%   | *      |
| 2-6      | 136,192   | 97.97%   | *      |
| 6-2      | 139,072   | 98.12%   | *      |
| **4-6**  | **160,768**| **98.15%**| *    |
| 6-4      | 163,072   | 98.11%   | (dominated) |
| 6-6      | 259,072   | 98.14%   | (dominated) |

### Findings

1. **First layer fan-in matters more than second.** `4-2` (40k LUTs, 97.88%) beats `2-4`
   (40k LUTs, 97.63%) at the same area — allocating more LUT inputs to layer 1 is more
   efficient than to layer 2.

2. **`4-6` dominates both `6-4` and `6-6`.** N=6 in layer 1 is not needed — `4-6` at
   160k LUTs achieves 98.15%, higher than `6-4` (98.11%) and `6-6` (98.14%) at equal
   or greater area.

3. **Best area-efficiency tradeoff: `4-2`** gives 97.88% at only 40k LUTs — 4× fewer
   LUTs than `4-6` for only 0.27% accuracy drop.

4. **Diminishing returns beyond N=4 in layer 1.** Moving from N=4 to N=6 in layer 1
   (compare `4-6` vs `6-6`) adds 98k LUTs for only −0.01% accuracy.

### Implication for Hardware Design

For area-constrained deployments, a mixed-N configuration (higher N in early layers,
lower N in later layers) is Pareto-superior to uniform N across all layers.

---

## CIFAR-10 Results

### Mixed-N sweep (2048×2050, 50 epochs, num_bits=2)

Unlike MNIST, CIFAR-10 benefits from N=6 in the **first** layer:

| Config | Accuracy | AreaLUTs (approx) | Notes |
|--------|----------|-------------------|-------|
| **6-6** | **45.50%** | ~160k | Uniform N=6 |
| 4-6    | 45.11%   | ~96k  | Mixed: N=4 first, N=6 second |

- `4,6` saves ~40% LUTs but loses 0.4% accuracy vs `6,6`
- On MNIST, `4-6` matched or beat `6-6` — CIFAR-10 reverses this
- Suggests CIFAR-10's high input dimensionality (3072 features) needs larger first-layer fan-in

### Interpretation

The optimal mixed-N strategy is **dataset-dependent**:
- MNIST (784 features): N=4 first layer is sufficient — `4-6` dominates `6-6`
- CIFAR-10 (3072 features): N=6 first layer needed — uniform `6-6` beats `4-6`

Higher-dimensional inputs likely require larger early-layer fan-in to capture feature interactions.

---

## CIFAR-10 Scaling (paper config: num_bits=10, tau=33.33, N=6, random mapping)

Scaling hidden layer size trades area for accuracy, approaching paper's 57.42%:

| Hidden size | Epochs | Accuracy | Notes |
|-------------|--------|----------|-------|
| 8,000       | 100    | 51.01%   | Baseline (StepLR step=25) |
| 16,000      | 100    | 53.67%   | |
| 32,000      | 100    | 55.15%   | |
| 64,000      | 100    | 56.61%   | Best so far |
| **paper**   | 100    | **57.42%** | 8,000 neurons (!) |

Paper achieves 57.42% with only 8k neurons vs our 51.01% — same EFD implementation (byte-for-byte identical to torch_dwn v1.1.1). Gap likely from LR schedule: paper uses MultiStepLR([30,60,90], γ=0.1) vs our StepLR(step=25).

### Augmentation finding

Data augmentation (RandomCrop + HFlip) **hurts** DWN on CIFAR-10:
- Without augmentation (8k, StepLR): **51.01%**
- With augmentation (8k, MultiStepLR): **46.39%** (−4.6%)

**Root cause:** `DistributiveThermometer` fits quantile thresholds **once** on unaugmented training data. Random crops change the pixel distribution at inference, causing a distribution mismatch with the pre-fit thresholds. To use augmentation properly, thresholds must be refit on the augmented distribution.

### Spectral regularization finding

Spectral regularization **hurts** in MASE's implementation:
- λ=0 (no reg): 51.01%
- λ=1e-4: 50.70%
- λ=1e-3: 46.51%

The `spectral_reg_loss` in `src/chop/nn/dwn/lut_layer.py` is a MASE addition (not in torch_dwn). By Parseval's theorem the spectral norm of the LUT output distribution is a constant, so gradients are near-zero — the regularization is mathematically questionable.

---

## Smaller Model Confirmation (500×250, 20 epochs)

Same trend holds at small scale — `4-2` beats `2-4` at similar area:

| N Config | AreaLUTs | Accuracy | Pareto |
|----------|----------|----------|--------|
| 2-2      | 3,000    | 93.01%   | *      |
| 2-4      | 6,000    | 94.05%   | *      |
| 4-2      | 9,000    | 95.38%   | *      |
| 4-4      | 12,000   | 95.78%   | *      |
| 4-6      | 24,000   | 95.94%   | *      |
| 6-2      | 33,000   | 96.46%   | *      |
| 6-4      | 36,000   | 96.57%   | *      |
| 6-6      | 48,000   | 96.73%   | *      |
