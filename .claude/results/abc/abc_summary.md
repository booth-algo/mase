# ABC Boolean Minimisation Results

**Tool**: ABC (`strash; dc2`) — equivalence-preserving, bit-exact
**Date**: 2026-03-10
**Binary**: `~/.local/bin/abc`

## AND Node Counts After Minimisation

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

## Key Observations
- AND-node count scales ~4× per fan-in step (n=2→4→6)
- JSC n=2 (797 nodes) vs n=6 (17,538 nodes): 22× fewer AND nodes at equal accuracy
- Scaling is predictable: can estimate minimised size from n and neuron count
