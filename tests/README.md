# Tests — the pyramid

```
        visual smoke     visualization/* + pointcloud-viewer   ← thin, MANUAL, eyeball the maps
      integration        tests/integration/*                   ← module pairs / pipeline chains
   unit                  tests/unit/*                          ← equivalence-class, fast, no GPU/data
```

Wide base, thin top: many cheap unit tests, fewer integration tests, a handful of manual
visual smokes. A bug should be caught by the cheapest layer that can see it.

## unit/ — equivalence-class
One representative per **equivalence class** of inputs (plus boundaries), not exhaustive cases.
The contribution — the eval math (`metrics`, `harness.aggregate`, `scoring`) — lives here, with
no dataset and no GPU: perfect map → AU-PRO ≈ 1, random → the diagonal; top-k score over valid
pixels; the preprocess output contract. Fast, deterministic.

## integration/ — module-pair / pipeline chains
If modules A and B sit in the same pipeline, the **A → B chain** must hold on the same
inputs/outputs the units promise (shape, dtype, mask conventions across the boundary). These
catch interface drift per-unit tests miss — e.g. `preprocess → store schema` agreeing on fields.

Run: `pytest tests/ -q` (from the repo root, in the `mirage` env).
