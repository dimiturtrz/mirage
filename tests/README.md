# Tests — the pyramid

```
        visual smoke     visualization/* + pointcloud-viewer   ← thin, MANUAL, eyeball the maps
      e2e (RUN_E2E)      tests/e2e/*                           ← real pipeline, 1 cat, GPU+data+torchvision
     integration         tests/integration/*                   ← module pairs / pipeline chains (CI-safe)
   unit                  tests/unit/*                          ← equivalence-class, fast, no GPU/data
```

Wide base, thin top: many cheap unit tests, fewer integration tests, a few gated e2e smokes, a
handful of manual visual smokes. A bug should be caught by the cheapest layer that can see it.

## unit/ — equivalence-class
One representative per **equivalence class** of inputs (plus boundaries), not exhaustive cases.
The contribution — the eval math (`metrics`, `harness.aggregate`, `scoring`) — lives here, with
no dataset and no GPU: perfect map → AU-PRO ≈ 1, random → the diagonal; top-k score over valid
pixels; the preprocess output contract. Fast, deterministic.

## integration/ — module-pair / pipeline chains
If modules A and B sit in the same pipeline, the **A → B chain** must hold on the same
inputs/outputs the units promise (shape, dtype, mask conventions across the boundary). These
catch interface drift per-unit tests miss — e.g. `preprocess → store schema` agreeing on fields.
CI-safe: synthetic/stubbed inputs only, no dataset, no torchvision, no GPU.

## e2e/ — the real pipeline on a real slice (gated)
A handful of smokes that run the **actual method end-to-end** on **one small category** — real
backbone / real store / real GPU — and assert the aggregate is sane (`au_pro ∈ (0,1]`, no NaN,
shapes). This is the layer the synthetic tiers can't reach: it catches "works on fixtures, breaks
on real data/torchvision/CUDA" **before** an expensive multi-category run. Kept tiny (tiny coreset,
one cat) so it's a smoke, not a benchmark.

Gated on `RUN_E2E` so it skips both CI (no torchvision/data/GPU there) and the normal local loop:

```
RUN_E2E=1 uv run pytest tests/e2e -q      # run the smokes (dev box, before a GPU window)
uv run pytest tests -q                     # everything except e2e (CI + normal local)
```

Imports of torchvision-backed methods live **inside** the test bodies so collection stays clean
where torchvision is absent.

## What each new test guards
| test | layer | seam it pins |
|---|---|---|
| `integration/test_method_class_to_harness` | integration | the f7w contract: a config → `AnomalyMethod` **class** → `build_method` → `harness.aggregate` produces a valid, localizing result (not just closures) |
| `integration/test_btf_pipeline` | integration | `FpfhBank.fit` + the parallel batch `score_maps` → `scoring` → `aggregate` on a synthetic cloud (no torchvision) — pins the qfm speedup seam |
| `e2e/test_run_smoke::patchcore` (+ `--amp`) | e2e | real backbone + store + GPU end-to-end; the `--amp` case also closes the "bf16 path not GPU-verified" gap |
| `e2e/test_run_smoke::fused` / `::btf` | e2e | the hybrid (PatchCore-GPU + FPFH-CPU) and geometry paths on real data |

Already CI-safe and covered as units (synthetic): `harness.aggregate` over a fake `(fit,score)`
(`test_harness`), the full train loop stubbed on CPU (`test_train_cpu`), `Trainer` on a tiny model
(`test_trainer`), `preprocess → dataset` (`integration/test_preprocess_to_dataset`).

Run: `pytest tests/ -q` (from the repo root, in the `mirage` env).
