# learning/

My study material + quiz logs for mirage — my ramp into 3D perception. One file per topic:
the **theory writeup** (what I study), then a dated **quiz log** (questions, my answers, honest
assessment) appended when I ask to be quizzed.

These are the "theory" half of the build (see [../docs/PLAN.md](../docs/PLAN.md)) — learning-in-public,
public at the Stage-0 flip. The 3D-geometry topics are the real ramp; the DL building blocks
(conv/ViT/transformers/VAE/contrastive/distillation) I already have and don't re-study.

## The circuit (how each topic is learned)
1. **Research** — ground the topic (internal knowledge + web) → `../research/` (cited raw material).
2. **Theory** — the study writeup here → `<date>_<topic>.md`.
3. **Quiz (on demand)** — when I say *"quiz me,"* open-form questions on that theory.
4. **Log** — my answers + honest assessment + score appended to the topic file.
5. **Sharpen** — set the next concrete step.

**Honest by rule:** domain facts are source-grounded, not asserted from memory; wrong answers stay
in the log — the ramp is real, the self-eval is honest.

## Planned topics (build order)
1. ✅ **FPFH + point-cloud neighborhoods** — the BTF baseline's foundation ([2026-06-25](2026-06-25_fpfh-and-neighborhoods.md)).
2. ⬜ **Memory bank + coreset + AU-PRO** — PatchCore skeleton + the eval metric (the contribution).
3. ⬜ **Reconstruction pitfalls + anomaly synthesis** — why naive AEs fail here; DRAEM → DAS3D (the VAE + the engine).
4. ⬜ **Point-cloud backbones** — PointNet++ → PointMAE (the M3DM ceiling).
