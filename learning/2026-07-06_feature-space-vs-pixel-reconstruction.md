# Why feature-space beats pixel reconstruction — the Stage-0 headline

The single most important thing to explain cold: *why* the intuitive method (rebuild the object, flag
what won't rebuild) **failed at localization (AU-PRO 0.095 ≈ random)**, and why "compare to normal in
feature space" **works (0.908)**. Everything else (PatchCore, fusion, the SOTA) hangs off this.

## The setup (unsupervised, train-on-normal)
No labelled defects to train on — the train set is **good only**. So you learn a model of *normal* and
flag *deviation*. Two families of "deviation."

## Approach A — pixel reconstruction (the intuitive one)
Train an autoencoder/VAE to rebuild good samples. At test, anomaly score = **per-pixel reconstruction
residual** `‖input − rebuilt‖²`. Intuition: it learned to rebuild normal, so it *can't* rebuild a
defect → big residual there.

**It fails — and anti-localizes (residual ~9× LOWER on the bagel's actual defect than on normal rim).**
Why: the residual measures **reconstruction difficulty**, which tracks **geometric/visual complexity**,
NOT **defect-ness**:
- A **normal-but-complex** region (the curved bagel rim, structured-light grazing-noise) is *hard to
  rebuild* → high residual → **false positive.**
- A **smooth defect** (a shallow dent, a discoloured patch) *rebuilds fine* → low residual → **false
  negative.**

So the score is dominated by "how hard is this to draw," not "is this wrong." In *raw-pixel space,
complexity swamps the signal.*

## Approach B — compare to normal, in FEATURE space (the fix, 0.908)
Two variants, both land at **0.908 AU-PRO**:
- **Memory bank (PatchCore):** run every good patch through a **frozen pretrained backbone** (ImageNet
  wide-resnet), store those patch **features** in a bank. Score a test patch by the **distance to its
  nearest stored normal feature.** A normal-but-complex rim matches *a normal rim in the bank* → small
  distance → not flagged. A defect has **no close normal match** → large distance → flagged. *No
  reconstruction at all.*
- **Feature reconstruction (feature-recon):** reconstruct the **features**, not the pixels; residual in
  feature space. Also 0.908.

## The insight (the thing to say in an interview)
**Both feature methods = 0.908; both pixel methods ≈ 0.095. So the winning ingredient is NOT
"memory-bank vs reconstruction" — it's the SPACE you measure deviation in.**

A frozen backbone maps images to a **semantic feature space where normal variation (complexity, texture,
curvature) is already compressed** — normal patches cluster tightly. A defect falls **outside** that
normal cluster. So in feature space, *distance-from-normal ≈ defect-ness*. In pixel space,
*distance-from-normal ≈ complexity.* Same idea ("compare to normal"), different space, opposite result.

**Reconstruction was never the wrong idea — raw-pixel space was.** Move the exact same reconstruction
into feature space (feature-recon) and it localizes fine. The lesson generalises: *the representation you
measure deviation in matters more than the algorithm that measures it.*

## Connections
- This is *why* the SOTA is feature-/memory-based, not pixel-reconstruction (measured here, not assumed).
- PatchCore's remaining pieces (coreset, locally-aware features) are efficiency/robustness on top of this
  core idea — a separate topic.
- The RGB+3D **fusion** gap to SOTA is: the *feature space* above is rgb-only; add a geometry feature
  space (FPFH / point-transformer) and fuse → the last ~8pp. Same principle, second modality.

---
## Quiz log
*(appended when quizzed — questions, my answers, honest score)*
