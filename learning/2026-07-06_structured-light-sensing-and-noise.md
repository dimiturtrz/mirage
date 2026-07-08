# Structured-light sensing + its noise — where the position map comes from, and how it lies

Lesson 1 said *what* the data is (a position map: a grid of `(x,y,z)`). This one is *how it's made* and
*how it's wrong.* You cannot judge a defect detector without knowing which "defects" are really the
scanner lying. Half of anomaly-method failure is sensor noise mistaken for signal.

## How a structured-light scanner gets 3D from cameras
A camera alone gives 2D — no depth. Structured light adds depth by **projecting a known pattern** onto the
object and watching how it deforms.

1. A **projector** throws a known pattern (stripes / sinusoidal fringes) onto the surface.
2. One or more **cameras**, offset from the projector by a known **baseline**, photograph the lit surface.
3. A flat surface shows the stripes undistorted; a curved/tilted surface **bends** them. The *amount a
   stripe shifts* at each pixel encodes how far that point is.
4. **Triangulation:** projector ray + camera ray + known baseline = one 3D point per pixel (same geometry
   as stereo, but the projector replaces the second camera — the pattern gives guaranteed texture, so
   correspondence is easy even on blank surfaces).

Zivid does this with **phase-shifted fringes** (several patterns projected in sequence, sub-pixel phase
decoded) → sub-100µm `(x,y,z)` per pixel. That's why every valid pixel already has full `(x,y,z)` — the
scanner triangulated it. (Lesson 1's "already un-projected" = this.)

## The core geometry — why noise is not uniform
Depth `z` comes from **disparity** `d` (the stripe shift) via `z ∝ (baseline · focal) / d`. Because `z`
depends on `1/d`, the **depth error grows with `z²`**: `δz ∝ z² · δd / (baseline · focal)`. Two
consequences that matter for every method downstream:
- **Far points are noisier than near points** (quadratically). Depth precision is not constant over the
  scan.
- **Bigger baseline = better depth precision, but more occlusion** (the two views share less surface).
  It's a trade, not a free lunch.

## The noise modes (the ones that masquerade as defects)
These are the failure modes to *name on sight* — each looks defect-like to a naive detector:

- **Flying pixels** — at a **depth discontinuity** (object edge, hole rim) one pixel straddles foreground
  and background; triangulation averages them → a 3D point floating in *empty space* between surfaces.
  Looks like a spike of geometry where there is none. (This is why preprocessing removes isolated points,
  not just background.)
- **Grazing-angle noise** — where the surface tilts nearly parallel to the camera ray, the projected
  stripe smears over many pixels → phase decode is mushy → **noisy, biased `z`.** The curved bagel *rim*
  is exactly this: high geometric complexity **and** high sensor noise, stacked. (This is the physical
  reason pixel-reconstruction false-positives on the rim — lesson 4.)
- **Missing data / dropouts** — **specular** (shiny/metal) surfaces mirror the pattern away from the
  camera; **dark** surfaces absorb it; **occlusion** hides points from projector or camera. No return →
  the pixel is background `(0,0,0)` even though surface is there. A *hole in the data*, not in the object.
- **Multipath / inter-reflection** — in concave regions the pattern bounces surface→surface before
  returning; the decoded phase is corrupted → warped geometry in corners and cavities.
- **Quantization / speckle** — laser/projector speckle + finite sensor resolution add a fine high-freq
  jitter on `z` everywhere, worst where signal is weak (dark, far, grazing).

## Why this lesson is load-bearing for the project
- **Preprocessing is sensor-aware, not arbitrary.** `xyz != 0` drops dropouts; flying-pixel/isolated-point
  removal targets the discontinuity artifact; both are *undoing known sensor lies*, not generic cleanup.
- **The pixel-reconstruction failure is half sensor.** "Residual tracks complexity" (lesson 4) is really
  "residual tracks *complexity + sensor noise*" — the rim is noisy **because** it's grazing-angle. Naming
  the sensor mode makes the headline finding precise, not hand-wavy.
- **Sim-to-real needs a sensor model (Stage 1/2).** A synthetic defect rendered on a *clean* CAD surface
  has none of this noise → a detector trained on it faces a domain gap that is **mostly sensor
  statistics**, not defect appearance. To close the gap you must *inject* flying pixels, grazing noise,
  dropouts into the synthetic render. This is the concrete reason the twin is hard and the reason
  "physics-based" in the pitch is not decoration.
- **A calibrated detector must down-weight noisy regions.** Depth error is `z²`- and angle-dependent →
  the *same* geometric deviation is more trustworthy near/flat than far/grazing. Robust scoring should
  know that (ties to calibration, lesson 6).

## One-line takeaways
- Structured light = projector pattern + camera + triangulation → `(x,y,z)` per pixel.
- Depth noise is **not uniform**: grows as `z²`, worst at edges (flying pixels), grazing angles, shiny/dark
  (dropouts), and concavities (multipath).
- Every one of those noise modes *looks like a defect* — knowing them by name is what separates "detected
  an anomaly" from "detected the scanner."

## Glossary (the names worth knowing — mechanism is the load-bearing part)
| term | one-line |
|---|---|
| **Structured light** | depth by projecting a *known* pattern + watching it deform; projector replaces stereo's 2nd camera |
| **Triangulation** | 3D point = apex where projector ray + camera ray cross; known baseline + 2 angles → solve the triangle |
| **Baseline `B`** | fixed projector↔camera gap; bigger = better depth precision, more occlusion |
| **Disparity `d`** | pattern shift at a pixel (px) — the raw measured thing |
| **Depth `z`** | distance to surface; *computed* from `d` via `z = B·f/d`; never measured directly |
| **Phase-shift fringes** | Zivid pattern — sinusoids in sequence, sub-pixel phase decode → unique label per point → correspondence on blank surface |
| **`z²` law** | `δz ∝ z²·δd/(B·f)` — depth noise grows with square of range; 2× distance → 4× error |

**The mnemonic — what does the surface do to the projected light?** straddle / smear / kill / bounce / jitter:
- **edge** → straddle two depths → **flying pixel** (speck in empty space)
- **tilt (grazing)** → fringe smears → **grazing-angle noise** (rough on smooth)
- **shiny/dark/hidden** → no return → **dropout** ((0,0,0) hole = false missing-material)
- **concave** → light bounces surface→surface → **multipath** (warped/bent corners)
- **always** → interference + quantization → **speckle** (the `δd` floor feeding the `z²` law)

---
## Quiz log

### 2026-07-06 — quiz 1 · score ~3.1/6 (52%)
- **Q1 pipeline** ⚠️0.6 — got "known pattern → deformation reveals shape." Missed **triangulation** (ray+ray+baseline) as the mechanical step; "curves so it's circular" too loose.
- **Q2 z vs d** ✅0.7 — z=depth, d=pattern shift ✅. Missed: scanner *measures* d, *computes* z via `z=B·f/d`.
- **Q3 z² law** ⚠️0.6 — right intuition (far → small pixel diff = big distance). WRONG label ("Pythagoras" — it's the `1/d` relation). Missed the consequence: trust near/flat, distrust far.
- **Q4 flying pixels** ❌0.3 — said "random positions" (the recurring miss). They're at **edges/depth-discontinuities**, mechanism = pixel **straddles** fg+bg → averaged depth → floats in empty space. Got filtering-isolated only.
- **Q5 the rim** ⚠️0.3 — named "grazing" ✅, zero mechanism/lesson-4 link. Rim tilts → fringe smears → noisy; that's *why* pixel-recon false-positives (complexity + sensor noise stacked).
- **Q6 sim-to-real** ⚠️0.6 — correct instinct (missing = noise/reflections) but hedged; sharpen to *specific structured-light modes must be injected*, gap = sensor statistics not defect look.

**Recurring error (same as lesson 1):** names the *outcome*, not the *mechanism + location*. L1: "aggregate views" vs single-view. L2: "random positions" vs "at edges, by straddling." Fix drilled: for each mode, **[where] + [physical cause] + [why it mimics a defect]**. Intuition-first, names as tags (student's own framing — endorsed).
