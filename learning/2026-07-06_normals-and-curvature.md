# Surface normals + curvature — recovering geometry from bare points

Lesson 1: the data is a **set of points** (no surface, no connectivity stored). Lesson 2: those points are
**noisy** in structured ways. This lesson: how you **recover local surface shape** from that bare, noisy
point set — the geometry half of everything downstream. Normals + curvature are the two numbers that turn
"a pile of coordinates" into "a surface you can reason about." FPFH (already covered, `2026-06-25`) is
*built on top of* normals — this is the ground under it.

## The problem
A point cloud stores **only positions**. It does NOT store: which way the surface faces, whether it's flat
or curved, where the edges are. Yet detecting a **dent** or **bump** is exactly a question about *surface
shape*, not position. So you must **compute** shape from the raw points. Two local quantities do it:
- **Normal** — which way the surface **faces** at a point (a direction).
- **Curvature** — how much the surface **bends** at a point (a scalar).

Both are **local** — defined by a point and its **neighbourhood** (the k nearest points, or all points within
radius r). No global mesh needed.

## Surface normal — the direction the surface faces
A **normal** is the unit vector **perpendicular** to the surface at a point (points "straight out" of the
surface). On a table it points up; on a sphere it points radially outward everywhere.

**How you compute it from points (this is the load-bearing bit — it's PCA):**
1. Take a point `p` and its **k nearest neighbours** (say k=30).
2. Those ~30 points form a little local patch. If the surface is locally smooth, they roughly lie on a
   **plane**. Find that best-fit plane.
3. **PCA (principal component analysis)** on the neighbourhood does it: compute the 3×3 **covariance
   matrix** of the neighbour positions (how they spread in x, y, z), take its **eigenvectors**.
   - The **two eigenvectors with the largest eigenvalues** span the plane (the directions the patch
     spreads *most* — along the surface).
   - The **eigenvector with the smallest eigenvalue** is the **normal** — the direction the patch spreads
     *least* (perpendicular to the plane, because a flat patch has ~zero thickness that way).

So a normal is "the direction your local neighbourhood is thinnest." That's why PCA (the eigenvectors of
the neighbourhood spread) gives it directly. **This is why linear algebra is a foundation** (field-map F):
surface normals ARE an eigenvector problem.

**The sign ambiguity:** PCA gives the normal *line* but not which way along it (in vs out). You fix the
sign by pointing all normals toward the **sensor/viewpoint** — we know where the scanner was, and a
visible surface must face it. (Our position map makes this trivial: the scanner is the origin side.)

## Curvature — how much it bends
Once PCA gives you the three eigenvalues `λ0 ≤ λ1 ≤ λ2`, curvature falls out of the **same computation**:
```
surface variation  σ = λ0 / (λ0 + λ1 + λ2)
```
- **Flat patch** → neighbours lie almost perfectly on the plane → `λ0 ≈ 0` (near-zero thickness
  perpendicular) → `σ ≈ 0`. Low curvature.
- **Sharp/curved patch** (edge, corner, dent lip) → neighbours DON'T fit one plane → `λ0` is a real
  fraction → `σ` large. High curvature.

So one PCA per point gives **both** the normal (smallest eigenvector) and curvature (smallest-eigenvalue
ratio). One neighbourhood, one eigendecomposition, both numbers. Cheap, and the whole cloud does it in
parallel (batched — every point's neighbourhood independent → GPU-friendly).

## Why this matters for anomaly detection
- **Defects are curvature/normal events.** A dent = a local normal that points the wrong way + a curvature
  spike at its lip. A scratch = a thin high-curvature line. A bump = normals splaying outward. Raw *position*
  barely moves (a 0.2mm dent is invisible in xyz magnitude); the *normal/curvature* screams. This is why
  geometry features beat raw coordinates for surface defects.
- **FPFH sits directly on normals** (`2026-06-25`): FPFH describes a point by the **distribution of normal
  differences** between it and its neighbours — i.e. "how do the surrounding normals twist relative to
  mine." No normals → no FPFH. This lesson is the missing floor under that note.
- **Normals feed the good detectors' geometry channel.** The RGB+3D fusion SOTA (lesson 7) computes point
  features from normals/FPFH as the "3D half" that fuses with the rgb half. The ~8pp gap to SOTA is
  literally "add a normal-based geometry feature space."

## The sensor-noise coupling (ties to lesson 2)
Normals are a **neighbourhood** estimate → they **amplify sensor noise**:
- **Grazing-angle noise** (lesson 2) jitters the neighbour positions → the fitted plane wobbles → **normal
  estimate is noisy exactly where the sensor was already noisy** (curved rims). The rim gets a double hit:
  physically curved (real high curvature) AND noisy normals (sensor). This is a *third* reason the rim is
  where pixel-methods false-positive.
- **Neighbourhood size (k or r) is a bias/variance knob:** too small → normals fit noise (jittery); too
  large → normals smooth over real features (a small dent averaged away). Choosing k is choosing the scale
  of defect you can see. No universal k — it's set by expected defect size vs point density.

## One-line takeaways
- **Normal** = direction surface faces = **smallest-eigenvalue eigenvector** of the neighbourhood (PCA).
- **Curvature** = how much it bends = **smallest eigenvalue / sum** (surface variation), *same* PCA.
- Defects are **normal/curvature** events, not position events — that's why geometry features win.
- Normal estimates **inherit sensor noise** (grazing rim → noisy normals); neighbourhood size sets the
  defect scale you can resolve.

---
## Quiz log
*(appended when quizzed — questions, my answers, honest score)*
