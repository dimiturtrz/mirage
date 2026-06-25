# Point-cloud neighborhoods, normals, and FPFH (the BTF foundation)

The first 3D-geometry ramp topic. Goal: understand the machinery under the **BTF baseline**
(`baselines/btf/`, the quarantined SOTA-floor) — classical geometric features in a memory bank.
Everything here is Open3D one-liners in practice; the point is to know *what they compute and why*.

Tie-in: BTF = "Back to the Feature" (Horwitz & Hoshen, VAND/CVPR-W 2023) — hand-crafted **FPFH**
descriptors + a PatchCore-style memory bank ≈ deep SOTA on MVTec 3D-AD (AU-PRO 0.964), no DL.
This note is the *features* half; the *memory bank + AU-PRO* half is the next topic.

Lesson plan:
1. neighborhoods — kNN / radius, KD-tree, why search at all (organized grid vs unordered points)
2. surface normals — PCA on the local neighborhood; the orientation ambiguity
3. PFH → FPFH — the Darboux frame, the 3 angular features, the histogram, FPFH's speedup
4. FPFH as an anomaly signal — why a dent/hole moves the descriptor

---

## L1 — neighborhoods (the substrate)

Every geometric descriptor is **local**: it summarizes a point's neighborhood. So step zero is
"who are my neighbors." Two definitions:
- **kNN** — the k nearest points. Adaptive density, fixed count.
- **radius (ball)** — all points within r. Fixed physical scale, variable count.
- **hybrid** (Open3D default) — radius capped at max_nn. Best of both.

**Cost.** Naive neighbor search is O(N²). A **KD-tree** (axis-aligned binary space partition) gets
each query to ~O(log N). Open3D builds one under the hood (`o3d.geometry.KDTreeFlann`).

**Our data has a shortcut — and a subtlety.** MVTec scans are an *organized* grid (256×256), so
on-grid neighbors are free (adjacent pixels). But adjacency in *pixel* space ≠ adjacency in *3D*
space at depth edges (two pixels next to each other can be far apart in xyz — the flying-pixel
problem again). FPFH/BTF work on the **unordered point cloud** (`xyz[valid]`, ~50k points/sample
after our cleanup), so they pay for the KD-tree and get true 3D neighbors. The organized grid is
the VAE's friend (fixed tensor); the point cloud is the geometric-feature methods' friend.

**Scale is a hyperparameter.** Neighborhood radius sets what "local" means: too small → noise, too
big → the descriptor blurs across the whole object and stops being discriminative. The same defect
looks different at different radii. (BTF picks a radius per the point spacing.)

## L2 — surface normals (orientation of the local surface)

A **normal** is the unit vector perpendicular to the surface at a point. Compute it from the
neighborhood:
1. Take the point's k neighbors → their **3×3 covariance matrix**.
2. **PCA** (eigendecomposition). The two large eigenvalues span the local tangent plane; the
   **eigenvector of the smallest eigenvalue** = the direction of least spread = **the normal**.
   (A locally-flat patch has almost no variance perpendicular to itself.)

**Orientation ambiguity.** PCA gives the normal's *line*, not its *sign* — it could point in or out.
Resolve by consistency or by pointing toward the sensor (we know the scanner location → 
`orient_normals_towards_camera_location`). Matters because FPFH uses the relative angle between two
normals; flipped signs corrupt it.

**Curvature, free.** The eigenvalues also give surface variation: `λ_min / (λ_0+λ_1+λ_2)` ≈ how
non-flat the patch is. A crack/edge spikes it. (A cheap anomaly cue on its own.)

Open3D: `pcd.estimate_normals(KDTreeSearchParamHybrid(radius, max_nn))`.

## L3 — PFH → FPFH (the descriptor)

The idea: describe a point by **how its surface bends relative to its neighbors** — captured through
the relationship between *normals*, not raw coordinates (so it's pose-invariant).

**The Darboux frame** between a source point `p_s` (normal `n_s`) and a target `p_t` (normal `n_t`),
`d = p_t − p_s`:
```
u = n_s
v = u × d / ‖u × d‖
w = u × v
```
This is a local coordinate frame anchored at `p_s`. In it, three angles describe the pair:
```
α = v · n_t                  (how n_t tilts off the v axis)
φ = u · d / ‖d‖              (direction to the neighbor vs the surface normal)
θ = atan2(w · n_t, u · n_t)  (twist of n_t around u)
```
These three numbers are **translation- and rotation-invariant** — they depend only on the relative
geometry of the two surface patches. That's the whole trick: a bagel rotated/shifted gives the same
features, so "normal" is learned once regardless of pose.

**PFH (Point Feature Histogram).** For point `p`: compute (α, φ, θ) for **every pair** in its
neighborhood, bin into a 3D histogram. Discriminative but **O(k²)** per point — too slow.

**FPFH (Fast PFH)** — the speedup BTF uses:
1. **SPFH** (Simplified PFH): for each point, compute (α, φ, θ) only between `p` and **its direct
   neighbors** (not all pairs). O(k).
2. **FPFH(p) = SPFH(p) + (1/k) Σ_i (1/d_i) · SPFH(neighbor_i)** — add the distance-weighted SPFHs of
   the neighbors. Re-uses each point's SPFH, so the whole cloud is O(N·k), not O(N·k²).
The result: a fixed-length histogram per point (Open3D: **33-D** = 11 bins × 3 angles).
`o3d.pipelines.registration.compute_fpfh_feature(pcd, KDTreeSearchParamHybrid(...))`.

(FPFH was invented for *registration* — matching points between two scans — which is why it lives in
Open3D's registration module. Anomaly detection borrows it as a per-point surface descriptor.)

## L4 — FPFH as an anomaly signal (why BTF works)

BTF's pipeline:
1. On all **normal** training scans: compute FPFH per point → throw them in a **memory bank**
   (coreset-subsampled, next topic).
2. At test: each point's FPFH → **nearest-neighbor distance** to the bank → anomaly score.
   Far from every normal descriptor = anomalous. Per-point → a pixel-level map → AU-PRO.

**Why geometric defects light up.** A dent, hole, or crack changes the **local surface shape** →
changes the **normals** → changes the **angular features** → the FPFH histogram moves away from
anything seen in normal data → large NN distance. RGB never sees a same-color dent; FPFH does. That
asymmetry is the whole reason 3D beats RGB-only on these defects.

**Why it's the right baseline for us.** No training, no pretraining, fast, near-SOTA, **geometry-only**
— so it's an apples-ish floor for our **geometry-first VAE**, and it's the memory-bank "commodity
pipeline" the plan says to *use, not rebuild*. We run it quarantined, score it through our own eval,
and report our deployable VAE against it (the systole nnU-Net move).

---

## Takeaways
- **Neighborhood → normal → pairwise normal-angles → histogram.** That chain is FPFH; each step is
  PCA / cross-products / binning, no learning.
- **Normals = PCA smallest-eigenvector**, sign fixed toward the sensor; eigenvalues also give free
  curvature.
- **FPFH is pose-invariant** (relative angles only) and **fast** (SPFH + weighted neighbor sum,
  O(N·k)). 33-D per point in Open3D.
- **Defects move the descriptor** because they bend the local surface → BTF flags them by
  nearest-neighbor distance to a bank of normal descriptors.
- For us: BTF = the geometry baseline; the cloud (not the grid) is its substrate; Open3D supplies
  normals + FPFH directly.

## Open questions / to verify when building BTF
- ⚠️ exact neighborhood radii BTF uses (point-spacing-relative) — read the 3D-ADS repo.
- ⚠️ whether BTF fuses RGB (the paper's "almost" — strongest variant adds raw color) — confirm which
  variant we quarantine.
- How our cleanup (centroid clip, 256-resize) changes point density vs BTF's native-resolution
  assumptions — may need to run FPFH on the *pre-resize* cloud.

### Quiz log
_(none yet — "quiz me" to add one)_
