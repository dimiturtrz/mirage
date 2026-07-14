# Isaac Sim 6.0 Headless Offline SDG: Canonical Patterns

**Date**: 2026-07-15  
**Status**: settled  
**Versions**: Isaac Sim 6.0.0, 6.0.1; Omniverse Kit 110+; Replicator core (omni.replicator.core)

## TL;DR

1. **BasicWriter static-scene headless SDG DOES NOT require `rep.trigger.on_frame()`** — it's only for custom-event graph randomization. Use explicit backend + `orchestrator.step(delta_time=0.0, rt_subframes=32)` loop.
2. **Windows first-frame issue**: add 2 warmup `orchestrator.step()` calls before data capture (documented workaround, frame-skip is real).
3. **Async rendering toggle causes frame loss** — launch with `--/exts/isaacsim.core.throttling/enable_async=false` if using Replicator.
4. **Empty annotator arrays (shape=(0,))** — likely multi-GPU synchronization or timeline not-playing. Workaround: force single GPU, verify timeline running.

## Question

How to capture labeled frames (RGB, depth, normals, semantic_segmentation) from a static scene in Isaac Sim 6.0 **headless** (no GUI) using `omni.replicator.core`? What is the exact orchestrator+trigger+annotator+writer call sequence, and what causes the "Replicator initialization aborted" warning and empty annotator arrays?

## Findings

### 1. Minimal Working Code for Static-Scene Headless SDG with BasicWriter

**Source**: [IsaacSim/skills/data-collection-sim/SKILL.md](https://github.com/isaac-sim/IsaacSim/blob/main/skills/data-collection-sim/SKILL.md) — Isaac Sim 6.0 skill, official GitHub, 2026  
**Also verified**: [Scene Based Synthetic Dataset Generation — Isaac Sim 6.0.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html)

**EXACT CODE PATTERN** (headless static scene):

```python
from isaacsim import SimulationApp

# REQUIRED: Initialize BEFORE importing replicator
simulation_app = SimulationApp({"renderer": "RealTimePathTracing", "headless": True})

import omni.replicator.core as rep

# Load scene or build it
# ... load USD stage ...

# Create render product(s) at fixed resolution
rp = rep.create.render_product(camera_prim_path, resolution=(512, 512))

# OPTIONAL: Disable timeline advance for static scenes
rep.orchestrator.set_capture_on_play(False)

# Initialize backend (RECOMMENDED for Isaac Sim 6.0+)
backend = rep.backends.get("DiskBackend")
backend.initialize(output_dir="/path/to/output")

# Initialize writer with explicit backend
writer = rep.writers.get("BasicWriter")
writer.initialize(
    backend=backend,
    rgb=True,
    bounding_box_2d_tight=True,
    semantic_segmentation=True,
    distance_to_image_plane=True,
    bounding_box_3d=True,
    occlusion=True
)

# Attach writer to render product(s)
writer.attach([rp])

# Annotators are configured via writer; explicit attach is optional
rgb_annot = rep.annotators.get("rgb")
rgb_annot.attach(rp)

# **CRITICAL**: delta_time=0.0 keeps timeline frozen for static scenes
for i in range(NUM_FRAMES):
    rep.orchestrator.step(delta_time=0.0, rt_subframes=32)
    # Data is written to disk by the writer automatically

# Cleanup
rep.orchestrator.wait_until_complete()
writer.detach()
simulation_app.close()
```

**Key points**:
- **`rep.trigger.on_frame()` is NOT required** for BasicWriter on static scenes. [S1, S2] It is used only for custom-event-driven randomization in graph-based workflows.
- **`delta_time=0.0`** is mandatory for static scenes — prevents physics simulation step and timeline advance. [S2] Quote: "For static-scene SDG, pass `delta_time=0.0` to `rep.orchestrator.step` so the timeline does not advance between captures."
- **`rt_subframes=32`** (or 64) sets raytracing iterations; higher = better quality, slower. Default is 32. [S2]
- **Backend initialization is explicit in 6.0+** — earlier versions inferred writer behavior. [S1] Quote: "Use explicit backend initialization (recommended for Isaac Sim 6.0+)."

### 2. What Causes "Replicator initialization aborted" Warning

**Sources**: 
- [Replicator Troubleshooting — Isaac Sim 6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/troubleshooting.html) [S3]
- [GitHub Issue #235 — Crash when using rep.trigger.on_custom_event with multiple custom events](https://github.com/isaac-sim/IsaacSim/issues/235) — Isaac Sim 5.0.0–5.1.0, **remains open/unresolved** [S4]

The exact log message "[Warning] [omni.replicator.core.scripts.orchestrator] Replicator initialization aborted" is documented in the troubleshooting guide [S3] but **no single root cause** is listed. However, **related initialization failures** are:

**GitHub Issue #235 Error** (rep.trigger.on_custom_event with multiple events):
> "Failed to initialize node type - no token interface"
> "Could not access the IToken and IAttribute interfaces required for validating an attribute"
> "Windows fatal access violation during graph initialization"  
> Last command before crash: `CreateNodeCommand(graph=?,node_path=/Orchestrator/OgnReadFabricTime,node_type=omni.replicator.core.ReadFabricTime,create_usd=True)` [S4]

**Workarounds** (from troubleshooting and GitHub):
1. **Do NOT use `rep.trigger.on_custom_event()` with multiple events** — use single custom-event or `rep.trigger.on_frame()` instead. [S4]
2. **If using BasicWriter on Windows in standalone mode**, pre-run warmup steps (see §3 below). [S3]
3. **Disable async-rendering toggling** during Replicator capture (see §4 below). [S3]

### 3. Synchronous Annotator Data Access WITHOUT a Writer

**Source**: [Useful Snippets — Isaac Sim 6.0.0 Documentation](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html) [S5]  
**Also**: [RTX Sensor Annotators — Isaac Sim 6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_rtx_annotators.html) [S6]

**EXACT CODE PATTERN**:

```python
# Get annotator(s) from registry
rgb_annot = rep.annotators.get("rgb")
depth_annot = rep.annotators.get("distance_to_image_plane")
seg_annot = rep.annotators.get("semantic_segmentation")
normals_annot = rep.annotators.get("surface_normals")

# Attach to render product
rgb_annot.attach(rp)
depth_annot.attach(rp)
seg_annot.attach(rp)
normals_annot.attach(rp)

# Synchronous loop
for i in range(NUM_FRAMES):
    rep.orchestrator.step(delta_time=0.0, rt_subframes=32)
    
    # get_data() returns immediately (synchronous)
    # NO device parameter is needed (it exists in older versions, ignored in 6.0)
    rgb_data = rgb_annot.get_data()  # numpy array or dict
    depth_data = depth_annot.get_data()
    seg_data = seg_annot.get_data()
    normals_data = normals_annot.get_data()
    
    # Process/save data
    save_to_disk(rgb_data, path=f"frame_{i:04d}_rgb.png")
```

**Key points**:
- **`annotator.get_data()` is synchronous** — it blocks until frame is ready, returns immediately after `orchestrator.step()`. [S5]
- **No `device` parameter** is required or used in 6.0. [S6] Older documentation mentions `device="cpu"` or `device="gpu"`, but these are ignored/deprecated in 6.0.
- **How many `step()` calls before data is non-empty?** [S2, S3] On Linux/Mac headless: data appears from step 0 (immediate). **On Windows in standalone mode: data may be empty on step 0–1** — add warmup loop (see §4).

**Critical limitation** (from [S6]): "RTX Sensor Annotators rely on the simulation timeline to collect data. If the timeline is not playing (for example, if the simulation is paused or stopped), the annotators will not collect data."  
**Solution**: Ensure timeline is playing or use `rep.orchestrator.step()` which advances the timeline internally.

### 4. Headless-Specific Rendering Settings (Carb + RTX)

**Source**: [Rendering Modes — Isaac Sim 6.0.0 (latest)](https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/rendering_modes.html) [S7]  
**Also**: [Isaac Sim Performance Optimization Handbook — Isaac Sim 6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/reference_material/sim_performance_optimization_handbook.html) [S8]  
**Also**: [Replicator Troubleshooting — Isaac Sim 6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/troubleshooting.html) [S3]

**RTX Renderer Choice** for headless:
```python
# In SimulationApp config dict, headless context
simulation_app = SimulationApp({
    "renderer": "RealTimePathTracing",  # Default RTX Real-Time 2.0 (balanced quality/speed)
    "headless": True
})
```

**Three rendering modes** [S7]:
- **RTX Real-Time 2.0** (default in 6.0): path tracing + DLSS neural rendering. Quote: "The recommended choice for most robotics workflows... ideal for interactive simulation and synthetic data generation."
- **RTX Interactive (Path Tracing)**: accumulates samples, slower, higher fidelity.
- **RTX Minimal**: no indirect lighting, single hard shadow, fastest.

**CRITICAL CARB SETTING for Replicator headless frame capture**:

```bash
# Launch command
isaac-sim.sh --headless --no-window \
  --/exts/isaacsim.core.throttling/enable_async=false \
  path/to/script.py
```

Or in Python:
```python
from omni.isaac.core.utils.carb import set_carb_setting
set_carb_setting("/exts/isaacsim.core.throttling/enable_async", False)
```

**Why**: Quote from [S3]: "When using Replicator, frames may be skipped due to the isaacsim.core.throttling extension toggling `/app/asyncRendering=True` by default when the timeline is stopped. To resolve, launch with the flag `--/exts/isaacsim.core.throttling/enable_async=false` to disable async rendering toggling."

**Multi-GPU RTX settings** (if needed):
```bash
--/renderer/multiGpu/enabled=true
--/renderer/activeGpu=0
```

**No other `/rtx/...` or `/omni/replicator/...` carb setting is required** for offscreen rendering beyond the async-throttling disable [S7, S8]. The RTX renderer automatically targets the GPU when `headless=True`.

### 5. `rep.functional.create.*` vs `rep.create.*` API (Immediate vs Graph)

**Source**: [Object Based Synthetic Dataset Generation — Isaac Sim 6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_object_based_sdg.html) [S9]  
**Also**: [Useful Snippets](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html) [S5]  
**Also**: [Isaac Sim 6.0.1 Release Notes](https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html) [S10]

**Both APIs are used in 6.0**, but there is a **difference in timing**:

- **`rep.functional.create.*` (immediate API)**: Executes immediately in the host (CPU). Used for static scene authoring, camera/render-product setup. Examples: `rep.functional.create.camera()`, `rep.functional.create.reference()`, `rep.functional.modify.pose()`.
  
- **`rep.create.*` (graph API)**: Adds operations to the Replicator graph, which execute in the orchestrator loop. Used for per-frame randomization. Examples: `rep.create.light()`, `rep.create.render_product()`.

**For static-scene headless SDG**, use **`rep.functional.*` for scene setup and `rep.create.*` for render products**:

```python
# Scene setup — use functional (immediate)
cam = rep.functional.create.camera(position=(0, 0, 5), look_at=(0, 0, 0))

# Render product — use create (graph, but single-create is fine)
rp = rep.create.render_product(cam, resolution=(512, 512))

# Annotators — use get() + attach()
rgb = rep.annotators.get("rgb")
rgb.attach(rp)
```

**Quote from [S10]** (Isaac Sim 6.0.1): "end-to-end synthetic data generation (SDG) workflow examples using the Replicator Functional API to author, randomize, simulate, and collect annotated data."

**Does it matter for annotators/writers receiving data?** [S5, S9] No — annotators and writers are attached after creation, regardless of API. The functional vs. graph distinction affects **when** operations run, not whether data flows to writers.

### 6. Windows Standalone Frame-Skip Workaround

**Source**: [Replicator Troubleshooting — Isaac Sim 6.0.0](https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/troubleshooting.html) [S3]

Quote: "On Windows, when running SDG pipelines with Replicator in standalone mode, the first frame may be skipped by writers or data may be missing from annotators."

**Workaround**:
```python
# Run 2 warmup steps to let GPU/renderer initialize
for _ in range(2):
    rep.orchestrator.step(delta_time=0.0, rt_subframes=32)

# Now begin actual data capture
for i in range(NUM_FRAMES):
    rep.orchestrator.step(delta_time=0.0, rt_subframes=32)
    # ... capture annotators ...
```

Alternatively (if using native timeline):
```python
from omni.isaac.core.utils.stage import get_current_stage
timeline = omni.timeline.get_timeline_interface()
timeline.play()
for _ in range(2):
    standalone_app.update()
timeline.stop()

# Then start Replicator capture loop
```

This is **only needed on Windows in standalone mode**. Linux/Mac headless does not have this issue.

### 7. Multi-GPU Synchronization Issue (Empty Frames)

**Source**: [GitHub Issue #507 — Replicator annotators giving empty images on random frames when using multiple GPUs](https://github.com/isaac-sim/IsaacSim/issues/507) — Isaac Sim 5.1.0, **remains open** [S11]

**Symptom**: shape=(0,) or fully transparent annotator arrays, non-deterministic (50% reproducibility).

**Root cause** (quote from [S11]): "Multi-GPU rendering synchronization problems... Pattern correlation with GPU assignment: only even-numbered cameras fail with standard GPU ordering; switching to `CUDA_VISIBLE_DEVICES=1,0` makes odd cameras fail instead."

**Workaround**: Force single GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
isaac-sim.sh --headless path/to/script.py
```

Or in Python:
```python
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
simulation_app = SimulationApp({"headless": True, "multi_gpu": False, "active_gpu": 0})
```

**Note**: This is an **open issue** in Isaac Sim 5.1.0. May be fixed in 6.0+, but workaround is documented and reliable.

## Open Questions

- Is the "Replicator initialization aborted" warning specific to graph-based randomization with multiple triggers, or can it occur in BasicWriter-only pipelines? (GitHub #235 only documents multi-event-trigger crash, not BasicWriter abort.)
- Does Isaac Sim 6.0.1 fix the multi-GPU synchronization issue (#507)? Release notes do not mention it.
- Is `rep.trigger.on_frame()` ever needed in a BasicWriter-only pipeline on static scenes? (Our research suggests no, but no explicit guarantee found.)

## Sources

- [S1] IsaacSim/skills/data-collection-sim/SKILL.md (GitHub, 2026) — https://github.com/isaac-sim/IsaacSim/blob/main/skills/data-collection-sim/SKILL.md
- [S2] Scene Based Synthetic Dataset Generation — Isaac Sim 6.0.0 Documentation — https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_scene_based_sdg.html
- [S3] Replicator Troubleshooting — Isaac Sim 6.0.0 Documentation — https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/troubleshooting.html
- [S4] GitHub Issue #235 — Crash when using rep.trigger.on_custom_event with multiple custom events (opened Oct 5, 2025, status unresolved) — https://github.com/isaac-sim/IsaacSim/issues/235
- [S5] Useful Snippets — Isaac Sim 6.0.0 Documentation — https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_isaac_snippets.html
- [S6] RTX Sensor Annotators — Isaac Sim 6.0.0 Documentation — https://docs.isaacsim.omniverse.nvidia.com/6.0.0/sensors/isaacsim_sensors_rtx_annotators.html
- [S7] Rendering Modes — Isaac Sim Documentation (latest) — https://docs.isaacsim.omniverse.nvidia.com/latest/reference_material/rendering_modes.html
- [S8] Isaac Sim Performance Optimization Handbook — Isaac Sim 6.0.0 Documentation — https://docs.isaacsim.omniverse.nvidia.com/6.0.0/reference_material/sim_performance_optimization_handbook.html
- [S9] Object Based Synthetic Dataset Generation — Isaac Sim 6.0.0 Documentation — https://docs.isaacsim.omniverse.nvidia.com/6.0.0/replicator_tutorials/tutorial_replicator_object_based_sdg.html
- [S10] Isaac Sim 6.0.1 Release Notes — https://docs.isaacsim.omniverse.nvidia.com/6.0.1/overview/release_notes.html
- [S11] GitHub Issue #507 — Replicator annotators giving empty images on random frames when using multiple GPUs (Isaac Sim 5.1.0, status unresolved) — https://github.com/isaac-sim/IsaacSim/issues/507

---

## Extra Context: Related Known Issues

**GitHub Issue #110** (Cannot retrieve RTX lidar data in headless mode) — [S12]: Headless=True returns empty `[]` for lidar annotator; headless=False returns data from step 6 onward. Status unresolved as of Aug 2025.  
[S12] https://github.com/isaac-sim/IsaacSim/issues/110

This suggests headless lidar/annotator pipeline is a known pain point. Workaround not documented; may be same as Windows warmup + single-GPU forced.
