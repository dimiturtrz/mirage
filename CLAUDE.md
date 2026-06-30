# Project Instructions for AI Agents — mirage

A physics-based synthetic-data engine for 3D perception: generate defect scenes, detect anomalies, measure the sim-to-real gap honestly. Edge-deployable. Personal learning project.

**Source of truth = [`docs/PLAN.md`](docs/PLAN.md).** Read it before proposing work. Sibling project (same philosophy, different domain): [systole](../cardiac-seg/).

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:3216161c -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   git push     # beads syncs via git (.beads/issues.jsonl is tracked); no separate bd push needed
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->

## Non-negotiable conventions (from docs/PLAN.md)

- **Engine-first.** The center is the synthetic-data generation engine + honest sim-to-real eval — NOT a "3D perception stack." Anomaly = the task; calibration + sim-to-real eval = the contribution; 3D point clouds = the deliberate ramp (new skill, say so).
- **Use, don't reimplement.** MMDetection3D / OpenPCDet / Open3D / anomaly methods are dependencies, not rebuilds.
- **Headline metrics decided up front** — AUPRO/AUROC (real ceiling) · sim-to-real gap (pp) · ECE (calibration under shift). The triad (real / synth-only / synth+DA) IS the result, not a leaderboard number.
- **Carried, not reinvented** — calibration+eval harness, closed-loop curriculum, train-vs-deploy parity tests come from prior acoustic work; retarget them, don't rebuild.
- **No speculative folders.** A piece (`pipeline/`, `pointcloud-viewer/`, `synth-gen-viz/`) is added only when it's real.
- **Honesty rules.** Cite prior art; verify every ⚠️ in PLAN.md against a primary source before it lands in README; claim edge deploy (real), never production-scale serving; 3D is a ramp.
- **Build private → flip public at Stage 0** (MVTec 3D-AD + eval harness presentable). First commit = that.

## Build & Test

Env is **uv** (like the siblings): `uv sync --extra features` (creates `.venv`, resolves torch/torchvision cu130 for the RTX 5090), `uv run pytest tests/ -q`, one-root `paths.yaml` config.

## Architecture Overview

_See [`docs/PLAN.md`](docs/PLAN.md) → "Structure". Three pieces: `pipeline/` (data → model → anomaly → eval), `pointcloud-viewer/` (in-browser ONNX), `synth-gen-viz/` (the engine made visible). Added only when real._

## Conventions & Patterns

- Data lives OUT of the repo (licensing + size); one-root `paths.yaml`, per-source adapters → common schema.
- `learning/<date>_<topic>.md` for theory ramp; build log = git history.
- Test layout: unit (equivalence-class) + integration (module-pairs).
