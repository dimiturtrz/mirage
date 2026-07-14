# Project Instructions for AI Agents — mirage

A physics-based synthetic-data engine for 3D perception: generate defect scenes, detect anomalies, measure the sim-to-real gap rigorously. Edge-deployable. Personal learning project.

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

- **Engine-first.** The center is the synthetic-data generation engine + robust sim-to-real eval — NOT a "3D perception stack." Anomaly = the task; calibration + sim-to-real eval = the contribution; 3D point clouds = the deliberate ramp (new skill, say so).
- **Use, don't reimplement.** MMDetection3D / OpenPCDet / Open3D / anomaly methods are dependencies, not rebuilds.
- **Headline metrics decided up front** — AUPRO/AUROC (real ceiling) · sim-to-real gap (pp) · ECE (calibration under shift). The triad (real / synth-only / synth+DA) IS the result, not a leaderboard number.
- **Carried, not reinvented** — calibration+eval harness, closed-loop curriculum, train-vs-deploy parity tests come from prior acoustic work; retarget them, don't rebuild.
- **No speculative folders.** A piece (`pipeline/`, `pointcloud-viewer/`, `synth-gen-viz/`) is added only when it's real.
- **Integrity rules.** Cite prior art; verify every ⚠️ in PLAN.md against a primary source before it lands in README; claim edge deploy (real), never production-scale serving; 3D is a ramp.
- **Build private → flip public at Stage 0** (MVTec 3D-AD + eval harness presentable). First commit = that.

## Build & Test

Env is **uv** (like the siblings): `uv sync --extra features` (creates `.venv`, resolves torch/torchvision cu130 for the RTX 5090), `uv run pytest tests/ -q`, one-root `paths.yaml` config.

## Static analysis — the gates (CI blocks dev→main)

Principles are enforced **mechanically**, not by review — a **ratcheting** gate per axis: a rule graduates to blocking once it hits zero, then any regression fails CI (fix, don't suppress; no limit-bumps, no `noqa` spray). Six axes:

| Axis | Tool | What it catches | Config |
|------|------|-----------------|--------|
| Style/bugs | `ruff@0.15.13` (enforced on `core surfscan`) | logging-not-print (T201), named constants (PLR2004), keyword-only bools (FBT), specific-exception (BLE001/S110), low-complexity/strategy (C901/PLR09xx), config-objects (PLR0913), imports-at-top (PLC0415), dead noqa (RUF100) | `[tool.ruff]` |
| Dead code | `vulture@2.16` | unused funcs/attrs — blocks conf≥80, warns conf≥60 | `[tool.vulture]` |
| Import layers | `import-linter` (grimp) | `core` = independent kernel, imports no `surfscan` | `[tool.importlinter]` |
| Arch fitness | `devtools/graph.py --assert` (grimp+networkx) | god-module (fan-in AND fan-out both >8), god-file (>750 lines), import cycle; line-floor/chokepoint advisory | `[tool.structure]` |
| Module shape | `ast-grep` (`devtools/sgconfig.yml`) | no import-time side-effect call (`matplotlib.use` exempt); **everything-in-a-class** — every top-level `def` is a method on the class that owns it (`main` exempt) | `devtools/sg-rules/` |
| Duplication | `jscpd` (advisory) | copy-paste over the DRY threshold | `devtools/jscpd.json` |
| Shape contracts | `devtools/shape_contracts.py --assert` (enforced) | public array/tensor boundary carries a `jaxtyping` shape (`Float[Tensor, "b c h w"]`), not bare `np.ndarray`/`Tensor`; `@shapecheck` (jaxtyping+beartype) makes it live at runtime | `[tool.shape_contracts]` |

**noqa policy: bare `# noqa: RULE`** — no prose reasons; minimal comments, prefer self-documenting names. **Everything-in-a-class** (qc5, matching systole): a top-level function belongs as a method on the class that owns its state/responsibility; a pure stateless helper is a `@staticmethod` on that class. `main`/`_main` exempt. The rule catches decorated top-level defs too (a `@contextmanager`/`@torch.no_grad()` free func can't slip past).

## Scaffolding

Guardrails provisioned by **sdlc-scaffold** via copier — `.copier-answers.yml` pins the version. The gate config, `devtools/`, and the nox/CI/pre-commit runners are **template-owned**: don't hand-edit them to pass a gate — fix upstream in the scaffold and `uvx copier update`, or edit only within `# >>> LOCAL-SLOT` regions. `copier update` pulls scaffold improvements as reviewable steps.

## Architecture Overview

_See [`docs/PLAN.md`](docs/PLAN.md) → "Structure". Three pieces: `pipeline/` (data → model → anomaly → eval), `pointcloud-viewer/` (in-browser ONNX), `synth-gen-viz/` (the engine made visible). Added only when real._

## Conventions & Patterns

- Data lives OUT of the repo (licensing + size); one-root `paths.yaml`, per-source adapters → common schema.
- Three doc layers: `research/` = external / field synthesis (theirs) · `learning/<date>_<topic>.md` = the study ramp / general understanding · `interpretations/<task>/<date>_<topic>.md` = sense-making of *our own* results (per task, `converging/` for cross-task). Build log = git history; portfolio READMEs carry the result + link to the interpretation.
- Test layout: unit (equivalence-class) + integration (module-pairs).
