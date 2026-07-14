# devtools — the guardrail tools shipped with this project

Each tool is engine + config, invoked identically in `noxfile.py` and CI. Targets are the `packages`
list (space-separated below as `<packages>`). Only the tools this project enabled are shipped.

- **graph.py** — import-graph arch fitness (god-module = fan-in AND fan-out over a degree / import cycle / god-file / **test-mirror gap**). Gate: `python -m devtools.graph --assert <packages>` (thresholds in `pyproject [tool.structure]`). Explorer: drop `--assert` for ranked fan-in/out/bottleneck/chokepoint tables.
- **test-mirror rule** — every logic module needs a unit test. `[tool.structure] test_layout` sets *where*: `"mirror"` (default, strict — `<pkg>/<path>/foo.py` → `tests/unit/<pkg>/<path>/test_foo.py`, one home per module), `"flat"` (a `test_foo.py` anywhere under `tests/`), or `"off"`. `__init__`/`__main__` and coverage-**omitted** shells (`[tool.coverage] omit`, via **omit.py**) are exempt — a non-unit-testable shell isn't forced to carry a stub.
- **sg-rules/** + **sgconfig.yml** — ast-grep module-shape rules (behaviour in a class, no import-time side effects). `uvx --from ast-grep-cli ast-grep scan -c devtools/sgconfig.yml <packages>` (enforced).
- **jscpd.json** — jscpd copy-paste (DRY) threshold. `npx jscpd <packages> --config devtools/jscpd.json` (advisory).
- **lcom.py** — LCOM4 cohesion: ranks concrete stateful classes whose methods split into disjoint-state groups. `python -m devtools.lcom <packages>`.
- **data_clumps.py** — Fowler data clumps: param sets that travel together across signatures (Introduce Parameter Object). `python -m devtools.data_clumps <packages>`.
- **state_candidates.py** — namespace classes with latent shared instance state. `python -m devtools.state_candidates <packages>`.

  The three class-shape tools are ADVISORY explorers — they print a ranked report and always exit 0, never blocking.

## Directional layer contracts — import-linter

`graph.py`'s structural checks are layer-AGNOSTIC and catch *cycles*, but not a one-way *forbidden*
import (`core -> trainer` is no cycle). That directional axis is import-linter's.

Shipped: `[tool.importlinter]` in `pyproject.toml` carries `root_packages` + a kernel-independence
starter contract (the first package imports none of the others) inside a `# >>> LOCAL-SLOT: import-contracts`
region — add your viewer/trainer or domain contracts there. Enforced in nox/CI/pre-commit via `lint-imports`.

