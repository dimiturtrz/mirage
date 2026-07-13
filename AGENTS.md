# Agent Instructions — mirage

Project context + conventions live in [`CLAUDE.md`](CLAUDE.md); plan in [`docs/PLAN.md`](docs/PLAN.md). This file = shell/tooling rules + beads.

## Non-Interactive Shell Commands

**ALWAYS use non-interactive flags** with file operations to avoid hanging on confirmation prompts.

Shell commands like `cp`, `mv`, and `rm` may be aliased to include `-i` (interactive) mode on some systems, causing the agent to hang indefinitely waiting for y/n input.

**Use these forms instead:**
```bash
# Force overwrite without prompting
cp -f source dest           # NOT: cp source dest
mv -f source dest           # NOT: mv source dest
rm -f file                  # NOT: rm file

# For recursive operations
rm -rf directory            # NOT: rm -r directory
cp -rf source dest          # NOT: cp -r source dest
```

**Other commands that may prompt:**
- `scp` - use `-o BatchMode=yes` for non-interactive
- `ssh` - use `-o BatchMode=yes` to fail instead of prompting
- `apt-get` - use `-y` flag
- `brew` - use `HOMEBREW_NO_AUTO_UPDATE=1` env var

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
   bd dolt push
   git push
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

## Static analysis — the gates (CI blocks dev→main)

Principles enforced **mechanically** via a **ratcheting** gate per axis — a rule graduates to blocking once at zero, then any regression fails CI (fix, don't suppress). Run before pushing:

```bash
uvx ruff@0.15.13 check core surfscan          # style/bugs (enforced)
uvx vulture@2.16 --min-confidence 80          # dead code (blocks ≥80)
uvx --from import-linter lint-imports         # core = independent kernel (imports no surfscan)
uv run python -m devtools.graph --assert      # arch fitness: god-module / god-file / cycle
uvx --from ast-grep-cli ast-grep scan -c devtools/sgconfig.yml core surfscan   # no import-time side-effects
npx --yes jscpd@4 core surfscan --config devtools/jscpd.json                    # duplication (advisory)
```

**noqa policy: bare `# noqa: RULE`** (no prose). **surfscan is functional-core** — systole's "everything-in-a-class" rule is deliberately not ported. Full detail in `CLAUDE.md` → "Static analysis — the gates".
