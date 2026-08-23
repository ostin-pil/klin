# Lifecycle manifest — klin

Project configuration for the lifecycle-kit skills. Modelled on the
`claude-plugins` manifest, because klin is the same kind of repo: a Claude Code
plugin marketplace hosted as a monorepo, with a Python tool alongside it.

klin has a remote and one writer, so the full lifecycle applies unchanged.
Branches are born off `origin/main`, nothing merges locally, and the forge's PR
state is the authoritative merge signal.

**klin's sessions are its own.** Work here is never part of a consuming
project's session. The decision to extract this tooling belongs in the project
that made it; how the tooling got built is logged here. Two repos, two logs,
two numbering sequences.

```yaml
product_name: klin

# git integration
remote: origin
integration_ref: origin/main
local_main: main
pr_base: main
requires_remote: true

# forge
forge: github
forge_url: https://github.com
forge_repo: ostin-pil/klin

# branches & worktrees
branch_pattern: "feature/session-{n}-{topic}"
branch_glob: "feature/*"
docs_log_branch: "docs/session-{n}-log"
worktree_dir: .claude/worktrees
worktree_pattern: "session-{n}-{topic}"
worktree_ignore: .git/info/exclude
orphan_branch_globs: ["feature/session-*", "worktree-agent-*", "fix/*"]

# session logs
log_dir: sessions
log_archive: sessions/archive
log_index: sessions/INDEX.md
log_pattern: "{date}_session_{n}[_{suffix}].md"
log_glob: "sessions/[0-9]*_session*.md"
log_presence_regex: '^sessions/[0-9]\{4\}-[0-9]\{2\}-[0-9]\{2\}_session.*\.md$'

# build / test gate — run in order, abort on the first failure.
# No compile step; the suite is the correctness gate. `--with .` installs the
# package from the checkout, so the gate always tests working-tree code rather
# than whatever happens to be installed on the machine.
code_globs: ["klin/**/*.py", "tests/**/*.py"]
build_commands: none
test_commands:
  - uv run --with pytest --with pyyaml --with . python -m pytest tests -q

# merge
merge_strategy: merge

# prose gate — the user-facing docs, scoped by .prose-mint.toml
prose_gate: uvx --from prose-mint prose-mint bulk --strict .
prose_rule: none

# code review
code_reviewer: none
review_command: none

# commit / PR convention — matches claude-plugins, the sibling repo
commit_convention: "prefix(topic): short description"
commit_trailers: "Claude-Session: <url>"
subject_max: 72

# project knowledge & docs
issues_file: none
research_dir: research
reports_dir: reports
jargon_terms: [CLI, JSONL, SPDX, CC0, CC-BY, DRM, MCP, PR]
plan_doc: none
workflow_rule: none
scratch_paths: [.claude/settings.local.json, .claude/worktrees/]
```

## Open items for this manifest

- `workflow_rule` is `none` because lifecycle-kit ships its own bundled
  invariants doc and klin adds nothing to them. Point this at a local copy only
  if that stops being true.
- CI is not wired yet. The token in use lacks the `workflow` scope, so a push
  that adds `.github/workflows/` is rejected by GitHub; run
  `gh auth refresh -s workflow` before adding it.
- `research_dir` and `reports_dir` do not exist yet. The `report` skill creates
  the latter on first run.
