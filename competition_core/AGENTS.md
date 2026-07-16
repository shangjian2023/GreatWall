# Competition Core Agent Instructions

Read the repository-root `CLAUDE.md` and this directory's `README.md` before editing.

- Keep training truth in training YAML and `training_quality_gate` only.
- Detection modules and configs must never import or accept condition text, target sequences, poisoned data, or clean reference models.
- Do not add synthetic dataset fallback behavior.
- Preserve the RTX 4060 resource profile and shard-resume support.
- Run `python -m pytest competition_core/tests -q` and `python -m ruff check competition_core` after changes.
