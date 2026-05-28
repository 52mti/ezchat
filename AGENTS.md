# Workspace Rules

These rules apply to the entire workspace.

## Protected Directories

- `/jiekou`
- `/lgb`

## Required Behavior

- Never modify, create, delete, rename, or move any files under `/jiekou` or `/lgb`.
- These two directories are read-only for Codex.
- Codex may only read files from `/jiekou` and `/lgb`, or call/use the existing code and interfaces found there.
- Any new code, scripts, notes, or temporary files must be created outside `/jiekou` and `/lgb`.
- If a task would require changing `/jiekou` or `/lgb`, stop and ask the user first instead of making the change.
