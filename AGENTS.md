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

## Packaging Rules (打包规则)

- All packaging/build operations (打包) MUST be performed inside the official project directory: `D:\Program Files (x86)\appkaifa\jixiebijiaoben`.
- NEVER package from a Codex worktree, agent worktree, or any other copy/checkout of the project (禁止使用 Codex worktree 中的工程副本打包).
- Before running any build/packaging command, verify the current working directory is the official directory above. If it is not, stop and ask the user first.
- Reason: a package was once built from a Codex worktree copy and shipped to a customer, which broke the customer's environment. This must never happen again.
