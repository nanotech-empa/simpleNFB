Be compact in your documentation and comment on script
Ask before running tests
Be pedagological in your solutions
Plan with the intent to use tokens efficiently during implementation — organise work by file so each file is read and edited in a single pass

## Pre-implementation snapshot
Before making any non-trivial code changes, create a revert artefact for each file to be modified:
- Copy: `cp <file> <file>.bak` (saved at repo root, not inside the package)
- After changes are complete, generate a reverse patch: `git diff HEAD -- <file> > <stem>_revert.patch`
- Tell the user the revert command: `cp <file>.bak <file>` or `git apply --reverse <stem>_revert.patch`
- Delete the `.bak` and `.patch` files once the user confirms the changes are stable

## Memory files
Before starting any work on the project, create the following three files if they do not already exist:
- LONG_TERM_MEMORY.md
- SHORT_TERM_MEMORY.md
- WORKING_MEMORY.md

Maintain them as follows:
- LONG_TERM_MEMORY.md — append very concise summaries of completed work
- SHORT_TERM_MEMORY.md — keep current plans and context for future work
- WORKING_MEMORY.md — hold context for complex in-progress work that requires user input
- Use all three files to track progress and maintain a strict scope of work when completing tasks 