# Working with subagents

Subagents (the Agent tool) run on a small, fast model and cost far less than you. Every step you take re-reads the whole
conversation, so the cheapest plan is the one with the fewest steps of your own.

- Delegate investigation (find the code involved in the issue and explain how it works; write a minimal script that
  reproduces the bug and report its output) and verification (run the reproduction and the relevant tests; report only
  what fails). Make the code changes yourself.
- Run subagents in the foreground (do not use run_in_background) and launch independent ones in the same turn, so their
  reports come back together and nothing needs polling.
- Ask each subagent for a short report, at most 150 words: only the facts you need, with file paths, line numbers and
  exact error messages.
- Do not keep a task list (TaskCreate / TaskUpdate / TaskList / TaskGet) for this job; keep your plan in your own messages.
- Subagent reports can be wrong: check a fact yourself before your fix depends on it.
