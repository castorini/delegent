<!-- Delegent instruction. One part per subtask tool; the harness gives the frontier agent the parts for the tools
     it has (Claude Code's Agent tool on coding, Delegate on search). Marker lines are not shown to the model. -->
<!-- part: Agent -->
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
<!-- part: Delegate -->
You have a research assistant: the Delegate tool hands it a focused subtask; it searches the same corpus with its own
search engine and returns a short report with evidence, the docids and its confidence. It is fast, and its
high-confidence reports are usually right. Work through it:
- Delegate all searching, including checks of a candidate ("does the corpus confirm that X ...?"). Run independent
  subtasks in parallel.
- Do not run broad searches or read whole documents yourself. To confirm one fact, use a single targeted search, or read
  only the part of the document around it (get_document with an offset).
- Each report ends with an automatic lookup of its evidence in the cited documents: a located passage is verbatim there.
- Give your final answer as soon as the evidence covers every condition of the question.
