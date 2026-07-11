---
mode: agent
---

# Fix a Bug

Use when the user reports a failure (crash, wrong metric, low FPS, etc.).

## Steps

1. **Reproduce first.** Ask for exact command, expected vs. actual, and
   any error output. Do not guess.
2. **Locate.** Search the codebase for the symptom (error message,
   metric name, module). Use `codebase` and `search`.
3. **Diagnose.** Read the relevant code + tests. State the root cause in
   one sentence before touching anything.
4. **Fix minimally.** Change only what's needed. Do not "improve" nearby
   code.
5. **Add a regression test** if the bug was in pure logic (aggregator,
   tracker, phash). UI / capture bugs get a manual repro note in the
   commit message instead.
6. **Run** `ruff check src tests` and `pytest -q`.
7. **Commit** — `fix: <one-line summary> (closes #<issue>)`.
8. **Report** — root cause, fix summary, test added (or manual repro).
