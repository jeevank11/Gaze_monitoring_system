---
mode: agent
---

# Write a Test

Use when the user asks to add tests to a module.

## Steps

1. **Identify the module** and its pure vs. side-effecting boundary.
   - Pure logic (aggregator, tracker, phash, engagement gates) → unit test
     with hand-crafted inputs.
   - Side-effect wrappers (capture, sink, UI) → integration test with
     fixtures / temp DBs. Never open the real webcam in a test.
2. **Use fixtures** from `tests/fixtures/` (add if missing).
3. **Cover:**
   - Happy path
   - Empty / degenerate input
   - Boundary values (thresholds, timeouts, edge frames)
   - Privacy invariant (if applicable — e.g., aggregator never emits
     a per-person row)
4. **Name tests** descriptively: `test_<module>_<behavior>`.
5. **Run** `pytest -q -k <module>` and confirm they pass.
6. **Commit** — `test: <module> — <what is now covered>`.
