---
mode: agent
---

# Add a Feature

Use when the user asks to add a new capability (e.g., "add gender aggregate
export", "add scene-change ad detector").

## Steps

1. **Restate the requirement** in one sentence. Confirm with the user.
2. **Locate the sprint** — is this scope creep on the current sprint, or a
   new sprint? Update `docs/ARCHITECTURE.md` if a new sprint is added.
3. **Design first, code second.** List:
   - Files to be created / modified
   - New dependencies (must be pre-approved by the user)
   - New config keys in `gaze_analytics.config`
   - New DB columns (must be aggregate-only)
   - Tests to be added
4. **Implement** on `feature/hackathon-mvp`.
5. **Test** — `ruff check src tests`, `pytest -q`.
6. **Privacy grep** — run the check from
   `.github/agents/gaze-analytics.agent.md`.
7. **Commit** — `sprint-N: add <feature>`.
8. **Report** — files touched, tests added, next suggested sprint.

## Reject if

- The feature requires face recognition, ReID, embeddings, or persistence
  of raw frames.
- The feature requires a cloud API call at runtime.
- The feature adds >100 MB of dependencies.
