# Gaze Analytics — Agent Learnings

> Persistent memory across chats. Append short, dated bullets. Newest at the top.
> Keep entries factual: what was tried, what worked, what didn't, why.

## 2026-07-10 — Repository bootstrap

- Target hardware: Intel Core i7-1185G7 (Tiger Lake, Iris Xe iGPU, **no NPU**).
  OpenVINO device selection: `AUTO:GPU,CPU` with `PERFORMANCE_HINT=LATENCY`.
- Python 3.12.10 installed via winget. **Do not upgrade to 3.13/3.14** —
  OpenVINO wheels only ship for 3.10–3.12 as of 2025.x.
- GitHub CLI browser device-code auth is **blocked on the Intel corporate
  network** (proxy silently drops `github.com/login/device/code`). Use a PAT
  (`gh auth login --with-token`) instead when pushing.
- Git identity: `Jeevank11 / jeevan.k@intel.com`.

## Conventions locked in

- Branches: `main` (stable) ← `develop` ← `feature/hackathon-mvp` (active).
- Commits per sprint, prefix `sprint-N:`.
- All raw frames must be discarded after processing — enforced by convention
  *and* by the `tests/test_privacy_smoke.py` grep test (to be added in
  sprint 2 or 3).
- SQLite schema is aggregates-only. Any migration that adds a per-person
  column must be rejected in review.
