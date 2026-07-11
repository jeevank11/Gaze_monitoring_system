# Pitch — Privacy-Preserving Gaze Analytics

*(90-second version to open the demo, plus a 3-minute technical version.)*

---

## The 90-second pitch

> Digital signage operators want to know *who's watching* and *for how long*
> — but doing that with face recognition creates massive privacy problems.
>
> We built an analytics pipeline that answers the same business questions —
> viewers, dwell time, gender ratio, engagement per content — **without ever
> storing a frame, without recognizing anyone, and without a single cloud
> call.** Everything runs on the signage box's own Intel CPU and iGPU.
>
> The system automatically detects when the content on screen changes, so
> operators get "which ad performed best" for **anything** they play — no
> configuration required. Play a YouTube video with ad breaks, the system
> segments the ads on its own and shows attention dropping during them.
>
> On an Intel Core i7 with Iris Xe iGPU, the whole pipeline runs at
> **≥15 FPS at 720p using under 15 W** — well below 2 TOPS of compute,
> meaning it will run just as well on Intel Atom or Compute Stick class
> hardware in a real signage deployment.
>
> Privacy isn't a policy line in this project — it's architectural, it's
> tested by CI, and it's demonstrable: while the pipeline is running, the
> data folder contains zero images and zero video files. Ever.

---

## The 3-minute technical version

### Problem framing (30 s)
Show the problem statement. Point out that the ask contains a genuine
tension: measure engagement *without* face recognition, *without* storing
frames, *on low-TOPS hardware*. Most teams will hand-wave one of these.

### Architecture (60 s)
Draw the diagram from `docs/ARCHITECTURE.md`.
- Webcam → OpenVINO detectors → attention gate → aggregator → SQLite → dashboard.
- Screen capture → perceptual hash → auto content segmentation.
- Frames discarded at the top of every loop iteration.

### Live demo (60 s)
1. Start the pipeline. Show the debug overlay (bboxes + attention badge).
2. Play a YouTube video with an ad break on the second monitor.
3. Show the dashboard filling in per-segment metrics **live**.
4. Point at the segments table: "This row is the ad break — attention
   dropped from 88% to 22%."
5. Open File Explorer on `data/`. Filter to `*.jpg`, `*.mp4`. **Empty.**

### Privacy proof (20 s)
Open `tests/test_privacy_smoke.py`. Run it. It grep-tests the whole source
tree for banned APIs and passes. This is CI-enforced on every commit.

### Performance (20 s)
Open `docs/BENCHMARKS.md`. One table: device × FPS × latency × Watts.
Highlight the iGPU row. Compare to a discrete-GPU baseline (implicit:
"we don't need a $500 GPU to solve this problem").

### Deployment (10 s)
Show `.github/skills/deploy/SKILL.md`. Same code, four ways to ship:
local script, Windows `.exe`, Docker, systemd. Judges' choice.

---

## Anticipated questions

- **Q: Isn't gender classification a privacy concern too?**
  A: We only emit it as an aggregate over a 5-second window and only
  after a viewer has already been attending for ≥1 s. No per-person value
  is ever stored. If the operator prefers, `GAZE_ENABLE_GENDER=false`
  disables it entirely at zero cost to the rest of the pipeline.

- **Q: What stops someone from adding face recognition later?**
  A: The privacy smoke test in CI fails the build. Also, the `.github/`
  agent + instruction files instruct any Copilot session on this repo
  to refuse such suggestions.

- **Q: How well does the ad detector work on non-YouTube content?**
  A: The generic pHash scene-change detector catches any hard cut. The
  YouTube-specific "Ad" badge detector adds ad-vs-content classification;
  on other sources, segments are tagged `unknown` and metrics are still
  correct — you just lose the ad-vs-content breakdown.

- **Q: What if the camera fails?**
  A: `webcam.py` logs the failure, backs off, and retries. The dashboard
  keeps serving the last known aggregates. Content segmentation is
  independent and keeps working.
