# SKILL: Run the Dashboard

Launch the Streamlit UI that reads aggregate metrics from
`data/metrics.sqlite` and shows live viewer counts, dwell time,
attention rate, and per-content-segment engagement.

## Prerequisites

- The analytics pipeline is running (see `run-analytics/SKILL.md`).
- `data/metrics.sqlite` exists (created automatically on first analytics run).

## Run

```powershell
streamlit run dashboard/📺_Gaze_Live.py
```

Opens `http://localhost:8501` in the browser.

## For the dual-monitor demo

- Move the browser window to the **external monitor** and press `F11` for
  fullscreen — this is the "operator" screen.
- On the **laptop screen**, play whatever content you want the audience
  (you) to view — YouTube, PPT, image, video. The webcam watches whoever
  is looking at the laptop screen; the screen-capture module tags metrics
  with the auto-detected content segment.

## Dashboard sections

1. **Live tiles** — viewers now, attending now, avg dwell (5-min rolling).
2. **Trend chart** — viewers & attention over the last hour.
3. **Content segments table** — auto-detected segments with thumbnail,
   duration, avg dwell, attention rate.
4. **Aggregate breakdown** — gender ratio, attention rate: content vs. ads.
5. **Privacy tile** — running counter proving zero frames written to disk.
