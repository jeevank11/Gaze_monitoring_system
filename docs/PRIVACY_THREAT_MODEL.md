# Privacy Threat Model

This document is the contract with the audience and with the reviewer.
If a proposed change would break any of the guarantees below, it is
**rejected in review**.

---

## What we collect

- **Live camera frames** — used only to run inference in the current
  process. Never written to disk. Never sent off the device. Overwritten
  by the next frame.
- **Live screen captures** — used only to compute a perceptual hash of
  the content on screen. The full screenshot is overwritten by the next
  grab. Only the 64-bit hash (a number) and an optional 160×90 preview
  of the *content screen* (not of any person) are persisted.
- **Aggregate counters** — timestamped rows of viewer count, attending
  count, average dwell time, gender counts, and segment tags.

## What we DO NOT collect

- ❌ No raw camera frames on disk, in memory beyond one frame, or on the network.
- ❌ No face recognition. No face embeddings. No face IDs.
- ❌ No re-identification of returning viewers. Tracker IDs are integer
  counters that die when the process exits.
- ❌ No per-person records of any kind — every value stored is an aggregate
  over a 5-second window.
- ❌ No audio.
- ❌ No cloud calls at runtime. No telemetry. No third-party APIs.
- ❌ No OCR / logo recognition on the screen capture (only visual hashing).

---

## Where the data lives

| Data | Location | Lifetime |
|---|---|---|
| Camera frame | RAM (single numpy buffer) | ~66 ms (until next `read()`) |
| Screen capture | RAM (single numpy buffer) | ~1 s (until next capture) |
| Perceptual hash | RAM + SQLite | Persists per segment |
| Tracker ID | RAM (process-local integer) | Until face leaves FoV |
| Content thumbnail | SQLite (optional, 160×90 PNG b64) | Persists per segment |
| Aggregate metrics | SQLite (`data/metrics.sqlite`) | Persists |

## Where the data goes

- **By default: nowhere.** Everything stays in `data/metrics.sqlite` on the
  signage device.
- If `GAZE_MQTT_URL` is set, aggregate rows (only) may be published to that
  broker. Never raw frames. Never faces. Never per-person values.

---

## Enforcement

1. **Code review contract** — see `.github/instructions/copilot-instructions.md`.
2. **Automated grep** — `tests/test_privacy_smoke.py` fails CI if any
   banned API (`cv2.imwrite`, `VideoWriter`, `pickle.dump`, `face_recognition`,
   `deepface`, `dlib.face_recognition`, unmarked `Image.save`) appears in `src/`.
3. **Schema contract** — the SQLite schema in
   `src/gaze_analytics/storage/schema.sql` is aggregate-only. Any migration
   adding a per-person or per-frame column must be rejected.
4. **Demo proof** — during a live demo, run
   `Get-ChildItem data\ -Include *.jpg,*.png,*.mp4 -Recurse` and show it
   returns nothing while the pipeline is running.

---

## Regulatory alignment (informational)

- **GDPR (EU) Art. 4/5** — no personal data is processed because there is
  no identifier that can single out a natural person: frames are ephemeral,
  no IDs persist across process restarts, aggregates are non-identifying.
- **India DPDP Act 2023** — same reasoning: no "personal data" is collected
  or stored under the Act's definition.
- **CCPA/CPRA (California)** — no sale/sharing of personal information; no
  personal information collected in the first place.

*(This is an architectural claim, not legal advice. A real deployment should
still complete a Data Protection Impact Assessment.)*

---

## Threat scenarios and mitigations

| Threat | Mitigation |
|---|---|
| Attacker steals the signage box | Only aggregate SQLite is present. No faces. No frames. |
| Malicious update injects `imwrite` | Blocked by `test_privacy_smoke.py` in CI. |
| Camera stream intercepted on the wire | The stream never leaves the process — no wire traffic. |
| Aggregates re-identify a rare viewer (e.g., only person present at 3 a.m.) | 5-second windows plus counter-only fields; no bbox or bounding data emitted. |
| Content thumbnail leaks something sensitive on screen | Operator can disable thumbnails: `GAZE_KEEP_SEGMENT_THUMBNAIL=false`. |
| Ad-badge detector misclassifies content | Type tag is `unknown` on failure; metrics still correct. |
