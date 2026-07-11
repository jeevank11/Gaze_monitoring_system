-- Sprint 5 will create this DB. Schema is intentionally aggregate-only.
-- No per-person, no bbox, no image data. Reviewer contract:
-- any migration adding a per-person column MUST be rejected.

CREATE TABLE IF NOT EXISTS metrics (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                  TEXT    NOT NULL,          -- ISO-8601 UTC
    window_seconds      INTEGER NOT NULL,
    viewers             INTEGER NOT NULL,          -- max viewers in window
    attending           INTEGER NOT NULL,          -- max attending in window
    avg_dwell_ms        INTEGER NOT NULL,
    male_count          INTEGER NOT NULL DEFAULT 0,
    female_count        INTEGER NOT NULL DEFAULT 0,
    segment_id          TEXT,                      -- auto content segment
    segment_type        TEXT                       -- 'content' | 'ad' | 'unknown'
);

CREATE INDEX IF NOT EXISTS idx_metrics_ts ON metrics(ts);
CREATE INDEX IF NOT EXISTS idx_metrics_segment ON metrics(segment_id);

CREATE TABLE IF NOT EXISTS segments (
    id                  TEXT    PRIMARY KEY,       -- e.g. 'auto_2026-07-10_11-03-24'
    started_at          TEXT    NOT NULL,
    ended_at            TEXT,
    duration_ms         INTEGER,
    phash               TEXT,                      -- 64-bit hash as hex string
    thumbnail_b64       TEXT,                      -- optional 160x90 preview
    segment_type        TEXT                       -- 'content' | 'ad' | 'unknown'
);
