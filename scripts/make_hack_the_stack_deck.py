"""Generate the Hack the Stack pitch deck for Gaze Analytics.

Reproducible: edit this file and re-run to regenerate the .pptx. All content
lives here as Python data (no external template needed).

Usage
-----
    .\\.venv\\Scripts\\python.exe scripts\\make_hack_the_stack_deck.py

Output
------
    presentations/GazeAnalytics_HackTheStack.pptx

This is a build-time tool, not a runtime component. `python-pptx` is a dev
convenience and is NOT tracked in `pyproject.toml`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt

# ---- Design tokens --------------------------------------------------------

NAVY = RGBColor(0x0F, 0x1C, 0x3F)
INDIGO = RGBColor(0x1E, 0x3A, 0x8A)
SKY = RGBColor(0x0E, 0xA5, 0xE9)
INK = RGBColor(0x0F, 0x17, 0x2A)
SLATE = RGBColor(0x47, 0x55, 0x69)
MUTED = RGBColor(0x64, 0x74, 0x8B)
LIGHT = RGBColor(0xF8, 0xFA, 0xFC)
BORDER = RGBColor(0xE2, 0xE8, 0xF0)
ACCENT = RGBColor(0x22, 0xC5, 0x5E)
WARN = RGBColor(0xF5, 0x9E, 0x0B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


# ---- Slide content model --------------------------------------------------


@dataclass
class Bullet:
    text: str
    sub: str | None = None


@dataclass
class Slide:
    kind: str  # "title" | "content" | "twocol" | "kpi" | "closing"
    title: str
    subtitle: str = ""
    bullets: list[Bullet] = field(default_factory=list)
    right_bullets: list[Bullet] = field(default_factory=list)
    right_title: str = ""
    footer: str = ""
    kpis: list[tuple[str, str, str]] = field(default_factory=list)


# ---- Content --------------------------------------------------------------

DECK: list[Slide] = [
    Slide(
        kind="title",
        title="Gaze Analytics",
        subtitle=(
            "Signage Engagement — Privacy-preserving audience analytics "
            "on Intel CPU / iGPU\n\nHack the Stack · 2026"
        ),
    ),
    Slide(
        kind="content",
        title="The problem",
        subtitle="Digital signage is blind to who actually watches it.",
        bullets=[
            Bullet(
                "Operators buy screens, pipe content, and hope.",
                "Zero closed-loop measurement — even Nielsen-style panels are proxies.",
            ),
            Bullet(
                "Existing computer-vision solutions ship raw frames to the cloud.",
                "Face recognition, ReID and embeddings — regulatory nightmare (GDPR, DPDP, BIPA).",
            ),
            Bullet(
                "Edge boxes exist but cost $1k+ and lock you into a vendor SDK.",
                "No inspection, no privacy audit, no on-prem story.",
            ),
        ],
        footer="Signage market $30B+ · <5% instrumented today",
    ),
    Slide(
        kind="content",
        title="Our answer",
        subtitle="A privacy-by-design pipeline that runs on the laptop you already own.",
        bullets=[
            Bullet(
                "100 % on-device — Intel CPU + Iris Xe iGPU via OpenVINO.",
                "No cloud, no telemetry, no third-party inference calls.",
            ),
            Bullet(
                "Frames are ephemeral.",
                "No imwrite, no video writer, no numpy.save — grepped in CI.",
            ),
            Bullet(
                "Only aggregates leave RAM.",
                "SQLite stores counters, ratios, pHash fingerprints, tiny 160×90 thumbnails.",
            ),
            Bullet(
                "No face recognition · no embeddings · no ReID.",
                "Tracker is bbox geometry + Kalman motion. IDs die with the process.",
            ),
        ],
        footer="Privacy contract is enforced by a smoke test that greps banned APIs.",
    ),
    Slide(
        kind="twocol",
        title="Live demo — two panes",
        subtitle="One codebase, two audience-mode dashboards.",
        right_title="👁️ Gaze Insights",
        bullets=[
            Bullet("📺 Gaze Live", "Control plane + real-time counters"),
            Bullet("Start / stop pipeline as a detached subprocess."),
            Bullet("Toggle detection / gaze / age-gender modules."),
            Bullet("Pick device (CPU / GPU / AUTO) and privacy mode."),
            Bullet("Live KPI tiles + trend, updated every 5 s."),
        ],
        right_bullets=[
            Bullet("Impressions, Attention time, Avg viewing duration, Attention rate."),
            Bullet("Audience engagement trend — auto-rebinned 5 s → 1 h."),
            Bullet("Top content by attention time — leaderboard with thumbnails."),
            Bullet("Gender split donut over the selected window."),
            Bullet("Everything from SQLite aggregates — zero per-person records."),
        ],
    ),
    Slide(
        kind="kpi",
        title="Metrics we produce",
        subtitle="Every number below comes from aggregate SQL only.",
        kpis=[
            ("Impressions", "SUM(viewers)", "How many audience-seconds."),
            ("Attention time", "SUM(attending × Δt)", "Seconds actively looking."),
            ("Avg viewing duration", "MEAN(avg_dwell)", "How long an average viewer stays."),
            ("Attention rate", "attending / viewers", "% of impressions that convert to attention."),
            ("Peak concurrent", "MAX(viewers)", "Highest simultaneous audience."),
            ("M/F ratio", "SUM(male / female)", "Demographic tilt — no identities stored."),
        ],
    ),
    Slide(
        kind="content",
        title="Architecture",
        subtitle="Six stages, all on-device.",
        bullets=[
            Bullet(
                "1 · Capture — webcam frames at 720p (OpenCV VideoCapture).",
                "Optional screen capture for pHash content segmentation.",
            ),
            Bullet(
                "2 · Detect — person-detection-retail-0013 INT8 (crowd) + face-detection-retail-0004 INT8.",
                "Tiled inference for wide-angle rooms.",
            ),
            Bullet(
                "3 · Track — Kalman motion + IoU association. IDs are int counters.",
                "No embeddings, no ReID — safe on identity even across occlusion.",
            ),
            Bullet(
                "4 · Gaze — head-pose-estimation-adas-0001 (FP16) + gaze-estimation-adas-0002 (FP16, near tier only).",
                "Screen dwell computed from head/eye vectors, not identity.",
            ),
            Bullet(
                "5 · Aggregate — 5-second windows, keyed by content segment ID (pHash).",
                "Only counters + ratios written to SQLite.",
            ),
            Bullet(
                "6 · Dashboard — Streamlit multipage · Plotly · Pandas resample.",
                "Auto-refresh every 5 s, cached 30 s.",
            ),
        ],
    ),
    Slide(
        kind="twocol",
        title="Stack — no magic, all Intel",
        subtitle="Runs on the machine already in your bag.",
        right_title="Data & UI",
        bullets=[
            Bullet("Python 3.12, typed + ruff-clean + pytest-covered."),
            Bullet("OpenVINO 2024 · AUTO device (GPU preferred)."),
            Bullet("Open Model Zoo — 5 INT8/FP16 models, ~50 MB total."),
            Bullet("psutil-backed subprocess supervisor for lifecycle."),
            Bullet("Target: Intel Core i7-1185G7 + Iris Xe iGPU."),
        ],
        right_bullets=[
            Bullet("SQLite — a single file, zero admin, portable."),
            Bullet("Pandas + Plotly for time-series aggregation."),
            Bullet("Streamlit multipage with shared theme.py."),
            Bullet("Auto-refresh via streamlit-autorefresh (5 s)."),
            Bullet("Everything in-process — no message broker, no queue."),
        ],
    ),
    Slide(
        kind="kpi",
        title="Real-time budget hit",
        subtitle="Benchmarked on Intel Core i7-1185G7 · Iris Xe iGPU · 720p input.",
        kpis=[
            ("≥ 15 FPS", "End-to-end", "Capture → detect → track → gaze → sink."),
            ("< 50 MB", "Model footprint", "5 OMZ models, INT8/FP16."),
            ("< 5 %", "Idle CPU", "Between inference passes."),
            ("~ 30 s", "Cold-start", "First model compile is cached."),
            ("100 %", "On-device", "Zero cloud calls, zero telemetry."),
            ("121", "Tests passing", "Privacy smoke test in CI."),
        ],
    ),
    Slide(
        kind="content",
        title="What Hack the Stack enables next",
        subtitle="From working demo to shippable product.",
        bullets=[
            Bullet(
                "🎯 Fleet mode.",
                "Publish aggregates to a central store — same privacy contract, N screens.",
            ),
            Bullet(
                "🎯 Content A/B.",
                "Rotate creatives, join to attention time — the leaderboard becomes a decision tool.",
            ),
            Bullet(
                "🎯 Age × dwell heatmap.",
                "Aggregate demography by content — never per-person.",
            ),
            Bullet(
                "🎯 Retail integration.",
                "OpenVINO Model Server on edge · sync SQLite → data warehouse over WireGuard.",
            ),
            Bullet(
                "🎯 Certification.",
                "Third-party privacy audit — the codebase is already grep-provable.",
            ),
        ],
    ),
    Slide(
        kind="closing",
        title="Gaze Analytics",
        subtitle=(
            "Signage engagement, on-device, provably private.\n"
            "Repo · Demo · Team — ready at the booth."
        ),
    ),
]


# ---- Rendering helpers ----------------------------------------------------


def _bg(slide, color: RGBColor) -> None:
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H
    )
    bg.fill.solid()
    bg.fill.fore_color.rgb = color
    bg.line.fill.background()
    bg.shadow.inherit = False
    slide.shapes._spTree.remove(bg._element)
    slide.shapes._spTree.insert(2, bg._element)


def _gradient_hero(slide) -> None:
    """Approximate the app's gradient with three stacked stripes."""
    stripes = [
        (NAVY, 0.0, 0.45),
        (INDIGO, 0.45, 0.85),
        (SKY, 0.85, 1.0),
    ]
    for color, y0, y1 in stripes:
        rect = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            0,
            int(SLIDE_H * y0),
            SLIDE_W,
            int(SLIDE_H * (y1 - y0)),
        )
        rect.fill.solid()
        rect.fill.fore_color.rgb = color
        rect.line.fill.background()


def _add_text(
    slide,
    left: Emu,
    top: Emu,
    width: Emu,
    height: Emu,
    text: str,
    *,
    size: int = 18,
    bold: bool = False,
    color: RGBColor = INK,
    align: PP_ALIGN = PP_ALIGN.LEFT,
    anchor: MSO_ANCHOR = MSO_ANCHOR.TOP,
    font: str = "Segoe UI",
):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = 0
    tf.margin_top = tf.margin_bottom = 0
    for i, line in enumerate(text.split("\n")):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        run.font.name = font
        run.font.size = Pt(size)
        run.font.bold = bold
        run.font.color.rgb = color
    return box


def _add_pill(slide, left: Emu, top: Emu, text: str) -> None:
    pill_w = Inches(2.4)
    pill_h = Inches(0.4)
    pill = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, pill_w, pill_h
    )
    pill.fill.solid()
    pill.fill.fore_color.rgb = ACCENT
    pill.line.fill.background()
    tf = pill.text_frame
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = "Segoe UI"
    run.font.size = Pt(11)
    run.font.bold = True
    run.font.color.rgb = WHITE


def _add_footer(slide, text: str) -> None:
    if not text:
        return
    _add_text(
        slide,
        Inches(0.6),
        Inches(7.0),
        Inches(12.1),
        Inches(0.35),
        text,
        size=10,
        color=MUTED,
        align=PP_ALIGN.LEFT,
    )


def _bullet_block(
    slide,
    bullets: list[Bullet],
    left: Emu,
    top: Emu,
    width: Emu,
    *,
    row_h: Emu = Inches(0.85),
) -> None:
    for i, b in enumerate(bullets):
        y = top + row_h * i
        # dot
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, left, y + Inches(0.14), Inches(0.16), Inches(0.16)
        )
        dot.fill.solid()
        dot.fill.fore_color.rgb = SKY
        dot.line.fill.background()
        # primary line
        _add_text(
            slide,
            left + Inches(0.35),
            y,
            width - Inches(0.35),
            Inches(0.4),
            b.text,
            size=18,
            bold=True,
            color=INK,
        )
        # sub line
        if b.sub:
            _add_text(
                slide,
                left + Inches(0.35),
                y + Inches(0.38),
                width - Inches(0.35),
                Inches(0.4),
                b.sub,
                size=13,
                color=SLATE,
            )


# ---- Slide builders -------------------------------------------------------


def _render_title(pres: Presentation, s: Slide) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _gradient_hero(slide)

    # Badge
    badge = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.9),
        Inches(0.9),
        Inches(0.85),
        Inches(0.85),
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    badge.line.fill.background()
    tf = badge.text_frame
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = "📺"
    run.font.size = Pt(36)

    _add_text(
        slide,
        Inches(2.0),
        Inches(0.95),
        Inches(10),
        Inches(0.5),
        "GAZE ANALYTICS",
        size=12,
        color=SKY,
        bold=True,
    )
    _add_text(
        slide,
        Inches(2.0),
        Inches(1.35),
        Inches(10),
        Inches(0.6),
        "Signage Engagement",
        size=22,
        color=WHITE,
        bold=True,
    )

    # Big title
    _add_text(
        slide,
        Inches(0.9),
        Inches(2.8),
        Inches(11.5),
        Inches(1.6),
        s.title,
        size=64,
        bold=True,
        color=WHITE,
    )
    # Sub
    _add_text(
        slide,
        Inches(0.9),
        Inches(4.6),
        Inches(11.5),
        Inches(1.6),
        s.subtitle,
        size=18,
        color=RGBColor(0xE0, 0xF2, 0xFE),
    )

    _add_pill(slide, Inches(0.9), Inches(6.5), "PRIVACY-BY-DESIGN · ON-DEVICE")


def _render_content(pres: Presentation, s: Slide) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _bg(slide, LIGHT)

    # Top accent bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.15)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = SKY
    bar.line.fill.background()

    _add_text(
        slide,
        Inches(0.6),
        Inches(0.45),
        Inches(12.1),
        Inches(0.6),
        s.title,
        size=32,
        bold=True,
        color=INK,
    )
    if s.subtitle:
        _add_text(
            slide,
            Inches(0.6),
            Inches(1.1),
            Inches(12.1),
            Inches(0.5),
            s.subtitle,
            size=15,
            color=SLATE,
        )

    _bullet_block(
        slide, s.bullets, Inches(0.7), Inches(1.85), Inches(12.0)
    )
    _add_footer(slide, s.footer)


def _render_twocol(pres: Presentation, s: Slide) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _bg(slide, LIGHT)

    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.15)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = SKY
    bar.line.fill.background()

    _add_text(
        slide,
        Inches(0.6),
        Inches(0.45),
        Inches(12.1),
        Inches(0.6),
        s.title,
        size=32,
        bold=True,
        color=INK,
    )
    if s.subtitle:
        _add_text(
            slide,
            Inches(0.6),
            Inches(1.1),
            Inches(12.1),
            Inches(0.5),
            s.subtitle,
            size=15,
            color=SLATE,
        )

    # Two card backgrounds
    card_top = Inches(1.85)
    card_h = Inches(4.9)
    card_w = Inches(5.9)
    for i, x in enumerate([Inches(0.6), Inches(6.85)]):
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, card_top, card_w, card_h
        )
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = BORDER
        card.line.width = Pt(1)

    # Left col
    if s.bullets:
        _add_text(
            slide,
            Inches(0.9),
            card_top + Inches(0.2),
            Inches(5.4),
            Inches(0.45),
            s.bullets[0].text,
            size=18,
            bold=True,
            color=INK,
        )
        if s.bullets[0].sub:
            _add_text(
                slide,
                Inches(0.9),
                card_top + Inches(0.65),
                Inches(5.4),
                Inches(0.35),
                s.bullets[0].sub,
                size=12,
                color=MUTED,
            )
        _bullet_block(
            slide,
            s.bullets[1:],
            Inches(0.9),
            card_top + Inches(1.1),
            Inches(5.4),
            row_h=Inches(0.75),
        )

    # Right col
    if s.right_title:
        _add_text(
            slide,
            Inches(7.15),
            card_top + Inches(0.2),
            Inches(5.4),
            Inches(0.45),
            s.right_title,
            size=18,
            bold=True,
            color=INK,
        )
    _bullet_block(
        slide,
        s.right_bullets,
        Inches(7.15),
        card_top + Inches(0.85),
        Inches(5.4),
        row_h=Inches(0.75),
    )


def _render_kpi(pres: Presentation, s: Slide) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _bg(slide, LIGHT)

    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.15)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = SKY
    bar.line.fill.background()

    _add_text(
        slide,
        Inches(0.6),
        Inches(0.45),
        Inches(12.1),
        Inches(0.6),
        s.title,
        size=32,
        bold=True,
        color=INK,
    )
    if s.subtitle:
        _add_text(
            slide,
            Inches(0.6),
            Inches(1.1),
            Inches(12.1),
            Inches(0.5),
            s.subtitle,
            size=15,
            color=SLATE,
        )

    # 3x2 grid of KPI tiles
    tile_w = Inches(4.0)
    tile_h = Inches(2.4)
    gap = Inches(0.15)
    x0 = Inches(0.6)
    y0 = Inches(1.9)

    for idx, (label, value, sub) in enumerate(s.kpis[:6]):
        row, col = divmod(idx, 3)
        x = x0 + (tile_w + gap) * col
        y = y0 + (tile_h + gap) * row

        tile = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, y, tile_w, tile_h
        )
        tile.fill.solid()
        tile.fill.fore_color.rgb = WHITE
        tile.line.color.rgb = BORDER
        tile.line.width = Pt(1)

        _add_text(
            slide,
            x + Inches(0.25),
            y + Inches(0.2),
            tile_w - Inches(0.4),
            Inches(0.4),
            label.upper(),
            size=11,
            bold=True,
            color=MUTED,
        )
        _add_text(
            slide,
            x + Inches(0.25),
            y + Inches(0.65),
            tile_w - Inches(0.4),
            Inches(0.9),
            value,
            size=28,
            bold=True,
            color=INK,
        )
        _add_text(
            slide,
            x + Inches(0.25),
            y + Inches(1.6),
            tile_w - Inches(0.4),
            Inches(0.7),
            sub,
            size=12,
            color=SLATE,
        )


def _render_closing(pres: Presentation, s: Slide) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _gradient_hero(slide)
    _add_text(
        slide,
        Inches(0.9),
        Inches(2.5),
        Inches(11.5),
        Inches(1.6),
        s.title,
        size=64,
        bold=True,
        color=WHITE,
    )
    _add_text(
        slide,
        Inches(0.9),
        Inches(4.3),
        Inches(11.5),
        Inches(1.6),
        s.subtitle,
        size=20,
        color=RGBColor(0xE0, 0xF2, 0xFE),
    )
    _add_pill(slide, Inches(0.9), Inches(6.4), "Thank you · Q & A")


RENDERERS = {
    "title": _render_title,
    "content": _render_content,
    "twocol": _render_twocol,
    "kpi": _render_kpi,
    "closing": _render_closing,
}


def main() -> Path:
    pres = Presentation()
    pres.slide_width = SLIDE_W
    pres.slide_height = SLIDE_H

    for s in DECK:
        RENDERERS[s.kind](pres, s)

    out_dir = Path("presentations")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "GazeAnalytics_HackTheStack.pptx"
    pres.save(out_path)
    return out_path


if __name__ == "__main__":
    path = main()
    print(f"Wrote {path}  ({path.stat().st_size / 1024:.1f} KB)")
