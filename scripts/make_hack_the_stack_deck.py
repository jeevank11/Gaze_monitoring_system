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
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
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
SKY_SOFT = RGBColor(0xE0, 0xF2, 0xFE)

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

FONT = "Segoe UI"


# ---- Slide content model --------------------------------------------------


@dataclass
class Bullet:
    text: str
    sub: str | None = None


@dataclass
class Slide:
    kind: str  # "title" | "content" | "twocol" | "kpi" | "pipeline" | "closing"
    title: str
    subtitle: str = ""
    bullets: list[Bullet] = field(default_factory=list)
    right_bullets: list[Bullet] = field(default_factory=list)
    right_title: str = ""
    footer: str = ""
    kpis: list[tuple[str, str, str]] = field(default_factory=list)


# ---- Content --------------------------------------------------------------

DECK: list[Slide] = [
    # 1 — Title
    Slide(
        kind="title",
        title="See who's watching.\nProve nothing else.",
        subtitle="Privacy-preserving signage analytics on Intel CPU / iGPU.",
    ),
    # 2 — Problem
    Slide(
        kind="content",
        title="Signage is blind.",
        subtitle="Operators buy screens. They can't prove anyone watched.",
        bullets=[
            Bullet(
                "No measurement.",
                "Even Nielsen-style panels are proxies.",
            ),
            Bullet(
                "Cloud CV is a liability.",
                "Face recognition, ReID, embeddings — GDPR · DPDP · BIPA.",
            ),
            Bullet(
                "Edge boxes are opaque.",
                "$1k+ vendor lock-in. No audit path.",
            ),
        ],
        footer="Signage market $30B+ · <5% instrumented today.",
    ),
    # 3 — Solution
    Slide(
        kind="content",
        title="One idea: aggregates only.",
        subtitle="Watch the room. Never remember a face.",
        bullets=[
            Bullet(
                "100 % on-device.",
                "Intel CPU + Iris Xe via OpenVINO. Zero cloud. Zero telemetry.",
            ),
            Bullet(
                "Frames die at 30 ms.",
                "No imwrite, no video writer — enforced by a CI grep.",
            ),
            Bullet(
                "Only counters persist.",
                "SQLite: numbers, ratios, 160×90 thumbnails. No identities. Ever.",
            ),
        ],
        footer="Privacy contract lives in a test that fails the build.",
    ),
    # 4 — Pipeline
    Slide(
        kind="pipeline",
        title="How it works.",
        subtitle="Six stages · all on the laptop.",
        bullets=[
            Bullet("Capture", "720p webcam"),
            Bullet("Detect", "OMZ INT8"),
            Bullet("Track", "Kalman + IoU"),
            Bullet("Gaze", "Head + eye"),
            Bullet("Aggregate", "5 s windows"),
            Bullet("Dashboard", "Streamlit"),
        ],
        footer="OpenVINO · 5 OMZ models · ~50 MB total.",
    ),
    # 5 — Live demo
    Slide(
        kind="twocol",
        title="Two dashboards. One process.",
        subtitle="Run the pipeline. Read the trends. Same codebase.",
        bullets=[
            Bullet("📺 Gaze Live", "Real-time control plane"),
            Bullet("Start / stop as a detached subprocess."),
            Bullet("Toggle detection · gaze · demographics."),
            Bullet("Pick device — CPU / GPU / AUTO."),
            Bullet("Live KPIs, refreshed every 5 s."),
        ],
        right_title="👁️ Gaze Insights",
        right_bullets=[
            Bullet("Historical analytics"),
            Bullet("Impressions · attention time · dwell · rate."),
            Bullet("Engagement trend — 5 s → 1 h auto-rebin."),
            Bullet("Top content leaderboard with thumbnails."),
            Bullet("Gender split — SQL aggregates only."),
        ],
    ),
    # 6 — Metrics
    Slide(
        kind="kpi",
        title="What we measure.",
        subtitle="Every number below is aggregate SQL — zero per-person records.",
        kpis=[
            ("Impressions", "SUM viewers", "Audience-seconds delivered."),
            ("Attention time", "SUM attending × Δt", "Seconds actively looking."),
            ("Avg dwell", "MEAN avg_dwell", "How long a viewer stays."),
            ("Attention rate", "attending / viewers", "% converting look → focus."),
            ("Peak concurrent", "MAX viewers", "Biggest simultaneous audience."),
            ("Gender split", "SUM(m / f)", "Aggregate tilt. No identities."),
        ],
    ),
    # 7 — Performance
    Slide(
        kind="kpi",
        title="Real-time budget hit.",
        subtitle="Intel Core i7-1185G7 · Iris Xe iGPU · 720p input.",
        kpis=[
            ("≥ 15 FPS", "End-to-end", "Capture → detect → track → gaze → sink."),
            ("< 50 MB", "Model footprint", "5 OMZ models · INT8 / FP16."),
            ("< 5 %", "Idle CPU", "Between inference passes."),
            ("~ 30 s", "Cold start", "First model compile is cached."),
            ("100 %", "On-device", "Zero cloud calls · zero telemetry."),
            ("121 / 121", "Tests green", "Privacy grep runs in CI."),
        ],
    ),
    # 8 — Future scope
    Slide(
        kind="content",
        title="Where this goes next.",
        subtitle="Same privacy contract. Wider surface.",
        bullets=[
            Bullet(
                "🚀 Fleet federation.",
                "Aggregates from N screens. DP-noise on outbound counters.",
            ),
            Bullet(
                "🎯 Attention heatmaps.",
                "Saccade-level overlays on content thumbnails.",
            ),
            Bullet(
                "📺 DOOH integration.",
                "Programmatic ad platforms · IAB OpenRTB · measurable inventory.",
            ),
            Bullet(
                "🧠 Silicon expansion.",
                "Intel NPU · Arc dGPU · low-cost Movidius tier.",
            ),
            Bullet(
                "🛡️ Certification.",
                "GDPR · DPDP · BIPA · CPRA — grep-provable dossier.",
            ),
        ],
        footer="North star: 10 000 screens instrumented · 0 frames ever leaving the device.",
    ),
    # 9 — Closing
    Slide(
        kind="closing",
        title="Signage engagement,\non-device, provably private.",
        subtitle="Ready at the booth.  Repo · Demo · Team.",
    ),
]


# ---- Rendering primitives -------------------------------------------------


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
    """Stacked stripes approximating the app hero gradient."""
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
    font: str = FONT,
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


def _pill(
    slide,
    left: Emu,
    top: Emu,
    width: Emu,
    height: Emu,
    text: str,
    *,
    fill: RGBColor = ACCENT,
    fg: RGBColor = WHITE,
    size: int = 11,
) -> None:
    p = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height
    )
    p.fill.solid()
    p.fill.fore_color.rgb = fill
    p.line.fill.background()
    tf = p.text_frame
    tf.margin_left = tf.margin_right = Inches(0.1)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    para = tf.paragraphs[0]
    para.alignment = PP_ALIGN.CENTER
    run = para.add_run()
    run.text = text
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = True
    run.font.color.rgb = fg


def _chrome(slide, index: int, total: int) -> None:
    """Top accent bar, section counter, and bottom brand footer."""
    # Top accent bar
    bar = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, Inches(0.12)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = SKY
    bar.line.fill.background()

    # Section counter pill (top-right)
    _pill(
        slide,
        left=Inches(11.6),
        top=Inches(0.4),
        width=Inches(1.15),
        height=Inches(0.36),
        text=f"{index:02d} / {total:02d}",
        fill=NAVY,
        fg=WHITE,
        size=10,
    )

    # Bottom hairline
    line = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, Inches(7.15), SLIDE_W, Inches(0.02)
    )
    line.fill.solid()
    line.fill.fore_color.rgb = BORDER
    line.line.fill.background()

    # Brand mark (bottom-left)
    _add_text(
        slide,
        Inches(0.6),
        Inches(7.22),
        Inches(6.0),
        Inches(0.28),
        "GAZE ANALYTICS  ·  Privacy by design",
        size=9,
        bold=True,
        color=MUTED,
    )
    # Page counter (bottom-right)
    _add_text(
        slide,
        Inches(11.4),
        Inches(7.22),
        Inches(1.4),
        Inches(0.28),
        f"{index} / {total}",
        size=9,
        color=MUTED,
        align=PP_ALIGN.RIGHT,
    )


def _title_block(slide, s: Slide) -> None:
    _add_text(
        slide,
        Inches(0.6),
        Inches(0.55),
        Inches(10.5),
        Inches(0.75),
        s.title,
        size=34,
        bold=True,
        color=INK,
    )
    if s.subtitle:
        _add_text(
            slide,
            Inches(0.6),
            Inches(1.25),
            Inches(12.1),
            Inches(0.5),
            s.subtitle,
            size=15,
            color=SLATE,
        )


def _slide_footer(slide, text: str) -> None:
    if not text:
        return
    _add_text(
        slide,
        Inches(0.6),
        Inches(6.75),
        Inches(12.1),
        Inches(0.35),
        text,
        size=11,
        color=MUTED,
    )


def _bullet_block(
    slide,
    bullets: list[Bullet],
    left: Emu,
    top: Emu,
    width: Emu,
    *,
    row_h: Emu = Inches(0.95),
    primary_size: int = 18,
    sub_size: int = 13,
) -> None:
    for i, b in enumerate(bullets):
        y = top + row_h * i
        # dot
        dot = slide.shapes.add_shape(
            MSO_SHAPE.OVAL, left, y + Inches(0.16), Inches(0.18), Inches(0.18)
        )
        dot.fill.solid()
        dot.fill.fore_color.rgb = SKY
        dot.line.fill.background()
        # primary
        _add_text(
            slide,
            left + Inches(0.4),
            y,
            width - Inches(0.4),
            Inches(0.42),
            b.text,
            size=primary_size,
            bold=True,
            color=INK,
        )
        if b.sub:
            _add_text(
                slide,
                left + Inches(0.4),
                y + Inches(0.42),
                width - Inches(0.4),
                Inches(0.42),
                b.sub,
                size=sub_size,
                color=SLATE,
            )


# ---- Slide builders -------------------------------------------------------


def _render_title(pres: Presentation, s: Slide, index: int, total: int) -> None:
    _ = index, total  # title has no chrome
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _gradient_hero(slide)

    # Decorative rings (top-right)
    for radius_in, opacity_color in [
        (3.6, RGBColor(0x1E, 0x3A, 0x8A)),
        (2.6, RGBColor(0x2C, 0x4B, 0xA6)),
        (1.6, RGBColor(0x3B, 0x5D, 0xC2)),
    ]:
        d = Inches(radius_in * 2)
        ring = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            SLIDE_W - Inches(radius_in) - Inches(-1.0),
            Inches(-1.0),
            d,
            d,
        )
        ring.fill.background()
        ring.line.color.rgb = opacity_color
        ring.line.width = Pt(1.25)

    # Brand mark (top-left)
    badge = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.9),
        Inches(0.9),
        Inches(0.85),
        Inches(0.85),
    )
    badge.fill.solid()
    badge.fill.fore_color.rgb = WHITE
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
        Inches(0.4),
        "GAZE ANALYTICS",
        size=12,
        color=SKY,
        bold=True,
    )
    _add_text(
        slide,
        Inches(2.0),
        Inches(1.32),
        Inches(10),
        Inches(0.6),
        "Signage Engagement",
        size=20,
        color=WHITE,
        bold=True,
    )

    # Big title
    _add_text(
        slide,
        Inches(0.9),
        Inches(2.9),
        Inches(11.5),
        Inches(2.4),
        s.title,
        size=54,
        bold=True,
        color=WHITE,
    )
    # Subtitle
    _add_text(
        slide,
        Inches(0.9),
        Inches(5.4),
        Inches(11.5),
        Inches(0.9),
        s.subtitle,
        size=18,
        color=SKY_SOFT,
    )

    _pill(
        slide,
        left=Inches(0.9),
        top=Inches(6.4),
        width=Inches(2.4),
        height=Inches(0.42),
        text="HACK THE STACK · 2026",
        fill=ACCENT,
        fg=WHITE,
        size=11,
    )
    _pill(
        slide,
        left=Inches(3.45),
        top=Inches(6.4),
        width=Inches(2.6),
        height=Inches(0.42),
        text="PRIVACY BY DESIGN",
        fill=WHITE,
        fg=NAVY,
        size=11,
    )


def _render_content(pres: Presentation, s: Slide, index: int, total: int) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _bg(slide, LIGHT)
    _chrome(slide, index, total)
    _title_block(slide, s)

    # Body card
    card = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE,
        Inches(0.6),
        Inches(2.0),
        Inches(12.13),
        Inches(4.55),
    )
    card.fill.solid()
    card.fill.fore_color.rgb = WHITE
    card.line.color.rgb = BORDER
    card.line.width = Pt(1)

    # Left accent bar on card
    strip = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE,
        Inches(0.6),
        Inches(2.0),
        Inches(0.12),
        Inches(4.55),
    )
    strip.fill.solid()
    strip.fill.fore_color.rgb = SKY
    strip.line.fill.background()

    _bullet_block(
        slide,
        s.bullets,
        Inches(1.05),
        Inches(2.3),
        Inches(11.5),
        row_h=Inches(1.3),
        primary_size=20,
        sub_size=14,
    )
    _slide_footer(slide, s.footer)


def _render_twocol(pres: Presentation, s: Slide, index: int, total: int) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _bg(slide, LIGHT)
    _chrome(slide, index, total)
    _title_block(slide, s)

    card_top = Inches(2.0)
    card_h = Inches(4.55)
    card_w = Inches(5.95)
    positions = [
        (Inches(0.6), s.bullets, "", SKY),
        (Inches(6.78), s.right_bullets, s.right_title, INDIGO),
    ]

    for x, items, header_title, header_color in positions:
        # Card
        card = slide.shapes.add_shape(
            MSO_SHAPE.ROUNDED_RECTANGLE, x, card_top, card_w, card_h
        )
        card.fill.solid()
        card.fill.fore_color.rgb = WHITE
        card.line.color.rgb = BORDER
        card.line.width = Pt(1)

        # Header strip
        strip = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE,
            x,
            card_top,
            card_w,
            Inches(0.4),
        )
        strip.fill.solid()
        strip.fill.fore_color.rgb = header_color
        strip.line.fill.background()

        # If items has [0] as (title, sub) pair (bullets slot with heading), use it
        header_source = items[0] if (items and not header_title) else None
        header_text = header_title or (header_source.text if header_source else "")
        header_sub = (
            header_source.sub if header_source and header_source.sub else ""
        )
        _add_text(
            slide,
            x + Inches(0.35),
            card_top + Inches(0.05),
            card_w - Inches(0.7),
            Inches(0.3),
            header_text,
            size=14,
            bold=True,
            color=WHITE,
        )

        body_items = items[1:] if header_source else items
        if header_sub:
            _add_text(
                slide,
                x + Inches(0.35),
                card_top + Inches(0.55),
                card_w - Inches(0.7),
                Inches(0.35),
                header_sub,
                size=13,
                bold=True,
                color=INK,
            )
            body_top = card_top + Inches(1.0)
        else:
            body_top = card_top + Inches(0.6)

        _bullet_block(
            slide,
            body_items,
            x + Inches(0.35),
            body_top,
            card_w - Inches(0.7),
            row_h=Inches(0.78),
            primary_size=15,
            sub_size=12,
        )
    _slide_footer(slide, s.footer)


def _render_kpi(pres: Presentation, s: Slide, index: int, total: int) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _bg(slide, LIGHT)
    _chrome(slide, index, total)
    _title_block(slide, s)

    tile_w = Inches(4.0)
    tile_h = Inches(2.15)
    gap = Inches(0.15)
    x0 = Inches(0.6)
    y0 = Inches(2.05)

    # Alternate accent colors so the grid feels lively
    accent_cycle = [SKY, INDIGO, ACCENT, SKY, INDIGO, ACCENT]

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

        # Colored top strip
        strip = slide.shapes.add_shape(
            MSO_SHAPE.RECTANGLE, x, y, tile_w, Inches(0.08)
        )
        strip.fill.solid()
        strip.fill.fore_color.rgb = accent_cycle[idx]
        strip.line.fill.background()

        _add_text(
            slide,
            x + Inches(0.28),
            y + Inches(0.22),
            tile_w - Inches(0.5),
            Inches(0.35),
            label.upper(),
            size=11,
            bold=True,
            color=MUTED,
        )
        _add_text(
            slide,
            x + Inches(0.28),
            y + Inches(0.6),
            tile_w - Inches(0.5),
            Inches(0.9),
            value,
            size=28,
            bold=True,
            color=INK,
        )
        _add_text(
            slide,
            x + Inches(0.28),
            y + Inches(1.5),
            tile_w - Inches(0.5),
            Inches(0.55),
            sub,
            size=12,
            color=SLATE,
        )
    _slide_footer(slide, s.footer)


def _render_pipeline(
    pres: Presentation, s: Slide, index: int, total: int
) -> None:
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _bg(slide, LIGHT)
    _chrome(slide, index, total)
    _title_block(slide, s)

    # 6 nodes across the slide
    n = len(s.bullets)
    x_start = Inches(0.6)
    avail = SLIDE_W - Inches(1.2)
    circle_d = Inches(1.15)
    slot_w = avail / n

    center_y = Inches(3.9)
    circle_top = center_y - circle_d / 2

    node_centers: list[Emu] = []

    for i, b in enumerate(s.bullets):
        slot_left = x_start + slot_w * i
        cx = slot_left + slot_w / 2
        node_centers.append(cx)

        # Circle
        circle = slide.shapes.add_shape(
            MSO_SHAPE.OVAL,
            cx - circle_d / 2,
            circle_top,
            circle_d,
            circle_d,
        )
        circle.fill.solid()
        # Alternate SKY / INDIGO for rhythm
        circle.fill.fore_color.rgb = SKY if i % 2 == 0 else INDIGO
        circle.line.fill.background()

        # Number inside circle
        tf = circle.text_frame
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(i + 1)
        run.font.name = FONT
        run.font.size = Pt(28)
        run.font.bold = True
        run.font.color.rgb = WHITE

        # Stage label (below circle)
        _add_text(
            slide,
            slot_left,
            center_y + circle_d / 2 + Inches(0.15),
            slot_w,
            Inches(0.4),
            b.text,
            size=16,
            bold=True,
            color=INK,
            align=PP_ALIGN.CENTER,
        )
        if b.sub:
            _add_text(
                slide,
                slot_left,
                center_y + circle_d / 2 + Inches(0.6),
                slot_w,
                Inches(0.4),
                b.sub,
                size=11,
                color=MUTED,
                align=PP_ALIGN.CENTER,
            )

        # Stage tag above circle
        _add_text(
            slide,
            slot_left,
            circle_top - Inches(0.45),
            slot_w,
            Inches(0.3),
            f"STAGE 0{i + 1}",
            size=9,
            bold=True,
            color=MUTED,
            align=PP_ALIGN.CENTER,
        )

    # Connector arrows between circles
    for i in range(n - 1):
        cx1 = node_centers[i] + circle_d / 2 + Inches(0.08)
        cx2 = node_centers[i + 1] - circle_d / 2 - Inches(0.08)
        line = slide.shapes.add_connector(
            MSO_CONNECTOR.STRAIGHT, cx1, center_y, cx2, center_y
        )
        line.line.color.rgb = SLATE
        line.line.width = Pt(1.5)
        # Arrowhead via a small triangle
        arrow = slide.shapes.add_shape(
            MSO_SHAPE.RIGHT_TRIANGLE,
            cx2 - Inches(0.05),
            center_y - Inches(0.08),
            Inches(0.12),
            Inches(0.16),
        )
        arrow.rotation = 30
        arrow.fill.solid()
        arrow.fill.fore_color.rgb = SLATE
        arrow.line.fill.background()

    _slide_footer(slide, s.footer)


def _render_closing(
    pres: Presentation, s: Slide, index: int, total: int
) -> None:
    _ = index, total
    slide = pres.slides.add_slide(pres.slide_layouts[6])
    _gradient_hero(slide)

    # Big title
    _add_text(
        slide,
        Inches(0.9),
        Inches(2.3),
        Inches(11.5),
        Inches(2.6),
        s.title,
        size=48,
        bold=True,
        color=WHITE,
    )
    _add_text(
        slide,
        Inches(0.9),
        Inches(5.0),
        Inches(11.5),
        Inches(0.8),
        s.subtitle,
        size=18,
        color=SKY_SOFT,
    )
    _pill(
        slide,
        left=Inches(0.9),
        top=Inches(6.15),
        width=Inches(2.6),
        height=Inches(0.5),
        text="THANK YOU · Q & A",
        fill=ACCENT,
        fg=WHITE,
        size=13,
    )


RENDERERS = {
    "title": _render_title,
    "content": _render_content,
    "twocol": _render_twocol,
    "kpi": _render_kpi,
    "pipeline": _render_pipeline,
    "closing": _render_closing,
}


def main(force: bool = False) -> Path:
    out_dir = Path("presentations")
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / "GazeAnalytics_HackTheStack.pptx"

    # Safety guard: refuse to overwrite manual edits.
    # If the .pptx has been modified more recently than this script,
    # a human has probably touched it in PowerPoint.
    if out_path.exists() and not force:
        script_mtime = Path(__file__).stat().st_mtime
        pptx_mtime = out_path.stat().st_mtime
        if pptx_mtime > script_mtime:
            raise SystemExit(
                f"Refusing to overwrite {out_path} — it is newer than this "
                f"script (someone edited it in PowerPoint). "
                f"Pass --force to overwrite, or rename/move the pptx first."
            )

    pres = Presentation()
    pres.slide_width = SLIDE_W
    pres.slide_height = SLIDE_H

    total = len(DECK)
    for i, s in enumerate(DECK, start=1):
        RENDERERS[s.kind](pres, s, i, total)

    pres.save(out_path)
    return out_path


if __name__ == "__main__":
    import sys

    force_flag = "--force" in sys.argv
    path = main(force=force_flag)
    print(f"Wrote {path}  ({path.stat().st_size / 1024:.1f} KB)")
