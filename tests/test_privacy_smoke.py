"""Privacy invariant smoke tests.

These tests grep the source tree for banned APIs that would violate the
"frames are ephemeral" contract. Any commit that introduces one of these
without an explicit ``# noqa: PRIVACY`` exemption will fail CI.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"

BANNED_PATTERNS = {
    "cv2.imwrite":          r"\bcv2\.imwrite\b",
    "cv2.VideoWriter":      r"\bcv2\.VideoWriter\b",
    "PIL Image.save":       r"\.save\s*\(",  # heuristic; refined by exemption
    "numpy.save":           r"\bnp(?:umpy)?\.save\b",
    "pickle.dump":          r"\bpickle\.dump\b",
    "face_recognition":     r"\bimport\s+face_recognition\b",
    "deepface":             r"\bimport\s+deepface\b",
    "dlib.face_recognition":r"\bdlib\.face_recognition",
}

# The rendered thumbnail path (sprint 4) is the ONE place PIL .save is
# permitted — and only for a 160x90 thumbnail of the *content screen*,
# never a face. That module will mark its call with `# noqa: PRIVACY`.
EXEMPTION_MARKER = "# noqa: PRIVACY"


def _iter_py_files() -> list[Path]:
    return [p for p in SRC.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_banned_apis_in_src() -> None:
    offenders: list[str] = []
    for py in _iter_py_files():
        text = py.read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if EXEMPTION_MARKER in line:
                continue
            for name, pattern in BANNED_PATTERNS.items():
                if re.search(pattern, line):
                    offenders.append(f"{py.relative_to(SRC.parent)}:{line_no} {name} -> {line.strip()}")
    assert not offenders, (
        "Privacy invariant violated. Banned APIs found:\n  " + "\n  ".join(offenders)
    )
