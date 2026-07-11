"""Downloads OpenVINO Open Model Zoo models into ``models/``.

Requires ``openvino-dev`` (installed via ``requirements.txt``), which
provides the ``omz_downloader`` CLI.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

MODELS = [
    "person-detection-retail-0013",
    "face-detection-retail-0004",
    "head-pose-estimation-adas-0001",
    "gaze-estimation-adas-0002",
    "age-gender-recognition-retail-0013",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"


def main() -> int:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name in MODELS:
        print(f"\n=== downloading {name} ===")
        # omz_downloader picks up default precisions (FP16 + FP16-INT8 where available)
        rc = subprocess.call(
            [
                "omz_downloader",
                "--name",
                name,
                "--output_dir",
                str(MODELS_DIR),
            ]
        )
        if rc != 0:
            print(f"!! omz_downloader failed for {name} (exit {rc})", file=sys.stderr)
            return rc
    print("\nAll models downloaded to", MODELS_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())
