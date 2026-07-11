"""Capture layer — webcam and screen frame sources.

Every producer here MUST guarantee that frame buffers are not persisted.
Callers process the frame, then let it fall out of scope.
"""

from __future__ import annotations
