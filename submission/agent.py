"""TechJam submission entry point.

Exports `Agent`; the implementation lives in `src/`. Runtime is standard
library only and requires no network access.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow both `from agent import Agent` and `from submission.agent import Agent`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.agent import Agent  # noqa: E402

__all__ = ["Agent"]
