"""CP6 entry point: CP5 defaults plus validated exact-category retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_cp5_variant import cp5_main


def cp6_main() -> None:
    if "--category-filter" not in sys.argv[1:]:
        sys.argv.append("--category-filter")
    cp5_main()


if __name__ == "__main__":
    cp6_main()
