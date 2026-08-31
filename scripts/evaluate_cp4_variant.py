"""CP4-named entry point for the configurable evaluator."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_cp3_variant import main


def cp4_main() -> None:
    defaults = {
        "--cross-candidates": "20",
        "--cross-buying-weight": "0.15",
        "--cross-browsing-weight": "0",
        "--cross-constrained-browsing-weight": "0",
        "--cross-override-weight": "0",
    }
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
            sys.argv.extend((flag, value))
    main()


if __name__ == "__main__":
    cp4_main()
