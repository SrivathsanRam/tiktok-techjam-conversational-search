"""CP5-named entry point with the selected protocol-aware defaults."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from scripts.evaluate_cp3_variant import main


def cp5_main() -> None:
    switches = ("--no-cross-encoder", "--dialogue-cards")
    for switch in switches:
        if switch not in sys.argv[1:]:
            sys.argv.append(switch)
    defaults = {
        "--dialogue-tiebreak": "popularity",
        "--opening-output-k": "1",
        "--ambiguous-output-k": "1",
    }
    for flag, value in defaults.items():
        if not any(arg == flag or arg.startswith(flag + "=") for arg in sys.argv[1:]):
            sys.argv.extend((flag, value))
    main()


if __name__ == "__main__":
    cp5_main()
