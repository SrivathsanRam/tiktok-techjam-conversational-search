"""Assemble submission/ in the layout docs/submission_rules.md recommends.

    submission/
      agent.py          entry file exporting Agent
      requirements.txt  runtime dependencies (none)
      README.md         setup, reproduction, limitations, disclosures
      src/              the runtime modules, copied from starter/

`starter/` stays the single source of truth: this script copies its modules
into `submission/src/`, rewriting the package prefix, then imports the result
and runs a short session so a broken package cannot be shipped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ENTRY_POINT = '''"""TechJam submission entry point.

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
'''
SMOKE_TEST = '''
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, {submission!r})
from agent import Agent

products = [
    {{"parent_asin": "A", "title": "Black leather belt",
      "categories": ["Accessories", "Belts"], "features": ["100% Leather", "Imported"],
      "details": {{"department": "mens"}}, "store": "Example",
      "description": ["Belt"], "rating_number": 10}},
    {{"parent_asin": "B", "title": "Blue cotton belt",
      "categories": ["Accessories", "Belts"], "features": ["100% Cotton"],
      "details": {{"department": "mens"}}, "store": "Example",
      "description": ["Belt"], "rating_number": 20}},
]
with tempfile.TemporaryDirectory() as directory:
    catalog = Path(directory) / "catalog.jsonl"
    catalog.write_text("".join(json.dumps(p) + "\\n" for p in products), encoding="utf-8")
    agent = Agent(catalog)
    agent.reset("s", {{"preference_tags": []}})
    ranked = agent.respond("s", "I'm looking for belts. A key requirement is: 100% Leather.", 1, 10)
    assert ranked["recommendations"][0]["parent_asin"] == "A", ranked
    assert isinstance(ranked["message"], str), ranked
    assert ranked["usage"] == {{"prompt_tokens": 0, "completion_tokens": 0}}, ranked

    # One result per page, so the second turn must page to the unseen product.
    agent.reset("r", {{"preference_tags": []}})
    first = agent.respond("r", "I'm looking for belts. A key requirement is: 100% Leather.", 1, 1)
    second = agent.respond("r", "I don't have an additional preference for other.", 2, 1)
    assert first["recommendations"][0]["parent_asin"] == "A", first
    assert second["recommendations"][0]["parent_asin"] == "B", second
print("submission smoke test passed")
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("starter"))
    parser.add_argument("--output", type=Path, default=Path("submission"))
    parser.add_argument("--readme", type=Path, default=Path("README.md"))
    parser.add_argument(
        "--requirements", type=Path, default=Path("requirements.txt")
    )
    args = parser.parse_args()

    package = args.output / "src"
    if args.output.exists():
        shutil.rmtree(args.output)
    package.mkdir(parents=True)

    copied: list[str] = []
    for module in sorted(args.source.glob("*.py")):
        text = module.read_text(encoding="utf-8")
        # starter/ is the canonical package; inside the bundle it is src/.
        text = text.replace("from starter.", "from src.").replace(
            "import starter.", "import src."
        )
        (package / module.name).write_text(text, encoding="utf-8")
        copied.append(module.name)
    weights = args.source / "reranker_weights.json"
    if weights.exists():
        shutil.copy2(weights, package / weights.name)
        copied.append(weights.name)

    (args.output / "agent.py").write_text(ENTRY_POINT, encoding="utf-8")
    shutil.copy2(args.requirements, args.output / "requirements.txt")
    shutil.copy2(args.readme, args.output / "README.md")

    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as handle:
        handle.write(SMOKE_TEST.format(submission=str(args.output.resolve())))
        smoke_path = handle.name
    result = subprocess.run(
        [sys.executable, smoke_path], capture_output=True, text=True
    )
    Path(smoke_path).unlink()
    if result.returncode != 0:
        raise SystemExit(
            f"packaged submission failed its smoke test:\n{result.stdout}{result.stderr}"
        )

    print(json.dumps({
        "output": str(args.output),
        "modules": copied,
        "entry_point": str(args.output / "agent.py"),
        "smoke_test": result.stdout.strip(),
    }, indent=2))


if __name__ == "__main__":
    main()
