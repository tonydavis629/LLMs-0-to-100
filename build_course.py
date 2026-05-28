#!/usr/bin/env python3
"""Build every bundled slide deck in the course."""

from __future__ import annotations

import sys
from pathlib import Path

from slides.build import build_module


def discover_modules(slides_dir: Path) -> list[Path]:
    modules = []
    for path in slides_dir.iterdir():
        source_dir = path / "source"
        if (
            path.is_dir()
            and path.name.startswith("module_")
            and (source_dir / "slides.md").exists()
            and (source_dir / "config.js").exists()
        ):
            modules.append(path)
    return sorted(modules)


def main() -> int:
    repo_root = Path(__file__).resolve().parent
    slides_dir = repo_root / "slides"
    modules = discover_modules(slides_dir)

    if not modules:
        print("No slide modules with source/slides.md and source/config.js were found.", file=sys.stderr)
        return 1

    for module_dir in modules:
        print(f"Building {module_dir.relative_to(repo_root)}")
        build_module(module_dir)

    suffix = "" if len(modules) == 1 else "s"
    print(f"Built {len(modules)} slide module bundle{suffix}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
