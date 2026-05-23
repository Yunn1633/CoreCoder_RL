#!/usr/bin/env python3
"""Install the CoreCoder_RL slime/CoreCoder overlay into a slime checkout."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slime-root", required=True, type=Path, help="Path to the slime repository root")
    parser.add_argument("--overlay-root", type=Path, default=Path(__file__).resolve().parents[1] / "patches" / "corecoder_slime_overlay")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    overlay_root = args.overlay_root.resolve()
    slime_root = args.slime_root.resolve()
    if not overlay_root.exists():
        raise SystemExit(f"overlay root does not exist: {overlay_root}")
    if not slime_root.exists():
        raise SystemExit(f"slime root does not exist: {slime_root}")

    for src in sorted(overlay_root.rglob("*.py")):
        rel = src.relative_to(overlay_root)
        target = slime_root / rel
        print(f"{src} -> {target}")
        if not args.dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
