"""Command-line entry point for NablaGuard artifacts and package metadata."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from nablaguard import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the stable CLI parser."""

    parser = argparse.ArgumentParser(
        prog="nabla", description="Verification and debugging for differentiable programs."
    )
    parser.add_argument("--version", action="version", version=f"NablaGuard {__version__}")
    subcommands = parser.add_subparsers(dest="command")
    inspect_parser = subcommands.add_parser("inspect", help="inspect a failure artifact")
    inspect_parser.add_argument("artifact", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the NablaGuard CLI."""

    parser = build_parser()
    arguments = parser.parse_args(argv)
    if arguments.command == "inspect":
        metadata_path = arguments.artifact / "metadata.json"
        if not metadata_path.is_file():
            parser.error(f"not a NablaGuard artifact: {arguments.artifact}")
        print(json.dumps(json.loads(metadata_path.read_text(encoding="utf-8")), indent=2))
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
