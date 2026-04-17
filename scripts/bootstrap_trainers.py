from __future__ import annotations

import argparse
import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trainer_versions import TRAINERS


def checkout_path(name: str) -> Path:
    trainer = TRAINERS[name]
    return ROOT / trainer["directory"]


def _git_command(*args: str) -> list[str]:
    return ["git", *args]


def _run_git(command: list[str], dry_run: bool) -> None:
    if dry_run:
        print("dry-run:", " ".join(command))
        return
    subprocess.run(command, check=True)


def ensure_checkout(name: str, dry_run: bool = False) -> None:
    trainer = TRAINERS[name]
    path = checkout_path(name)

    if path.exists():
        if not path.is_dir():
            raise RuntimeError(f"{path} exists and is not a directory")
        if not (path / ".git").exists():
            raise RuntimeError(f"{path} exists but is not a git repository")

    commands: list[list[str]] = []
    if not path.exists():
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)
        commands.append(_git_command("clone", trainer["repo"], str(path)))
    commands.append(_git_command("-C", str(path), "fetch", "--all"))
    commands.append(_git_command("-C", str(path), "checkout", trainer["commit"]))

    for command in commands:
        _run_git(command, dry_run=dry_run)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Bootstrap pinned external trainer checkouts.")
    parser.add_argument(
        "--trainer",
        choices=sorted(TRAINERS),
        help="Bootstrap one trainer instead of all trainers.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the git commands without running them.",
    )
    args = parser.parse_args(argv)

    trainer_names = [args.trainer] if args.trainer else list(TRAINERS)
    for trainer_name in trainer_names:
        ensure_checkout(trainer_name, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
