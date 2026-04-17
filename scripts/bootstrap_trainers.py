from __future__ import annotations

import sys
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trainer_versions import TRAINERS


def ensure_checkout(name: str) -> None:
    trainer = TRAINERS[name]
    path = Path(trainer["directory"])
    if not path.exists():
        subprocess.run(["git", "clone", trainer["repo"], str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "fetch", "--all"], check=True)
    subprocess.run(["git", "-C", str(path), "checkout", trainer["commit"]], check=True)


def main() -> None:
    for trainer_name in TRAINERS:
        ensure_checkout(trainer_name)


if __name__ == "__main__":
    main()
