from __future__ import annotations

from pathlib import Path
import sys

import typer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.storage import find_final_safetensors


def export_final_weight(run_dir: Path) -> Path:
    return find_final_safetensors(run_dir)


def main(run_dir: Path) -> None:
    print(export_final_weight(run_dir))


if __name__ == "__main__":
    typer.run(main)
