from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.bootstrap_demo_dataset import bootstrap_demo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a local demo dataset for smoke testing.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--count", type=int, default=40)
    parser.add_argument("--dataset-id", default=None)
    parser.add_argument("--trigger-word", default="mrblbust")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    kwargs = {"count": args.count, "trigger_word": args.trigger_word, "force": args.force}
    if args.dataset_id is not None:
        kwargs["dataset_id"] = args.dataset_id

    try:
        output_root = bootstrap_demo_dataset(args.output_root, **kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
    print(output_root)


if __name__ == "__main__":
    main()
