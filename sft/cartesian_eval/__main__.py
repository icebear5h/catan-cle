"""Build the immutable panel: python -m sft.cartesian_eval [--source PATH] [--output PATH]."""

import argparse
import json
from pathlib import Path

from .dataset import DEFAULT_OUTPUT, DEFAULT_SOURCE, build_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_dataset(args.output, source=args.source), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
