"""Execute upstream train.py in a private one-GPU Ray runtime (evaluation only)."""

from __future__ import annotations

import importlib
import os
import runpy
from pathlib import Path
from typing import Protocol, cast


class RayRuntime(Protocol):
    def init(
        self, *, address: str, num_gpus: int, include_dashboard: bool,
        runtime_env: dict[str, dict[str, str]],
    ) -> object: ...

    def shutdown(self) -> None: ...


RAY = cast("RayRuntime", importlib.import_module("ray"))


def main() -> None:
    root = Path(os.environ["CATAN_MILES_ROOT"])
    RAY.init(address="local", num_gpus=1, include_dashboard=False, runtime_env={
        "env_vars": {"PYTHONPATH": os.environ["PYTHONPATH"], "TOKENIZERS_PARALLELISM": "false"},
    })
    try:
        runpy.run_path(str(root / "train.py"), run_name="__main__")
    finally:
        RAY.shutdown()


if __name__ == "__main__":
    main()
