"""Console entry point for verify_spatial_extension."""

from __future__ import annotations

from . import main

# Guarded so runpy imports under another name (the offline-import test) never run the CLI.
if __name__ == "__main__":
    raise SystemExit(main())
