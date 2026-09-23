"""Run the pinned ms-swift Catan semantic vision-SFT pipeline."""

from __future__ import annotations

from sft.ms_swift_plugin import CatanSwiftSft, assert_ms_swift_version


def main() -> None:
    assert_ms_swift_version()
    CatanSwiftSft().main()


if __name__ == "__main__":
    main()
