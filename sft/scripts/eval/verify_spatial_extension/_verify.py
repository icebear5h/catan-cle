"""verify."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import modal

from sft.json_types import JsonDict, as_dict, json_list, json_path
from sft.launchers.spatial.modal_spatial_continuation import (
    LOCAL_RUN_ROOT,
    PANEL_BUDGETS,
    PROJECT_ROOT,
    read_json,
)
from sft.launchers.spatial.modal_spatial_extension import (
    DEFAULT_RUN_NAME,
)
from sft.scripts.eval import verify_spatial_extension as verify

from ._base import SCORER_FILES
from ._panels import verify_panel, verify_training
from ._receipts import (
    atomic_write,
    completed_post,
    download_receipts,
    json_bytes,
    load_launch,
    local_path,
    require,
    status_report,
    validate_result,
)


def verify_local(run: Path, launch: JsonDict) -> JsonDict:
    result = read_json(local_path(run, "result.json"))
    validate_result(result, launch)
    post = read_json(local_path(run, "post/result.json"))
    completed_post(result, post, launch)
    for relative in SCORER_FILES:
        require(verify.sha256_file(PROJECT_ROOT / relative)
                == json_path(launch, "source_sha256", relative),
                f"scorer source changed: {relative}")
    history = verify_training(run, launch, result, as_dict(post["checkpoint_audit"]))
    panels: JsonDict = {
        label: as_dict(verify_panel(run, launch, post, label)) for label in PANEL_BUDGETS
    }
    report: JsonDict = {"status": "PASS", "stage": "post", "run_name": run.name,
              "checkpoint": post["checkpoint"], "panels": panels,
              "teacher_forced_history": json_list(history),
              "scorer_sha256": {key: json_path(launch, "source_sha256", key) for key in SCORER_FILES},
              "receipts_sha256": {key: verify.sha256_file(local_path(run, key))
                                  for key in ("launch.json", "result.json", "post/result.json")}}
    atomic_write(run, "post-verification.json", json_bytes(report))
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=LOCAL_RUN_ROOT / DEFAULT_RUN_NAME)
    parser.add_argument("--download", action="store_true", help="read receipts from Modal volumes")
    parser.add_argument("--status-only", action="store_true", help="show coordinator receipt; skip verification")
    args = parser.parse_args(argv)
    try:
        run, launch = load_launch(args.run_dir)
        result = (download_receipts(run, launch, status_only=args.status_only) if args.download
                  else read_json(local_path(run, "result.json")))
        validate_result(result, launch)
        if args.status_only or result["status"] != "completed":
            print(json.dumps(status_report(result), sort_keys=True))
            return 1 if result["status"] == "failed" else 0 if args.status_only else 2
        report = verify_local(run, launch)
        counts = " ".join(f"{label}={as_dict(panel)['correct']}/{as_dict(panel)['rows']}"
                          for label, panel in as_dict(report["panels"]).items())
        print(f"PASS checkpoint-256 {counts}\n{run / 'post-verification.json'}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, modal.exception.Error) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
