"""Location and strict loading of the versioned decision-bucket suite."""

from __future__ import annotations

from pathlib import Path

import yaml

from evals.decision_buckets.models import DecisionBucketSuite


def default_bucket_suite_path() -> Path:
    return Path(__file__).resolve().parents[1] / "suites" / "decision_spot_checks_v1.yaml"


def load_decision_bucket_suite(path: str | Path | None = None) -> DecisionBucketSuite:
    suite_path = Path(path) if path is not None else default_bucket_suite_path()
    with suite_path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Decision bucket suite {suite_path} must contain a YAML mapping")
    return DecisionBucketSuite.model_validate(data)
