"""Every authored suite file carries a lifecycle status; exactly one bundle is active."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest
import yaml

from cle.harness.communication import load_communication_suite
from cle.harness.shared_suite import default_shared_suite_path, load_shared_prompt_suite
from cle.harness.suite import load_context_suite

SUITES_DIR = Path("cle/harness/suites")


def _status_by_file() -> dict[str, str]:
    result = {}
    for path in sorted(SUITES_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert "status" in data, f"{path.name} is missing a status field"
        result[path.name] = data["status"]
    return result


def test_exactly_one_active_suite_and_it_is_the_default():
    statuses = _status_by_file()
    active = [name for name, status in statuses.items() if status == "active"]
    assert active == [default_shared_suite_path().name]
    assert set(statuses.values()) <= {"active", "legacy", "deprecated"}


def test_legacy_pair_defaults_are_marked_legacy_not_deprecated():
    statuses = _status_by_file()
    assert statuses["catan_v11.yaml"] == "legacy"
    assert statuses["communication_v5.yaml"] == "legacy"


def test_deprecated_files_carry_a_banner_comment():
    for name, status in _status_by_file().items():
        first_line = (SUITES_DIR / name).read_text(encoding="utf-8").splitlines()[0]
        if status == "deprecated":
            assert first_line.startswith("# DEPRECATED:"), name
        elif status == "legacy":
            assert first_line.startswith("# LEGACY:"), name


def test_loading_deprecated_suite_warns_and_active_does_not():
    with pytest.warns(DeprecationWarning, match="catan_v4.yaml"):
        load_context_suite(SUITES_DIR / "catan_v4.yaml")
    with pytest.warns(DeprecationWarning, match="communication_v1.yaml"):
        load_communication_suite(SUITES_DIR / "communication_v1.yaml")
    with pytest.warns(DeprecationWarning, match="shared_rl_v1.yaml"):
        load_shared_prompt_suite(SUITES_DIR / "shared_rl_v1.yaml")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert load_shared_prompt_suite().status == "active"
        assert load_context_suite().status == "legacy"
        assert load_communication_suite().status == "legacy"
