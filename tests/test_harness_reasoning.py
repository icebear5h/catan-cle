import pytest

from cle.harness.reasoning import (
    native_reasoning_enabled,
    native_reasoning_request,
    native_reasoning_returned,
    reasoning_token_count,
    validate_native_reasoning_request,
)


def test_native_reasoning_defaults_are_explicit_and_retain_evidence():
    assert validate_native_reasoning_request(None) == {
        "effort": "xhigh",
        "exclude": False,
    }
    assert native_reasoning_request("off") == {"enabled": False}
    assert not native_reasoning_enabled({"enabled": False})


def test_native_reasoning_validation_rejects_unobservable_or_ambiguous_requests():
    with pytest.raises(ValueError, match="exclude must be false"):
        validate_native_reasoning_request({"effort": "high", "exclude": True})
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_native_reasoning_request({"effort": "high", "max_tokens": 100})
    with pytest.raises(ValueError, match="cannot also set"):
        validate_native_reasoning_request({"enabled": False, "effort": "low"})


def test_native_reasoning_evidence_accepts_text_details_or_reported_tokens():
    usage = {"completion_tokens_details": {"reasoning_tokens": 12}}

    assert reasoning_token_count(usage) == 12
    assert native_reasoning_returned("trace", (), {})
    assert native_reasoning_returned("", ({"type": "reasoning.text"},), {})
    assert native_reasoning_returned("", (), usage)
    assert not native_reasoning_returned("", (), {})
