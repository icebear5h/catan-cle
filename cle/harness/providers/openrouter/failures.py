"""Safe public summaries for OpenRouter TLS and HTTP transport failures."""

from __future__ import annotations

import html
import json
import re
import ssl
from collections.abc import Sequence
from urllib.parse import quote, quote_plus, unquote

import httpx

from cle.harness.models import ModelRequest


def _classify_tls_error(exc: BaseException) -> str | None:
    """Inspect the active exception chain; any other SSL error fails closed."""
    classification = None
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, ssl.SSLError):
            reason = getattr(current, "reason", None)
            if isinstance(current, ssl.SSLCertVerificationError) or not (
                reason == "SSLV3_ALERT_BAD_RECORD_MAC"
                or (
                    reason is None
                    and "[SSL: SSLV3_ALERT_BAD_RECORD_MAC]" in str(current)
                )
            ):
                return "other_tls"
            classification = "SSLV3_ALERT_BAD_RECORD_MAC"
        # Explicit causes supersede incidental context, just as in a traceback.
        current = current.__cause__ or (
            current.__context__ if not current.__suppress_context__ else None
        )
    return classification


class OpenRouterTLSFailure(RuntimeError):
    """Safe public summary; the original exception is retained only as a cause."""

    def __init__(
        self, request: ModelRequest, *, model: str, attempts: int
    ) -> None:
        self.context_id = request.decision_id
        self.session_id = request.session_id
        self.model = model
        self.attempts = attempts
        super().__init__(
            "OpenRouter TLS connection failed (SSLV3_ALERT_BAD_RECORD_MAC) "
            f"after {attempts} transport attempt(s)."
        )


class OpenRouterHTTPFailure(RuntimeError):
    """Bounded provider diagnostics, not a general-purpose free-text privacy filter."""

    def __init__(
        self,
        request: ModelRequest,
        *,
        model: str,
        attempts: int,
        response: httpx.Response,
        redactions: tuple[str, ...] = (),
    ) -> None:
        self.context_id = request.decision_id
        self.session_id = request.session_id
        self.model = model
        self.attempts = attempts
        self.status_code = response.status_code
        self.provider_message: str | None = None
        self.provider_request_id: str | None = None

        secrets = set(redactions)
        credential_headers: Sequence[tuple[str, str]]
        try:
            credential_headers = response.request.headers.multi_items()
        except RuntimeError:  # Direct callers may supply a response without a request.
            credential_headers = ()
        for name, value in credential_headers:
            if re.search(r"auth|key|token|secret|cookie|credential", name, re.I):
                secrets.add(value)
                secrets.update(part.strip("\"'") for part in re.split(r"[=;,\s]+", value))
        echoes = {message.content for message in request.messages}
        if request.board_presentation is not None and request.board_presentation.kind == "text":
            echoes.add(request.board_presentation.content)
        echoes.update(line.strip() for text in tuple(echoes) for line in text.splitlines())

        def sanitize(value: object, *, limit: int, identifier: bool = False) -> str | None:
            if not isinstance(value, str) or not value.strip() or len(value) > 2048:
                return None
            if identifier and (
                len(value) > limit or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:-]*", value)
            ):
                return None
            # Check echoes before replacing secrets; never truncate before redaction.
            for texts, is_echo in ((echoes, True), (secrets, False)):
                for text in sorted(texts, key=len, reverse=True):
                    if not text or len(text) > len(value):
                        continue
                    variants = {text, html.escape(text), quote(text, safe=""), quote_plus(text)}
                    for ascii_only in (False, True):
                        escaped = json.dumps(text, ensure_ascii=ascii_only)[1:-1]
                        variants.update((escaped, escaped.replace("/", "\\/")))
                        variants.add(json.dumps(escaped)[1:-1])
                    for variant in sorted(variants, key=len, reverse=True):
                        if variant in value:
                            if is_echo or identifier:
                                return None
                            value = value.replace(variant, "[REDACTED]")
            decoded = value
            for _ in range(2):
                decoded = unquote(html.unescape(decoded))
            decoded = " ".join(decoded.split())
            if any(text and text in decoded for text in secrets | echoes):
                return None
            # Suppress obvious payloads, including escaped ones, rather than trying
            # to salvage arbitrary raw/metadata/flagged-input strings as diagnostics.
            if re.search(
                r"[{}<>]|\[\s*[\[\"']|data:|\\(?:u[0-9a-f]{4}|[nrtbf])|%[0-9a-f]{2}|"
                r"\b(?:messages|content|prompt|input|flagged_input|request|body|payload|headers|"
                r"authorization|cookie|api[_-]?key|metadata|raw)"
                r"[\s\"'\\]*[:=]",
                decoded,
                re.I,
            ):
                return None
            value = " ".join(value.split())
            if not value or not value.isprintable():
                return None
            return value if len(value) <= limit else value[:limit - 3] + "..."

        # httpx has already buffered the response. Limit parsing, not the wire read.
        try:
            data = response.json() if len(response.content) <= 16 * 1024 else None
        except (ValueError, RecursionError, httpx.ResponseNotRead):
            data = None
        if isinstance(data, dict) and isinstance(data.get("error"), dict):
            self.provider_message = sanitize(data["error"].get("message"), limit=512)
        for name in ("x-request-id", "x-openrouter-request-id"):
            self.provider_request_id = sanitize(
                response.headers.get(name), limit=128, identifier=True
            )
            if self.provider_request_id is not None:
                break
        summary = (
            f"OpenRouter HTTP{self.status_code} failed after {attempts} transport attempt(s): "
            + (self.provider_message or "no usable structured error.message reason returned.")
        )
        if self.provider_request_id is not None:
            summary += f" (request ID: {self.provider_request_id})"
        super().__init__(summary)