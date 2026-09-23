"""Keep correctness probes away from live services and persisted game data."""
import socket
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import unquote, urlsplit

import pytest

from cle.harness.communication import default_communication_suite_path
from cle.harness.suite import default_suite_path


@pytest.fixture(autouse=True)
def isolated_audit_io(
    monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[None]:
    violations: list[str] = []
    temporary_root = tmp_path_factory.getbasetemp().resolve()
    connect_sqlite = sqlite3.connect

    def no_network(*args: object, **kwargs: object) -> None:
        violations.append("network access")
        raise RuntimeError("Correctness audits must not contact live services")

    def temporary_sqlite(
        database: str | Path, *args: object, **kwargs: object
    ) -> sqlite3.Connection:
        name = str(database)
        if name != ":memory:":
            path = unquote(urlsplit(name).path) if name.startswith("file:") else name
            if not Path(path).resolve().is_relative_to(temporary_root):
                violations.append(f"non-test SQLite database: {name}")
                raise RuntimeError("Correctness audits may open only temporary databases")
        return connect_sqlite(database, *args, **kwargs)

    for method in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(socket.socket, method, no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(sqlite3, "connect", temporary_sqlite)
    monkeypatch.setenv("CATAN_CONTEXT_SUITE", str(default_suite_path()))
    monkeypatch.setenv("CATAN_COMMUNICATION_SUITE", str(default_communication_suite_path()))
    yield
    if violations:
        pytest.fail(f"Audit attempted forbidden I/O even if its error was caught: {violations}")
