"""Build, validate, and persist the deterministic replay-source lock."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from data_pipeline.board_recognition.source_lock._audit import audit_replay_file
from data_pipeline.board_recognition.source_lock._config import (
    DEFAULT_LEAKAGE_LEDGER,
    DEFAULT_REPLAY_DIR,
    LEGACY_LEAKAGE_LEDGER_PATHS,
    SOURCE_LOCK_SCHEMA,
    ReplayLoader,
    ReplaySourceAuditError,
    ReplayStepper,
)
from data_pipeline.board_recognition.source_lock._contracts import load_leakage_ledger
from data_pipeline.board_recognition.source_lock._digests import (
    canonical_sha256,
    file_sha256,
    normalize_game_id,
    repository_relative,
)
from data_pipeline.json_coerce import as_dict, as_int, as_list, as_str
from data_pipeline.json_types import JsonDict
from evals.catan_board_bench.builder import load_colonist_replay, step_replay


def build_replay_source_lock(
    *,
    replay_dir: Path = DEFAULT_REPLAY_DIR,
    leakage_ledger: Path = DEFAULT_LEAKAGE_LEDGER,
    minimum_accepted: int = 40,
    loader: ReplayLoader = load_colonist_replay,
    stepper: ReplayStepper = step_replay,
) -> JsonDict:
    """Audit every raw payload and build a deterministic replay-source lock."""

    if minimum_accepted < 1:
        raise ValueError("minimum_accepted must be positive")
    if not replay_dir.is_dir():
        raise FileNotFoundError(replay_dir)
    replay_paths = sorted(replay_dir.glob("*.json"), key=lambda path: normalize_game_id(path))
    if not replay_paths:
        raise ReplaySourceAuditError(f"no replay payloads found in {replay_dir}")

    ledger, benchmark_ids = load_leakage_ledger(leakage_ledger)
    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    excluded_benchmark: list[JsonDict] = []
    seen_game_ids: set[str] = set()
    for path in replay_paths:
        game_id = normalize_game_id(path)
        if game_id in seen_game_ids:
            raise ReplaySourceAuditError(f"duplicate replay game ID: {game_id}")
        seen_game_ids.add(game_id)
        if game_id in benchmark_ids:
            excluded_benchmark.append(
                {
                    "game_id": game_id,
                    "path": repository_relative(path),
                    "sha256": file_sha256(path),
                }
            )
            continue
        evidence = audit_replay_file(path, loader=loader, stepper=stepper)
        if evidence["status"] == "accepted":
            accepted.append(evidence)
        else:
            rejected.append(evidence)

    lock: JsonDict = {
        "schema": SOURCE_LOCK_SCHEMA,
        "source_root": repository_relative(replay_dir),
        "leakage": {
            "benchmark": ledger.get("benchmark"),
            "ledger_path": repository_relative(leakage_ledger),
            "ledger_sha256": file_sha256(leakage_ledger),
            "excluded_game_ids": [game_id for game_id in sorted(benchmark_ids)],
        },
        "minimum_accepted": minimum_accepted,
        "accepted": list(accepted),
        "rejected": list(rejected),
        "excluded_benchmark": list(excluded_benchmark),
        "counts": {
            "raw_payloads": len(replay_paths),
            "accepted": len(accepted),
            "rejected": len(rejected),
            "excluded_benchmark": len(excluded_benchmark),
        },
    }
    identity_payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    lock["lock_sha256"] = canonical_sha256(identity_payload)
    if len(accepted) < minimum_accepted:
        raise ReplaySourceAuditError(
            f"only {len(accepted)} non-benchmark replays passed strict reconstruction; "
            f"require at least {minimum_accepted}"
        )
    return lock


def source_lock_identity_variants(lock: JsonDict) -> tuple[JsonDict, ...]:
    """Return current and declared path-only migration identities for one lock."""

    ledger_paths = (
        str(as_dict(lock["leakage"])["ledger_path"]),
        *LEGACY_LEAKAGE_LEDGER_PATHS,
    )
    variants: list[JsonDict] = []
    for ledger_path in dict.fromkeys(ledger_paths):
        variant = as_dict(json.loads(json.dumps(lock)))
        as_dict(variant["leakage"])["ledger_path"] = ledger_path
        identity_payload = {key: value for key, value in variant.items() if key != "lock_sha256"}
        variant["lock_sha256"] = canonical_sha256(identity_payload)
        serialized = (json.dumps(variant, indent=2, sort_keys=True) + "\n").encode()
        variants.append(
            {
                "ledger_path": ledger_path,
                "lock_sha256": variant["lock_sha256"],
                "file_sha256": hashlib.sha256(serialized).hexdigest(),
            }
        )
    return tuple(variants)


def source_lock_matches_metadata(
    lock: JsonDict,
    *,
    lock_sha256: str,
    file_sha256_value: str,
) -> bool:
    """Match generated metadata across the one declared package-path migration."""

    return any(
        variant["lock_sha256"] == lock_sha256
        and variant["file_sha256"] == file_sha256_value
        for variant in source_lock_identity_variants(lock)
    )


def validate_replay_source_lock(
    lock: JsonDict,
    *,
    minimum_accepted: int | None = None,
) -> JsonDict:
    if lock.get("schema") != SOURCE_LOCK_SCHEMA:
        raise ReplaySourceAuditError(f"unsupported source lock schema: {lock.get('schema')}")
    stored_sha = lock.get("lock_sha256")
    identity_payload = {key: value for key, value in lock.items() if key != "lock_sha256"}
    if stored_sha != canonical_sha256(identity_payload):
        raise ReplaySourceAuditError("replay source lock identity hash mismatch")
    required = minimum_accepted if minimum_accepted is not None else as_int(lock["minimum_accepted"])
    accepted = [as_dict(row) for row in as_list(lock.get("accepted", []))]
    if len(accepted) < required:
        raise ReplaySourceAuditError(
            f"source lock has {len(accepted)} accepted replays; requires {required}"
        )
    if any(row.get("status") != "accepted" for row in accepted):
        raise ReplaySourceAuditError("accepted source lock entries contain a rejected row")
    accepted_ids = [as_str(row["game_id"]) for row in accepted]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ReplaySourceAuditError("accepted source lock game IDs are not unique")
    benchmark_ids = set(as_list(as_dict(lock["leakage"])["excluded_game_ids"]))
    leaked = sorted(set(accepted_ids) & benchmark_ids)
    if leaked:
        raise ReplaySourceAuditError(f"benchmark games entered the source lock: {leaked}")
    return {
        "valid": True,
        "accepted": len(accepted),
        "rejected": len(as_list(lock.get("rejected", []))),
        "excluded_benchmark": len(as_list(lock.get("excluded_benchmark", []))),
        "lock_sha256": stored_sha,
    }


def write_replay_source_lock(path: Path, lock: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


__all__ = [
    "build_replay_source_lock",
    "source_lock_identity_variants",
    "source_lock_matches_metadata",
    "validate_replay_source_lock",
    "write_replay_source_lock",
]
