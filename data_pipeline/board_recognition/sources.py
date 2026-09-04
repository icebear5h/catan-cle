"""Fail-closed source locking for replay-backed board-recognition data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

from evals.catan_board_bench.builder import (
    CatanObservationSuite,
    load_colonist_replay,
    step_replay,
)
from evals.catan_board_bench.paths import DATASETS_DIR


JsonDict = dict[str, Any]
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPLAY_DIR = PROJECT_ROOT / "artifacts" / "raw" / "colonist" / "replays"
DEFAULT_LEAKAGE_LEDGER = (
    DATASETS_DIR / "catan_board_bench_100" / "leakage" / "benchmark_game_ids.json"
)
DEFAULT_SOURCE_LOCK = (
    PROJECT_ROOT / "data" / "curriculum" / "board_recognition" / "replay_sources_v1.json"
)
SOURCE_LOCK_SCHEMA = "catan_board_recognition_replay_sources/v1"
LEGACY_LEAKAGE_LEDGER_PATHS = (
    "data_pipeline/catan_board_bench/datasets/catan_board_bench_100/"
    "leakage/benchmark_game_ids.json",
)

ReplayLoader = Callable[..., Any]
ReplayStepper = Callable[..., JsonDict]

NONVISUAL_INFO_KINDS = {"replayed_trade_closure", "observed_replay_state"}
NONVISUAL_TRADE_ACTIONS = {
    "OFFER_TRADE",
    "COUNTER_OFFER",
    "ACCEPT_TRADE",
    "REJECT_TRADE",
    "CLEAR_TRADE_RESPONSE",
    "CONFIRM_TRADE",
    "CLOSE_TRADE",
}
FINAL_SCORE_FIELDS = {
    "public_vp",
    "actual_vp",
    "has_largest_army",
    "has_longest_road",
    "longest_road_length",
}


class ReplaySourceAuditError(RuntimeError):
    """Raised when the replay source corpus cannot satisfy fail-closed gates."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def repository_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def normalize_game_id(path: Path) -> str:
    return path.stem.removesuffix("_sample")


def load_leakage_ledger(path: Path = DEFAULT_LEAKAGE_LEDGER) -> tuple[JsonDict, set[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"required benchmark leakage ledger is missing: {path}")
    payload = json.loads(path.read_text())
    game_ids = payload.get("benchmark_game_ids")
    if not isinstance(game_ids, list) or not game_ids:
        raise ReplaySourceAuditError("benchmark leakage ledger has no game IDs")
    return payload, {str(game_id) for game_id in game_ids}


def visible_board_facts(contract: JsonDict) -> JsonDict:
    """Return only facts that can alter the rendered public board image."""

    return {
        "tiles": [
            {
                "id": row["id"],
                "resource": row["resource"],
                "number": row["number"],
                "robber": bool(row["has_robber"]),
            }
            for row in contract["tiles"]
        ],
        "nodes": [
            {
                "id": row["id"],
                "color": row["color"],
                "building": row["building"],
            }
            for row in contract["nodes"]
        ],
        "edges": [
            {
                "id": row["id"],
                "owner": row["road_color"],
            }
            for row in contract["edges"]
        ],
        "ports": [
            {
                "id": row["id"],
                "kind": row["kind"],
                "resource": row["resource"],
            }
            for row in contract["ports"]
        ],
    }


def validate_public_board_contract(contract: JsonDict) -> None:
    expected_counts = {"tiles": 19, "nodes": 54, "edges": 72, "ports": 9}
    actual_counts = {name: len(contract.get(name, [])) for name in expected_counts}
    if actual_counts != expected_counts:
        raise ReplaySourceAuditError(
            f"public board topology mismatch: expected={expected_counts} actual={actual_counts}"
        )
    if sum(bool(tile.get("has_robber")) for tile in contract["tiles"]) != 1:
        raise ReplaySourceAuditError("public board contract must contain exactly one robber")
    colors = [player.get("color") for player in contract.get("players", [])]
    if len(colors) != 4 or len(set(colors)) != 4:
        raise ReplaySourceAuditError(f"expected four distinct replay colors, received {colors}")


def _safe_json(value: Any) -> Any:
    return json.loads(json.dumps(value, default=str, sort_keys=True))


def diagnostic_is_board_safe(issue: JsonDict) -> bool:
    """Return whether one replay diagnostic is proven not to affect pixels."""

    severity = str(issue.get("severity", "error"))
    kind = str(issue.get("kind", ""))
    action_type = str(issue.get("action_type", ""))
    if severity == "info" and kind in NONVISUAL_INFO_KINDS:
        return True
    if (
        severity == "warning"
        and kind == "forced_replay_overlay"
        and action_type in NONVISUAL_TRADE_ACTIONS
    ):
        return True
    if severity == "warning" and kind == "forced_final_state_sync":
        field = str((issue.get("details") or {}).get("field", ""))
        return field in FINAL_SCORE_FIELDS
    return False


def summarize_diagnostics(issues: list[JsonDict]) -> JsonDict:
    counts: dict[str, int] = {}
    for issue in issues:
        key = ":".join(
            (
                str(issue.get("severity", "unknown")),
                str(issue.get("kind", "unknown")),
                str(issue.get("action_type", "none")),
            )
        )
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def audit_replay_file(
    path: Path,
    *,
    loader: ReplayLoader = load_colonist_replay,
    stepper: ReplayStepper = step_replay,
) -> JsonDict:
    """Reconstruct one replay sequentially and return deterministic evidence."""

    source_path = path.resolve()
    game_id = normalize_game_id(source_path)
    evidence: JsonDict = {
        "game_id": game_id,
        "path": repository_relative(source_path),
        "sha256": file_sha256(source_path),
    }
    try:
        state = loader(source_path, quiet=True)
        parsed_actions = state.replay_data.get("parsed_actions", [])
        total_steps = len(parsed_actions)
        suite = CatanObservationSuite()
        seen_board_hashes: set[str] = set()

        initial_contract = suite.public_board_contract(
            state.current_game,
            sample={"id": f"source_audit_{game_id}_000000", "index": 0},
            source={"kind": "colonist_replay", "game_id": game_id, "replay_step": 0},
        )
        validate_public_board_contract(initial_contract)
        seen_board_hashes.add(canonical_sha256(visible_board_facts(initial_contract)))

        calls = 0
        while state.replay_index < total_steps:
            before = state.replay_index
            result = stepper(state, quiet=True)
            calls += 1
            if isinstance(result, tuple):
                raise ReplaySourceAuditError(f"replay step returned HTTP-style error: {result}")
            if not isinstance(result, dict):
                raise ReplaySourceAuditError(
                    f"replay step returned unsupported result {type(result)!r}"
                )
            if result.get("error"):
                raise ReplaySourceAuditError(f"replay step failed: {result['error']}")
            if state.replay_index <= before and not result.get("finished"):
                raise ReplaySourceAuditError(
                    f"replay made no progress at source step {before}/{total_steps}"
                )

            contract = suite.public_board_contract(
                state.current_game,
                sample={
                    "id": f"source_audit_{game_id}_{state.replay_index:06d}",
                    "index": state.replay_index,
                },
                source={
                    "kind": "colonist_replay",
                    "game_id": game_id,
                    "replay_step": state.replay_index,
                },
            )
            validate_public_board_contract(contract)
            seen_board_hashes.add(canonical_sha256(visible_board_facts(contract)))
            if calls > total_steps + 8:
                raise ReplaySourceAuditError("replay exceeded the sequential step bound")

        issues = [_safe_json(issue) for issue in getattr(state, "replay_semantic_issues", [])]
        blocking_issues = [issue for issue in issues if not diagnostic_is_board_safe(issue)]
        if blocking_issues:
            evidence.update(
                {
                    "status": "rejected",
                    "reason": "board_unsafe_replay_diagnostics",
                    "diagnostic_count": len(issues),
                    "diagnostic_counts": summarize_diagnostics(issues),
                    "blocking_diagnostic_count": len(blocking_issues),
                    "blocking_diagnostics": blocking_issues,
                    "parsed_action_count": total_steps,
                    "completed_replay_steps": state.replay_index,
                    "unique_board_states": len(seen_board_hashes),
                }
            )
            return evidence

        if state.replay_index != total_steps:
            raise ReplaySourceAuditError(
                f"replay stopped at {state.replay_index}/{total_steps} parsed actions"
            )

        evidence.update(
            {
                "status": "accepted",
                "parsed_action_count": total_steps,
                "completed_replay_steps": state.replay_index,
                "engine_action_count": len(state.current_game.state.actions),
                "unique_board_states": len(seen_board_hashes),
                "diagnostic_count": len(issues),
                "diagnostic_counts": summarize_diagnostics(issues),
                "blocking_diagnostic_count": 0,
            }
        )
        return evidence
    except Exception as exc:
        evidence.update(
            {
                "status": "rejected",
                "reason": "reconstruction_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )
        return evidence


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
    accepted = []
    rejected = []
    excluded_benchmark = []
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
            "excluded_game_ids": sorted(benchmark_ids),
        },
        "minimum_accepted": minimum_accepted,
        "accepted": accepted,
        "rejected": rejected,
        "excluded_benchmark": excluded_benchmark,
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
        str(lock["leakage"]["ledger_path"]),
        *LEGACY_LEAKAGE_LEDGER_PATHS,
    )
    variants = []
    for ledger_path in dict.fromkeys(ledger_paths):
        variant = json.loads(json.dumps(lock))
        variant["leakage"]["ledger_path"] = ledger_path
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
    required = minimum_accepted if minimum_accepted is not None else int(lock["minimum_accepted"])
    accepted = lock.get("accepted", [])
    if len(accepted) < required:
        raise ReplaySourceAuditError(
            f"source lock has {len(accepted)} accepted replays; requires {required}"
        )
    if any(row.get("status") != "accepted" for row in accepted):
        raise ReplaySourceAuditError("accepted source lock entries contain a rejected row")
    accepted_ids = [row["game_id"] for row in accepted]
    if len(accepted_ids) != len(set(accepted_ids)):
        raise ReplaySourceAuditError("accepted source lock game IDs are not unique")
    benchmark_ids = set(lock["leakage"]["excluded_game_ids"])
    leaked = sorted(set(accepted_ids) & benchmark_ids)
    if leaked:
        raise ReplaySourceAuditError(f"benchmark games entered the source lock: {leaked}")
    return {
        "valid": True,
        "accepted": len(accepted),
        "rejected": len(lock.get("rejected", [])),
        "excluded_benchmark": len(lock.get("excluded_benchmark", [])),
        "lock_sha256": stored_sha,
    }


def write_replay_source_lock(path: Path, lock: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
