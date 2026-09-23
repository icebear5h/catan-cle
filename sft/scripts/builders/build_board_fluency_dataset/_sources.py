from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TypeAlias, TypedDict

from data_pipeline.board_recognition.sources import canonical_sha256, file_sha256
from sft.board.board_fluency_scoring import SFT_SCHEMA
from sft.board.symbolic_board_tasks import decode_state
from sft.board.symbolic_board_tasks._types import DecodedState
from sft.json_types import JsonDict, JsonLikeDict, JsonValue, as_dict
from sft.scripts.builders import build_board_fluency_review as build_board_fluency_review
from sft.scripts.builders.build_symbolic_board_dataset._types import SourceRecord

review = build_board_fluency_review

ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "artifacts/generated/sft/symbolic_board_fluency_sft_v1"
REVIEW = review.OUTPUT / "review.jsonl"
REVIEW_SHA256 = "40469f58d060dd9a1d24354df42d959971172ba6e0de5e70b153cdda747e5b12"
INVENTORY = ROOT / "artifacts/generated/board_recognition/replay_v1/ms_swift_bidirectional_v1/trainable_tokens.json"
VERSION = "symbolic_board_fluency_sft_v1"
SCHEMA = SFT_SCHEMA
SEED = 20260915
OPERATIONS = tuple(review.OPERATION_FAMILY)
PIP_OPERATIONS = frozenset({"resource_pip_totals", "resource_pip_argmax"})
SPLITS = ("train", "validation", "test")
COUNTS = dict(train=3200, validation=370, test=370, validation_eval=190)
ATLAS_PATTERN = re.compile(r"<[NETP][0-9_]+>")
require = review.require

# Split profiles and token exposure are aggregate reports; state reuse
# histograms key on integers, so they are plain dicts rather than JSON objects.
Profile: TypeAlias = dict[str, object]
Exposure: TypeAlias = dict[str, object]


class ValidationReport(TypedDict):
    checks: dict[str, int]
    split_disjoint_keys: list[str]
    review_overlap: int
    heldout_graph_coverage: dict[str, JsonLikeDict]
    profiles: dict[str, Profile]
    first_1024: Exposure
    all_train: Exposure


class Summary(TypedDict):
    """The keys shared by build and validate results that `main` prints."""

    valid: bool
    corpus_status: str
    counts: dict[str, int]
    checks: dict[str, int]
    profiles: dict[str, Profile]
    first_1024: Exposure


class BuildSummary(Summary):
    dry_run: bool
    output_dir: str
    eligibility: JsonLikeDict


class ValidateSummary(Summary):
    schema: str
    admitted_for_training: bool
    train_jsonl: str
    manifest_sha256: str
    files: JsonValue
    review_overlap: int


@dataclass
class Donor:
    """The source-backed portion of review.Donor, without fictitious v2 lineage."""

    state: JsonDict
    data: DecodedState
    provenance: JsonDict
    state_hash: str
    terrain_hash: str


def donor_for(record: SourceRecord) -> Donor:
    state = as_dict(record["state"])
    data = decode_state(state)
    return Donor(state, data, record["provenance"], canonical_sha256(state),
                 canonical_sha256(data["tiles"]))


def meta(row: JsonDict) -> JsonDict:
    return as_dict(row["metadata"])


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path.resolve()), "sha256": file_sha256(path)}
