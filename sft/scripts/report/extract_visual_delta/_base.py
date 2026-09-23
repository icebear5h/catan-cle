"""Shared imports, constants and types for this package."""

from __future__ import annotations

from typing import NotRequired, TypedDict

RANKS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)

THRESHOLDS = (0.5, 0.9, 0.95, 0.99)


class SvdSummary(TypedDict):
    """Spectrum and factor-quality statistics for one decomposed matrix."""

    matrix_shape: list[int]
    maximum_rank: int
    numerical_rank_fp32: int
    singular_values: list[float]
    rank_energy_fraction: dict[str, float]
    energy_threshold_ranks: dict[str, int]
    saved_factor_rank: int
    saved_factor_relative_frobenius_error: float
    svd_energy_relative_error: float
    output_basis_orthogonality_max_abs: float
    input_basis_orthogonality_max_abs: float


class DeltaStats(TypedDict):
    """Dense-delta statistics for one visual tensor."""

    shape: list[int]
    numel: int
    base_storage_dtype: str
    trained_storage_dtype: str
    delta_norm: float
    delta_energy: float
    relative_delta_norm: float | None
    changed_fraction: float
    reconstruction_bit_exact: bool
    reconstruction_max_abs_error: float


class TensorRow(DeltaStats):
    """One report row: delta statistics, provenance and, for matrices, the SVD."""

    name: str
    trained_key: str
    base_key: str
    component: str
    matrix_kind: str | None
    svd: NotRequired[SvdSummary]
    factor_file: NotRequired[str]
    factor_sha256: NotRequired[str]


class GroupSummary(TypedDict):
    tensor_count: int
    numel: int
    matrix_count: int
    delta_norm: float
    matrix_delta_energy: float
    nonmatrix_delta_energy: float
    matrix_energy_weighted_rank_capture: dict[str, float]


class ShardProvenance(TypedDict):
    file: str
    sha256: str
    bytes: int


class DeltaReport(TypedDict):
    """`report.json`; the NotRequired keys are filled in as extraction advances."""

    schema: str
    status: str
    model_id: str
    hf_revision: str
    base_load_dtype: str
    factor_rank: int
    dense_delta_dtype: str
    svd_dtype: str
    base_shards: list[ShardProvenance]
    base_index_sha256: str
    trained_path: str
    trained_sha256: str
    meaning: str
    scope: str
    factor_convention: str
    protection_warning: str
    tensors: list[TensorRow]
    dense_delta_sha256: NotRequired[str]
    source_dtypes: NotRequired[dict[str, int]]
    summary: NotRequired[dict[str, GroupSummary]]
    rank16_linear_sha256: NotRequired[str]
    elapsed_seconds: NotRequired[float]
