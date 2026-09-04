"""Canonical local paths and portable dataset-asset resolution for SFT."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts"
GENERATED_SFT_ROOT = ARTIFACTS_ROOT / "generated" / "sft"
SFT_FIXTURES_ROOT = ARTIFACTS_ROOT / "fixtures" / "sft"
SFT_DIAGNOSTICS_ROOT = ARTIFACTS_ROOT / "diagnostics" / "sft"
SFT_RUNS_ROOT = ARTIFACTS_ROOT / "runs" / "sft"
SFT_CONFIG_ROOT = PROJECT_ROOT / "configs" / "sft"
RENDERER_STYLE_CONFIG = SFT_CONFIG_ROOT / "renderer_style.json"
RENDER_CONTRACT_FIXTURE_DIR = SFT_FIXTURES_ROOT / "render_contracts"


def resolve_dataset_asset(dataset_path: Path, asset: str | Path) -> Path:
    """Resolve an image or other asset referenced by a dataset row.

    Relative paths are interpreted against the dataset file first, then against
    the repository root. This supports self-contained datasets (`images/x.png`)
    and repository-relative references (`artifacts/fixtures/...`) without
    embedding one developer's absolute checkout path.
    """

    asset_path = Path(asset).expanduser()
    if asset_path.is_absolute():
        return asset_path.resolve()

    dataset_candidate = (dataset_path.resolve().parent / asset_path).resolve()
    if dataset_candidate.exists():
        return dataset_candidate

    repository_candidate = (PROJECT_ROOT / asset_path).resolve()
    if repository_candidate.exists():
        return repository_candidate

    return dataset_candidate


def resolve_dataset_image(image_root: Path, asset: str | Path) -> Path:
    """Resolve one relative image strictly under an explicit image root."""

    root = image_root.expanduser().resolve()
    asset_path = Path(asset)
    if asset_path.is_absolute():
        raise ValueError(f"dataset image reference must be relative to image_root: {asset}")
    candidate = (root / asset_path).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"dataset image reference escapes image_root: {asset}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def repository_relative_path(path: Path) -> str:
    """Return a portable repository-relative path when possible."""

    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(resolved)
