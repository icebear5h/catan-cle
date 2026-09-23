import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sft.scripts.report.inspect_token_rows import (
    INPUT_ROWS_KEY,
    OUTPUT_ROWS_KEY,
    inspect_adapter,
    inspect_rows,
    load_token_rows,
    main,
)

FAMILY_COUNTS = {"N": 54, "E": 72, "T": 19, "P": 9}
HIDDEN = 64
TWIN_SOURCE, TWIN_COPY, ANTI_ALIGNED = 5, 9, 130
# 64-d random rows are far from orthogonal, so the synthetic twin threshold sits well above
# the chance level of same-family pairs while the planted copy still lands near cosine 1.
TWIN_THRESHOLD = 0.8


def synthetic_tokens() -> list[str]:
    tokens = [f"<N{index:02d}>" for index in range(FAMILY_COUNTS["N"])]
    tokens += [f"<E{index:02d}_{index + 1:02d}>" for index in range(FAMILY_COUNTS["E"])]
    tokens += [f"<T{index:02d}>" for index in range(FAMILY_COUNTS["T"])]
    tokens += [f"<P{index:02d}>" for index in range(FAMILY_COUNTS["P"])]
    return tokens


def synthetic_rows(seed: int = 7) -> torch.Tensor:
    """Rows scattered around four family centroids, plus one twin pair and one anti-aligned row."""
    generator = torch.Generator().manual_seed(seed)
    centroids = {
        family: torch.nn.functional.normalize(torch.randn(HIDDEN, generator=generator), dim=0)
        for family in FAMILY_COUNTS
    }
    families = [token[1] for token in synthetic_tokens()]
    base = torch.stack([centroids[family] for family in families])
    noise = torch.randn(len(families), HIDDEN, generator=generator)
    noise = 2.0 * torch.nn.functional.normalize(noise, dim=-1)
    rows = base + noise
    rows[TWIN_COPY] = rows[TWIN_SOURCE] + 0.01 * torch.randn(HIDDEN, generator=generator)
    rows[ANTI_ALIGNED] = -base[ANTI_ALIGNED] + noise[ANTI_ALIGNED]
    return rows


def test_inspect_rows_finds_planted_twins_and_anti_aligned_row() -> None:
    tokens = synthetic_tokens()
    rows = synthetic_rows()
    assert tokens[ANTI_ALIGNED].startswith("<T")

    result = inspect_rows(rows, tokens, twin_threshold=TWIN_THRESHOLD)

    assert result["row_count"] == 154
    assert result["hidden_size"] == HIDDEN
    assert result["zero_norm_tokens"] == []
    ((twin_a, twin_b, twin_cosine),) = result["twins"]
    assert (twin_a, twin_b) == (tokens[TWIN_SOURCE], tokens[TWIN_COPY])
    assert twin_cosine > 0.99
    assert result["tokens_with_twin"] == 2
    assert result["max_pair_tokens"] == [tokens[TWIN_SOURCE], tokens[TWIN_COPY]]
    assert result["max_pair"] == twin_cosine
    # At the default threshold the planted pair is still the strongest twin.
    assert inspect_rows(rows, tokens)["twins"][0][:2] == [tokens[TWIN_SOURCE], tokens[TWIN_COPY]]

    below = result["rows_below_family_floor"]
    assert below[0][0] == tokens[ANTI_ALIGNED]
    assert below[0][1] < 0
    assert [loading for _, loading in below] == sorted(loading for _, loading in below)

    assert set(result["family_centroid_cosines"]) == {"NE", "NT", "NP", "ET", "EP", "TP"}
    assert all(abs(value) < 0.9 for value in result["family_centroid_cosines"].values())
    assert set(result["own_family_median"]) == set(FAMILY_COUNTS)
    assert all(median > 0.3 for median in result["own_family_median"].values())
    assert set(result["own_family_loading"]) == set(tokens)
    assert set(result["mean_direction_loading"]) == set(tokens)
    assert all(
        result["mean_direction_loading"][token] < 0 for token in result["rows_negative_on_mean"]
    )
    assert -0.1 < result["pairwise_cosine_mean"] < 0.2
    assert 0 < result["pairwise_cosine_sd"] < 0.3


def test_inspect_rows_is_deterministic() -> None:
    tokens = synthetic_tokens()
    rows = synthetic_rows()
    assert inspect_rows(rows, tokens) == inspect_rows(rows.clone(), tokens)


def test_inspect_rows_rejects_mismatched_inventory() -> None:
    tokens = synthetic_tokens()
    with pytest.raises(ValueError, match="153 rows"):
        inspect_rows(synthetic_rows()[:153], tokens)
    with pytest.raises(ValueError, match="cannot derive a family"):
        inspect_rows(synthetic_rows(), ["<X00>"] + tokens[1:])


def test_missing_key_error_names_present_token_keys(tmp_path: Path) -> None:
    path = tmp_path / "half.safetensors"
    save_file({INPUT_ROWS_KEY: torch.zeros(3, 4)}, str(path))
    with pytest.raises(ValueError, match="embed_tokens.token_adapter"):
        load_token_rows(path)


def test_cli_writes_json_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    tokens = synthetic_tokens()
    rows = synthetic_rows()
    adapter = tmp_path / "adapter_model.safetensors"
    save_file(
        {INPUT_ROWS_KEY: rows.contiguous(), OUTPUT_ROWS_KEY: rows.flip(0).contiguous()},
        str(adapter),
    )
    inventory = tmp_path / "trainable_tokens.json"
    inventory.write_text(json.dumps({"tokens": tokens}))
    output = tmp_path / "report.json"

    argv = [
        "--adapter",
        f"synthetic={adapter}",
        "--token-inventory",
        str(inventory),
        "--twin-threshold",
        str(TWIN_THRESHOLD),
        "--output",
        str(output),
    ]
    assert main(argv) == 0

    captured = capsys.readouterr()
    assert captured.out == ""
    table_lines = captured.err.strip().splitlines()
    assert len(table_lines) == 3
    assert table_lines[1].startswith("synthetic  input")
    assert table_lines[2].startswith("synthetic  output")

    report = json.loads(output.read_text())
    assert report["token_count"] == 154
    assert report["twin_threshold"] == TWIN_THRESHOLD
    entry = report["adapters"]["synthetic"]
    assert entry["path"] == str(adapter)
    assert entry["input"] == inspect_rows(rows, tokens, twin_threshold=TWIN_THRESHOLD)
    assert entry == {"path": str(adapter), **inspect_adapter(adapter, tokens, twin_threshold=0.8)}
    # The output side was saved reversed, so the twin pair lands on mirrored token indices.
    mirrored = [tokens[153 - TWIN_COPY], tokens[153 - TWIN_SOURCE]]
    assert entry["output"]["max_pair_tokens"] == mirrored
    assert entry["output"]["tokens_with_twin"] == 2

    assert main(argv + ["--quiet"]) == 0
    assert capsys.readouterr().err == ""


def test_cli_rejects_unlabeled_adapter() -> None:
    with pytest.raises(SystemExit):
        main(["--adapter", "/nowhere/adapter_model.safetensors"])
