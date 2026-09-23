from flask import Flask
from flask.testing import FlaskClient

from playground.game_viewer.routes.bench import bench_bp


def _client() -> FlaskClient:
    app = Flask(__name__)
    app.register_blueprint(bench_bp)
    return app.test_client()


def test_sft_data_route_exposes_active_bidirectional_training_projection() -> None:
    response = _client().get(
        "/api/catan-board-bench/sft-data",
        query_string={"split": "train", "limit": 2},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 12_288
    summary = payload["summary"]
    assert summary["direction_counts"] == {"forward": 8_192, "inverse": 4_096}
    assert summary["direction_ratio"] == "2:1"
    assert summary["row_count"] == 12_288
    assert summary["rows_per_state"] == 12
    assert summary["source_schema"] == "catan_board_recognition_ms_swift_bidirectional/v1"
    assert len(summary["source_sha256"]) == 64
    assert summary["state_count"] == 1_024
    assert summary["target_coverage"] == 154
    assert summary["trainable_tokens"] == 154
    assert len(payload["rows"]) == 2
    assert payload["rows"][0]["row_kind"] == "forward"
    assert payload["rows"][0]["prompt"].startswith("<image>\n<T")
    assert payload["rows"][0]["image_url"].startswith(
        "/api/catan-board-bench/sft-image/"
    )
    readiness = payload["launch_readiness"]
    assert readiness["status"] == "blocked"
    assert readiness["labeled_rows"] == 4_104
    assert readiness["directional_rows"] == 1_032
    assert readiness["robber_rows"] == 3_072
    assert readiness["topology_rows"] == 390
    assert readiness["missing_stages"] == []
    assert [stage["rows"] for stage in readiness["stages"]] == [
        1_032,
        3_885,
        3_405,
        8_070,
    ]
    assert readiness["projection"] == {
        "effective_batch_size": 8,
        "optimizer_steps": 1_536,
        "smoke_inclusive_hours": 14.9,
        "steady_state_hours": 14.9,
    }
    smoke = readiness["smoke"]
    assert smoke["run_id"] == "ap-xy0bhh0Mhoc2kj3M8NoyII"
    assert smoke["loss"] == 6.0777270793914795
    assert [row["stage"] for row in smoke["stage_metrics"]] == [
        "spatial_grounding",
        "clean_board_grounding",
        "pieces_and_colors",
        "real_game_distribution",
    ]
    assert smoke["reload_validation"]["valid"] is True


def test_sft_data_route_filters_inverse_grounding_rows_and_serves_images() -> None:
    client = _client()
    response = client.get(
        "/api/catan-board-bench/sft-data",
        query_string={
            "split": "train",
            "row_kind": "inverse",
            "entity_type": "node",
            "query": "WO11/B4/WO8",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1
    row = payload["rows"][0]
    assert row["answer"] == "<N00>"
    assert row["description_style"] == "node_compact_resource_number"
    assert row["target_token"] == "<N00>"

    image = client.get(row["image_url"])
    assert image.status_code == 200
    assert image.content_type == "image/png"
    assert client.get("/api/catan-board-bench/sft-image/../metadata.json").status_code in {
        400,
        404,
    }


def test_sft_data_route_rejects_unknown_split() -> None:
    response = _client().get(
        "/api/catan-board-bench/sft-data",
        query_string={"split": "dev"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Unknown SFT split: dev"


def test_spatial_localization_route_exposes_stage1_patch_targets() -> None:
    client = _client()
    response = client.get(
        "/api/catan-board-bench/spatial-localization-data",
        query_string={"stage": "stage1", "split": "train", "limit": 2},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["summary"]["row_count"] == 24_080
    assert payload["summary"]["atlas_tokens"] == 154
    assert payload["summary"]["marker_groups_per_board"] == 40
    assert payload["distributions"]["task_type"] == {
        "marker_to_token": 12_040,
        "token_to_marker": 12_040,
    }
    assert [stage["id"] for stage in payload["stages"]] == [
        "stage1",
        "stage2",
        "probes",
    ]
    row = payload["rows"][0]
    assert row["spatial_target"]["token"] == row["target_token"]
    assert len(row["spatial_target"]["bbox"]) == 4
    image = client.get(row["image_url"])
    assert image.status_code == 200
    assert image.content_type == "image/png"


def test_spatial_localization_route_filters_stage2_hard_negatives() -> None:
    response = _client().get(
        "/api/catan-board-bench/spatial-localization-data",
        query_string={
            "stage": "stage2",
            "split": "validation",
            "relationship": "left_of",
            "polarity": "hard_negative",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] > 0
    assert all(row["relationship"] == "left_of" for row in payload["rows"])
    assert all(row["polarity"] == "hard_negative" for row in payload["rows"])


def test_spatial_localization_probes_do_not_accept_train_split() -> None:
    response = _client().get(
        "/api/catan-board-bench/spatial-localization-data",
        query_string={"stage": "probes", "split": "train"},
    )

    assert response.status_code == 400
    assert "unavailable" in response.get_json()["error"]


def test_piece_viewer_exposes_reweighted_training_and_color_diagnostic() -> None:
    client = _client()
    for split, expected in (("train", 17_708), ("color_diagnostic", 8_192)):
        response = client.get(
            "/api/catan-board-bench/spatial-localization-data",
            query_string={"dataset": "node_edge_readout_reweighted_v1", "split": split, "limit": 1},
        )
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["total"] == expected
        assert len(payload["summary"]["source_sha256"]) == 64
        row = payload["rows"][0]
        assert row["queried_token"] in row["prompt"]
        assert row["piece"]
        assert client.get(row["image_url"]).content_type == "image/png"


def test_v3_viewer_keeps_queried_location_distinct_from_piece_location() -> None:
    client = _client()
    response = client.get(
        "/api/catan-board-bench/spatial-localization-data",
        query_string={"dataset": "spatial_localization_v3", "polarity": "hard_negative", "limit": 1},
    )
    assert response.status_code == 200
    row = response.get_json()["rows"][0]
    assert row["queried_token"] != row["target_token"]
    assert row["queried_token"] in row["prompt"]
    image = client.get(row["image_url"])
    assert image.status_code == 200
    assert image.content_type == "image/png"


def test_piece_viewer_rejects_unknown_datasets_and_image_traversal() -> None:
    client = _client()
    assert client.get(
        "/api/catan-board-bench/spatial-localization-data?dataset=../../.."
    ).status_code == 400
    assert client.get(
        "/api/catan-board-bench/spatial-localization-image/test.png?dataset=../../.."
    ).status_code == 400
    assert client.get(
        "/api/catan-board-bench/spatial-localization-image/../metadata.json?dataset=node_edge_readout_reweighted_v1"
    ).status_code in {400, 404}


def test_sft_eval_route_exposes_final_checkpoint_results() -> None:
    response = _client().get(
        "/api/catan-board-bench/sft-eval",
        query_string={"limit": 2},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] == 1_080
    assert len(payload["rows"]) == 2
    assert payload["summary"]["correct"] == 436
    assert payload["summary"]["attempted"] == 1_080
    assert payload["summary"]["exact_accuracy"] == 436 / 1_080
    assert payload["summary"]["by_category"]["inverse.port"]["correct"] == 63
    assert payload["summary"]["spatial_token_return"] == {
        "correct": 0,
        "exact_accuracy": 0.0,
        "total": 20,
    }
    assert payload["checkpoint"]["name"] == "catan-qwen3.8-27b-spatial-sft"
    assert payload["rows"][0]["image_url"].startswith(
        "/api/catan-board-bench/sft-image/"
    )


def test_sft_eval_route_filters_exact_failures() -> None:
    response = _client().get(
        "/api/catan-board-bench/sft-eval",
        query_string={
            "category": "inverse.edge",
            "correctness": "incorrect",
            "density_bin": "dense",
        },
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["total"] > 0
    assert all(row["category"] == "inverse.edge" for row in payload["rows"])
    assert all(row["density_bin"] == "dense" for row in payload["rows"])
    assert all(row["correct"] is False for row in payload["rows"])


def test_sft_eval_route_rejects_unknown_correctness_filter() -> None:
    response = _client().get(
        "/api/catan-board-bench/sft-eval",
        query_string={"correctness": "maybe"},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Unknown correctness filter: maybe"
