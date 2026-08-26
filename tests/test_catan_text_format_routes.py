from __future__ import annotations

from flask import Flask

from playground.game_viewer.routes.bench import bench_bp


def _client():
    app = Flask(__name__)
    app.register_blueprint(bench_bp)
    return app.test_client()


def test_text_format_v3_overview_reports_frozen_scores_and_question_lock() -> None:
    response = _client().get("/api/catan-board-bench/text-format-v3")

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["format_id"] == "indexed_tile_rows_v3"
    assert payload["board_count"] == 12
    assert payload["question_count"] == 60
    assert payload["question_lock"]["byte_identical_to_source"] is True
    assert payload["development"]["baseline"]["exact"] == 40
    assert payload["development"]["winner"]["exact"] == 60
    assert payload["development"]["winner"]["requests"] == 60
    assert payload["transfer"]["paired_questions"] == 46
    assert payload["transfer"]["baseline"]["exact"] == 34
    assert payload["transfer"]["winner"]["exact"] == 46


def test_text_format_v3_sample_returns_exact_serialization_and_questions() -> None:
    response = _client().get(
        "/api/catan-board-bench/text-format-v3/sample",
        query_string={"sample_id": "ascii_board_00"},
    )

    assert response.status_code == 200
    payload = response.get_json()
    assert payload["sample_id"] == "ascii_board_00"
    assert payload["format"] == "indexed_tile_rows"
    assert payload["version"] == "v3"
    assert "ASCII VARIANT=tile_rows" in payload["text"]
    assert "QI|TILE_NEIGHBORS|" in payload["text"]
    assert "QI|ROLL_SOURCE|" in payload["text"]
    assert payload["query_index_counts"]["TILE_NEIGHBORS"] == 19
    assert len(payload["questions"]) == 6
    assert payload["questions"][0]["question"] == "Which tile is directly LEFT of T06?"


def test_text_format_v3_sample_rejects_unknown_id() -> None:
    response = _client().get(
        "/api/catan-board-bench/text-format-v3/sample",
        query_string={"sample_id": "../../etc/passwd"},
    )

    assert response.status_code == 404
    assert "Unknown text-format v3 sample id" in response.get_json()["error"]
