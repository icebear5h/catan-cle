"""Browsing the CatanBoardBench examples, samples, and annotations."""

from __future__ import annotations

from flask import Response, jsonify, request, send_file

from evals.catan_board_bench.annotations import (
    annotation_payload_for_contract,
    contract_to_render_state,
)

from .blueprint import bench_bp
from .common import (
    _int_arg,
)
from .paths import (
    BENCH_DIR,
)
from .qa import (
    _compact_qa,
    _contract_context,
    _load_contract,
    _qa_rows,
    _question_detail,
    _sample_context,
    _sample_ids,
)


@bench_bp.route("/api/catan-board-bench/examples", methods=["GET"])
def list_catan_board_bench_examples() -> Response | tuple[Response, int]:
    """Return compact QA rows for picker controls."""

    category = request.args.get("category")
    sample_id = request.args.get("sample_id")
    limit = _int_arg("limit", 100)
    offset = _int_arg("offset", 0)

    rows = [
        _compact_qa(qa)
        for qa in _qa_rows()
        if (not category or qa["category"] == category)
        and (not sample_id or qa["sample_id"] == sample_id)
    ]
    categories = sorted({str(qa["category"]) for qa in _qa_rows()})
    samples = sorted({str(qa["sample_id"]) for qa in rows})
    sample_counts = {sample: 0 for sample in samples}
    for row in rows:
        sample_counts[str(row["sample_id"])] += 1

    return jsonify(
        {
            "examples": rows[offset : offset + limit],
            "total": len(rows),
            "offset": offset,
            "limit": limit,
            "categories": categories,
            "samples": samples,
            "sample_counts": sample_counts,
        }
    )


@bench_bp.route("/api/catan-board-bench/example", methods=["GET"])
def get_catan_board_bench_example() -> Response | tuple[Response, int]:
    """Return one QA row with its engine contract context."""

    qa_id = request.args.get("id")
    index = _int_arg("index", 0)
    rows = _qa_rows()

    if qa_id:
        qa = next((row for row in rows if row["id"] == qa_id), None)
        if qa is None:
            return jsonify({"error": f"Unknown CatanBoardBench QA id: {qa_id}"}), 404
    else:
        if index < 0 or index >= len(rows):
            return jsonify({"error": f"Index must be in [0, {len(rows) - 1}]"}), 400
        qa = rows[index]

    contract = _load_contract(qa["contract_path"])

    return jsonify(
        {
            **qa,
            "image_url": f"/api/catan-board-bench/image/{qa['image_path']}",
            "annotations_url": f"/api/catan-board-bench/annotations?id={qa['id']}",
            "contract_context": _contract_context(contract, qa),
            "render_state": contract_to_render_state(contract),
            "total": len(rows),
        }
    )


@bench_bp.route("/api/catan-board-bench/sample", methods=["GET"])
def get_catan_board_bench_sample() -> Response | tuple[Response, int]:
    """Return one benchmark image/contract with every QA row for that sample."""

    sample_id = request.args.get("sample_id")
    index = _int_arg("index", 0)
    rows = _qa_rows()
    sample_ids = _sample_ids(rows)

    if sample_id:
        if sample_id not in sample_ids:
            return jsonify({"error": f"Unknown CatanBoardBench sample id: {sample_id}"}), 404
    else:
        if index < 0 or index >= len(sample_ids):
            return jsonify({"error": f"Index must be in [0, {len(sample_ids) - 1}]"}), 400
        sample_id = sample_ids[index]

    sample_rows = [row for row in rows if row["sample_id"] == sample_id]
    if not sample_rows:
        return jsonify({"error": f"No QA rows for CatanBoardBench sample id: {sample_id}"}), 404

    first = sample_rows[0]
    contract = _load_contract(first["contract_path"])
    return jsonify(
        {
            "sample_id": sample_id,
            "sample_index": sample_ids.index(sample_id),
            "sample_count": len(sample_ids),
            "image_path": first["image_path"],
            "image_url": f"/api/catan-board-bench/image/{first['image_path']}",
            "contract_path": first["contract_path"],
            "annotations_url": f"/api/catan-board-bench/annotations?id={first['id']}",
            "render_state": contract_to_render_state(contract),
            "contract_context": _sample_context(contract),
            "questions": [_question_detail(contract, row) for row in sample_rows],
        }
    )


@bench_bp.route("/api/catan-board-bench/annotations", methods=["GET"])
def get_catan_board_bench_annotations() -> Response | tuple[Response, int]:
    """Return frontend-aligned bbox/point annotations for one QA row."""

    qa_id = request.args.get("id")
    index = _int_arg("index", 0)
    image_size = _int_arg("size", 512)
    rows = _qa_rows()

    if qa_id:
        qa = next((row for row in rows if row["id"] == qa_id), None)
        if qa is None:
            return jsonify({"error": f"Unknown CatanBoardBench QA id: {qa_id}"}), 404
    else:
        if index < 0 or index >= len(rows):
            return jsonify({"error": f"Index must be in [0, {len(rows) - 1}]"}), 400
        qa = rows[index]

    contract = _load_contract(qa["contract_path"])
    payload = annotation_payload_for_contract(contract, image_size=image_size)
    return jsonify(
        {
            "qa_id": qa["id"],
            "sample_id": qa["sample_id"],
            "category": qa["category"],
            "question": qa["question"],
            "answer": qa["answer"],
            **payload,
        }
    )


@bench_bp.route("/api/catan-board-bench/image/<path:image_path>", methods=["GET"])
def get_catan_board_bench_image(image_path: str) -> Response | tuple[Response, int]:
    """Serve benchmark board images from the generated CatanBoardBench directory."""

    path = (BENCH_DIR / image_path).resolve()
    try:
        path.relative_to(BENCH_DIR.resolve())
    except ValueError:
        return jsonify({"error": "Image path is outside the benchmark directory"}), 400
    if not path.exists():
        return jsonify({"error": f"Image not found: {image_path}"}), 404
    return send_file(path)
