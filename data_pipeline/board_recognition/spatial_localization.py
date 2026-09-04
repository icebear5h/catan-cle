"""Build the two-stage empty-board spatial-localization curriculum.

Stage 1 overlays shuffled A/B/C/D markers on nearby atlas locations and emits
both marker-to-token and token-to-marker questions with an exact normalized
spatial target.  Stage 2 removes the markers, expands the canonical spatial
relation bank with yes/no balanced per entity and relationship, and retains a
deterministic 25% replay slice from stage 1.

Train rows in both stages are written in a deterministic shuffled order so the
trainer's sequential sampler does not see one board per batch. Validation,
test, and probe rows keep their canonical order. Probes are emitted at two dot
sizes: a sub-patch dot and a marker-sized dot.

The generated JSONL is native TRL prompt/completion data.  It intentionally
keeps ``curriculum_stage=spatial_grounding`` for compatibility with the older
four-stage corpus while adding the narrower ``grounding_stage`` field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont

from data_pipeline.board_recognition.replay_dataset import (
    DEFAULT_STYLE_PATH,
    file_sha256,
    load_render_style,
    read_jsonl,
    validate_replay_v1_dataset,
)
from data_pipeline.board_recognition.spatial_robber import spatial_query_bank
from evals.catan_board_bench.annotations import (
    EDGE_BOX_PAD,
    NODE_BOX_SIZE,
    PORT_SHIP_SIZE,
    PORT_SHIP_X_OFFSET,
    PORT_SHIP_Y_OFFSET,
    TILE_HEIGHT,
    TILE_WIDTH,
    _bbox_to_pixels,
    _hex_to_pixel,
    _node_positions,
    _pixel_transform,
    _rect,
    contract_to_render_state,
)
from evals.catan_board_bench.render import _view_box_with_padding
from evals.catan_board_bench.tokens import atlas_metadata, canonical_edge


JsonDict = dict[str, Any]
EXPORT_SCHEMA = "catan_spatial_localization/v1"
ROW_SCHEMA = "catan_spatial_localization_row/v1"
DEFAULT_OUTPUT_NAME = "spatial_localization_v1"
MARKERS = ("A", "B", "C", "D")
ENTITY_ORDER = ("node", "edge", "tile", "port")
TRAIN_RELATION_REPETITIONS = 8
MARKER_REPLAY_FRACTION = 0.25

PROMPT_PREFIXES = (
    "",
    "Answer briefly. ",
    "Using the board, ",
    "On this Catan board, ",
    "Check the atlas: ",
    "Read the pictured board. ",
    "Use the fixed board orientation. ",
    "Give only the requested answer. ",
)

PROBE_DOT_SCALES = {
    "heldout_gray_dot_small": 0.012,
    "heldout_gray_dot_large": 0.023,
}

MARKER_STYLES = {
    "train_circle": ("circle", (255, 224, 72), (15, 20, 28)),
    "train_square": ("square", (86, 216, 255), (15, 20, 28)),
    "validation_diamond": ("diamond", (255, 132, 207), (15, 20, 28)),
    "test_triangle": ("triangle", (172, 255, 120), (15, 20, 28)),
}


class SpatialLocalizationError(RuntimeError):
    """Raised when the curriculum cannot satisfy its deterministic contract."""


def _stable_rank(*parts: Any) -> int:
    payload = ":".join(str(part) for part in parts)
    return int(hashlib.sha256(payload.encode()).hexdigest()[:16], 16)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: Iterable[JsonDict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _token_kind(token: str) -> str:
    return {"N": "node", "E": "edge", "T": "tile", "P": "port"}[token[1]]


def _atlas_tokens_by_kind() -> dict[str, list[str]]:
    atlas = atlas_metadata()
    return {
        kind: sorted(row["token"] for row in atlas[f"{kind}s"])
        for kind in ENTITY_ORDER
    }


def atlas_regions(
    contract: JsonDict,
    *,
    image_size: int,
    view_padding_factor: float,
) -> dict[str, JsonDict]:
    """Return renderer-aligned normalized regions for all 154 atlas tokens."""

    render_state = contract_to_render_state(contract)
    view_box = _view_box_with_padding(render_state, view_padding_factor)
    transform = _pixel_transform(view_box, image_size)
    node_positions = _node_positions(render_state)
    regions: dict[str, JsonDict] = {}

    def add(token: str, kind: str, bbox_svg: JsonDict, point_svg: JsonDict) -> None:
        bbox = _bbox_to_pixels(bbox_svg, transform)
        center = [
            int(round((point_svg["x"] - transform["min_x"]) * transform["scale"] + transform["offset_x"])),
            int(round((point_svg["y"] - transform["min_y"]) * transform["scale"] + transform["offset_y"])),
        ]
        normalized_bbox = [max(0.0, min(1.0, value / image_size)) for value in bbox]
        normalized_center = [max(0.0, min(1.0, value / image_size)) for value in center]
        regions[token] = {
            "token": token,
            "entity_type": kind,
            "bbox": normalized_bbox,
            "center": normalized_center,
            "bbox_pixels": bbox,
            "center_pixels": center,
        }

    for placed in render_state.get("tiles", []):
        tile = placed["tile"]
        center = _hex_to_pixel(placed["coordinate"])
        if tile["type"] == "PORT":
            add(
                f"<P{tile['id']:02d}>",
                "port",
                _rect(
                    center["x"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_X_OFFSET,
                    center["y"] - PORT_SHIP_SIZE / 2 + PORT_SHIP_Y_OFFSET,
                    PORT_SHIP_SIZE,
                    PORT_SHIP_SIZE,
                ),
                center,
            )
        else:
            add(
                f"<T{tile['id']:02d}>",
                "tile",
                _rect(
                    center["x"] - TILE_WIDTH / 2,
                    center["y"] - TILE_HEIGHT / 2,
                    TILE_WIDTH,
                    TILE_HEIGHT,
                ),
                center,
            )

    for node_id, point in node_positions.items():
        add(
            f"<N{node_id:02d}>",
            "node",
            _rect(
                point["x"] - NODE_BOX_SIZE / 2,
                point["y"] - NODE_BOX_SIZE / 2,
                NODE_BOX_SIZE,
                NODE_BOX_SIZE,
            ),
            point,
        )

    for edge in contract["edges"]:
        left, right = canonical_edge(tuple(edge["id"]))
        p1, p2 = node_positions[left], node_positions[right]
        point = {"x": (p1["x"] + p2["x"]) / 2, "y": (p1["y"] + p2["y"]) / 2}
        add(
            f"<E{left:02d}_{right:02d}>",
            "edge",
            {
                "x1": min(p1["x"], p2["x"]) - EDGE_BOX_PAD,
                "y1": min(p1["y"], p2["y"]) - EDGE_BOX_PAD,
                "x2": max(p1["x"], p2["x"]) + EDGE_BOX_PAD,
                "y2": max(p1["y"], p2["y"]) + EDGE_BOX_PAD,
            },
            point,
        )

    expected = _atlas_tokens_by_kind()
    expected_tokens = {token for values in expected.values() for token in values}
    if set(regions) != expected_tokens:
        missing = sorted(expected_tokens - set(regions))
        extra = sorted(set(regions) - expected_tokens)
        raise SpatialLocalizationError(f"atlas region mismatch: missing={missing} extra={extra}")
    return regions


def nearby_marker_groups(regions: dict[str, JsonDict], *, group_size: int = 4) -> list[list[str]]:
    """Partition each entity type into deterministic spatially local groups."""

    if not 2 <= group_size <= len(MARKERS):
        raise ValueError(f"group_size must be between 2 and {len(MARKERS)}")
    groups: list[list[str]] = []
    by_kind = _atlas_tokens_by_kind()
    for kind in ENTITY_ORDER:
        remaining = set(by_kind[kind])
        group_count = math.ceil(len(remaining) / group_size)
        base_size, larger_groups = divmod(len(remaining), group_count)
        group_sizes = [base_size + (index < larger_groups) for index in range(group_count)]
        for current_group_size in group_sizes:
            anchor = min(remaining)
            ax, ay = regions[anchor]["center"]
            nearest = sorted(
                remaining,
                key=lambda token: (
                    (regions[token]["center"][0] - ax) ** 2
                    + (regions[token]["center"][1] - ay) ** 2,
                    token,
                ),
            )[:current_group_size]
            groups.append(nearest)
            remaining.difference_update(nearest)
        if remaining:
            raise SpatialLocalizationError(f"failed to group every {kind} token")
    return groups


def _font(radius: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, int(radius * 1.25)))
    except OSError:
        try:
            return ImageFont.load_default(size=max(18, int(radius * 1.25)))
        except TypeError:
            return ImageFont.load_default()


def entity_marker_polygon(geometry: JsonDict, radius: int) -> list[tuple[float, float]] | None:
    """Polygon for an entity-shaped marker, or None to fall back to the style glyph.

    Edges get a bar along the edge at its true angle, so the marker stage asks
    the model to ground a thin slanted shape the way a road is drawn; tiles
    get a hexagon at tile scale. Nodes keep the style glyph.
    """

    kind = geometry.get("kind")
    if kind == "edge" and geometry.get("endpoints"):
        (x1, y1), (x2, y2) = geometry["endpoints"]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return None
        ux, uy = dx / length, dy / length
        nx, ny = -uy, ux
        half = radius * 0.55
        inset = length * 0.12
        ax, ay = x1 + ux * inset, y1 + uy * inset
        bx, by = x2 - ux * inset, y2 - uy * inset
        return [(ax + nx * half, ay + ny * half), (bx + nx * half, by + ny * half), (bx - nx * half, by - ny * half), (ax - nx * half, ay - ny * half)]
    if kind == "tile":
        cx, cy = geometry["center"]
        size = radius * 1.8
        return [(cx + size * math.cos(math.radians(60 * index + 30)), cy + size * math.sin(math.radians(60 * index + 30))) for index in range(6)]
    return None


def render_markers(
    base_image: Image.Image,
    assignments: Sequence[tuple[str, tuple[int, int]] | tuple[str, tuple[int, int], JsonDict]],
    *,
    style_name: str,
    entity_shaped: bool = False,
) -> Image.Image:
    """Overlay readable markers without changing the underlying board layout.

    With ``entity_shaped`` an assignment may carry a third element describing
    the entity (``{"kind": "edge", "endpoints": [...]}`` or ``{"kind": "tile",
    "center": ...}``); those markers take the entity's shape and the letter
    sits at the centre as before.
    """

    if style_name not in MARKER_STYLES:
        raise ValueError(f"unknown marker style: {style_name}")
    shape, fill, text_fill = MARKER_STYLES[style_name]
    image = base_image.convert("RGB").copy()
    draw = ImageDraw.Draw(image)
    radius = max(16, round(min(image.size) * 0.023))
    font = _font(radius)
    outline = (255, 255, 255)
    for assignment in assignments:
        marker, (cx, cy) = assignment[0], assignment[1]
        geometry = assignment[2] if len(assignment) > 2 else {}
        if marker not in MARKERS:
            raise ValueError(f"unknown marker: {marker}")
        entity_polygon = entity_marker_polygon(geometry, radius) if entity_shaped else None
        if entity_polygon is not None:
            draw.polygon(entity_polygon, fill=fill, outline=outline)
        elif shape == "circle":
            points: Any = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.ellipse(points, fill=fill, outline=outline, width=3)
        elif shape == "square":
            points = [cx - radius, cy - radius, cx + radius, cy + radius]
            draw.rounded_rectangle(points, radius=4, fill=fill, outline=outline, width=3)
        elif shape == "diamond":
            points = [(cx, cy - radius), (cx + radius, cy), (cx, cy + radius), (cx - radius, cy)]
            draw.polygon(points, fill=fill, outline=outline)
        else:
            points = [(cx, cy - radius), (cx + radius, cy + radius), (cx - radius, cy + radius)]
            draw.polygon(points, fill=fill, outline=outline)
        box = draw.textbbox((0, 0), marker, font=font)
        width, height = box[2] - box[0], box[3] - box[1]
        draw.text((cx - width / 2, cy - height / 2 - box[1]), marker, font=font, fill=text_fill)
    return image


def _marker_style(split: str, board_index: int) -> str:
    if split == "train":
        return "train_circle" if board_index % 2 == 0 else "train_square"
    if split == "validation":
        return "validation_diamond"
    return "test_triangle"


def _control_regions(regions: dict[str, JsonDict]) -> dict[str, JsonDict]:
    controls: dict[str, JsonDict] = {}
    for tokens in _atlas_tokens_by_kind().values():
        for index, token in enumerate(tokens):
            controls[token] = regions[tokens[(index + 1) % len(tokens)]]
    return controls


def _spatial_target(region: JsonDict, control: JsonDict) -> JsonDict:
    return {
        "token": region["token"],
        "entity_type": region["entity_type"],
        "bbox": region["bbox"],
        "center": region["center"],
        "control_bbox": control["bbox"],
        "control_token": control["token"],
    }


def _training_row(
    *,
    row_id: str,
    image_name: str,
    prompt: str,
    answer: str,
    grounding_stage: str,
    task_type: str,
    metadata: JsonDict,
    spatial_target: JsonDict | None = None,
) -> JsonDict:
    row = {
        "schema": ROW_SCHEMA,
        "row_id": row_id,
        "curriculum_stage": "spatial_grounding",
        "grounding_stage": grounding_stage,
        "task_family": "spatial_localization",
        "task_type": task_type,
        "images": [image_name],
        "messages": [
            {"role": "user", "content": f"<image>\n{prompt}"},
            {"role": "assistant", "content": answer},
        ],
        **metadata,
    }
    if spatial_target is not None:
        row["spatial_targets"] = [spatial_target]
    return row


def _marker_geometry(token: str, regions: dict[str, JsonDict], contract: JsonDict) -> JsonDict:
    kind = regions[token]["entity_type"]
    if kind == "edge":
        edge = next(entry for entry in contract["edges"] if entry["token"] == token)
        return {"kind": "edge", "endpoints": [tuple(regions[node]["center_pixels"]) for node in edge["node_tokens"]]}
    return {"kind": kind, "center": tuple(regions[token]["center_pixels"])}


def marker_rows_for_board(
    *,
    state: JsonDict,
    contract: JsonDict,
    base_image: Image.Image,
    output_images: Path,
    board_index: int,
    view_padding_factor: float,
    entity_shaped: bool = False,
) -> list[JsonDict]:
    """Render one marker set and return both QA directions for all 154 tokens."""

    split = state["split"]
    image_size = int(state["image_size"][0])
    regions = atlas_regions(
        contract,
        image_size=image_size,
        view_padding_factor=view_padding_factor,
    )
    controls = _control_regions(regions)
    rows: list[JsonDict] = []
    style = _marker_style(split, board_index)
    for group_index, group in enumerate(nearby_marker_groups(regions)):
        offset = _stable_rank(state["sample_id"], group_index, "markers") % len(MARKERS)
        marker_order = MARKERS[offset:] + MARKERS[:offset]
        token_to_marker = {token: marker_order[index] for index, token in enumerate(group)}
        image_name = f"{split}_{state['sample_id']}_markers_{group_index:02d}.png"
        assignments = [
            (token_to_marker[token], tuple(regions[token]["center_pixels"]), _marker_geometry(token, regions, contract))
            for token in group
        ]
        marked = render_markers(base_image, assignments, style_name=style, entity_shaped=entity_shaped)
        marked.save(output_images / image_name)

        for token in group:
            marker = token_to_marker[token]
            target = _spatial_target(regions[token], controls[token])
            common = {
                "split": split,
                "state_id": state["sample_id"],
                "entity_type": regions[token]["entity_type"],
                "target_token": token,
                "marker": marker,
                "marker_style": style,
                "marker_shape": "entity" if entity_shaped else "glyph",
                "marker_group": group,
            }
            rows.append(
                _training_row(
                    row_id=f"{state['sample_id']}_g{group_index:02d}_{token[1:-1]}_marker_to_token",
                    image_name=image_name,
                    prompt=f"Which atlas token is marked {marker}? Answer with one token.",
                    answer=token,
                    grounding_stage="marked_localization",
                    task_type="marker_to_token",
                    metadata=common,
                    spatial_target=target,
                )
            )
            rows.append(
                _training_row(
                    row_id=f"{state['sample_id']}_g{group_index:02d}_{token[1:-1]}_token_to_marker",
                    image_name=image_name,
                    prompt=f"Which marker identifies {token}? Answer with one letter.",
                    answer=marker,
                    grounding_stage="marked_localization",
                    task_type="token_to_marker",
                    metadata=common,
                    spatial_target=target,
                )
            )
    if len(rows) != 154 * 2:
        raise SpatialLocalizationError(f"expected 308 marker rows, received {len(rows)}")
    return rows


def _relation_row(
    fact: JsonDict,
    fact_index: int,
    repetition: int,
    empty_states: Sequence[JsonDict],
) -> JsonDict:
    state = empty_states[_stable_rank(fact_index, repetition, "state") % len(empty_states)]
    prefix = PROMPT_PREFIXES[repetition % len(PROMPT_PREFIXES)]
    return _training_row(
        row_id=f"relation_{fact_index:04d}_r{repetition}",
        image_name=f"unmarked_{state['sample_id']}.png",
        prompt=prefix + fact["prompt"],
        answer=fact["answer"],
        grounding_stage="unmarked_orientation",
        task_type=fact_index_type(fact),
        metadata={
            "split": state["split"],
            "state_id": state["sample_id"],
            "entity_type": _token_kind(fact["tokens"][0]),
            "relationship": fact["relationship"],
            "polarity": fact["polarity"],
            "tokens": fact["tokens"],
        },
    )


def _balancing_rows(
    facts: Sequence[JsonDict],
    empty_states: Sequence[JsonDict],
    *,
    repetitions: int,
) -> list[JsonDict]:
    """Top up the minority yes/no polarity per entity and relationship.

    The canonical bank holds more hard negatives than positives for adjacency
    and connectivity, so an unbalanced stage teaches "no" as a prior. Extra
    repetitions continue the repetition index, which keeps row ids unique and
    cycles the prompt prefixes and boards exactly like the base rows.
    """

    grouped: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(
        lambda: {"positive": [], "hard_negative": []}
    )
    for fact_index, fact in enumerate(facts):
        if fact["polarity"] in ("positive", "hard_negative"):
            key = (_token_kind(fact["tokens"][0]), fact["relationship"])
            grouped[key][fact["polarity"]].append(fact_index)
    extra: list[JsonDict] = []
    for key in sorted(grouped):
        positive = grouped[key]["positive"]
        negative = grouped[key]["hard_negative"]
        if not positive or not negative or len(positive) == len(negative):
            continue
        minority, majority = (
            (positive, negative) if len(positive) < len(negative) else (negative, positive)
        )
        deficit = (len(majority) - len(minority)) * repetitions
        for offset in range(deficit):
            fact_index = minority[offset % len(minority)]
            repetition = repetitions + offset // len(minority)
            extra.append(_relation_row(facts[fact_index], fact_index, repetition, empty_states))
    return extra


def _relation_rows(
    empty_states: Sequence[JsonDict],
    *,
    repetitions: int,
    balance_polarity: bool = False,
) -> list[JsonDict]:
    bank = spatial_query_bank()
    facts = [row for family in sorted(bank) for row in bank[family]]
    rows: list[JsonDict] = []
    for fact_index, fact in enumerate(facts):
        for repetition in range(repetitions):
            rows.append(_relation_row(fact, fact_index, repetition, empty_states))
    if balance_polarity:
        rows.extend(_balancing_rows(facts, empty_states, repetitions=repetitions))
    return rows


def fact_index_type(fact: JsonDict) -> str:
    entity = _token_kind(fact["tokens"][0])
    relation = fact["relationship"]
    if fact["polarity"] == "token_return":
        return f"{entity}_direction_token"
    if relation in {"above", "below", "left_of", "right_of"}:
        return f"{entity}_direction_{'yes' if fact['polarity'] == 'positive' else 'no'}"
    return f"{entity}_{relation}_{'yes' if fact['polarity'] == 'positive' else 'no'}"


def _copy_unmarked_images(
    dataset_root: Path,
    output_images: Path,
    states: Sequence[JsonDict],
) -> None:
    for state in states:
        source = dataset_root / state["image_path"]
        destination = output_images / f"unmarked_{state['sample_id']}.png"
        if not destination.exists():
            shutil.copy2(source, destination)


def neutral_probe_rows(
    *,
    state: JsonDict,
    contract: JsonDict,
    base_image: Image.Image,
    output_images: Path,
    view_padding_factor: float,
) -> list[JsonDict]:
    """Create held-out gray-dot probes for every node and edge location.

    Each location is probed at two dot sizes: a sub-patch dot and a dot the size
    of the training markers, so a probe failure separates marker-style transfer
    from resolution.
    """

    image_size = int(state["image_size"][0])
    regions = atlas_regions(
        contract,
        image_size=image_size,
        view_padding_factor=view_padding_factor,
    )
    controls = _control_regions(regions)
    rows = []
    tokens = [
        token
        for kind in ("node", "edge")
        for token in _atlas_tokens_by_kind()[kind]
    ]
    for probe_style, scale in PROBE_DOT_SCALES.items():
        for token in tokens:
            region = regions[token]
            cx, cy = region["center_pixels"]
            radius = max(8, round(image_size * scale))
            image = base_image.convert("RGB").copy()
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(196, 196, 196),
                outline=(25, 25, 25),
                width=max(2, radius // 4),
            )
            size = probe_style.rsplit("_", 1)[-1]
            image_name = f"probe_{state['split']}_{state['sample_id']}_{token[1:-1]}_{size}.png"
            image.save(output_images / image_name)
            rows.append(
                _training_row(
                    row_id=f"probe_{state['sample_id']}_{token[1:-1]}_{size}",
                    image_name=image_name,
                    prompt=(
                        f"Which {region['entity_type']} location contains the gray dot? "
                        "Answer with one atlas token."
                    ),
                    answer=token,
                    grounding_stage="marked_localization",
                    task_type="neutral_probe_token_return",
                    metadata={
                        "split": state["split"],
                        "state_id": state["sample_id"],
                        "entity_type": region["entity_type"],
                        "target_token": token,
                        "probe_style": probe_style,
                        "probe_dot_radius_px": radius,
                        "eval_variant": "original",
                    },
                    spatial_target=_spatial_target(region, controls[token]),
                )
            )
    if len(rows) != 126 * len(PROBE_DOT_SCALES):
        raise SpatialLocalizationError(
            f"expected {126 * len(PROBE_DOT_SCALES)} node/edge probes, received {len(rows)}"
        )
    return rows


def _weighted_marker_train_rows(rows: Sequence[JsonDict]) -> list[JsonDict]:
    weighted: list[JsonDict] = []
    for row in rows:
        weighted.append(dict(row, sampling_repeat=0))
        if row["entity_type"] in {"node", "edge"}:
            weighted.append(dict(row, sampling_repeat=1))
    return weighted


def _deterministic_shuffle(rows: Sequence[JsonDict], salt: str) -> list[JsonDict]:
    """Return a seeded permutation so sequential batches mix boards and tasks."""

    def key(row: JsonDict) -> tuple[int, str, int]:
        repeat = int(row.get("sampling_repeat", 0))
        return (
            _stable_rank(row["row_id"], repeat, row.get("replay_source", ""), salt),
            row["row_id"],
            repeat,
        )

    return sorted(rows, key=key)


def _deterministic_replay(rows: Sequence[JsonDict], count: int) -> list[JsonDict]:
    ordered = sorted(rows, key=lambda row: (_stable_rank(row["row_id"], "replay"), row["row_id"]))
    if count > len(ordered):
        raise SpatialLocalizationError("marker replay request exceeds available unique rows")
    return [dict(row, grounding_stage="unmarked_orientation", replay_source="marked_localization") for row in ordered[:count]]


def _summarize_rows(rows: Sequence[JsonDict]) -> JsonDict:
    dimensions = {}
    for key in ("grounding_stage", "task_type", "entity_type", "relationship", "polarity"):
        dimensions[key] = dict(sorted(Counter(str(row.get(key, "unknown")) for row in rows).items()))
    return {"rows": len(rows), "dimensions": dimensions}


def export_spatial_localization_curriculum(
    dataset_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    style_path: str | Path = DEFAULT_STYLE_PATH,
    overwrite: bool = False,
    entity_markers: bool = False,
) -> JsonDict:
    """Export marked localization, unmarked orientation, and held-out probes."""

    dataset_root = Path(dataset_dir).resolve()
    output = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (dataset_root / DEFAULT_OUTPUT_NAME).resolve()
    )
    validate_replay_v1_dataset(dataset_root, rerender=False)
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"output already exists: {output}")
        if output.parent != dataset_root:
            raise SpatialLocalizationError("refusing to overwrite output outside the replay dataset")
        shutil.rmtree(output)
    images_dir = output / "images"
    images_dir.mkdir(parents=True)

    style = load_render_style(Path(style_path))
    states = [row for row in read_jsonl(dataset_root / "manifest.jsonl") if row["density_bin"] == "empty"]
    states_by_split = {
        split: [row for row in states if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    if {split: len(rows) for split, rows in states_by_split.items()} != {
        "train": 43,
        "validation": 5,
        "test": 5,
    }:
        raise SpatialLocalizationError("replay_v1 empty-board split counts changed")
    primary_empty_states = [state for rows in states_by_split.values() for state in rows]
    _copy_unmarked_images(dataset_root, images_dir, primary_empty_states)

    marker_rows: dict[str, list[JsonDict]] = {}
    for split, split_states in states_by_split.items():
        rows: list[JsonDict] = []
        for board_index, state in enumerate(split_states):
            contract_path = dataset_root / state["contract_path"]
            if file_sha256(contract_path) != state["sha256"]["contract"]:
                raise SpatialLocalizationError(f"contract changed: {contract_path}")
            contract = json.loads(contract_path.read_text())
            with Image.open(dataset_root / state["image_path"]) as source:
                base_image = source.convert("RGB").copy()
            rows.extend(
                marker_rows_for_board(
                    state=state,
                    contract=contract,
                    base_image=base_image,
                    output_images=images_dir,
                    board_index=board_index,
                    view_padding_factor=style.view_padding_factor,
                    entity_shaped=entity_markers,
                )
            )
        marker_rows[split] = rows

    stage1 = {
        "train": _deterministic_shuffle(
            _weighted_marker_train_rows(marker_rows["train"]), "stage1"
        ),
        "validation": marker_rows["validation"],
        "test": marker_rows["test"],
    }
    train_relations = _relation_rows(
        states_by_split["train"],
        repetitions=TRAIN_RELATION_REPETITIONS,
        balance_polarity=True,
    )
    replay_count = round(len(train_relations) * MARKER_REPLAY_FRACTION / (1 - MARKER_REPLAY_FRACTION))
    stage2 = {
        "train": _deterministic_shuffle(
            train_relations + _deterministic_replay(marker_rows["train"], replay_count),
            "stage2",
        ),
        "validation": _relation_rows(states_by_split["validation"], repetitions=1),
        "test": _relation_rows(states_by_split["test"], repetitions=1),
    }

    probes = {}
    for split in ("validation", "test"):
        state = states_by_split[split][0]
        contract = json.loads((dataset_root / state["contract_path"]).read_text())
        with Image.open(dataset_root / state["image_path"]) as source:
            base_image = source.convert("RGB").copy()
        probes[split] = neutral_probe_rows(
            state=state,
            contract=contract,
            base_image=base_image,
            output_images=images_dir,
            view_padding_factor=style.view_padding_factor,
        )

    files = {}
    for stage_name, splits in (("stage1", stage1), ("stage2", stage2)):
        for split, rows in splits.items():
            path = output / stage_name / f"{split}.jsonl"
            _write_jsonl(path, rows)
            files[f"{stage_name}/{split}.jsonl"] = {
                **_summarize_rows(rows),
                "sha256": file_sha256(path),
            }
    for split, rows in probes.items():
        path = output / "probes" / f"{split}.jsonl"
        _write_jsonl(path, rows)
        files[f"probes/{split}.jsonl"] = {
            **_summarize_rows(rows),
            "sha256": file_sha256(path),
        }

    metadata = {
        "schema": EXPORT_SCHEMA,
        "source_dataset": str(dataset_root),
        "source_manifest_sha256": file_sha256(dataset_root / "manifest.jsonl"),
        "style_sha256": file_sha256(Path(style_path)),
        "atlas_counts": {kind: len(tokens) for kind, tokens in _atlas_tokens_by_kind().items()},
        "marker_groups_per_board": sum(
            math.ceil(len(tokens) / len(MARKERS)) for tokens in _atlas_tokens_by_kind().values()
        ),
        "node_edge_sampling_multiplier": 2,
        "stage2_marker_replay_fraction": MARKER_REPLAY_FRACTION,
        "stage2_polarity_balanced": True,
        "train_row_order": "deterministic_shuffle",
        "probe_dot_scales": PROBE_DOT_SCALES,
        "files": files,
    }
    _write_json(output / "metadata.json", metadata)
    return metadata


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--style-path", type=Path, default=DEFAULT_STYLE_PATH)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--entity-markers",
        action="store_true",
        help="Draw edge markers as bars along the edge and tile markers at tile scale instead of one glyph for every entity.",
    )
    args = parser.parse_args(argv)
    result = export_spatial_localization_curriculum(
        args.dataset_dir,
        output_dir=args.output_dir,
        style_path=args.style_path,
        overwrite=args.overwrite,
        entity_markers=args.entity_markers,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
