"""Bidirectional marker questions and held-out gray-dot probes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from data_pipeline.board_recognition import spatial_localization as api
from data_pipeline.board_recognition.replay_dataset import JsonDict
from data_pipeline.json_coerce import as_int, as_list, as_str


def _center_pixels(region: JsonDict) -> tuple[int, int]:
    """The integer (x, y) pixel centre of one atlas region."""

    x, y = (as_int(part) for part in as_list(region["center_pixels"]))
    return x, y


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

    split = as_str(state["split"])
    image_size = as_int(as_list(state["image_size"])[0])
    regions = api.atlas_regions(contract, image_size=image_size, view_padding_factor=view_padding_factor)
    controls = api._control_regions(regions)
    rows: list[JsonDict] = []
    style = api._marker_style(split, board_index)
    for group_index, group in enumerate(api.nearby_marker_groups(regions)):
        offset = api._stable_rank(state["sample_id"], group_index, "markers") % len(api.MARKERS)
        marker_order = api.MARKERS[offset:] + api.MARKERS[:offset]
        token_to_marker = {token: marker_order[index] for index, token in enumerate(group)}
        image_name = f"{split}_{state['sample_id']}_markers_{group_index:02d}.png"
        assignments = [
            (
                token_to_marker[token],
                _center_pixels(regions[token]),
                api._marker_geometry(token, regions, contract),
            )
            for token in group
        ]
        marked = api.render_markers(base_image, assignments, style_name=style, entity_shaped=entity_shaped)
        marked.save(output_images / image_name)

        for token in group:
            marker = token_to_marker[token]
            target = api._spatial_target(regions[token], controls[token])
            common: JsonDict = {
                "split": split,
                "state_id": state["sample_id"],
                "entity_type": regions[token]["entity_type"],
                "target_token": token,
                "marker": marker,
                "marker_style": style,
                "marker_shape": "entity" if entity_shaped else "glyph",
                "marker_group": [token for token in group],
            }
            rows.append(
                api._training_row(
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
                api._training_row(
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
        raise api.SpatialLocalizationError(f"expected 308 marker rows, received {len(rows)}")
    return rows


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

    image_size = as_int(as_list(state["image_size"])[0])
    regions = api.atlas_regions(contract, image_size=image_size, view_padding_factor=view_padding_factor)
    controls = api._control_regions(regions)
    rows = []
    tokens = [token for kind in ("node", "edge") for token in api._atlas_tokens_by_kind()[kind]]
    for probe_style, scale in api.PROBE_DOT_SCALES.items():
        for token in tokens:
            region = regions[token]
            cx, cy = _center_pixels(region)
            radius = max(8, round(image_size * scale))
            image = base_image.convert("RGB").copy()
            draw = ImageDraw.Draw(image)
            draw.ellipse(
                [cx - radius, cy - radius, cx + radius, cy + radius],
                fill=(196, 196, 196), outline=(25, 25, 25), width=max(2, radius // 4),
            )
            size = probe_style.rsplit("_", 1)[-1]
            image_name = f"probe_{state['split']}_{state['sample_id']}_{token[1:-1]}_{size}.png"
            image.save(output_images / image_name)
            rows.append(
                api._training_row(
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
                    spatial_target=api._spatial_target(region, controls[token]),
                )
            )
    if len(rows) != 126 * len(api.PROBE_DOT_SCALES):
        raise api.SpatialLocalizationError(
            f"expected {126 * len(api.PROBE_DOT_SCALES)} node/edge probes, received {len(rows)}"
        )
    return rows
