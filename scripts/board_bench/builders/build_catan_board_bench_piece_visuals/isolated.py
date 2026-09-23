"""Isolated single-primitive examples composed from frontend assets."""

from __future__ import annotations

from evals.catan_board_bench.tokens import color_token, resource_token
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.constants import (
    COLORS,
    NUMBERS,
    PORTS,
    RESOURCES,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.images import (
    render_node_image,
    render_port_image,
    render_road_image,
    render_robber_image,
    render_tile_image,
)
from scripts.board_bench.builders.build_catan_board_bench_piece_visuals.writer import DatasetWriter

__all__ = [
    "add_node_examples",
    "add_port_examples",
    "add_road_examples",
    "add_robber_examples",
    "add_tile_examples",
]


def add_tile_examples(writer: DatasetWriter, variant: int) -> None:
    for resource in RESOURCES:
        for number in NUMBERS:
            sample_id = f"isolated_tile_{resource.lower()}_{number}_v{variant:02d}"
            image, tile_bbox, number_bbox = render_tile_image(
                resource, number, writer.image_size, variant
            )
            writer.add_sample(
                sample_id=sample_id,
                category="isolated_tile_resource_number",
                image=image,
                question="What resource and dice number are shown on this tile?",
                answer=f"{resource_token(resource)} {number}",
                target={
                    "resource": resource,
                    "resource_token": resource_token(resource),
                    "number": number,
                    "tile_bbox": list(tile_bbox),
                    "number_bbox": list(number_bbox) if number_bbox is not None else None,
                },
            )

    sample_id = f"isolated_tile_desert_v{variant:02d}"
    image, tile_bbox, _ = render_tile_image(None, None, writer.image_size, variant)
    writer.add_sample(
        sample_id=sample_id,
        category="isolated_tile_resource_number",
        image=image,
        question="What resource and dice number are shown on this tile?",
        answer="<DESERT>",
        target={
            "resource": None,
            "resource_token": "<DESERT>",
            "number": None,
            "tile_bbox": list(tile_bbox),
        },
    )


def add_road_examples(writer: DatasetWriter, variant: int) -> None:
    for color in [None, *COLORS]:
        color_slug = "empty" if color is None else color.lower()
        sample_id = f"isolated_road_{color_slug}_v{variant:02d}"
        image, road_bbox, angle = render_road_image(color, writer.image_size, variant)
        writer.add_sample(
            sample_id=sample_id,
            category="isolated_road_owner",
            image=image,
            question="Who owns the road shown?",
            answer="EMPTY" if color is None else color_token(color),
            target={
                "color": color,
                "color_token": color_token(color) if color else None,
                "angle": angle,
                "road_bbox": list(road_bbox) if road_bbox is not None else None,
            },
        )


def add_node_examples(writer: DatasetWriter, variant: int) -> None:
    sample_id = f"isolated_node_empty_v{variant:02d}"
    image, marker_bbox = render_node_image(None, None, writer.image_size, variant)
    writer.add_sample(
        sample_id=sample_id,
        category="isolated_node_occupancy",
        image=image,
        question="What building, if any, is shown on this node?",
        answer="EMPTY",
        target={"color": None, "building": None, "marker_bbox": list(marker_bbox)},
    )
    for color in COLORS:
        for building in ["SETTLEMENT", "CITY"]:
            sample_id = f"isolated_node_{color.lower()}_{building.lower()}_v{variant:02d}"
            image, building_bbox = render_node_image(color, building, writer.image_size, variant)
            writer.add_sample(
                sample_id=sample_id,
                category="isolated_node_occupancy",
                image=image,
                question="What building, if any, is shown on this node?",
                answer=f"{color_token(color)} <{building}>",
                target={
                    "color": color,
                    "color_token": color_token(color),
                    "building": building,
                    "building_token": f"<{building}>",
                    "building_bbox": list(building_bbox),
                },
            )


def add_port_examples(writer: DatasetWriter, variant: int) -> None:
    for resource in PORTS:
        slug = "generic" if resource is None else resource.lower()
        sample_id = f"isolated_port_{slug}_v{variant:02d}"
        image, port_bbox = render_port_image(resource, writer.image_size, variant)
        writer.add_sample(
            sample_id=sample_id,
            category="isolated_port_trade_type",
            image=image,
            question="What trade port is shown?",
            answer="GENERIC 3:1" if resource is None else f"{resource_token(resource)} 2:1",
            target={
                "resource": resource,
                "resource_token": resource_token(resource) if resource else None,
                "ratio": "3:1" if resource is None else "2:1",
                "port_bbox": list(port_bbox),
            },
        )


def add_robber_examples(writer: DatasetWriter, variant: int) -> None:
    for resource in [None, *RESOURCES]:
        slug = "desert" if resource is None else resource.lower()
        for robber in [True, False]:
            sample_id = f"isolated_robber_{'on' if robber else 'absent'}_{slug}_v{variant:02d}"
            image, tile_bbox, robber_bbox = render_robber_image(
                resource, robber, writer.image_size, variant
            )
            writer.add_sample(
                sample_id=sample_id,
                category="isolated_robber_presence",
                image=image,
                question="Is the robber shown on this tile?",
                answer="YES" if robber else "NO",
                target={
                    "resource": resource,
                    "resource_token": resource_token(resource),
                    "robber": robber,
                    "tile_bbox": list(tile_bbox),
                    "robber_bbox": list(robber_bbox) if robber_bbox is not None else None,
                },
            )
