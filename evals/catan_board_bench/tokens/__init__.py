"""Catan-specific added vocabulary tokens.

These are intended to be added as regular tokenizer tokens, not chat/control
special tokens. They give the trainable model atomic symbols for the fixed
Catan atlas: tiles, nodes, edges, and ports.

The vocabulary tables, the derived atlas, and the exported manifest live in
sibling modules. Every pre-split name stays importable from this package.
"""

from __future__ import annotations

import json as json
from dataclasses import asdict as asdict
from dataclasses import dataclass as dataclass
from pathlib import Path as Path
from typing import Any as Any
from typing import Dict as Dict
from typing import List as List
from typing import Tuple as Tuple

from cle.game_engine.board_tokens import _check_range as _check_range
from cle.game_engine.board_tokens import canonical_edge as canonical_edge
from cle.game_engine.board_tokens import edge_token as edge_token
from cle.game_engine.board_tokens import node_token as node_token
from cle.game_engine.board_tokens import port_token as port_token
from cle.game_engine.board_tokens import tile_token as tile_token
from cle.game_engine.models.enums import CITY as CITY
from cle.game_engine.models.enums import RESOURCES as RESOURCES
from cle.game_engine.models.enums import ROAD as ROAD
from cle.game_engine.models.enums import SETTLEMENT as SETTLEMENT

# Names the pre-split module also exposed, kept importable at this path.
from cle.game_engine.models.enums import ActionType as ActionType
from cle.game_engine.models.map import BASE_MAP_TEMPLATE as BASE_MAP_TEMPLATE
from cle.game_engine.models.map import NUM_EDGES as NUM_EDGES
from cle.game_engine.models.map import NUM_NODES as NUM_NODES
from cle.game_engine.models.map import NUM_TILES as NUM_TILES
from cle.game_engine.models.map import PORT_DIRECTION_TO_NODEREFS as PORT_DIRECTION_TO_NODEREFS
from cle.game_engine.models.map import CatanMap as CatanMap
from cle.game_engine.models.map import LandTile as LandTile
from cle.game_engine.models.map import Port as Port
from cle.game_engine.models.map import initialize_tiles as initialize_tiles
from cle.game_engine.models.player import Color as Color
from evals.catan_board_bench.tokens.atlas import AtlasEdge as AtlasEdge
from evals.catan_board_bench.tokens.atlas import AtlasMetadata as AtlasMetadata
from evals.catan_board_bench.tokens.atlas import AtlasNode as AtlasNode
from evals.catan_board_bench.tokens.atlas import AtlasPort as AtlasPort
from evals.catan_board_bench.tokens.atlas import AtlasTile as AtlasTile
from evals.catan_board_bench.tokens.atlas import (
    _coordinate_for_tile as _coordinate_for_tile,
)
from evals.catan_board_bench.tokens.atlas import atlas_metadata as atlas_metadata
from evals.catan_board_bench.tokens.atlas import atlas_metadata_json as atlas_metadata_json
from evals.catan_board_bench.tokens.atlas import base_catan_map as base_catan_map
from evals.catan_board_bench.tokens.atlas import base_edges as base_edges
from evals.catan_board_bench.tokens.manifest import TokenAdder as TokenAdder
from evals.catan_board_bench.tokens.manifest import (
    add_tokens_to_tokenizer as add_tokens_to_tokenizer,
)
from evals.catan_board_bench.tokens.manifest import added_tokens as added_tokens
from evals.catan_board_bench.tokens.manifest import atlas_tokens as atlas_tokens
from evals.catan_board_bench.tokens.manifest import (
    recognition_token_inventory as recognition_token_inventory,
)
from evals.catan_board_bench.tokens.manifest import (
    recognition_trainable_tokens as recognition_trainable_tokens,
)
from evals.catan_board_bench.tokens.manifest import (
    semantic_recognition_token_inventory as semantic_recognition_token_inventory,
)
from evals.catan_board_bench.tokens.manifest import token_manifest as token_manifest
from evals.catan_board_bench.tokens.manifest import token_specs as token_specs
from evals.catan_board_bench.tokens.manifest import (
    write_token_manifest as write_token_manifest,
)
from evals.catan_board_bench.tokens.vocabulary import BOARD_OBJECTS as BOARD_OBJECTS
from evals.catan_board_bench.tokens.vocabulary import (
    RECOGNITION_CLASS_VOCABULARIES as RECOGNITION_CLASS_VOCABULARIES,
)
from evals.catan_board_bench.tokens.vocabulary import (
    RECOGNITION_HEADS as RECOGNITION_HEADS,
)
from evals.catan_board_bench.tokens.vocabulary import CatanTokenSpec as CatanTokenSpec
from evals.catan_board_bench.tokens.vocabulary import EdgeId as EdgeId
from evals.catan_board_bench.tokens.vocabulary import NodeId as NodeId
from evals.catan_board_bench.tokens.vocabulary import PortId as PortId
from evals.catan_board_bench.tokens.vocabulary import TileId as TileId
from evals.catan_board_bench.tokens.vocabulary import action_token as action_token
from evals.catan_board_bench.tokens.vocabulary import building_token as building_token
from evals.catan_board_bench.tokens.vocabulary import color_token as color_token
from evals.catan_board_bench.tokens.vocabulary import object_token as object_token
from evals.catan_board_bench.tokens.vocabulary import (
    recognition_answer_token as recognition_answer_token,
)
from evals.catan_board_bench.tokens.vocabulary import (
    recognition_answer_tokens as recognition_answer_tokens,
)
from evals.catan_board_bench.tokens.vocabulary import (
    recognition_query_token as recognition_query_token,
)
from evals.catan_board_bench.tokens.vocabulary import (
    recognition_query_tokens as recognition_query_tokens,
)
from evals.catan_board_bench.tokens.vocabulary import resource_token as resource_token
