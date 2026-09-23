"""Randomized multi-operation query packs over a single board.

The board-fluency corpus states a ~800-token board, asks one question, and throws
the board away -- about 0.4% of the ~243 valid queries each board admits, with a
loss-bearing token fraction near 5%. A pack states the board once and answers
many queries against it.
"""

from __future__ import annotations

import random as random
from collections import OrderedDict as OrderedDict
from collections import defaultdict as defaultdict
from dataclasses import dataclass as dataclass
from dataclasses import field as field
from typing import Dict as Dict
from typing import Iterable as Iterable
from typing import List as List
from typing import Sequence as Sequence
from typing import Tuple as Tuple

from sft.board.board_fluency_scoring import OPERATION_FAMILY as OPERATION_FAMILY
from sft.board.board_fluency_scoring import _parse_answer as _parse_answer
from sft.board.board_fluency_scoring import _strip_transport as _strip_transport
from sft.board.board_fluency_scoring import _typed_equal as _typed_equal
from sft.board.board_packs._compose import build_pack as build_pack
from sft.board.board_packs._compose import compose_pack as compose_pack
from sft.board.board_packs._compose import generate_packs as generate_packs
from sft.board.board_packs._model import _SHARED_KEYS as _SHARED_KEYS
from sft.board.board_packs._model import PACK_INSTRUCTIONS as PACK_INSTRUCTIONS
from sft.board.board_packs._model import SCHEMA as SCHEMA
from sft.board.board_packs._model import Pack as Pack
from sft.board.board_packs._model import PackEntry as PackEntry
from sft.board.board_packs._model import _share_signature as _share_signature
from sft.board.board_packs._model import render_query as render_query
from sft.board.board_packs._model import sample_entry_count as sample_entry_count
from sft.board.board_packs._scoring import PackScore as PackScore
from sft.board.board_packs._scoring import entry_correct as entry_correct
from sft.board.board_packs._scoring import parse_pack_answer as parse_pack_answer
from sft.board.board_packs._scoring import render_pack as render_pack
from sft.board.board_packs._scoring import score_pack as score_pack
from sft.scripts.builders.build_board_fluency_review import Candidate as Candidate
from sft.scripts.builders.build_board_fluency_review import Facts as Facts
from sft.scripts.builders.build_board_fluency_review import answer as gold_answer
from sft.scripts.builders.build_board_fluency_review import candidates_for as candidates_for
from sft.scripts.builders.build_board_fluency_review import question as render_question

__all__: list[str] = [
    "Candidate",
    "Dict",
    "Facts",
    "Iterable",
    "List",
    "OPERATION_FAMILY",
    "OrderedDict",
    "PACK_INSTRUCTIONS",
    "Pack",
    "PackEntry",
    "PackScore",
    "SCHEMA",
    "Sequence",
    "Tuple",
    "annotations",
    "build_pack",
    "candidates_for",
    "compose_pack",
    "dataclass",
    "defaultdict",
    "entry_correct",
    "field",
    "generate_packs",
    "gold_answer",
    "parse_pack_answer",
    "random",
    "render_pack",
    "render_query",
    "render_question",
    "sample_entry_count",
    "score_pack",
]
