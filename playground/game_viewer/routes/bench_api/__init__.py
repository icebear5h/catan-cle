"""CatanBoardBench route modules and the payload builders they share.

Importing this package registers every bench route except the ones that read
values the tests patch on ``playground.game_viewer.routes.bench`` itself.
"""

from . import board_bench, sft, spatial, text_format
from .blueprint import bench_bp
from .paths import INITIAL_SETTLEMENT_REASONING_ROOT
from .reasoning import reasoning_trace_payload, reasoning_traces_payload

__all__ = [
    "INITIAL_SETTLEMENT_REASONING_ROOT",
    "bench_bp",
    "board_bench",
    "reasoning_trace_payload",
    "reasoning_traces_payload",
    "sft",
    "spatial",
    "text_format",
]
