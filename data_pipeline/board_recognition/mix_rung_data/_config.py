"""Schema constants and the token/answer regexes."""


from __future__ import annotations

import re

from data_pipeline.json_types import JsonDict

EXPORT_SCHEMA = "catan_mixed_rung/v1"


DENSITY_BINS = ("empty", "setup", "sparse", "dense")


ATLAS_TOKEN_RE = re.compile(r"<[NETP][0-9_]+>")


WORD_RE = re.compile(r"[A-Za-z0-9:]+|;")


__all__ = ["ATLAS_TOKEN_RE", "DENSITY_BINS", "EXPORT_SCHEMA", "JsonDict", "WORD_RE"]
