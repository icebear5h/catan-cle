"""The single-page tuner markup, assembled from its two halves."""

from __future__ import annotations

from ._page_head import HTML_HEAD
from ._page_tail import HTML_TAIL

HTML = HTML_HEAD + HTML_TAIL
