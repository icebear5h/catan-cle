"""The one blueprint every CatanBoardBench route is registered on."""

from flask import Blueprint

__all__ = ["bench_bp"]

bench_bp = Blueprint("bench", __name__)
