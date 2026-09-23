"""Shapes the YouTube transcript API hands back."""

from typing import TypedDict


class TranscriptSegment(TypedDict):
    """One caption segment with its timestamp."""

    text: str
    start: float
    duration: float


class TranscriptChunk(TypedDict):
    """A sliding window of caption text with its time range."""

    text: str
    start: float
    end: float


class VideoMetadata(TypedDict):
    """The subset of the YouTube videos.list response this scraper keeps."""

    title: str
    channel: str
    published_at: str
    view_count: int
    like_count: int
    description: str


class ScrapedVideo(TypedDict):
    """One video's transcript, chunks and metadata."""

    video_id: str
    url: str
    metadata: VideoMetadata | None
    transcript: list[TranscriptSegment]
    chunks: list[TranscriptChunk]
    scraped_at: str


__all__ = [
    "ScrapedVideo",
    "TranscriptChunk",
    "TranscriptSegment",
    "VideoMetadata",
]
