"""
Catan Knowledge Corpus Builder

Orchestrates all data sources (YouTube transcripts, forums, rulebook) into a
single JSONL file suitable for continued pretraining with LLaMA-Factory.

Usage:
    python -m data_pipeline.training.pretraining.build_corpus [--output OUTPUT_PATH] [--skip-youtube] [--skip-forums]
"""

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import List, Dict

from data_pipeline.training.pretraining.sources.youtube import YouTubeCorpusBuilder
from data_pipeline.training.pretraining.sources.forums import ForumCorpusBuilder

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SOURCES_DIR = Path(__file__).parent / "sources"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "generated" / "pretraining" / "legacy_corpus"
DEFAULT_OUTPUT = OUTPUT_DIR / "catan_corpus.jsonl"

# Chunking config
MIN_CHUNK_CHARS = 200  # Skip chunks shorter than this
MAX_CHUNK_CHARS = 8000  # Split chunks longer than this
OVERLAP_CHARS = 200  # Overlap between split chunks


def load_rulebook() -> List[Dict[str, str]]:
    """Load the static rulebook text and split into sections."""
    rulebook_path = SOURCES_DIR / "rulebook.txt"
    if not rulebook_path.exists():
        print("[Rulebook] WARNING: rulebook.txt not found, skipping")
        return []

    text = rulebook_path.read_text()

    # Split on major section headers (ALL CAPS lines)
    sections = re.split(r"\n(?=[A-Z][A-Z\s&]{5,}\n)", text)

    results = []
    for section in sections:
        section = section.strip()
        if len(section) < MIN_CHUNK_CHARS:
            continue
        results.append({
            "source": "rulebook",
            "text": section,
        })

    print(f"[Rulebook] Loaded {len(results)} sections ({sum(len(r['text']) for r in results)} chars)")
    return results


def clean_text(text: str) -> str:
    """Normalize and clean text for pretraining."""
    # Normalize unicode
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", " - ").replace("\u2013", " - ")
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove excessive whitespace but keep paragraph structure
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def chunk_document(text: str, source: str) -> List[Dict[str, str]]:
    """Split a long document into training-sized chunks with overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [{"source": source, "text": text}]

    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(text):
        end = start + MAX_CHUNK_CHARS

        # Try to break at a paragraph or sentence boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind("\n\n", start + MAX_CHUNK_CHARS // 2, end)
            if para_break > start:
                end = para_break
            else:
                # Look for sentence break
                sent_break = text.rfind(". ", start + MAX_CHUNK_CHARS // 2, end)
                if sent_break > start:
                    end = sent_break + 1

        chunk_text = text[start:end].strip()
        if len(chunk_text) >= MIN_CHUNK_CHARS:
            chunks.append({
                "source": f"{source}/chunk_{chunk_idx}",
                "text": chunk_text,
            })
            chunk_idx += 1

        # Advance with overlap
        start = end - OVERLAP_CHARS if end < len(text) else len(text)

    return chunks


def deduplicate(documents: List[Dict[str, str]], similarity_threshold: int = 100) -> List[Dict[str, str]]:
    """Remove near-duplicate documents using prefix hashing."""
    seen_hashes = set()
    deduped = []

    for doc in documents:
        text = doc["text"]
        # Hash on first N chars (catches exact and near-duplicates)
        prefix = text[:similarity_threshold].lower().strip()
        h = hashlib.md5(prefix.encode()).hexdigest()

        if h not in seen_hashes:
            seen_hashes.add(h)
            deduped.append(doc)

    removed = len(documents) - len(deduped)
    if removed:
        print(f"[Dedup] Removed {removed} near-duplicate documents")

    return deduped


def build_corpus(
    output_path: Path = DEFAULT_OUTPUT,
    skip_youtube: bool = False,
    skip_forums: bool = False,
    skip_rulebook: bool = False,
) -> Path:
    """Build the complete Catan knowledge corpus.

    Args:
        output_path: Where to write the JSONL file
        skip_youtube: Skip YouTube transcript fetching
        skip_forums: Skip Reddit/BGG forum scraping
        skip_rulebook: Skip static rulebook text

    Returns:
        Path to the output JSONL file
    """
    all_docs: List[Dict[str, str]] = []

    # 1. Rulebook (always fast, no API needed)
    if not skip_rulebook:
        print("\n=== Loading Rulebook ===")
        all_docs.extend(load_rulebook())

    # 2. YouTube transcripts
    if not skip_youtube:
        print("\n=== Fetching YouTube Transcripts ===")
        yt_builder = YouTubeCorpusBuilder()
        yt_docs = yt_builder.build()
        all_docs.extend(yt_docs)

    # 3. Forum content
    if not skip_forums:
        print("\n=== Fetching Forum Content ===")
        forum_builder = ForumCorpusBuilder()
        forum_docs = forum_builder.build()
        all_docs.extend(forum_docs)

    # 4. Clean all text
    print("\n=== Cleaning Text ===")
    for doc in all_docs:
        doc["text"] = clean_text(doc["text"])

    # 5. Filter out short documents
    all_docs = [doc for doc in all_docs if len(doc["text"]) >= MIN_CHUNK_CHARS]

    # 6. Chunk long documents
    print("\n=== Chunking Documents ===")
    chunked_docs = []
    for doc in all_docs:
        chunked_docs.extend(chunk_document(doc["text"], doc["source"]))

    # 7. Deduplicate
    print("\n=== Deduplicating ===")
    final_docs = deduplicate(chunked_docs)

    # 8. Write JSONL
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for doc in final_docs:
            # LLaMA-Factory pt format: just {"text": "..."}
            f.write(json.dumps({"text": doc["text"]}) + "\n")

    # Stats
    total_chars = sum(len(doc["text"]) for doc in final_docs)
    estimated_tokens = total_chars // 4  # Rough estimate: ~4 chars per token
    print("\n=== Corpus Stats ===")
    print(f"Documents: {len(final_docs)}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated tokens: {estimated_tokens:,}")
    print(f"Output: {output_path}")

    # Source breakdown
    source_counts: Dict[str, int] = {}
    for doc in final_docs:
        src_type = doc["source"].split("/")[0]
        source_counts[src_type] = source_counts.get(src_type, 0) + 1
    print(f"Source breakdown: {json.dumps(source_counts, indent=2)}")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Build Catan knowledge corpus for continued pretraining")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSONL path")
    parser.add_argument("--skip-youtube", action="store_true", help="Skip YouTube transcripts")
    parser.add_argument("--skip-forums", action="store_true", help="Skip Reddit/BGG forums")
    parser.add_argument("--skip-rulebook", action="store_true", help="Skip rulebook text")
    args = parser.parse_args()

    build_corpus(
        output_path=args.output,
        skip_youtube=args.skip_youtube,
        skip_forums=args.skip_forums,
        skip_rulebook=args.skip_rulebook,
    )


if __name__ == "__main__":
    main()
