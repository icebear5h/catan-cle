"""
YouTube video scraper for Catan strategy content.

Extracts transcripts and metadata from Catan strategy videos.
"""

import os
import json
import re
from typing import List, Dict, Optional
from datetime import datetime
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build


class YouTubeScraper:
    """Scrape Catan strategy videos and extract transcripts."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize YouTube scraper.

        Args:
            api_key: YouTube Data API key (optional, for metadata)
        """
        self.api_key = api_key or os.getenv("YOUTUBE_API_KEY")
        self.youtube = None
        if self.api_key:
            self.youtube = build('youtube', 'v3', developerKey=self.api_key)

    def extract_video_id(self, url: str) -> str:
        """Extract video ID from YouTube URL."""
        patterns = [
            r'(?:youtube\.com/watch\?v=|youtu\.be/)([^&\n?#]+)',
            r'youtube\.com/embed/([^&\n?#]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        raise ValueError(f"Could not extract video ID from: {url}")

    def get_transcript(self, video_id: str) -> List[Dict[str, any]]:
        """
        Get transcript for a video.

        Returns:
            List of transcript segments with timestamps:
            [
                {"text": "...", "start": 12.5, "duration": 3.2},
                ...
            ]
        """
        try:
            api = YouTubeTranscriptApi()
            result = api.fetch(video_id)
            return [
                {"text": s.text, "start": s.start, "duration": s.duration}
                for s in result
            ]
        except Exception as e:
            print(f"Error fetching transcript for {video_id}: {e}")
            return []

    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """
        Get video metadata (title, views, etc) via YouTube API.

        Requires API key to be set.
        """
        if not self.youtube:
            return None

        try:
            response = self.youtube.videos().list(
                part='snippet,statistics',
                id=video_id
            ).execute()

            if not response['items']:
                return None

            item = response['items'][0]
            return {
                'title': item['snippet']['title'],
                'channel': item['snippet']['channelTitle'],
                'published_at': item['snippet']['publishedAt'],
                'view_count': int(item['statistics'].get('viewCount', 0)),
                'like_count': int(item['statistics'].get('likeCount', 0)),
                'description': item['snippet']['description']
            }
        except Exception as e:
            print(f"Error fetching metadata for {video_id}: {e}")
            return None

    def chunk_transcript(
        self,
        transcript: List[Dict],
        window_size: int = 30
    ) -> List[Dict]:
        """
        Chunk transcript into windows for analysis.

        Args:
            transcript: Raw transcript segments
            window_size: Size of sliding window in seconds

        Returns:
            List of chunks with combined text and time range
        """
        chunks = []
        current_chunk = []
        chunk_start = None

        for segment in transcript:
            if chunk_start is None:
                chunk_start = segment['start']

            current_chunk.append(segment['text'])

            # Check if window is complete
            if segment['start'] + segment['duration'] - chunk_start >= window_size:
                chunks.append({
                    'text': ' '.join(current_chunk),
                    'start': chunk_start,
                    'end': segment['start'] + segment['duration']
                })
                current_chunk = []
                chunk_start = None

        # Add final chunk if exists
        if current_chunk:
            chunks.append({
                'text': ' '.join(current_chunk),
                'start': chunk_start,
                'end': transcript[-1]['start'] + transcript[-1]['duration']
            })

        return chunks

    def scrape_video(self, video_url: str) -> Dict:
        """
        Scrape a single video for transcript and metadata.

        Returns:
            {
                'video_id': str,
                'url': str,
                'metadata': dict,
                'transcript': list,
                'chunks': list
            }
        """
        video_id = self.extract_video_id(video_url)
        print(f"Scraping video: {video_id}")

        # Get transcript
        transcript = self.get_transcript(video_id)
        if not transcript:
            print(f"  No transcript available")
            return None

        # Get metadata
        metadata = self.get_video_metadata(video_id)

        # Chunk transcript
        chunks = self.chunk_transcript(transcript)

        print(f"  Found {len(transcript)} segments, {len(chunks)} chunks")

        return {
            'video_id': video_id,
            'url': video_url,
            'metadata': metadata,
            'transcript': transcript,
            'chunks': chunks,
            'scraped_at': datetime.now().isoformat()
        }

    def search_catan_videos(
        self,
        query: str = "catan strategy",
        max_results: int = 50
    ) -> List[str]:
        """
        Search for Catan videos using YouTube API.

        Returns list of video URLs.
        """
        if not self.youtube:
            print("YouTube API key required for search")
            return []

        try:
            request = self.youtube.search().list(
                part='id',
                q=query,
                type='video',
                maxResults=max_results,
                order='relevance'
            )
            response = request.execute()

            video_ids = [
                item['id']['videoId']
                for item in response['items']
                if item['id']['kind'] == 'youtube#video'
            ]

            return [f"https://youtube.com/watch?v={vid}" for vid in video_ids]

        except Exception as e:
            print(f"Error searching videos: {e}")
            return []

    def save_transcript(self, result: Dict, output_dir: str = "data_pipeline/training/pretraining/output") -> str:
        """
        Save scraped video to markdown file in the transcript folder.

        Args:
            result: Output from scrape_video()
            output_dir: Directory to save transcript files

        Returns:
            Path to saved file
        """
        os.makedirs(output_dir, exist_ok=True)

        video_id = result['video_id']
        title = "Unknown Title"
        if result.get('metadata') and result['metadata'].get('title'):
            title = result['metadata']['title']

        filename = f"transcript_{video_id}.md"
        filepath = os.path.join(output_dir, filename)

        # Build full transcript text
        full_text = ' '.join(seg['text'] for seg in result['transcript'])

        lines = []
        lines.append(f"# {title}")
        lines.append(f"")
        lines.append(f"Source: {result['url']}")
        if result.get('metadata'):
            meta = result['metadata']
            if meta.get('channel'):
                lines.append(f"Channel: {meta['channel']}")
            if meta.get('view_count'):
                lines.append(f"Views: {meta['view_count']:,}")
        lines.append(f"Scraped: {result.get('scraped_at', 'unknown')}")
        lines.append(f"")
        lines.append(f"## Full Transcript")
        lines.append(f"")
        lines.append(full_text)
        lines.append(f"")
        lines.append(f"## Chunked Transcript (30s windows)")
        lines.append(f"")

        for i, chunk in enumerate(result['chunks'], 1):
            start_m, start_s = divmod(int(chunk['start']), 60)
            end_m, end_s = divmod(int(chunk['end']), 60)
            lines.append(f"### Chunk {i} [{start_m}:{start_s:02d} - {end_m}:{end_s:02d}]")
            lines.append(f"")
            lines.append(chunk['text'])
            lines.append(f"")

        with open(filepath, 'w') as f:
            f.write('\n'.join(lines))

        print(f"  Saved: {filepath} ({len(result['transcript'])} segments, {len(result['chunks'])} chunks)")
        return filepath


# Curated list of high-quality Catan strategy channels/videos
CURATED_VIDEOS = [
    # Add specific video URLs here
    # Format: (url, expert_name, notes)
]

CURATED_CHANNELS = [
    # Top Catan strategy channels
    # Format: (channel_id, channel_name, avg_elo_estimate)
]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m data_pipeline.ingestion.youtube_scraper <youtube_url> [youtube_url2 ...]")
        print("  Scrapes transcript and saves to data_pipeline/training/pretraining/output/transcript_{id}.md")
        sys.exit(1)

    scraper = YouTubeScraper()

    for url in sys.argv[1:]:
        try:
            result = scraper.scrape_video(url)
            if result:
                scraper.save_transcript(result)
        except Exception as e:
            print(f"Error scraping {url}: {e}")
