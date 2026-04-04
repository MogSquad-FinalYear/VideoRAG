"""
VideoRAG — MCP Server
FastMCP server exposing search tools as MCP protocol endpoints.
For this implementation, tools are called in-process by the agent.
"""
import json
import logging
from backend.services import indexing_service, embedding_service
from backend.config import FRAMES_DIR

logger = logging.getLogger(__name__)


class MCPTools:
    """
    MCP-style tool implementations.
    These mirror the agent tools but contain the actual logic.
    Called in-process for simplicity (can be deployed as separate MCP server later).
    """

    @staticmethod
    def search_by_caption(query: str, n: int = 5, video_id: str = None) -> list[dict]:
        """Search video frames by their BLIP-generated captions."""
        return indexing_service.search_captions(query, n=n, video_id=video_id)

    @staticmethod
    def search_by_image_embedding(query_embedding: list[float], n: int = 5, video_id: str = None) -> list[dict]:
        """Search frames by CLIP embedding similarity."""
        return indexing_service.search_images(query_embedding, n=n, video_id=video_id)

    @staticmethod
    def search_by_text_visual(query: str, n: int = 5, video_id: str = None) -> list[dict]:
        """Search frames by text→CLIP visual similarity."""
        embedding = embedding_service.embed_text(query)
        return indexing_service.search_images(embedding, n=n, video_id=video_id)

    @staticmethod
    def search_by_reference_image(image_path: str, n: int = 5, video_id: str = None) -> list[dict]:
        """Image-to-video: search for frames similar to a reference image."""
        embedding = embedding_service.embed_image(image_path)
        return indexing_service.search_images(embedding, n=n, video_id=video_id)

    @staticmethod
    def search_transcripts(query: str, n: int = 5, video_id: str = None) -> list[dict]:
        """Search speech transcripts."""
        return indexing_service.search_transcripts(query, n=n, video_id=video_id)

    @staticmethod
    def get_video_clip(video_id: str, start_time: float, end_time: float) -> dict:
        """Get frame paths for a specific time range."""
        frames_dir = FRAMES_DIR / video_id
        if not frames_dir.exists():
            return {"error": f"No frames for video {video_id}"}

        frame_files = sorted(frames_dir.glob("frame_*.jpg"))
        clip_frames = []
        for f in frame_files:
            num = int(f.stem.split("_")[1])
            if start_time <= num <= end_time:
                clip_frames.append({
                    "frame_path": f"/frames/{video_id}/{f.name}",
                    "timestamp": num,
                })

        return {
            "video_id": video_id,
            "start_time": start_time,
            "end_time": end_time,
            "frames": clip_frames,
        }

    @staticmethod
    def get_index_stats() -> dict:
        """Get statistics about indexed data."""
        return indexing_service.get_index_stats()


# Singleton instance
mcp_tools = MCPTools()
