"""
VideoRAG — Pydantic Models
Request/response schemas for all API endpoints.
"""

from __future__ import annotations
from typing import Optional
from pydantic import BaseModel, Field


# ── Execute (Video Upload) ───────────────────────────────────────────────────

class ExecuteResponse(BaseModel):
    task_id: str
    status: str = "processing"
    message: str = "Video upload received. Processing started."
    case_id: Optional[str] = None


# ── Task Status ──────────────────────────────────────────────────────────────

class TaskStatus(BaseModel):
    task_id: str
    status: str  # "processing" | "completed" | "failed"
    progress: float = 0.0  # 0.0 to 1.0
    message: str = ""
    video_id: Optional[str] = None
    captions_ready: bool = False


# ── Chat ─────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language question about the video(s)")
    video_id: Optional[str] = Field(None, description="Optional: scope query to a specific video")
    image_path: Optional[str] = Field(None, description="Optional: path to reference image for visual search")
    case_id: Optional[str] = Field(None, description="Optional: scope query to a legal case for contradiction checking")


class SearchResult(BaseModel):
    video_id: str
    frame_number: Optional[int] = None
    timestamp: Optional[float] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    score: float = 0.0
    content: str = ""  # caption or transcript text
    frame_path: Optional[str] = None
    source_index: str = ""  # "image" | "caption" | "speech"


class VideoClip(BaseModel):
    video_id: str
    start_time: float
    end_time: float
    frame_paths: list[str] = []
    description: str = ""


class CitationVerification(BaseModel):
    citation_id: str
    timestamp: float
    timestamp_text: str = ""
    verification_score: float = 0.0
    verification_result: str = "unverified"  # "verified" | "uncertain" | "unsupported"
    explanation: str = ""
    clip_hash: str = ""


class ContradictionResult(BaseModel):
    contradiction_id: str = ""
    speaker_id: str = ""
    statement_a: str = ""
    statement_b: str = ""
    confidence: float = 0.0
    explanation: str = ""
    video_a: Optional[str] = None
    video_b: Optional[str] = None
    timestamp_a: Optional[float] = None
    timestamp_b: Optional[float] = None


class ChatResponse(BaseModel):
    answer: str
    clips: list[VideoClip] = []
    sources: list[SearchResult] = []
    citations: list[CitationVerification] = []
    contradictions: list[ContradictionResult] = []


# ── Video Info ───────────────────────────────────────────────────────────────

class VideoInfo(BaseModel):
    video_id: str
    filename: str
    duration: Optional[float] = None
    resolution: Optional[str] = None
    fps: Optional[float] = None
    frame_count: Optional[int] = None
    status: str = "processing"  # "processing" | "completed" | "failed"
    uploaded_at: Optional[str] = None
    thumbnail: Optional[str] = None
    captions_ready: bool = False
    case_id: Optional[str] = None


# ── Speaker / Testimony Models ───────────────────────────────────────────────

class SpeakerRoleMapping(BaseModel):
    speaker_id: str
    role: str = "unknown"  # "judge" | "witness" | "counsel" | "unknown"
    label: str = ""  # Optional human-readable name


class SpeakerRoleMappingRequest(BaseModel):
    mappings: list[SpeakerRoleMapping]


class SpeakerMatch(BaseModel):
    video_id: str
    local_speaker_id: str
    canonical_name: str
    confidence: float = 0.0
    role: str = "unknown"
    confirmed_by_user: bool = False


class SpeakerMatchConfirmRequest(BaseModel):
    video_id: str
    local_speaker_id: str
    action: str  # "confirm" | "rename"
    corrected_name: Optional[str] = None
    role: Optional[str] = None
