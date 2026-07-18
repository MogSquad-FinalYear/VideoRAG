"""
VideoRAG — Speakers Router (Phase 6)
POST /videos/{video_id}/speakers — Set speaker-to-role mappings.
GET  /videos/{video_id}/speakers — Get current speaker-to-role mappings.
"""
import logging
from fastapi import APIRouter, HTTPException

from backend.models import SpeakerRoleMappingRequest
from backend.services import speaker_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/videos/{video_id}/speakers")
async def get_speakers(video_id: str):
    """Get speaker-to-role mappings for a video."""
    roles = speaker_service.get_speaker_roles(video_id)
    return {"video_id": video_id, "speakers": roles}


@router.post("/videos/{video_id}/speakers")
async def set_speakers(video_id: str, request: SpeakerRoleMappingRequest):
    """Set manual speaker-to-role mappings for a video.

    Example request body:
    {
        "mappings": [
            {"speaker_id": "SPEAKER_0", "role": "judge", "label": "Judge Smith"},
            {"speaker_id": "SPEAKER_1", "role": "witness", "label": "Jane Doe"},
            {"speaker_id": "SPEAKER_2", "role": "counsel", "label": "Defense Attorney"}
        ]
    }
    """
    if not request.mappings:
        raise HTTPException(status_code=400, detail="At least one mapping is required.")

    results = []
    for mapping in request.mappings:
        result = speaker_service.set_speaker_role(
            video_id=video_id,
            speaker_id=mapping.speaker_id,
            role=mapping.role,
            label=mapping.label,
        )
        results.append(result)

    return {"video_id": video_id, "updated": results}
