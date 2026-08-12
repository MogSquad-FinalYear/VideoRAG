"""
VideoRAG — Speakers Router (Phase 6)
POST /videos/{video_id}/speakers — Set speaker-to-role mappings.
GET  /videos/{video_id}/speakers — Get current speaker-to-role mappings.
GET  /cases/{case_id}/speaker-matches — Pending cross-session voiceprint matches.
POST /cases/{case_id}/speaker-matches/confirm — Confirm or correct a match.
"""
import logging
from fastapi import APIRouter, HTTPException

from backend.models import SpeakerRoleMappingRequest, SpeakerMatchConfirmRequest
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


@router.get("/cases/{case_id}/speaker-matches")
async def get_speaker_matches(case_id: str):
    """Get cross-session voiceprint matches for a case, for human confirmation.

    Novelty 1, Step 5: an auto-match must be confirmed (or corrected) by a
    user before it should be trusted by downstream contradiction detection.
    Returns both the matches still awaiting confirmation and the full list
    of canonical speakers already known in this case (for correction UIs).
    """
    pending = speaker_service.get_pending_matches(case_id)
    all_speakers = speaker_service.list_case_speakers(case_id)
    return {"case_id": case_id, "pending": pending, "all_speakers": all_speakers}


@router.post("/cases/{case_id}/speaker-matches/confirm")
async def confirm_speaker_match(case_id: str, request: SpeakerMatchConfirmRequest):
    """Confirm or correct a cross-session voiceprint match.

    action="confirm": accept the auto-match as-is.
    action="rename": the auto-match was wrong; reassign to `corrected_name`
        (linking to an existing case speaker or registering a new one).
    """
    if request.action not in ("confirm", "rename"):
        raise HTTPException(status_code=400, detail="action must be 'confirm' or 'rename'")
    if request.action == "rename" and not request.corrected_name:
        raise HTTPException(status_code=400, detail="corrected_name is required for action='rename'")

    try:
        result = speaker_service.confirm_speaker_match(
            case_id=case_id,
            video_id=request.video_id,
            local_speaker_id=request.local_speaker_id,
            action=request.action,
            corrected_name=request.corrected_name,
            role=request.role,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    return {"case_id": case_id, "result": result}
