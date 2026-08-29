"""
VideoRAG — Testimony Router (Phase 7)
GET /cases/{case_id}/contradictions — Get detected contradictions for a case.
GET /cases/{case_id}/testimony — Get all testimony statements for a case.
"""
import logging
from fastapi import APIRouter

from backend.services import testimony_db

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/cases/{case_id}/testimony")
async def get_testimony(case_id: str, limit: int = 200):
    """Get all testimony statements stored for a case across all sessions."""
    statements = testimony_db.get_all_statements(case_id, limit=limit)
    return {
        "case_id": case_id,
        "total_statements": len(statements),
        "statements": statements,
    }


@router.get("/cases/{case_id}/contradictions")
async def get_contradictions(case_id: str, speaker_id: str = None,
                             min_confidence: float = 0.0):
    """Get detected contradictions for a case.

    Args:
        case_id: The case identifier.
        speaker_id: Optional filter by speaker.
        min_confidence: Minimum confidence threshold (0.0-1.0).
    """
    contradictions = testimony_db.get_contradictions(
        case_id=case_id,
        speaker_id=speaker_id,
        min_confidence=min_confidence,
    )
    return {
        "case_id": case_id,
        "total_contradictions": len(contradictions),
        "contradictions": contradictions,
    }
