"""
VideoRAG — Chat Router
POST /chat — Query the AI agent about video content.
POST /chat/image — Upload a reference image for visual search.
"""
import os
import uuid
import logging
from pathlib import Path
import imagehash
import cv2
import numpy as np
from PIL import Image

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.config import DATA_DIR, FRAMES_DIR
from backend.models import ChatRequest, ChatResponse, VideoClip, SearchResult
from backend.agent.agent import run_agent

logger = logging.getLogger(__name__)
router = APIRouter()

# Simple conversation history per session (in-memory)
_conversations: dict[str, list] = {}


def _center_crop_variant(img: Image.Image, crop_ratio: float = 0.88) -> Image.Image:
    """Return a center-cropped variant to handle player borders/UI in screenshots."""
    w, h = img.size
    new_w = max(1, int(w * crop_ratio))
    new_h = max(1, int(h * crop_ratio))
    left = max(0, (w - new_w) // 2)
    top = max(0, (h - new_h) // 2)
    right = min(w, left + new_w)
    bottom = min(h, top + new_h)
    return img.crop((left, top, right, bottom))


def _compute_hashes(img: Image.Image) -> tuple:
    """Compute complementary perceptual hashes for robust near-duplicate matching."""
    rgb = img.convert("RGB")
    return imagehash.phash(rgb), imagehash.dhash(rgb)


def _to_gray_cv(img: Image.Image) -> np.ndarray:
    """Convert PIL image to OpenCV grayscale matrix."""
    rgb = np.array(img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _orb_match_score(uploaded_variants: list[Image.Image], frame_img: Image.Image) -> tuple[float, int]:
    """Return best ORB normalized score and good match count across uploaded variants."""
    orb = cv2.ORB_create(nfeatures=1200)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    frame_gray = _to_gray_cv(frame_img)
    kp_f, des_f = orb.detectAndCompute(frame_gray, None)
    if des_f is None or not kp_f:
        return 0.0, 0

    best_score = 0.0
    best_good = 0
    for variant in uploaded_variants:
        up_gray = _to_gray_cv(variant)
        kp_u, des_u = orb.detectAndCompute(up_gray, None)
        if des_u is None or not kp_u:
            continue

        try:
            matches = bf.knnMatch(des_u, des_f, k=2)
        except Exception:
            continue

        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

        denom = max(1, min(len(kp_u), len(kp_f)))
        score = len(good) / denom
        if score > best_score:
            best_score = score
            best_good = len(good)

    return best_score, best_good


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Send a natural language query to the AI agent."""
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    logger.info(f"Chat query: '{request.query}' (video_id={request.video_id})")

    result = run_agent(
        query=request.query,
        video_id=request.video_id,
        image_path=request.image_path,
    )

    # Convert to response model
    clips = [VideoClip(**c) for c in result.get("clips", [])]
    sources = []
    for s in result.get("sources", []):
        sources.append(SearchResult(
            video_id=s.get("video_id", ""),
            frame_number=s.get("frame_number"),
            timestamp=s.get("timestamp"),
            start_time=s.get("start_time"),
            end_time=s.get("end_time"),
            score=s.get("score", 0),
            content=s.get("content", ""),
            frame_path=s.get("frame_path", ""),
            source_index=s.get("source_index", ""),
        ))

    return ChatResponse(
        answer=result["answer"],
        clips=clips,
        sources=sources,
    )


@router.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    query: str = Form(default="Find frames similar to this image"),
    video_id: str = Form(default=None),
    image: UploadFile = File(...),
):
    """Upload a reference image and search for visually similar frames."""
    # Save uploaded image
    upload_dir = DATA_DIR / "uploads"
    upload_dir.mkdir(exist_ok=True)
    image_id = str(uuid.uuid4())[:8]
    ext = Path(image.filename).suffix or ".jpg"
    image_path = str(upload_dir / f"{image_id}{ext}")

    content = await image.read()
    with open(image_path, "wb") as f:
        f.write(content)

    logger.info(f"Image-to-video search: '{query}' with image {image_path}")

    # Validate image exists in videos using PERCEPTUAL HASH (pixel-level matching, not semantic)
    # This ensures only actual frames from the videos return matches, not semantic lookalikes
    try:
        PHASH_THRESHOLD = 14
        DHASH_THRESHOLD = 14
        COMBINED_THRESHOLD = 24
        ORB_GATE_PHASH_MAX = 22
        ORB_GATE_DHASH_MAX = 22
        ORB_SCORE_THRESHOLD = 0.28
        ORB_MATCH_COUNT_THRESHOLD = 60

        with Image.open(image_path) as uploaded_img:
            uploaded_base = uploaded_img.convert("RGB")

        uploaded_variants = [
            uploaded_base,
            _center_crop_variant(uploaded_base, 0.92),
            _center_crop_variant(uploaded_base, 0.84),
        ]
        uploaded_hashes = [_compute_hashes(v) for v in uploaded_variants]

        # Scan all indexed frames for hash matches
        frame_matches = []
        best_phash = 10**9
        best_dhash = 10**9

        candidate_dirs = []
        search_video_scope = video_id
        if video_id:
            scoped_dir = FRAMES_DIR / video_id
            if scoped_dir.exists() and scoped_dir.is_dir():
                candidate_dirs = [scoped_dir]
            else:
                logger.warning(f"Requested video_id '{video_id}' has no frame directory at {scoped_dir}")
                candidate_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]
                search_video_scope = None
        else:
            candidate_dirs = [d for d in FRAMES_DIR.iterdir() if d.is_dir()]

        for frame_dir in candidate_dirs:
            if not frame_dir.is_dir():
                continue

            matched_video_id = frame_dir.name
            for frame_file in frame_dir.glob("frame_*.jpg"):
                try:
                    with Image.open(frame_file) as frame_img:
                        frame_phash, frame_dhash = _compute_hashes(frame_img)

                    local_best_phash = 10**9
                    local_best_dhash = 10**9
                    for up_phash, up_dhash in uploaded_hashes:
                        p_dist = up_phash - frame_phash
                        d_dist = up_dhash - frame_dhash
                        local_best_phash = min(local_best_phash, p_dist)
                        local_best_dhash = min(local_best_dhash, d_dist)

                    best_phash = min(best_phash, local_best_phash)
                    best_dhash = min(best_dhash, local_best_dhash)

                    if (
                        local_best_phash <= PHASH_THRESHOLD
                        and local_best_dhash <= DHASH_THRESHOLD
                        and (local_best_phash + local_best_dhash) <= COMBINED_THRESHOLD
                    ):
                        frame_matches.append({
                            "video_id": matched_video_id,
                            "frame_file": frame_file.name,
                            "phash_distance": local_best_phash,
                            "dhash_distance": local_best_dhash,
                        })
                except Exception:
                    continue

        if not frame_matches and best_phash <= ORB_GATE_PHASH_MAX and best_dhash <= ORB_GATE_DHASH_MAX:
            # Fallback: local-feature ORB verification over frame files directly.
            # This avoids dependency on CLIP candidate quality and helps real screenshot matches.
            fallback_matches = []
            best_orb_score = 0.0
            best_orb_good = 0

            for frame_dir in candidate_dirs:
                if not frame_dir.is_dir():
                    continue
                matched_video_id = frame_dir.name

                for frame_file in frame_dir.glob("frame_*.jpg"):
                    try:
                        with Image.open(frame_file) as frame_img:
                            orb_score, orb_good = _orb_match_score(uploaded_variants, frame_img)

                        if orb_score > best_orb_score:
                            best_orb_score = orb_score
                        if orb_good > best_orb_good:
                            best_orb_good = orb_good

                        if orb_score >= ORB_SCORE_THRESHOLD and orb_good >= ORB_MATCH_COUNT_THRESHOLD:
                            fallback_matches.append({
                                "video_id": matched_video_id,
                                "frame_file": frame_file.name,
                                "orb_score": orb_score,
                                "orb_good": orb_good,
                            })
                    except Exception:
                        continue

            if fallback_matches:
                matched_video_ids = sorted({m["video_id"] for m in fallback_matches if m.get("video_id")})
                agent_video_id = search_video_scope
                if not agent_video_id and len(matched_video_ids) == 1:
                    agent_video_id = matched_video_ids[0]

                result = run_agent(
                    query=query,
                    video_id=agent_video_id,
                    image_path=image_path,
                )

                result_sources = [
                    src for src in result.get("sources", [])
                    if src.get("video_id") in matched_video_ids
                ]
                result_clips = [
                    clip for clip in result.get("clips", [])
                    if clip.get("video_id") in matched_video_ids
                ]

                clips = [VideoClip(**c) for c in result_clips]
                sources = []
                for s in result_sources:
                    sources.append(SearchResult(
                        video_id=s.get("video_id", ""),
                        frame_number=s.get("frame_number"),
                        timestamp=s.get("timestamp"),
                        start_time=s.get("start_time"),
                        end_time=s.get("end_time"),
                        score=s.get("score", 0),
                        content=s.get("content", ""),
                        frame_path=s.get("frame_path", ""),
                        source_index=s.get("source_index", ""),
                    ))

                logger.info(
                    f"Image validated via ORB fallback. matches={len(fallback_matches)}, "
                    f"best_orb_score={best_orb_score:.4f}, best_orb_good={best_orb_good}, videos={matched_video_ids}"
                )
                return ChatResponse(
                    answer=result["answer"],
                    clips=clips,
                    sources=sources,
                )

        if not frame_matches:
            best_orb_score = 0.0
            best_orb_good = 0

            logger.warning(
                "No hash matches found. "
                f"best_phash={best_phash}, best_dhash={best_dhash}, "
                f"thresholds=({PHASH_THRESHOLD},{DHASH_THRESHOLD},{COMBINED_THRESHOLD}); "
                f"ORB gate=({ORB_GATE_PHASH_MAX},{ORB_GATE_DHASH_MAX}); "
                f"best_orb_score={best_orb_score:.4f}, best_orb_good={best_orb_good}; "
                f"ORB thresholds=({ORB_SCORE_THRESHOLD},{ORB_MATCH_COUNT_THRESHOLD})"
            )
            return ChatResponse(
                answer="🔍 Image not found in the indexed videos. The uploaded image does not match any frames in the video library.",
                clips=[],
                sources=[],
            )

        matched_video_ids = sorted({m["video_id"] for m in frame_matches})
        logger.info(f"Found {len(frame_matches)} hash matches for uploaded image")

        # Match found - proceed with AI search to get semantic results
        agent_video_id = search_video_scope
        if not agent_video_id and len(matched_video_ids) == 1:
            agent_video_id = matched_video_ids[0]

        result = run_agent(
            query=query,
            video_id=agent_video_id,
            image_path=image_path,
        )

        # Keep response constrained to videos confirmed by hash match
        result_sources = [
            s for s in result.get("sources", [])
            if s.get("video_id") in matched_video_ids
        ]

        result_clips = [
            c for c in result.get("clips", [])
            if c.get("video_id") in matched_video_ids
        ]

        clips = [VideoClip(**c) for c in result.get("clips", [])]
        sources = []
        for s in result_sources:
            sources.append(SearchResult(
                video_id=s.get("video_id", ""),
                frame_number=s.get("frame_number"),
                timestamp=s.get("timestamp"),
                start_time=s.get("start_time"),
                end_time=s.get("end_time"),
                score=s.get("score", 0),
                content=s.get("content", ""),
                frame_path=s.get("frame_path", ""),
                source_index=s.get("source_index", ""),
            ))

        clips = [VideoClip(**c) for c in result_clips]

        return ChatResponse(
            answer=result["answer"],
            clips=clips,
            sources=sources,
        )
    
    except Exception as e:
        logger.error(f"Hash-based image validation failed: {e}")
        return ChatResponse(
            answer="🔍 Error validating image. Please try again.",
            clips=[],
            sources=[],
        )
