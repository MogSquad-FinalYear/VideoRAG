"""
VideoRAG — Execute Router
POST /execute — Upload and process a video file.
DELETE /videos/{video_id} — Delete a video and its indexed data.
"""
import uuid
import shutil
import threading
import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from backend.config import (
    VIDEOS_DIR,
    FRAMES_DIR,
    AUDIO_DIR,
    METADATA_DIR,
    FRAME_SAMPLE_FPS,
    MAX_FRAMES_PER_VIDEO,
    CAPTION_STRIDE,
    CAPTION_MAX_FRAMES,
    CAPTION_MODE,
)
from backend.models import ExecuteResponse
from backend.services import video_processor, embedding_service, captioning_service, transcription_service, indexing_service

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory task tracking
tasks: dict[str, dict] = {}

ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv"}


def _update_task(task_id: str, **kwargs):
    """Thread-safe task state update."""
    if task_id in tasks:
        tasks[task_id].update(kwargs)


def _select_caption_frames(frame_paths: list[str]) -> list[str]:
    """Select a sparse subset of frames for caption indexing."""
    caption_step = max(1, CAPTION_STRIDE)
    selected = frame_paths[::caption_step]
    if CAPTION_MAX_FRAMES > 0 and len(selected) > CAPTION_MAX_FRAMES:
        downsample_step = max(1, len(selected) // CAPTION_MAX_FRAMES)
        selected = selected[::downsample_step]
    return selected


def _generate_captions(video_id: str, frame_paths: list[str], valid_frame_paths: list[str], timestamps: list[float] = None) -> int:
    """Generate and index captions for selected key frames only."""
    try:
        caption_frame_paths = _select_caption_frames(frame_paths)
        if not caption_frame_paths:
            return 0

        logger.info(
            "Generating captions for %s key frames (from %s total) for video %s",
            len(caption_frame_paths),
            len(frame_paths),
            video_id,
        )
        captions = captioning_service.caption_batch(caption_frame_paths)
        caption_map = {
            path: caption
            for path, caption in zip(caption_frame_paths, captions)
            if caption
        }
        aligned_captions = [caption_map.get(path, "") for path in valid_frame_paths]
        indexing_service.index_captions(video_id, valid_frame_paths, aligned_captions, timestamps=timestamps)
        caption_count = sum(1 for c in aligned_captions if c)

        # Bug #4 fix: update captions_ready in metadata
        meta_path = METADATA_DIR / f"{video_id}.json"
        if meta_path.exists():
            import json
            with open(meta_path, "r") as f:
                meta = json.load(f)
            meta["captions_ready"] = True
            meta["caption_count"] = caption_count
            with open(meta_path, "w") as f:
                json.dump(meta, f, indent=2)

        return caption_count
    except Exception as e:
        logger.exception("Caption generation failed for video %s: %s", video_id, e)
        return 0


def _process_video(task_id: str, video_path: str, video_id: str, case_id: str = None):
    """Full video processing pipeline — runs in background thread."""
    try:
        total_steps = 10 if CAPTION_MODE == "sync" else 9
        current = 0

        # Step 1: Extract metadata
        _update_task(task_id, message="Extracting video metadata...", progress=current / total_steps)
        metadata = video_processor.get_video_metadata(video_path, video_id)
        current += 1

        # Step 2: Extract frames (Phase 2: adaptive scene-change sampling)
        _update_task(task_id, message="Extracting frames from video (adaptive sampling)...", progress=current / total_steps)
        try:
            frame_paths, timestamps = video_processor.extract_frames_adaptive(
                video_path,
                video_id,
                max_frames=MAX_FRAMES_PER_VIDEO,
            )
        except Exception as e:
            logger.warning(f"Adaptive sampling failed, falling back to uniform: {e}")
            frame_paths, timestamps = video_processor.extract_frames(
                video_path,
                video_id,
                fps=FRAME_SAMPLE_FPS,
                max_frames=MAX_FRAMES_PER_VIDEO,
            )
        if not frame_paths:
            _update_task(task_id, status="failed", message="No frames extracted from video.", progress=1.0)
            return
        current += 1

        # Step 3: Extract audio
        _update_task(task_id, message="Extracting audio track...", progress=current / total_steps)
        audio_path = video_processor.extract_audio(video_path, video_id)
        current += 1

        # Step 4: Generate CLIP embeddings
        # embed_batch returns (embeddings, valid_paths) — skip failed images
        _update_task(task_id, message=f"Generating visual embeddings for {len(frame_paths)} frames...", progress=current / total_steps)
        embeddings, valid_frame_paths = embedding_service.embed_batch(frame_paths)

        # Build timestamps aligned with valid_frame_paths
        valid_timestamps = []
        path_to_ts = dict(zip(frame_paths, timestamps))
        for vp in valid_frame_paths:
            valid_timestamps.append(path_to_ts.get(vp, 0.0))

        # Bug #1 fix: pass pre-computed timestamps directly
        indexing_service.index_frames(video_id, valid_frame_paths, embeddings, timestamps=valid_timestamps)
        current += 1

        caption_count = 0
        if CAPTION_MODE == "sync":
            caption_count = _generate_captions(video_id, frame_paths, valid_frame_paths, timestamps=valid_timestamps)
            current += 1
        elif CAPTION_MODE == "async":
            _update_task(task_id, captions_ready=False)
            threading.Thread(
                target=_generate_captions,
                args=(video_id, frame_paths, valid_frame_paths),
                kwargs={"timestamps": valid_timestamps},
                daemon=True,
            ).start()

        # Step 5: OCR extraction (Phase 3)
        _update_task(task_id, message="Extracting text from frames (OCR)...", progress=current / total_steps)
        try:
            from backend.services import ocr_service
            ocr_results = ocr_service.extract_text_batch(valid_frame_paths, stride=3)
            indexing_service.index_ocr(video_id, valid_frame_paths, ocr_results, timestamps=valid_timestamps)
        except ImportError:
            logger.warning("EasyOCR not installed, skipping OCR extraction.")
        except Exception as e:
            logger.warning(f"OCR extraction failed (non-fatal): {e}")
        current += 1

        # Step 6: Object Detection (Phase 4) + Scene Graph (Phase 5)
        _update_task(task_id, message="Running object detection (YOLO)...", progress=current / total_steps)
        try:
            from backend.services import detection_service, detection_db, scene_graph_service
            det_results = detection_service.detect_batch(valid_frame_paths, stride=2)
            detection_db.store_detections_batch(video_id, valid_frame_paths, det_results, timestamps=valid_timestamps)

            # Scene graph: extract spatial relations from detections
            from backend.services.indexing_service import _extract_frame_number
            for i, path in enumerate(valid_frame_paths):
                dets = det_results.get(path, [])
                if len(dets) >= 2:
                    frame_number = _extract_frame_number(path) or i
                    ts = valid_timestamps[i] if i < len(valid_timestamps) else 0.0
                    triples = scene_graph_service.process_frame_relations(dets)
                    if triples:
                        detection_db.store_relations(video_id, frame_number, ts, triples)
        except ImportError as ie:
            logger.warning(f"Detection dependencies not installed, skipping: {ie}")
        except Exception as e:
            logger.warning(f"Object detection failed (non-fatal): {e}")
        current += 1

        # Step 7: Transcribe audio + Speaker Tagging (Phase 6)
        segments = []
        if audio_path:
            _update_task(task_id, message="Transcribing audio...", progress=current / total_steps)
            segments = transcription_service.transcribe_audio(audio_path)

            # Phase 6: Speaker diarization (pyannote → gap-based fallback) + role detection
            try:
                from backend.services import speaker_service
                # Use pyannote if available, otherwise gap-based fallback
                segments = speaker_service.diarize_and_assign(segments, audio_path=audio_path)
                segments = speaker_service.auto_detect_roles_for_segments(segments)

                # Store auto-detected roles
                speaker_roles_seen = {}
                for seg in segments:
                    sid = seg.get("speaker_id", "SPEAKER_0")
                    role = seg.get("role", "unknown")
                    if sid not in speaker_roles_seen:
                        speaker_roles_seen[sid] = role
                        speaker_service.set_speaker_role(video_id, sid, role)

                # Cross-session voiceprint matching (Novelty 1, Steps 4-5)
                if case_id:
                    try:
                        speaker_mappings = speaker_service.match_and_register_speakers(
                            case_id=case_id,
                            video_id=video_id,
                            audio_path=audio_path,
                            segments=segments,
                        )
                        # Update segments with canonical speaker names for testimony storage
                        for seg in segments:
                            local_sid = seg.get("speaker_id")
                            if local_sid and local_sid in speaker_mappings:
                                seg["canonical_speaker"] = speaker_mappings[local_sid]
                        logger.info("Speaker matching complete: %s", speaker_mappings)
                    except Exception as e:
                        logger.warning("Voiceprint matching failed (non-fatal): %s", e)

                logger.info("Speaker tagging complete: %d speakers detected", len(speaker_roles_seen))
            except Exception as e:
                logger.warning("Speaker tagging failed (non-fatal): %s", e)

            indexing_service.index_transcripts(video_id, segments)
        else:
            _update_task(task_id, message="No audio track found, skipping transcription.", progress=current / total_steps)
        current += 1

        # Step 8: Store testimony in cross-session memory (Phase 7)
        _update_task(task_id, message="Storing testimony in case memory...", progress=current / total_steps)
        if segments and case_id:
            try:
                from backend.services import testimony_db
                session_id = video_id  # Each video is a session
                testimony_db.store_statements_batch(
                    case_id=case_id,
                    session_id=session_id,
                    video_id=video_id,
                    segments=segments,
                )
            except Exception as e:
                logger.warning("Testimony storage failed (non-fatal): %s", e)
        current += 1

        # Step 9: Contradiction scanning (Phase 7) — runs in background
        _update_task(task_id, message="Scanning for testimony contradictions...", progress=current / total_steps)
        if segments and case_id:
            try:
                from backend.services import contradiction_service
                threading.Thread(
                    target=contradiction_service.scan_for_contradictions,
                    args=(case_id, video_id, segments),
                    daemon=True,
                ).start()
            except Exception as e:
                logger.warning("Contradiction scanning failed (non-fatal): %s", e)
        current += 1

        if CAPTION_MODE == "off":
            caption_state = "captions disabled"
        elif CAPTION_MODE == "async":
            caption_state = "captions indexing in background"
        else:
            caption_state = f"{caption_count} captions indexed"

        # Bug #4 fix: track captions_ready status
        captions_ready = CAPTION_MODE == "sync" or CAPTION_MODE == "off"

        # Store case_id in metadata
        if case_id:
            meta_path = METADATA_DIR / f"{video_id}.json"
            if meta_path.exists():
                import json as _json
                with open(meta_path, "r") as f:
                    meta = _json.load(f)
                meta["case_id"] = case_id
                with open(meta_path, "w") as f:
                    _json.dump(meta, f, indent=2)

        _update_task(
            task_id,
            status="completed",
            message=f"Processing complete. {len(valid_frame_paths)} frames, {caption_state}.",
            progress=1.0,
            video_id=video_id,
            captions_ready=captions_ready,
        )
        logger.info(f"Video {video_id} processing completed successfully.")

    except Exception as e:
        logger.exception(f"Processing failed for video {video_id}")
        _update_task(task_id, status="failed", message=f"Processing failed: {str(e)}", progress=1.0)


@router.post("/execute", response_model=ExecuteResponse)
async def execute_upload(
    file: UploadFile = File(...),
    case_id: str = Form(default=None),
):
    """Upload a video file and start background processing.
    
    Args:
        file: Video file to upload.
        case_id: Optional case identifier to group videos for cross-session
                 testimony memory. If not provided, a unique case_id is
                 auto-generated per video.
    """
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}. Allowed: {ALLOWED_EXTENSIONS}")

    # Generate IDs
    video_id = str(uuid.uuid4())[:12]
    task_id = str(uuid.uuid4())[:12]
    if not case_id:
        case_id = f"case_{video_id}"  # Auto-generate case_id if not provided

    # Fix 9: Stream file to disk in chunks instead of loading entirely into RAM
    video_path = str(VIDEOS_DIR / f"{video_id}{ext}")
    try:
        chunk_size = 1024 * 1024  # 1MB chunks
        with open(video_path, "wb") as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
    except Exception as e:
        # Clean up partial file
        Path(video_path).unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")

    # Initialize task state
    tasks[task_id] = {
        "task_id": task_id,
        "video_id": video_id,
        "case_id": case_id,
        "status": "processing",
        "progress": 0.0,
        "message": "Video uploaded. Starting processing pipeline...",
    }

    # Start background processing
    thread = threading.Thread(
        target=_process_video,
        args=(task_id, video_path, video_id, case_id),
        daemon=True,
    )
    thread.start()

    return ExecuteResponse(
        task_id=task_id,
        status="processing",
        message="Video upload received. Processing started.",
        case_id=case_id,
    )


# Fix 14: Delete video endpoint
@router.delete("/videos/{video_id}")
async def delete_video(video_id: str):
    """Delete a video and all its indexed data."""
    deleted_items = []

    # Delete from ChromaDB indexes
    try:
        indexing_service.delete_video_from_indexes(video_id)
        deleted_items.append("indexes")
    except Exception as e:
        logger.error(f"Error deleting indexes for {video_id}: {e}")

    # Delete from detection/scene-graph DB
    try:
        from backend.services import detection_db
        detection_db.delete_video_data(video_id)
        deleted_items.append("detections")
    except Exception as e:
        logger.warning(f"Error deleting detection data for {video_id}: {e}")

    # Delete video file (try all extensions)
    for ext in ALLOWED_EXTENSIONS:
        vpath = VIDEOS_DIR / f"{video_id}{ext}"
        if vpath.exists():
            vpath.unlink()
            deleted_items.append("video_file")
            break

    # Delete frames directory
    frames_dir = FRAMES_DIR / video_id
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
        deleted_items.append("frames")

    # Delete audio file
    audio_file = AUDIO_DIR / f"{video_id}.wav"
    if audio_file.exists():
        audio_file.unlink()
        deleted_items.append("audio")

    # Delete metadata
    meta_file = METADATA_DIR / f"{video_id}.json"
    if meta_file.exists():
        meta_file.unlink()
        deleted_items.append("metadata")

    # Delete testimony data (Phase 7)
    try:
        from backend.services import testimony_db
        testimony_db.delete_video_testimony(video_id)
        deleted_items.append("testimony")
    except Exception as e:
        logger.warning(f"Error deleting testimony data for {video_id}: {e}")

    if not deleted_items:
        raise HTTPException(status_code=404, detail=f"Video {video_id} not found.")

    logger.info(f"Deleted video {video_id}: {deleted_items}")
    return {"status": "deleted", "video_id": video_id, "deleted": deleted_items}
