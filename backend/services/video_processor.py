"""
VideoRAG — Video Processing Service
Frame extraction via OpenCV, audio extraction via imageio-ffmpeg (no system FFmpeg needed).
"""
import cv2
import json
import math
import subprocess
import logging
from pathlib import Path
from datetime import datetime

from backend.config import FRAMES_DIR, AUDIO_DIR, METADATA_DIR, FRAME_SAMPLE_FPS

logger = logging.getLogger(__name__)


def get_ffmpeg_binary() -> str:
    """Get the bundled imageio-ffmpeg binary path (no system FFmpeg needed)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError("imageio-ffmpeg not installed. Run: pip install imageio-ffmpeg")


def extract_frames(video_path: str, video_id: str, fps: float = None, max_frames: int | None = None) -> tuple[list[str], list[float]]:
    """
    Extract frames from video at the configured FPS rate using OpenCV.
    Returns (frame_paths, timestamps) — real per-frame timestamps in seconds,
    computed from the actual OpenCV frame index at capture time.
    """
    if fps is None:
        fps = FRAME_SAMPLE_FPS

    frames_dir = FRAMES_DIR / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_interval = max(1, int(video_fps / fps))
    if max_frames and max_frames > 0 and total_frames > 0:
        # Enforce a hard cap so long videos do not explode processing time.
        capped_interval = max(1, math.ceil(total_frames / max_frames))
        frame_interval = max(frame_interval, capped_interval)
    frame_paths = []
    timestamps = []
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_interval == 0:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = str(frames_dir / frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            # Bug #1 fix: compute real timestamp from actual OpenCV frame index
            timestamps.append(round(frame_idx / video_fps, 2))
            saved_count += 1
        frame_idx += 1

    cap.release()
    logger.info(
        "Extracted %s frames from %s (interval=%s, total_frames=%s, video_fps=%.1f)",
        saved_count,
        video_path,
        frame_interval,
        total_frames,
        video_fps,
    )
    return frame_paths, timestamps


def extract_audio(video_path: str, video_id: str) -> str | None:
    """
    Extract audio from video as WAV using imageio-ffmpeg bundled binary.
    Returns path to WAV file, or None if video has no audio.
    """
    audio_path = str(AUDIO_DIR / f"{video_id}.wav")
    ffmpeg_bin = get_ffmpeg_binary()

    try:
        result = subprocess.run(
            [
                ffmpeg_bin, "-i", video_path,
                "-vn",                  # no video
                "-acodec", "pcm_s16le", # WAV format
                "-ar", "16000",         # 16kHz for Whisper
                "-ac", "1",             # mono
                "-y",                   # overwrite
                audio_path
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            # Check if video has no audio stream
            if "does not contain any stream" in result.stderr or "no audio" in result.stderr.lower():
                logger.warning(f"No audio stream in {video_path}")
                return None
            logger.error(f"FFmpeg error: {result.stderr[:500]}")
            return None

        logger.info(f"Extracted audio to {audio_path}")
        return audio_path

    except subprocess.TimeoutExpired:
        logger.error(f"Audio extraction timed out for {video_path}")
        return None
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return None


def get_video_metadata(video_path: str, video_id: str) -> dict:
    """Get video metadata (duration, resolution, fps) and save to JSON."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    metadata = {
        "video_id": video_id,
        "filename": Path(video_path).name,
        "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / max(cap.get(cv2.CAP_PROP_FPS), 1),
        "fps": cap.get(cv2.CAP_PROP_FPS),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "resolution": f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}",
        "uploaded_at": datetime.now().isoformat(),
    }
    cap.release()

    # Save metadata JSON
    meta_path = METADATA_DIR / f"{video_id}.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"Video metadata: {metadata['resolution']}, {metadata['duration']:.1f}s, {metadata['fps']:.1f}fps")
    return metadata


def extract_frames_adaptive(video_path: str, video_id: str, max_frames: int = None,
                            min_spacing_sec: float = 0.5, max_spacing_sec: float = 3.0,
                            scene_threshold: float = 0.4) -> tuple[list[str], list[float]]:
    """
    Phase 2: Adaptive scene-change sampling.
    Extracts keyframes where visual content changes significantly (scene cuts, motion).
    Falls back to uniform sampling if adaptive produces too few frames.

    Args:
        video_path: Path to video file.
        video_id: Unique identifier for the video.
        max_frames: Maximum number of frames to extract (default: MAX_FRAMES_PER_VIDEO).
        min_spacing_sec: Minimum time between keyframes (prevents over-sampling).
        max_spacing_sec: Maximum time between keyframes (ensures coverage of static scenes).
        scene_threshold: Histogram difference threshold to detect scene changes (0-1).

    Returns:
        (frame_paths, timestamps) tuple with real per-frame timestamps.
    """
    import numpy as np

    if max_frames is None:
        from backend.config import MAX_FRAMES_PER_VIDEO
        max_frames = MAX_FRAMES_PER_VIDEO

    frames_dir = FRAMES_DIR / video_id
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = 30.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / video_fps

    # For very short videos (< 5s), just use uniform sampling
    if duration < 5.0:
        cap.release()
        return extract_frames(video_path, video_id, max_frames=max_frames)

    min_spacing_frames = int(min_spacing_sec * video_fps)
    max_spacing_frames = int(max_spacing_sec * video_fps)

    # First pass: scan video and compute histogram differences
    prev_hist = None
    keyframe_indices = [0]  # Always include first frame
    last_keyframe_idx = 0

    frame_idx = 0
    # Read every Nth frame for efficiency (check every ~0.1 seconds)
    check_interval = max(1, int(video_fps * 0.1))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % check_interval == 0:
            # Compute grayscale histogram
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
            cv2.normalize(hist, hist)

            if prev_hist is not None:
                # Compare histograms using correlation (1 = identical, 0 = different)
                diff = 1.0 - cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

                frames_since_last = frame_idx - last_keyframe_idx

                # Mark as keyframe if:
                # 1. Significant scene change AND enough spacing, OR
                # 2. Too much time has passed (max_spacing enforcement)
                if (diff > scene_threshold and frames_since_last >= min_spacing_frames) or \
                   frames_since_last >= max_spacing_frames:
                    keyframe_indices.append(frame_idx)
                    last_keyframe_idx = frame_idx

            prev_hist = hist.copy()
        frame_idx += 1

    # Always include last frame
    if keyframe_indices[-1] < total_frames - 1:
        keyframe_indices.append(total_frames - 1)

    # If we got too many keyframes, subsample uniformly from the detected set
    if max_frames and len(keyframe_indices) > max_frames:
        step = len(keyframe_indices) / max_frames
        subsampled = [keyframe_indices[int(i * step)] for i in range(max_frames)]
        keyframe_indices = subsampled

    # If adaptive found too few frames (< 10% of budget), fall back to uniform
    min_reasonable = max(10, max_frames // 10)
    if len(keyframe_indices) < min_reasonable:
        cap.release()
        logger.info("Adaptive sampling found too few keyframes (%d), falling back to uniform.", len(keyframe_indices))
        return extract_frames(video_path, video_id, max_frames=max_frames)

    # Second pass: extract the selected keyframes
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    frame_paths = []
    timestamps = []
    keyframe_set = set(keyframe_indices)
    frame_idx = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx in keyframe_set:
            frame_filename = f"frame_{saved_count:06d}.jpg"
            frame_path = str(frames_dir / frame_filename)
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
            timestamps.append(round(frame_idx / video_fps, 2))
            saved_count += 1
        frame_idx += 1

    cap.release()
    logger.info(
        "Adaptive sampling: extracted %d keyframes from %s (total_frames=%d, duration=%.1fs)",
        saved_count, video_path, total_frames, duration,
    )
    return frame_paths, timestamps
