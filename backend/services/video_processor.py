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


def extract_frames(video_path: str, video_id: str, fps: float = None, max_frames: int | None = None) -> list[str]:
    """
    Extract frames from video at the configured FPS rate using OpenCV.
    Returns list of saved frame file paths.
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
            saved_count += 1
        frame_idx += 1

    cap.release()
    logger.info(
        "Extracted %s frames from %s (interval=%s, total_frames=%s)",
        saved_count,
        video_path,
        frame_interval,
        total_frames,
    )
    return frame_paths


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
