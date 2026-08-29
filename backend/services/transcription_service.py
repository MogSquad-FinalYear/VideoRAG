"""
VideoRAG — Whisper Transcription Service
Transcribes audio from videos with timestamps using OpenAI Whisper.
"""
import logging
import os
import shutil
from pathlib import Path

import torch
import whisper

from backend.config import WHISPER_MODEL, DATA_DIR

logger = logging.getLogger(__name__)

_whisper_model = None
_device = "cpu"
_ffmpeg_shimmed = False


def _ensure_ffmpeg_on_path():
    """openai-whisper shells out to a bare 'ffmpeg' command internally, but
    this project intentionally ships no system-wide FFmpeg dependency —
    video_processor.py already uses the imageio-ffmpeg bundled binary for
    muxing. Make that same binary resolvable as 'ffmpeg' on PATH via a
    small local shim, instead of requiring a system install.
    """
    global _ffmpeg_shimmed
    if _ffmpeg_shimmed:
        return
    try:
        import imageio_ffmpeg
        real_ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        shim_dir = DATA_DIR / "_bin"
        shim_dir.mkdir(parents=True, exist_ok=True)
        shim_path = shim_dir / "ffmpeg"
        if not shim_path.exists():
            try:
                os.symlink(real_ffmpeg, shim_path)
            except OSError:
                shutil.copy(real_ffmpeg, shim_path)
                os.chmod(shim_path, 0o755)
        if str(shim_dir) not in os.environ.get("PATH", "").split(os.pathsep):
            os.environ["PATH"] = str(shim_dir) + os.pathsep + os.environ.get("PATH", "")
        _ffmpeg_shimmed = True
    except Exception as e:
        logger.warning("Could not shim ffmpeg for Whisper: %s", e)


def load_whisper_model():
    """Load the Whisper model once, using GPU if available."""
    global _whisper_model, _device
    if _whisper_model is not None:
        return

    _ensure_ffmpeg_on_path()

    # Fix 4: Auto-detect CUDA instead of hardcoding CPU
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Loading Whisper model '{WHISPER_MODEL}' on {_device}...")
    _whisper_model = whisper.load_model(WHISPER_MODEL, device=_device)
    logger.info("Whisper model loaded successfully.")


def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribe audio file and return segments with timestamps.
    Each segment: {"text": str, "start_time": float, "end_time": float}
    """
    load_whisper_model()

    try:
        logger.info(f"Transcribing audio: {audio_path}")
        try:
            result = _whisper_model.transcribe(
                audio_path,
                language=None,  # auto-detect
                fp16=(_device == "cuda"),  # Use fp16 on GPU, fp32 on CPU
                verbose=False
            )
        except ValueError as e:
            # Some GPU/driver/cuDNN combinations produce NaN logits under
            # fp16 decoding (a well-documented openai-whisper compatibility
            # issue, unrelated to the audio content itself) — PyTorch's
            # distribution validation surfaces this as a ValueError. Retry
            # once in fp32 rather than losing the transcript entirely.
            if _device == "cuda" and "invalid values" in str(e):
                logger.warning("fp16 transcription produced invalid values, retrying in fp32: %s", e)
                result = _whisper_model.transcribe(
                    audio_path,
                    language=None,
                    fp16=False,
                    verbose=False
                )
            else:
                raise

        segments = []
        for seg in result.get("segments", []):
            segments.append({
                "text": seg["text"].strip(),
                "start_time": round(seg["start"], 2),
                "end_time": round(seg["end"], 2),
            })

        logger.info(f"Transcribed {len(segments)} segments from {audio_path}")
        return segments

    except Exception as e:
        logger.error(f"Transcription failed for {audio_path}: {e}")
        return []
