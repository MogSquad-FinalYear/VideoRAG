"""
VideoRAG — Whisper Transcription Service
Transcribes audio from videos with timestamps using OpenAI Whisper.
"""
import logging
import torch
import whisper

from backend.config import WHISPER_MODEL

logger = logging.getLogger(__name__)

_whisper_model = None
_device = "cpu"


def load_whisper_model():
    """Load the Whisper model once, using GPU if available."""
    global _whisper_model, _device
    if _whisper_model is not None:
        return

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
        result = _whisper_model.transcribe(
            audio_path,
            language=None,  # auto-detect
            fp16=(_device == "cuda"),  # Use fp16 on GPU, fp32 on CPU
            verbose=False
        )

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
