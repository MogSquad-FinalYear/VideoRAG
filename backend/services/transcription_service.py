"""
VideoRAG — Whisper Transcription Service
Transcribes audio from videos with timestamps using OpenAI Whisper.
"""
import logging
import whisper

from backend.config import WHISPER_MODEL

logger = logging.getLogger(__name__)

_whisper_model = None


def load_whisper_model():
    """Load the Whisper model once."""
    global _whisper_model
    if _whisper_model is not None:
        return

    logger.info(f"Loading Whisper model '{WHISPER_MODEL}' on CPU...")
    _whisper_model = whisper.load_model(WHISPER_MODEL, device="cpu")
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
            fp16=False,     # CPU mode
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
