# audio_stt.py
from typing import Optional
from google.genai import types
from llm_client import client, GEMINI_MODEL


def transcribe_audio_bytes(audio_bytes: bytes, mime_type: str = "audio/webm") -> Optional[str]:
    """
    Use Gemini to transcribe a short audio answer.
    audio_bytes: raw bytes from the mic recorder
    mime_type: usually 'audio/webm' from streamlit-mic-recorder.
    """
    if not audio_bytes:
        return None

    audio_part = types.Part.from_bytes(
        data=audio_bytes,
        mime_type=mime_type,
    )

    prompt = "Transcribe this interview answer to plain English. Do NOT add anything extra."

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[audio_part, prompt],
    )

    if not response or not response.text:
        return None

    return response.text.strip()
