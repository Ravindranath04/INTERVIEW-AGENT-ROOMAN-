# audio_stt.py
import os
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai.types import Part, GenerateContentConfig

load_dotenv()

# Try env var first, then Streamlit secrets (for Cloud)
API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("GEMINI_API_KEY is missing. Set it in .env (local) or Streamlit secrets (cloud).")

client = genai.Client(api_key=API_KEY)


def transcribe_audio_bytes(audio_bytes: bytes) -> str | None:
    """
    Use Gemini to transcribe the recorded audio to text.
    This is cloud-friendly (no ffmpeg, no local whisper, etc.)

    The mic_recorder component usually sends WEBM, so we mark mime_type as 'audio/webm'.
    """
    if not audio_bytes:
        return None

    try:
        # Prepare audio as a Part
        audio_part = Part.from_bytes(
            data=audio_bytes,
            mime_type="audio/webm",  # works for mic_recorder in most browsers
        )

        resp = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                audio_part,
                "Transcribe the speaker's answer into clear English text. "
                "Do not add extra commentary, just the transcription."
            ],
            config=GenerateContentConfig(
                response_mime_type="text/plain"
            ),
        )

        text = resp.text or ""
        text = text.strip()
        return text if text else None

    except Exception as e:
        # Surface the error in Streamlit so we can see it in the UI/logs
        st.error(f"Error while calling Gemini for transcription: {e}")
        return None
