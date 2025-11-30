# llm_client.py
import os
import json
from typing import Any, Dict

from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env or environment variables.")

# Configure Gemini client
client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"


def call_gemini_json(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """
    Call Gemini and force a strict JSON response.
    """
    full_prompt = f"""
You are a strict JSON generator.

SYSTEM INSTRUCTION:
{system_prompt}

USER INPUT:
{user_prompt}

Rules:
- Return ONLY valid JSON.
- No extra commentary or text outside JSON.
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )

    text = (response.text or "").strip()

    # Try to parse directly
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to trim to outermost JSON object
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Gemini did not return valid JSON. Raw response:\n" + text
        )
