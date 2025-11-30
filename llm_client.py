# llm_client.py
import os
import json
from typing import Any, Dict
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Load .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env or environment variables.")

client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.0-flash"


def call_gemini(prompt: str, *, temperature: float = 0.4) -> str:
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )
    return response.text


def call_gemini_json(system_prompt: str, user_prompt: str, *, temperature: float = 0.3) -> Dict[str, Any]:
    full_prompt = f"""
You are a strict JSON generator. Follow ALL these rules:
1. Output ONLY valid JSON.
2. Do not include any explanation or extra text.
3. Make sure quotes, commas, and braces are correct.

SYSTEM INSTRUCTION:
{system_prompt}

USER INPUT:
{user_prompt}
"""

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
        ),
    )

    text = response.text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except json.JSONDecodeError:
                pass
        raise ValueError(f"Gemini did not return valid JSON:\n{text}")
