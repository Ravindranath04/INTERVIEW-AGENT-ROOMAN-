# evaluator.py
from typing import Dict, Any, Optional
from llm_client import call_gemini_json

SYSTEM_PROMPT_EVAL = """
You are an HR Interview Evaluator.

You will receive:
- the question asked
- the candidate's transcribed answer (plain text)
- optional metadata: answer_duration_seconds, filler_word_count

You must:
1. Extract a STAR breakdown:
   - situation
   - task
   - action
   - result

2. Score these dimensions from 0 to 10:
   - relevance   (did they answer the question?)
   - content_depth (quality and detail of example)
   - star_completeness (how clear S, T, A, R are)
   - role_skill_match (how well it demonstrates skills relevant to the role)
   - grammar (clarity, correctness, fluency)
   - confidence (based on language, directness, any hints from metadata)
   - overall_impression (HR gut rating)

3. Provide short feedback:
   - strengths: list of 1-3 bullet points
   - areas_to_improve: list of 1-3 bullet points

You MUST return valid JSON with exactly this structure:

{
  "star": {
    "situation": "...",
    "task": "...",
    "action": "...",
    "result": "..."
  },
  "scores": {
    "relevance": 0-10,
    "content_depth": 0-10,
    "star_completeness": 0-10,
    "role_skill_match": 0-10,
    "grammar": 0-10,
    "confidence": 0-10,
    "overall_impression": 0-10
  },
  "feedback": {
    "strengths": ["...", "..."],
    "areas_to_improve": ["...", "..."]
  },
  "summary_comment": "One short HR-style summary sentence."
}
"""


def evaluate_answer(
    question: str,
    answer_transcript: str,
    *,
    answer_duration_seconds: Optional[float] = None,
    filler_word_count: Optional[int] = None,
    role_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single answer using Gemini.
    Includes grammar & confidence.
    """

    metadata_text_parts = []
    if answer_duration_seconds is not None:
        metadata_text_parts.append(f"answer_duration_seconds: {answer_duration_seconds}")
    if filler_word_count is not None:
        metadata_text_parts.append(f"filler_word_count: {filler_word_count}")
    if role_title:
        metadata_text_parts.append(f"role_title: {role_title}")

    metadata_block = "\n".join(metadata_text_parts) if metadata_text_parts else "None"

    user_prompt = f"""
QUESTION:
\"\"\"{question}\"\"\"

CANDIDATE ANSWER (TRANSCRIPT):
\"\"\"{answer_transcript}\"\"\"

METADATA:
{metadata_block}
"""

    result = call_gemini_json(SYSTEM_PROMPT_EVAL, user_prompt)
    return result
