# question_generator.py
from typing import Dict, List
from llm_client import call_gemini_json

SYSTEM_PROMPT_QUESTIONS = """
You are an AI Interview Question Designer.

Given:
- core_technical_skills
- soft_skills
- experience_level

Generate an interview plan with:
- behavioral_questions: list of 3-5 questions
- technical_questions: list of 3-5 questions
- culture_fit_questions: list of 2-3 questions

Each question should be a simple string.
Target level should match experience_level.
Return JSON with keys exactly:
{
  "behavioral_questions": [..],
  "technical_questions": [..],
  "culture_fit_questions": [..]
}
"""


def generate_interview_plan(jd_info: Dict) -> Dict:
    user_prompt = f"""
JD info:
{jd_info}
"""
    plan = call_gemini_json(SYSTEM_PROMPT_QUESTIONS, user_prompt)
    return plan


def get_next_question(
    asked_questions: List[str],
    plan: Dict,
    last_answer_score: float | None = None,
) -> str | None:
    """
    Very simple logic:
    - Ask behavioral questions first
    - Then technical
    - Then culture-fit
    - last_answer_score can later be used to adapt difficulty
    """
    all_questions: List[str] = (
        plan.get("behavioral_questions", [])
        + plan.get("technical_questions", [])
        + plan.get("culture_fit_questions", [])
    )

    for q in all_questions:
        if q not in asked_questions:
            return q

    return None  # no more questions
