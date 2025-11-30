# question_generator.py

from typing import Dict, List
from llm_client import call_gemini_json

SYSTEM_PROMPT_QUESTIONS = """
You are an AI Interview Question Designer.

Given:
- core_technical_skills
- soft_skills
- experience_level

You MUST generate an interview plan with:
- behavioral_questions: exactly 3 questions
- technical_questions: exactly 3 questions (only meaningful for technical roles)
- culture_fit_questions: exactly 2 questions

Rules:
- Each question must be a clear, single sentence.
- Do NOT leave any list empty.
- Tailor difficulty to experience_level (intern/fresher/junior/mid/senior).
- Questions should be realistic for HR interviews.

Return ONLY valid JSON with keys exactly:
{
  "behavioral_questions": [.. 3 items ..],
  "technical_questions": [.. 3 items ..],
  "culture_fit_questions": [.. 2 items ..]
}
"""


def generate_interview_plan(jd_info: Dict) -> Dict:
    """
    Create an interview plan based on parsed JD info.
    """
    user_prompt = f"""
JD info:
{jd_info}
"""
    plan = call_gemini_json(SYSTEM_PROMPT_QUESTIONS, user_prompt)
    return plan


def flatten_questions(plan: Dict) -> List[str]:
    """
    Turn the plan into a simple ordered list of questions:
    - Behavioral first
    - Then Technical
    - Then Culture-fit
    """
    behavioral = plan.get("behavioral_questions", []) or []
    technical = plan.get("technical_questions", []) or []
    culture = plan.get("culture_fit_questions", []) or []

    all_questions: List[str] = []
    all_questions.extend(behavioral)
    all_questions.extend(technical)
    all_questions.extend(culture)

    all_questions = [q for q in all_questions if isinstance(q, str) and q.strip()]
    return all_questions


def build_rounds(jd_info: Dict, plan: Dict) -> List[Dict]:
    """
    Build structured rounds based on role type + question groups.

    Rounds:
    - Round 1: Behavioral (always if questions available)
    - Round 2: Technical (only if role is technical AND questions exist)
    - Round 3: Culture Fit (if questions exist)
    """
    core_tech = jd_info.get("core_technical_skills", []) or []
    # Simple heuristic: if core tech skills exist -> technical role
    is_technical_role = len(core_tech) > 0

    rounds: List[Dict] = []

    behavioral = plan.get("behavioral_questions", []) or []
    technical = plan.get("technical_questions", []) or []
    culture = plan.get("culture_fit_questions", []) or []

    if behavioral:
        rounds.append(
            {
                "key": "behavioral",
                "name": "Behavioral Round",
                "questions": [q for q in behavioral if isinstance(q, str) and q.strip()],
            }
        )

    if is_technical_role and technical:
        rounds.append(
            {
                "key": "technical",
                "name": "Technical Round",
                "questions": [q for q in technical if isinstance(q, str) and q.strip()],
            }
        )

    if culture:
        rounds.append(
            {
                "key": "culture_fit",
                "name": "Culture Fit / HR Round",
                "questions": [q for q in culture if isinstance(q, str) and q.strip()],
            }
        )

    # Drop any empty rounds, just in case
    rounds = [r for r in rounds if r["questions"]]

    return rounds
