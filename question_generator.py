# question_generator.py
from typing import Dict, List, Any
from llm_client import call_gemini_json

SYSTEM_PROMPT_QUESTIONS = """
You are an experienced HR interviewer designing a realistic interview process.

You will receive:
- Parsed JD info (role_title, experience_level, core_technical_skills, soft_skills, role_type, responsibilities).

Design an interview flow that matches how REAL companies hire for this role.
Your job is to create 1–4 rounds, NOT a fixed pattern.

Examples of possible rounds:
- "Initial HR Screening"
- "Technical Deep Dive"
- "Problem-Solving & Projects"
- "Culture Fit with Manager"
- "Domain Knowledge Round"

Rules:

1) Number of rounds:
   - For interns/freshers: 1–2 rounds.
   - For junior: 1–3 rounds.
   - For mid/senior: 2–4 rounds.
   - For non-technical roles: focus more on behavioral, domain and culture; technical rounds only if relevant tools/skills exist in the JD.

2) Each round must include:
   - round_key: short snake_case id, e.g. "hr_screening", "technical_round_1".
   - round_name: human-friendly name, e.g. "Technical Round with Hiring Manager".
   - round_type: one of ["behavioral", "technical", "mixed", "culture", "domain"].
   - questions: 3–8 questions that fit that round_type and the JD.

3) Question style:
   - Sound like a real HR / hiring manager speaking to a candidate.
   - Use warm, professional wording.
   - Connect questions to the role_title and core_technical_skills when appropriate.
   - For technical roles, include both conceptual and practical/project-based questions.
   - For non-technical roles, focus on scenarios, behavior, ownership, stakeholder handling, etc.

4) Do NOT generate the same question twice.
   Do NOT generate questions that are irrelevant to the JD.

Return STRICT JSON:

{
  "rounds": [
    {
      "round_key": "hr_screening",
      "round_name": "Initial HR Screening",
      "round_type": "behavioral",
      "questions": ["...", "..."]
    },
    {
      "round_key": "technical_round_1",
      "round_name": "Technical Round with Hiring Manager",
      "round_type": "technical",
      "questions": ["...", "..."]
    }
  ]
}
"""


def generate_interview_plan(jd_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ask the LLM to design a realistic, JD-based interview process.
    """
    user_prompt = f"""
JD INFO:
{jd_info}
"""
    return call_gemini_json(SYSTEM_PROMPT_QUESTIONS, user_prompt)


def build_rounds(jd_info: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Convert the LLM-designed plan into the format used by app.py.

    We expect:
    plan = {
      "rounds": [
        {
          "round_key": "...",
          "round_name": "...",
          "round_type": "...",
          "questions": [...]
        },
        ...
      ]
    }

    We will normalize into:
    [
      { "key": round_key, "name": round_name, "questions": [...] },
      ...
    ]
    """
    raw_rounds = plan.get("rounds", []) or []
    rounds: List[Dict[str, Any]] = []

    for r in raw_rounds:
        round_key = r.get("round_key") or r.get("key")
        round_name = r.get("round_name") or r.get("name") or (round_key or "Interview Round")
        questions = r.get("questions") or []

        # Clean up questions
        clean_questions = [
            q for q in questions
            if isinstance(q, str) and q.strip()
        ]

        if not round_key:
            # Derive a fallback key from name
            round_key = round_name.lower().replace(" ", "_")

        if clean_questions:
            rounds.append(
                {
                    "key": round_key,
                    "name": round_name,
                    "questions": clean_questions,
                }
            )

    return rounds
