# report_generator.py

from typing import List, Dict, Any
from llm_client import call_gemini_json

# ---------- HR-FACING REPORT (Hire / Reject + Reasons) ----------

SYSTEM_PROMPT_REPORT = """
You are an HR manager summarizing an interview.

You will receive:
- role_title
- list of per-answer evaluations, each containing:
  - round_key
  - round_name
  - question
  - answer
  - evaluation (with scores, feedback, hr_comment, star)
  - timestamp

Your tasks:

1. Compute overall category scores (0-10) for:
   - technical_skill
   - behavioral_skill
   - communication_and_grammar
   - confidence
   - culture_fit
   - overall_recommendation_score

   Use the underlying per-answer scores:
   - role_skill_match, content_depth -> technical_skill
   - relevance, star_completeness -> behavioral_skill
   - grammar -> communication_and_grammar
   - confidence -> confidence
   - answers to culture/culture-fit questions -> culture_fit
   - overall_impression -> overall_recommendation_score

2. Decide a recommendation as one of:
   ["strong_hire", "hire", "hold", "reject"]

3. Explain the recommendation in 2–3 bullet points from an HR point of view,
   e.g. "Rejected because technical depth is too low for this role".

4. Write:
   - strengths: 3 bullet points (what this candidate does well)
   - weaknesses: 3 bullet points (where they may struggle)
   - final_verdict_line: one sentence HR verdict, e.g.
     "Overall verdict: Reject – communication is good but core technical skills are insufficient for a Backend Engineer role."

Return STRICT JSON with the following structure and nothing else:

{
  "aggregated_scores": {
    "technical_skill": 0-10,
    "behavioral_skill": 0-10,
    "communication_and_grammar": 0-10,
    "confidence": 0-10,
    "culture_fit": 0-10,
    "overall_recommendation_score": 0-10
  },
  "recommendation": "strong_hire" | "hire" | "hold" | "reject",
  "recommendation_reasons": ["...", "..."],
  "strengths": ["...", "..."],
  "weaknesses": ["...", "..."],
  "final_verdict_line": "..."
}
"""


def generate_candidate_report(
    role_title: str,
    evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    HR-facing report generator.
    Uses all per-question evaluations to produce:
    - aggregated scores
    - hire/hold/reject
    - reasons, strengths, weaknesses, verdict line
    """
    user_prompt = f"""
role_title: {role_title}

per_answer_evaluations:
{evaluations}
"""
    report = call_gemini_json(SYSTEM_PROMPT_REPORT, user_prompt)
    return report


# ---------- CANDIDATE-FACING FEEDBACK (How to Improve) ----------

SYSTEM_PROMPT_CANDIDATE_FEEDBACK = """
You are a friendly career coach.

You will receive:
- role_title
- list of per-answer evaluations, each containing:
  - question
  - answer
  - evaluation (with scores, feedback)

Your job is to create a short, clear feedback report for the candidate:
- Use positive, motivating tone.
- Do NOT mention numeric scores.
- Do NOT say "you were rejected" or "you were hired".
- Focus on:
  1) What they are doing well
  2) Where they can improve
  3) Concrete suggestions to practice and improve.

Return STRICT JSON with:

{
  "summary": "2-3 lines summarizing their overall performance in a kind, honest way.",
  "strengths": ["...", "..."],
  "improvement_areas": ["...", "..."],
  "suggested_actions": ["...", "..."]
}
"""


def generate_candidate_feedback(
    role_title: str,
    evaluations: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Candidate-facing feedback generator.
    Gives them:
    - brief summary
    - strengths
    - areas to improve
    - concrete next steps
    """
    user_prompt = f"""
role_title: {role_title}

per_answer_evaluations:
{evaluations}
"""
    feedback = call_gemini_json(SYSTEM_PROMPT_CANDIDATE_FEEDBACK, user_prompt)
    return feedback
