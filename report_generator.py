# report_generator.py
from typing import List, Dict, Any
from llm_client import call_gemini_json

SYSTEM_PROMPT_REPORT = """
You are an HR summarizer.

You will receive:
- role_title
- list of per-answer evaluations, each containing:
  - question
  - star breakdown
  - scores
  - feedback

You must:
1. Compute overall category scores (0-10) for:
   - technical_skill
   - behavioral_skill
   - communication_and_grammar
   - confidence
   - culture_fit (if possible)
   - overall_recommendation_score

2. Classify recommendation as one of:
   ["strong_hire", "hire", "hold", "reject"]

3. Write:
   - strengths: 3 bullet points
   - weaknesses: 3 bullet points
   - final_verdict_line: one sentence HR verdict.

Return strict JSON with the following structure:

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
    evaluations: list of dicts, each like:
    {
      "question": "...",
      "evaluation": { ... output of evaluate_answer ... }
    }
    """
    user_prompt = f"""
role_title: {role_title}

per_answer_evaluations:
{evaluations}
"""

    report = call_gemini_json(SYSTEM_PROMPT_REPORT, user_prompt)
    return report
