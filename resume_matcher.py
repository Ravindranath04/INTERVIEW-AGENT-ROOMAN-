# resume_matcher.py
from typing import Dict, Any
from llm_client import call_gemini_json

SYSTEM_PROMPT_RESUME_ANALYSIS = """
You are an experienced technical recruiter.

You will receive a candidate's resume text.
Extract:

- headline: 1-line summary of the candidate
- years_of_experience: approximate (string like "0", "1-2", "5+")
- core_technical_skills: list
- secondary_skills: list
- soft_skills: list
- key_projects: list of 2–5 short project descriptions
- roles_and_domains: list of previous roles/domains

Return STRICT JSON with:

{
  "headline": "",
  "years_of_experience": "",
  "core_technical_skills": [],
  "secondary_skills": [],
  "soft_skills": [],
  "key_projects": [],
  "roles_and_domains": []
}
"""


def analyze_resume(resume_text: str) -> Dict[str, Any]:
    user_prompt = f"""
RESUME TEXT:
\"\"\"{resume_text}\"\"\"
"""
    return call_gemini_json(SYSTEM_PROMPT_RESUME_ANALYSIS, user_prompt)


SYSTEM_PROMPT_RESUME_MATCH = """
You are an HR specialist evaluating how well a candidate's resume matches a job description.

You will receive:
- jd_info: parsed JD info
- resume_info: parsed resume info

You must:

1) Compute scores (0–100):
- skill_match_score
- experience_fit_score
- overall_fit_score

2) Identify:
- strong_matches: skills/areas where candidate matches or exceeds JD
- missing_critical_skills: 3–7 important skills/tools JD expects but resume doesn't show clearly
- optional_nice_to_have_skills: skills mentioned in JD but not mandatory
- overindexed_areas: areas where candidate is strong but not heavily needed in this JD

3) Candidate-facing:
- candidate_summary: 2–3 lines in friendly tone
- candidate_improvement_tips: 3–5 bullet points suggesting what to learn or highlight

4) HR-facing:
- hr_risk_flags: 2–4 bullet points of HR concerns
- hr_overall_comment: 1–2 lines HR summary

Return STRICT JSON:

{
  "scores": {
    "skill_match_score": 0-100,
    "experience_fit_score": 0-100,
    "overall_fit_score": 0-100
  },
  "strong_matches": [],
  "missing_critical_skills": [],
  "optional_nice_to_have_skills": [],
  "overindexed_areas": [],
  "candidate_summary": "",
  "candidate_improvement_tips": [],
  "hr_risk_flags": [],
  "hr_overall_comment": ""
}
"""


def match_resume_to_jd(jd_info: Dict[str, Any], resume_info: Dict[str, Any]) -> Dict[str, Any]:
    user_prompt = f"""
JD INFO:
{jd_info}

RESUME INFO:
{resume_info}
"""
    return call_gemini_json(SYSTEM_PROMPT_RESUME_MATCH, user_prompt)
