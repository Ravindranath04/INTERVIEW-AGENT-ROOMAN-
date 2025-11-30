# jd_analyzer.py

from llm_client import call_gemini_json

SYSTEM_PROMPT_JD = """
You analyse job descriptions for an AI interview agent.

Given a job description text, you must extract:
- core_technical_skills: list of strings
- secondary_technical_skills: list of strings
- soft_skills: list of strings
- experience_level: one of ["intern", "fresher", "junior", "mid", "senior"]
- role_title: short string
- summary: 2-3 line summary in plain English.

Return a JSON object with exactly these keys.
"""


def analyze_job_description(jd_text: str):
    user_prompt = f"""
Job Description:
\"\"\"{jd_text}\"\"\"
"""
    result = call_gemini_json(SYSTEM_PROMPT_JD, user_prompt)
    return result
