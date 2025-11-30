from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Dict, Any

from jd_analyzer import analyze_job_description
from question_generator import generate_interview_plan
from evaluator import evaluate_answer
from report_generator import generate_candidate_report

app = FastAPI()


class JDRequest(BaseModel):
    jd_text: str


class PlanRequest(BaseModel):
    jd_info: Dict[str, Any]


class EvaluateRequest(BaseModel):
    question: str
    answer_transcript: str
    answer_duration_seconds: float | None = None
    filler_word_count: int | None = None
    role_title: str | None = None


class ReportRequest(BaseModel):
    role_title: str
    evaluations: List[Dict[str, Any]]


@app.post("/analyze_jd")
def api_analyze_jd(body: JDRequest):
    jd_info = analyze_job_description(body.jd_text)
    return jd_info


@app.post("/generate_plan")
def api_generate_plan(body: PlanRequest):
    plan = generate_interview_plan(body.jd_info)
    return plan


@app.post("/evaluate_answer")
def api_evaluate_answer(body: EvaluateRequest):
    result = evaluate_answer(
        question=body.question,
        answer_transcript=body.answer_transcript,
        answer_duration_seconds=body.answer_duration_seconds,
        filler_word_count=body.filler_word_count,
        role_title=body.role_title,
    )
    return result


@app.post("/generate_report")
def api_generate_report(body: ReportRequest):
    report = generate_candidate_report(
        role_title=body.role_title,
        evaluations=body.evaluations,
    )
    return report
