# app.py
import streamlit as st

from jd_analyzer import analyze_job_description
from question_generator import generate_interview_plan, get_next_question
from evaluator import evaluate_answer
from report_generator import generate_candidate_report

# ---------- Streamlit Page Config ----------

st.set_page_config(page_title="AI HR Interview Agent", layout="wide")

# ---------- Session State Initialization ----------

if "jd_info" not in st.session_state:
    st.session_state.jd_info = None

if "plan" not in st.session_state:
    st.session_state.plan = None

if "asked_questions" not in st.session_state:
    st.session_state.asked_questions = []

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

if "current_question" not in st.session_state:
    st.session_state.current_question = None

if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False

# ---------- UI Layout ----------

st.title("🎙️ AI HR Interview Agent (Gemini-based)")

tab_hr, tab_candidate, tab_report = st.tabs(
    ["HR Setup", "Candidate Interview", "HR Report"]
)

# ---------- HR SETUP TAB ----------

with tab_hr:
    st.subheader("Step 1: Paste Job Description")

    jd_text = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the JD here...",
    )

    if st.button("Analyze JD and Generate Interview Plan"):
        if not jd_text.strip():
            st.error("Please paste a Job Description first.")
        else:
            with st.spinner("Analyzing JD and creating interview plan..."):
                try:
                    jd_info = analyze_job_description(jd_text)
                    plan = generate_interview_plan(jd_info)

                    st.session_state.jd_info = jd_info
                    st.session_state.plan = plan
                    st.session_state.asked_questions = []
                    st.session_state.evaluations = []
                    st.session_state.current_question = None
                    st.session_state.interview_finished = False

                    st.success("JD analyzed and interview plan created successfully.")
                    st.write("### Parsed JD Info")
                    st.json(jd_info)

                    st.write("### Generated Interview Plan")
                    st.json(plan)
                except Exception as e:
                    st.error(f"Error while analyzing JD or creating plan: {e}")

# ---------- CANDIDATE INTERVIEW TAB ----------

with tab_candidate:
    st.subheader("Step 2: Conduct Interview")

    if st.session_state.jd_info is None or st.session_state.plan is None:
        st.info("HR must first configure the JD and plan in the 'HR Setup' tab.")
    else:
        # Initialize first question if needed
        if (
            not st.session_state.current_question
            and not st.session_state.interview_finished
        ):
            next_q = get_next_question(
                st.session_state.asked_questions,
                st.session_state.plan,
                last_answer_score=None,
            )
            st.session_state.current_question = next_q

        # If we still have a question to ask
        if st.session_state.current_question and not st.session_state.interview_finished:
            st.markdown(f"**Question:** {st.session_state.current_question}")

            answer = st.text_area(
                "Your answer (candidate):",
                height=180,
                placeholder="Speak or type your answer here (voice integration can be added later)...",
            )

            col1, col2 = st.columns(2)
            with col1:
                answer_duration = st.number_input(
                    "Approx. duration of your answer in seconds (optional)",
                    min_value=0.0,
                    value=0.0,
                    step=1.0,
                )
            with col2:
                filler_count = st.number_input(
                    "Approx. number of filler words like 'um', 'uh' (optional)",
                    min_value=0,
                    value=0,
                    step=1,
                )

            if st.button("Submit Answer & Evaluate"):
                if not answer.strip():
                    st.error("Please enter an answer before submitting.")
                else:
                    with st.spinner("Evaluating answer..."):
                        try:
                            eval_result = evaluate_answer(
                                question=st.session_state.current_question,
                                answer_transcript=answer,
                                answer_duration_seconds=(
                                    answer_duration if answer_duration > 0 else None
                                ),
                                filler_word_count=(
                                    filler_count if filler_count > 0 else None
                                ),
                                role_title=st.session_state.jd_info.get("role_title"),
                            )

                            # Save evaluation
                            st.session_state.asked_questions.append(
                                st.session_state.current_question
                            )
                            st.session_state.evaluations.append(
                                {
                                    "question": st.session_state.current_question,
                                    "answer": answer,
                                    "evaluation": eval_result,
                                }
                            )

                            st.write("### Evaluation for this Answer")
                            st.json(eval_result)

                            # Use overall impression as a simple score for adaptiveness
                            overall_score = (
                                eval_result.get("scores", {}).get(
                                    "overall_impression", None
                                )
                            )

                            next_q = get_next_question(
                                st.session_state.asked_questions,
                                st.session_state.plan,
                                last_answer_score=overall_score,
                            )

                            if next_q is None:
                                st.success("No more questions. Interview finished.")
                                st.session_state.current_question = None
                                st.session_state.interview_finished = True
                            else:
                                st.session_state.current_question = next_q

                        except Exception as e:
                            st.error(f"Error during evaluation: {e}")

        elif st.session_state.interview_finished:
            st.success("Interview has been completed.")
            st.write(
                "You can now go to the **HR Report** tab to generate the final summary."
            )

# ---------- HR REPORT TAB ----------

with tab_report:
    st.subheader("Step 3: View HR Summary Report")

    if not st.session_state.evaluations:
        st.info("No evaluations yet. Conduct at least one Q&A in the Candidate tab.")
    else:
        if st.button("Generate Final HR Report"):
            with st.spinner("Generating HR report..."):
                try:
                    report = generate_candidate_report(
                        role_title=st.session_state.jd_info.get(
                            "role_title", "Unknown Role"
                        ),
                        evaluations=st.session_state.evaluations,
                    )

                    st.write("## Candidate HR Report")
                    st.json(report)

                except Exception as e:
                    st.error(f"Error while generating HR report: {e}")

        st.write("### Per-question Evaluations (Raw Data)")
        st.json(st.session_state.evaluations)
