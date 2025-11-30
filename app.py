# app.py
import json
from datetime import datetime

import streamlit as st

from file_utils import extract_text_from_pdf
from resume_matcher import analyze_resume, match_resume_to_jd
from jd_analyzer import analyze_job_description
from question_generator import generate_interview_plan, build_rounds
from evaluator import evaluate_answer
from report_generator import generate_candidate_report, generate_candidate_feedback
from audio_stt import transcribe_audio_bytes
from streamlit_mic_recorder import mic_recorder

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="AI Voice Interview Agent", layout="wide")


# ---------- Helper: make the agent speak the question ----------
def speak_text(text: str, key: str):
    """
    Use browser's SpeechSynthesis (Web Speech API) to speak text on the client side.
    We mark a 'spoken_key' in session_state to avoid re-speaking the same question on every rerun.
    """
    spoken_flag_key = f"spoken_{key}"
    if st.session_state.get(spoken_flag_key):
        return

    st.session_state[spoken_flag_key] = True

    escaped = json.dumps(text)
    st.markdown(
        f"""
        <script>
        const text = {escaped};
        if ("speechSynthesis" in window) {{
            const msg = new SpeechSynthesisUtterance(text);
            msg.rate = 1;
            msg.pitch = 1;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        }}
        </script>
        """,
        unsafe_allow_html=True,
    )


# ---------- Session State Initialization ----------
if "stage" not in st.session_state:
    # stages: onboarding -> analysis -> interview -> results
    st.session_state.stage = "onboarding"

if "profile" not in st.session_state:
    st.session_state.profile = None  # {name, company, role, experience_level}

if "resume_bytes" not in st.session_state:
    st.session_state.resume_bytes = None

if "jd_text" not in st.session_state:
    st.session_state.jd_text = ""

if "resume_info" not in st.session_state:
    st.session_state.resume_info = None

if "jd_info" not in st.session_state:
    st.session_state.jd_info = None

if "match_report" not in st.session_state:
    st.session_state.match_report = None

if "plan" not in st.session_state:
    st.session_state.plan = None

if "rounds" not in st.session_state:
    # List of { key, name, questions }
    st.session_state.rounds = []

if "current_round_index" not in st.session_state:
    st.session_state.current_round_index = 0  # which round candidate is on

if "question_index_in_round" not in st.session_state:
    st.session_state.question_index_in_round = 0  # which question within the round

if "evaluations" not in st.session_state:
    # List of { "round_key", "round_name", "question", "answer", "evaluation", "timestamp" }
    st.session_state.evaluations = []

if "interview_finished" not in st.session_state:
    st.session_state.interview_finished = False

if "candidate_started" not in st.session_state:
    st.session_state.candidate_started = False

# Threshold to go to next round (avg overall_impression out of 10)
ROUND_PASS_THRESHOLD = 6.0


# ---------- Reset helper ----------
def reset_everything():
    keys_to_reset = [
        "stage",
        "profile",
        "resume_bytes",
        "jd_text",
        "resume_info",
        "jd_info",
        "match_report",
        "plan",
        "rounds",
        "current_round_index",
        "question_index_in_round",
        "evaluations",
        "interview_finished",
        "candidate_started",
    ]
    for k in keys_to_reset:
        if k in st.session_state:
            del st.session_state[k]
    st.session_state.stage = "onboarding"


st.sidebar.title("⚙️ Controls")
if st.sidebar.button("🔁 Restart Interview Flow"):
    reset_everything()
    st.rerun()


# ======================================================================
#                           STAGE 1: ONBOARDING
# ======================================================================
def render_onboarding():
    st.title("🎙️ AI Voice Interview Agent")

    st.markdown(
        """
This agent will:
- Analyze your **resume** and the **job description**
- Build a **custom interview** based on the company and role
- Conduct the interview using **voice only** (HR-style questions)
- Evaluate your skills and give you a **feedback report**
        """
    )

    col1, col2 = st.columns(2)
    with col1:
        name = st.text_input("Full Name")
        email = st.text_input("Email (optional)")
        company = st.text_input("Company Name")
    with col2:
        role = st.text_input("Role Applying For")
        experience = st.selectbox(
            "Experience Level",
            ["Fresher", "Junior", "Mid", "Senior"],
        )

    resume_file = st.file_uploader("Upload Your Resume (PDF)", type=["pdf"])
    jd_text = st.text_area(
        "Paste the Job Description (JD)",
        height=220,
        placeholder="Paste the JD you are applying for...",
    )

    if st.button("Start Voice Interview Setup"):
        if not name or not company or not role or not resume_file or not jd_text.strip():
            st.error("Please fill all the required fields and upload your resume & JD.")
        else:
            st.session_state.profile = {
                "name": name,
                "email": email,
                "company": company,
                "role": role,
                "experience": experience,
            }
            st.session_state.resume_bytes = resume_file.read()
            st.session_state.jd_text = jd_text
            st.session_state.stage = "analysis"
            st.rerun()


# ======================================================================
#                           STAGE 2: ANALYSIS
# ======================================================================
def run_analysis():
    st.title("⚙️ Setting up your interview...")

    with st.spinner("Analyzing your resume and the job description..."):
        # Extract resume text
        resume_text = extract_text_from_pdf(st.session_state.resume_bytes)
        if not resume_text:
            st.error(
                "Could not read text from your resume PDF. "
                "Please upload a text-based PDF (not scanned image)."
            )
            if st.button("Back"):
                reset_everything()
                st.rerun()
            return

        # Analyze resume & JD
        resume_info = analyze_resume(resume_text)
        jd_info = analyze_job_description(st.session_state.jd_text)
        match_report = match_resume_to_jd(jd_info, resume_info)

        # Build interview plan & rounds based on JD
        plan = generate_interview_plan(jd_info)
        rounds = build_rounds(jd_info, plan)

        if not rounds:
            st.error(
                "The system could not build appropriate interview rounds for this JD. "
                "Try a different JD or simplify the description."
            )
            if st.button("Back"):
                reset_everything()
                st.rerun()
            return

        # Save to session
        st.session_state.resume_info = resume_info
        st.session_state.jd_info = jd_info
        st.session_state.match_report = match_report
        st.session_state.plan = plan
        st.session_state.rounds = rounds

        # Reset interview state
        st.session_state.current_round_index = 0
        st.session_state.question_index_in_round = 0
        st.session_state.evaluations = []
        st.session_state.interview_finished = False
        st.session_state.candidate_started = False

    # Move to interview stage
    st.session_state.stage = "interview"
    st.rerun()


# ======================================================================
#                           STAGE 3: INTERVIEW (VOICE)
# ======================================================================
def render_interview():
    profile = st.session_state.profile
    jd_info = st.session_state.jd_info
    match_report = st.session_state.match_report
    rounds = st.session_state.rounds

    name = profile.get("name", "Candidate")
    company = profile.get("company", "the company")
    role = profile.get("role", "this role")

    st.title("🎧 Voice Interview In Progress")

    total_rounds = len(rounds)

    # Intro screen before starting
    if not st.session_state.candidate_started:
        st.markdown(
            f"""
Hi **{name}**,  
We’ve analyzed your **resume** and the **{role}** role at **{company}**.

Now we’ll start a **voice-only interview** tailored to:
- The company’s JD
- Your skills and projects
- Your experience level
            """
        )

        if match_report:
            scores = match_report.get("scores", {})
            st.write("### Resume ↔ JD Match (Before Interview)")
            st.write(
                f"**Overall Fit (from resume only):** "
                f"{scores.get('overall_fit_score', 0)} / 100"
            )
            st.write(
                f"**Skill Match:** {scores.get('skill_match_score', 0)} / 100, "
                f"**Experience Fit:** {scores.get('experience_fit_score', 0)} / 100"
            )

        st.write("### Rounds in this Interview")
        for idx, rnd in enumerate(rounds, start=1):
            st.write(f"- **Round {idx}:** {rnd['name']} ({len(rnd['questions'])} questions)")

        st.info(
            "The agent will **speak each question**, and you will **answer using your microphone**.\n"
            "You can review the transcribed text briefly before submitting."
        )

        if st.button("🎙️ Start Voice Interview"):
            st.session_state.candidate_started = True
            st.session_state.interview_finished = False
            st.session_state.current_round_index = 0
            st.session_state.question_index_in_round = 0
            st.session_state.evaluations = []
            st.rerun()
        return

    # Main interview flow
    if st.session_state.candidate_started and not st.session_state.interview_finished:
        round_idx = st.session_state.current_round_index

        if round_idx >= total_rounds:
            st.session_state.interview_finished = True
            st.session_state.stage = "results"
            st.rerun()
            return

        current_round = rounds[round_idx]
        round_key = current_round["key"]
        round_name = current_round["name"]
        questions = current_round["questions"]
        total_q_in_round = len(questions)

        q_idx = st.session_state.question_index_in_round

        # Finished all Qs in this round: decide pass/fail
        if q_idx >= total_q_in_round:
            scores = []
            for ev in st.session_state.evaluations:
                if ev["round_key"] == round_key:
                    score = ev["evaluation"].get("scores", {}).get(
                        "overall_impression", None
                    )
                    if score is not None:
                        scores.append(score)

            avg_score = sum(scores) / len(scores) if scores else 0.0

            if avg_score >= ROUND_PASS_THRESHOLD and round_idx < total_rounds - 1:
                st.success(
                    f"You passed **{round_name}** with an average score of {avg_score:.2f}/10. "
                    "We will now move to the next round."
                )
                st.session_state.current_round_index += 1
                st.session_state.question_index_in_round = 0
                st.rerun()
                return
            else:
                if round_idx < total_rounds - 1:
                    st.error(
                        f"You did not meet the threshold to proceed beyond **{round_name}** "
                        f"(average score {avg_score:.2f}/10). The interview ends here."
                    )
                else:
                    st.success(
                        f"You have completed the final round (**{round_name}**). "
                        "Thank you for attending the interview."
                    )
                st.session_state.interview_finished = True
                st.session_state.stage = "results"
                st.rerun()
                return

        # Ask next question (voice)
        current_question = questions[q_idx]

        st.markdown(
            f"**{round_name}** — Question {q_idx + 1} of {total_q_in_round}"
        )
        st.markdown(current_question)

        # Make agent speak the question
        speak_text(
            current_question,
            key=f"round{round_idx}_q{q_idx}",
        )

        st.write("🎙️ Speak your answer below (click to start/stop):")

        audio = mic_recorder(
            start_prompt="Start recording",
            stop_prompt="Stop recording",
            key=f"mic_{round_idx}_{q_idx}",
        )

        # Keep transcript in session
        transcript_key = f"transcript_{round_idx}_{q_idx}"
        if transcript_key not in st.session_state:
            st.session_state[transcript_key] = ""

        if audio and audio.get("bytes"):
            with st.spinner("Transcribing your answer..."):
                try:
                    text = transcribe_audio_bytes(audio["bytes"])
                    if text:
                        st.session_state[transcript_key] = text
                        st.success(
                            "Transcription complete. You can quickly review your answer below."
                        )
                    else:
                        st.error(
                            "Could not transcribe audio. Please try speaking again."
                        )
                except Exception as e:
                    st.error(f"Error during transcription: {e}")

        # Show recognized text (editing optional)
        answer = st.text_area(
            "Transcribed answer (optional to edit before submitting):",
            value=st.session_state[transcript_key],
            height=160,
            key=f"answer_box_{round_idx}_{q_idx}",
        )

        button_label = (
            "Submit & Next Question"
            if q_idx < total_q_in_round - 1
            else "Submit & Finish This Round"
        )

        if st.button(
            button_label,
            key=f"candidate_submit_{round_idx}_{q_idx}",
        ):
            final_answer = answer.strip()
            if not final_answer:
                st.error(
                    "Please speak your answer and wait for transcription before submitting."
                )
            else:
                with st.spinner("Saving your answer and evaluating..."):
                    try:
                        eval_result = evaluate_answer(
                            question=current_question,
                            answer_transcript=final_answer,
                            answer_duration_seconds=None,
                            filler_word_count=None,
                            role_title=jd_info.get("role_title"),
                        )

                        st.session_state.evaluations.append(
                            {
                                "round_key": round_key,
                                "round_name": round_name,
                                "question": current_question,
                                "answer": final_answer,
                                "evaluation": eval_result,
                                "timestamp": datetime.now().isoformat(),
                            }
                        )

                        st.success("Your answer has been recorded.")
                        st.session_state.question_index_in_round += 1
                        st.rerun()

                    except Exception as e:
                        st.error(f"Error while evaluating your answer: {e}")

    # If finished, move to results
    if st.session_state.interview_finished:
        st.session_state.stage = "results"
        st.rerun()


# ======================================================================
#                           STAGE 4: RESULTS
# ======================================================================
def render_results():
    profile = st.session_state.profile
    name = profile.get("name", "Candidate")
    role = profile.get("role", "this role")
    company = profile.get("company", "the company")

    st.title("📊 Interview Results & Feedback")

    if not st.session_state.evaluations:
        st.info("No evaluations found. It seems the interview did not run fully.")
        if st.button("Restart"):
            reset_everything()
            st.rerun()
        return

    st.success(
        f"Your voice interview for **{role}** at **{company}** is complete, {name}."
    )

    # Candidate-facing feedback
    with st.spinner("Generating your personalized feedback report..."):
        try:
            candidate_feedback = generate_candidate_feedback(
                role_title=st.session_state.jd_info.get("role_title", role),
                evaluations=st.session_state.evaluations,
            )
        except Exception as e:
            candidate_feedback = None
            st.error(f"Could not generate candidate feedback: {e}")

    if candidate_feedback:
        st.write("### 🧑‍🎓 Overall Summary")
        st.write(candidate_feedback.get("summary", ""))

        st.write("### ✅ What You Did Well")
        for s in candidate_feedback.get("strengths", []):
            st.write("- ", s)

        st.write("### 📌 Where You Can Improve")
        for w in candidate_feedback.get("improvement_areas", []):
            st.write("- ", w)

        st.write("### 🚀 Suggested Practice & Next Steps")
        for a in candidate_feedback.get("suggested_actions", []):
            st.write("- ", a)

    # HR-style report in expander
    with st.expander("🧑‍💼 HR-style Report (Hire / Reject decision)", expanded=False):
        with st.spinner("Generating HR-style summary..."):
            try:
                hr_report = generate_candidate_report(
                    role_title=st.session_state.jd_info.get("role_title", role),
                    evaluations=st.session_state.evaluations,
                )

                st.write("### Recommendation")
                st.write(f"**Verdict:** {hr_report.get('recommendation', 'unknown')}")
                st.write(hr_report.get("final_verdict_line", ""))

                st.write("### Why (Top Reasons)")
                for r in hr_report.get("recommendation_reasons", []):
                    st.write("- ", r)

                st.write("### Strengths (HR View)")
                for s in hr_report.get("strengths", []):
                    st.write("- ", s)

                st.write("### Weaknesses (HR View)")
                for w in hr_report.get("weaknesses", []):
                    st.write("- ", w)

                st.write("### Aggregated Scores")
                st.json(hr_report.get("aggregated_scores", {}))

            except Exception as e:
                st.error(f"Could not generate HR report: {e}")

    if st.button("🔁 Start New Interview"):
        reset_everything()
        st.rerun()


# ======================================================================
#                           MAIN ROUTER
# ======================================================================
stage = st.session_state.stage

if stage == "onboarding":
    render_onboarding()
elif stage == "analysis":
    run_analysis()
elif stage == "interview":
    render_interview()
elif stage == "results":
    render_results()
else:
    st.error("Unknown stage. Resetting...")
    reset_everything()
    st.rerun()
