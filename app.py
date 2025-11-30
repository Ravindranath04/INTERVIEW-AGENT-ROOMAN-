# app.py
import json
from datetime import datetime

import streamlit as st

from jd_analyzer import analyze_job_description
from question_generator import (
    generate_interview_plan,
    flatten_questions,
    build_rounds,
)
from evaluator import evaluate_answer
from report_generator import generate_candidate_report
from audio_stt import transcribe_audio_bytes
from streamlit_mic_recorder import mic_recorder

# ---------- Streamlit Page Config ----------
st.set_page_config(page_title="AI HR Interview Agent", layout="wide")


# ---------- Small helper: make agent speak the question ----------
def speak_text(text: str, key: str):
    """
    Use browser's SpeechSynthesis (Web Speech API) to speak text on the client side.
    We track a key so we don't re-speak same question on every rerun.
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
if "jd_info" not in st.session_state:
    st.session_state.jd_info = None

if "plan" not in st.session_state:
    st.session_state.plan = None

if "all_questions" not in st.session_state:
    st.session_state.all_questions = []  # flat list, mainly for HR view

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

# Threshold to go to next round (avg overall impression out of 10)
ROUND_PASS_THRESHOLD = 6.0

# ---------- Sidebar: Role Selection ----------
st.sidebar.title("🧑‍💼 Role Selection")
user_role = st.sidebar.radio("Who is using this interface?", ["HR", "Candidate"])

st.title("🎙️ AI HR Interview Agent (Voice-based)")

# ======================================================================
#                           HR INTERFACE
# ======================================================================
if user_role == "HR":
    tab_hr_setup, tab_hr_report = st.tabs(
        ["⚙️ HR Setup (JD & Rounds)", "📊 HR Reports"]
    )

    # ---------- HR SETUP TAB ----------
    with tab_hr_setup:
        st.subheader("Step 1: Configure Interview Rounds for the Role")

        jd_text = st.text_area(
            "Job Description",
            height=220,
            placeholder="Paste the Job Description here...",
        )

        if st.button("Analyze JD and Generate Rounds", key="hr_generate_plan"):
            if not jd_text.strip():
                st.error("Please paste a Job Description first.")
            else:
                with st.spinner("Analyzing JD and creating interview plan..."):
                    try:
                        jd_info = analyze_job_description(jd_text)
                        plan = generate_interview_plan(jd_info)
                        questions = flatten_questions(plan)
                        rounds = build_rounds(jd_info, plan)

                        if not rounds:
                            st.error(
                                "Could not build any valid rounds. Try refining the JD or retry."
                            )
                        else:
                            st.session_state.jd_info = jd_info
                            st.session_state.plan = plan
                            st.session_state.all_questions = questions
                            st.session_state.rounds = rounds

                            # Reset candidate state
                            st.session_state.current_round_index = 0
                            st.session_state.question_index_in_round = 0
                            st.session_state.evaluations = []
                            st.session_state.interview_finished = False
                            st.session_state.candidate_started = False

                            st.success(
                                f"JD analyzed and {len(rounds)} interview round(s) created."
                            )

                            st.write("### Parsed JD Info")
                            st.json(jd_info)

                            st.write("### Generated Interview Plan (Structured)")
                            st.json(plan)

                            st.write("### Rounds & Questions")
                            st.json(rounds)

                            st.info(
                                "Now share this app with the candidate in 'Candidate' mode. "
                                "They will progress round by round based on performance."
                            )
                    except Exception as e:
                        st.error(f"Error while analyzing JD or creating plan: {e}")

        # Show current config if available
        if st.session_state.jd_info is not None and st.session_state.rounds:
            st.write("### Current Active JD")
            st.json(st.session_state.jd_info)

            st.write("### Current Rounds")
            st.json(st.session_state.rounds)

            st.write("### Flat Question List (all rounds)")
            st.json(st.session_state.all_questions)

    # ---------- HR REPORT TAB ----------
    with tab_hr_report:
        st.subheader("Step 2: View Candidate's Evaluation Report")

        if not st.session_state.evaluations:
            st.info(
                "No candidate responses yet. Ask the candidate to use the 'Candidate' interface "
                "and complete at least one round."
            )
        else:
            total_questions = sum(len(r["questions"]) for r in st.session_state.rounds)
            answered_q = len(st.session_state.evaluations)
            st.write(
                f"✅ Candidate answered {answered_q} out of ~{total_questions} questions."
            )

            # Per-round average scores
            st.write("### Per-Round Performance (avg overall impression)")

            round_scores = {}
            for ev in st.session_state.evaluations:
                r_key = ev["round_key"]
                score = ev["evaluation"].get("scores", {}).get(
                    "overall_impression", None
                )
                if score is not None:
                    round_scores.setdefault(r_key, []).append(score)

            for rnd in st.session_state.rounds:
                r_key = rnd["key"]
                scores = round_scores.get(r_key, [])
                if scores:
                    avg = sum(scores) / len(scores)
                    st.write(f"- **{rnd['name']}**: {avg:.2f} / 10 ({len(scores)} Qs)")
                else:
                    st.write(f"- **{rnd['name']}**: no scored answers")

            if st.button("Generate Final HR Report", key="hr_generate_report"):
                with st.spinner("Generating overall HR report..."):
                    try:
                        report = generate_candidate_report(
                            role_title=st.session_state.jd_info.get(
                                "role_title", "Unknown Role"
                            ),
                            evaluations=st.session_state.evaluations,
                        )

                        st.write("## Candidate HR Report (Overall Fit)")
                        st.json(report)

                    except Exception as e:
                        st.error(f"Error while generating HR report: {e}")

            st.write("### Detailed Per-question Evaluations (with rounds & timestamps)")
            st.json(st.session_state.evaluations)

    st.sidebar.info(
        "HR View:\n\n"
        "- Configure JD & rounds in 'HR Setup'\n"
        "- Candidate uses 'Candidate' role to attend interview (voice)\n"
        "- They progress to further rounds only if their previous round score is high enough\n"
        "- View aggregated evaluation in 'HR Reports'"
    )

# ======================================================================
#                        CANDIDATE INTERFACE (VOICE)
# ======================================================================
else:  # user_role == "Candidate"
    st.subheader("Candidate Interview (Voice)")

    if not st.session_state.rounds:
        st.warning(
            "The interview is not yet configured.\n\n"
            "Please ask the HR to first set up the Job Description and "
            "generate interview rounds in the HR interface."
        )
    else:
        total_rounds = len(st.session_state.rounds)

        # Intro screen
        if not st.session_state.candidate_started:
            info_lines = [f"There are {total_rounds} possible round(s) in this interview."]
            for idx, rnd in enumerate(st.session_state.rounds, start=1):
                info_lines.append(f"- Round {idx}: {rnd['name']}")

            st.info(
                "Welcome! This is an automated HR interview.\n\n"
                + "\n".join(info_lines)
                + "\n\nYou must perform well in each round to proceed to the next.\n"
                "Questions will be spoken aloud and your answers will be recorded via microphone."
            )
            if st.button("Start Interview", key="candidate_start"):
                st.session_state.candidate_started = True
                st.session_state.interview_finished = False
                st.session_state.current_round_index = 0
                st.session_state.question_index_in_round = 0
                st.session_state.evaluations = []
                st.rerun()

        # Main interview flow
        if st.session_state.candidate_started and not st.session_state.interview_finished:
            round_idx = st.session_state.current_round_index

            if round_idx >= total_rounds:
                st.session_state.interview_finished = True
            else:
                current_round = st.session_state.rounds[round_idx]
                round_key = current_round["key"]
                round_name = current_round["name"]
                questions = current_round["questions"]
                total_q_in_round = len(questions)

                q_idx = st.session_state.question_index_in_round

                # Finished all Qs in this round: decide pass/fail for next round
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
                            "You will now move to the next round."
                        )
                        st.session_state.current_round_index += 1
                        st.session_state.question_index_in_round = 0
                        st.rerun()
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

                else:
                    # ---- Ask current question (voice) and record candidate via mic ----
                    current_question = questions[q_idx]

                    st.markdown(
                        f"**{round_name}** — Question {q_idx + 1} of {total_q_in_round}"
                    )
                    st.markdown(current_question)

                    # Make agent speak question (HR voice via browser TTS)
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

                    # Keep transcript in session so it persists
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

                    # Show the recognized text (editing is optional)
                    answer = st.text_area(
                        "Transcribed answer (optional to edit):",
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
                                        role_title=st.session_state.jd_info.get(
                                            "role_title"
                                        ),
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
                                    st.error(
                                        f"Error while evaluating your answer: {e}"
                                    )

        # Completion message + candidate feedback
        if st.session_state.interview_finished and st.session_state.candidate_started:
            st.success(
                "Your interview is completed. HR will review your performance "
                "round by round.\n\nHere is a brief feedback summary to help you improve:"
            )

            if st.session_state.evaluations:
                # Lazy import to avoid circular import issues
                from report_generator import generate_candidate_feedback

                with st.spinner("Generating your feedback summary..."):
                    try:
                        candidate_feedback = generate_candidate_feedback(
                            role_title=st.session_state.jd_info.get(
                                "role_title", "this role"
                            ),
                            evaluations=st.session_state.evaluations,
                        )

                        st.write("### Overall Summary")
                        st.write(candidate_feedback.get("summary", ""))

                        st.write("### What You Did Well")
                        for s in candidate_feedback.get("strengths", []):
                            st.write("- ", s)

                        st.write("### Where You Can Improve")
                        for w in candidate_feedback.get("improvement_areas", []):
                            st.write("- ", w)

                        st.write("### Suggested Practice & Next Steps")
                        for a in candidate_feedback.get("suggested_actions", []):
                            st.write("- ", a)

                    except Exception as e:
                        st.error(f"Could not generate feedback right now: {e}")

    st.sidebar.info(
        "Candidate View (Voice):\n\n"
        "- Agent speaks each question.\n"
        "- You answer using your microphone.\n"
        "- Your responses are evaluated in the background and summarized at the end."
    )
