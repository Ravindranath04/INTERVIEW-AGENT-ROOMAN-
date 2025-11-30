# app.py
import json
from datetime import datetime
import io
import wave

import numpy as np
import pandas as pd
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
st.set_page_config(
    page_title="AI Voice Interview Agent",
    layout="wide",
    page_icon="🎧",
)


# ---------- Helper: make the agent speak the question ----------
def speak_text(text: str, key: str):
    """
    Use browser's SpeechSynthesis to speak the question.

    - Tries to auto-speak once per question (may be blocked by autoplay rules).
    - Always renders a 🔊 button that the user can click to play/replay the question.
    """
    # Decide whether to try auto-speak this rerun
    spoken_flag_key = f"spoken_{key}"
    auto_speak = False
    if not st.session_state.get(spoken_flag_key):
        st.session_state[spoken_flag_key] = True
        auto_speak = True  # first time for this question

    escaped = json.dumps(text)
    # Make sure the JS function name is a valid identifier
    safe_key = key.replace("-", "_").replace(" ", "_")

    st.markdown(
        f"""
        <script>
        const questionText_{safe_key} = {escaped};

        function speakQuestion_{safe_key}() {{
            if (!("speechSynthesis" in window)) {{
                alert("Speech synthesis is not supported in this browser.");
                return;
            }}
            const msg = new SpeechSynthesisUtterance(questionText_{safe_key});
            msg.rate = 1;
            msg.pitch = 1;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(msg);
        }}

        // Try auto-speak only once per question (may be blocked by browser)
        if ({str(auto_speak).lower()}) {{
            // small delay so page finishes rendering
            setTimeout(speakQuestion_{safe_key}, 300);
        }}
        </script>

        <div style="margin: 0.5rem 0 0.8rem 0;">
            <button
                type="button"
                onclick="speakQuestion_{safe_key}()"
                style="
                    border-radius: 999px;
                    border: 1px solid #38bdf8;
                    padding: 4px 12px;
                    background: rgba(15,23,42,0.9);
                    color: #e5e7eb;
                    font-size: 0.8rem;
                    cursor: pointer;
                "
            >
                🔊 Play question again
            </button>
        </div>
        """,
        unsafe_allow_html=True,
    )



# ---------- Voice level helper ----------
def compute_voice_level(audio_bytes: bytes) -> float:
    """
    Compute a simple voice 'intensity' level (0.0 - 1.0) from WAV audio bytes.
    This is based on RMS amplitude, not true pitch.
    """
    try:
        with wave.open(io.BytesIO(audio_bytes)) as wf:
            frames = wf.readframes(wf.getnframes())
            if not frames:
                return 0.0
            # Assuming 16-bit PCM
            audio_array = np.frombuffer(frames, dtype=np.int16)
            if audio_array.size == 0:
                return 0.0
            rms = np.sqrt(np.mean(audio_array.astype(np.float64) ** 2))
            norm_level = min(rms / 32768.0, 1.0)
            return float(norm_level)
    except Exception:
        return 0.0


# ---------- Dynamic round threshold based on experience ----------
def get_round_pass_threshold() -> float:
    """
    Decide how strict the round pass threshold should be
    based on candidate experience level.
    """
    profile = st.session_state.get("profile") or {}
    exp = (profile.get("experience") or "").lower()

    if "fresher" in exp or "intern" in exp:
        return 5.0
    elif "junior" in exp:
        return 6.0
    elif "mid" in exp:
        return 7.0
    elif "senior" in exp:
        return 8.0
    else:
        # default if unknown
        return 6.0


# ---------- Session State Initialization ----------
if "stage" not in st.session_state:
    # stages: onboarding -> analysis -> interview -> results
    st.session_state.stage = "onboarding"

if "profile" not in st.session_state:
    st.session_state.profile = None  # {name, company, role, experience_level, email}

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

if "high_contrast" not in st.session_state:
    st.session_state.high_contrast = True


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


# ---------- Sidebar ----------
st.sidebar.title("📋 Interview Controls")

stage_to_step = {"onboarding": 1, "analysis": 2, "interview": 3, "results": 4}
current_step = stage_to_step.get(st.session_state.stage, 1)
progress_ratio = current_step / 4.0

st.sidebar.markdown("#### Flow Progress")
st.sidebar.progress(progress_ratio)
st.sidebar.markdown(
    f"""
- **1. Candidate details** {'✅' if current_step > 1 else '⬤'}
- **2. Analysis & setup** {'✅' if current_step > 2 else '⬤'}
- **3. Voice interview** {'✅' if current_step > 3 else '⬤'}
- **4. Results & feedback** {'✅' if current_step > 4 else '⬤'}
"""
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Display")
hc_toggle = st.sidebar.toggle(
    "High contrast mode",
    value=st.session_state.high_contrast,
    help="Improves text visibility on dark backgrounds.",
    key="hc_toggle",
)
st.session_state.high_contrast = hc_toggle

if st.sidebar.button("🔁 Restart Interview Flow"):
    reset_everything()
    st.rerun()

st.sidebar.markdown("### ℹ️ Best practices")
st.sidebar.caption(
    "- Use a quiet environment and a clear microphone.\n"
    "- Wait for the question to finish before speaking.\n"
    "- You can edit the transcript before submitting your answer."
)


# ---------- Global styling (uses high-contrast toggle) ----------
main_text_color = "#e5e7eb" if st.session_state.high_contrast else "#9ca3af"
heading_color = "#f9fafb" if st.session_state.high_contrast else "#e5e7eb"
secondary_text = "#cbd5f5" if st.session_state.high_contrast else "#9ca3af"

st.markdown(
    f"""
    <style>
    /* Overall app background */
    .stApp {{
        background: radial-gradient(circle at top left, #020617 0, #020617 40%, #000 100%);
        color: {main_text_color} !important;
    }}

    /* Main container width tweak */
    .block-container {{
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }}

    /* Typography */
    h1, h2, h3, h4 {{
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        color: {heading_color} !important;
    }}

    body, .stApp, p, li {{
        color: {main_text_color} !important;
    }}

    small, .stCaption, label {{
        color: {secondary_text} !important;
    }}

    /* Glass cards */
    .glass-card {{
        background: rgba(15, 23, 42, 0.82);
        border-radius: 18px;
        padding: 1.25rem 1.5rem;
        border: 1px solid rgba(148, 163, 184, 0.35);
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.9);
    }}

    .section-label {{
        font-size: 0.8rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {secondary_text} !important;
        font-weight: 600;
        margin-bottom: 0.25rem;
    }}

    .tag-pill {{
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.7rem;
        background: rgba(15,23,42,0.9);
        border: 1px solid rgba(148,163,184,0.55);
        color: {main_text_color};
    }}

    .pill-dot {{
        width: 6px;
        height: 6px;
        border-radius: 999px;
        background: #22c55e;
    }}

    /* Buttons */
    .stButton>button {{
        border-radius: 999px;
        padding: 0.45rem 1.3rem;
        font-weight: 600;
        border: 1px solid rgba(148,163,184,0.5);
        background: radial-gradient(circle at top left, #1d4ed8, #4f46e5);
        color: white;
    }}
    .stButton>button:hover {{
        filter: brightness(1.08);
        border-color: #93c5fd;
    }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #020617 0%, #020617 50%, #020617 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.35);
    }}

    section[data-testid="stSidebar"] * {{
        color: {main_text_color} !important;
    }}

    /* Metric labels */
    [data-testid="stMetricLabel"] {{
        color: {secondary_text} !important;
    }}
    [data-testid="stMetricValue"] {{
        color: #ffffff !important;
    }}

    /* Mic recorder button styling */
    .mic-wrapper button {{
        border-radius: 999px !important;
        padding: 0.4rem 1.3rem !important;
        font-weight: 600 !important;
        border: 1px solid rgba(34,197,94,0.95) !important;
        background: radial-gradient(circle at top left, #22c55e, #16a34a) !important;
        color: #f9fafb !important;
        cursor: pointer !important;
        transition: transform 0.08s ease-out, box-shadow 0.08s ease-out, filter 0.08s !important;
        box-shadow: 0 10px 25px rgba(22,163,74,0.45) !important;
    }}

    .mic-wrapper button:hover {{
        filter: brightness(1.06) !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 14px 30px rgba(22,163,74,0.55) !important;
    }}

    .mic-wrapper button:active {{
        transform: translateY(0px) scale(0.99) !important;
        box-shadow: 0 8px 20px rgba(22,163,74,0.45) !important;
    }}

    /* Entrance animations */
    @keyframes fadeInUp {{
        from {{
            opacity: 0;
            transform: translateY(8px);
        }}
        to {{
            opacity: 1;
            transform: translateY(0);
        }}
    }}

    .fade-in {{
        animation: fadeInUp 0.45s ease-out;
    }}

    .fade-in-delayed {{
        animation: fadeInUp 0.6s ease-out;
    }}

    @media (prefers-reduced-motion: reduce) {{
        .fade-in, .fade-in-delayed {{
            animation: none !important;
        }}
    }}

    /* Improve readability inside cards */
    .glass-card * {{
        color: {main_text_color} !important;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ======================================================================
#                           STAGE 1: ONBOARDING
# ======================================================================
def render_onboarding():
    # Hero header
    st.markdown(
        """
        <div class="glass-card fade-in">
            <div class="section-label">AI Voice Interview Agent</div>
            <h1 style="margin-bottom:0.3rem;">Practice job interviews, end-to-end.</h1>
            <p style="max-width:680px; margin-top:0.35rem; font-size:0.95rem;">
                Upload your resume, paste a real job description, and experience an AI-driven, voice-based interview.
                At the end, you receive a structured HR-style summary and concrete skill feedback.
            </p>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-top:0.9rem;">
                <span class="tag-pill"><span class="pill-dot"></span>Resume–JD matching</span>
                <span class="tag-pill"><span class="pill-dot"></span>Dynamic multi-round flow</span>
                <span class="tag-pill"><span class="pill-dot"></span>Voice-only Q&A</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.markdown('<div class="section-label">Step 1</div>', unsafe_allow_html=True)
        st.subheader("Candidate details")

        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full name", key="name", placeholder="Alex Johnson")
            email = st.text_input("Email (optional)", key="email", placeholder="alex@email.com")
            company = st.text_input("Target company", key="company", placeholder="Acme Corp")
        with col2:
            role = st.text_input("Role applying for", key="role", placeholder="Backend Engineer")
            experience = st.selectbox(
                "Experience level",
                ["Fresher", "Junior", "Mid", "Senior"],
                key="experience",
            )

        st.markdown("")
        st.markdown('<div class="section-label">Step 2</div>', unsafe_allow_html=True)
        st.subheader("Resume & job description")

        resume_file = st.file_uploader(
            "Upload your resume (PDF)",
            type=["pdf"],
            key="resume_file",
        )

        jd_text = st.text_area(
            "Paste the job description (JD)",
            height=220,
            placeholder="Paste the full job description you want to practise for...",
            key="jd_text_input",
        )

        st.markdown(
            "<small>Files are processed in-memory only for this demo session.</small>",
            unsafe_allow_html=True,
        )

        st.markdown("")
        c1, c2 = st.columns([1, 3])
        with c1:
            start_clicked = st.button("▶ Start interview setup")
        with c2:
            st.caption("We will first analyse your resume and the JD to design a tailored interview plan.")

        if start_clicked:
            missing = []

            if not name.strip():
                missing.append("Full name")
            if not company.strip():
                missing.append("Target company")
            if not role.strip():
                missing.append("Role applying for")
            if not resume_file:
                missing.append("Resume (PDF)")
            if not jd_text.strip():
                missing.append("Job description")

            if missing:
                st.error("Please complete the following before continuing: " + ", ".join(missing))
                return

            # ✅ SAFELY STORE FILE ONCE
            resume_bytes = resume_file.read()
            if not resume_bytes:
                st.error("The uploaded resume appears to be empty. Please upload a valid PDF.")
                return

            # ✅ SAVE INTO SESSION STATE
            st.session_state.profile = {
                "name": name,
                "email": email,
                "company": company,
                "role": role,
                "experience": experience,
            }

            st.session_state.resume_bytes = resume_bytes
            st.session_state.jd_text = jd_text

            st.success("Resume and job description captured successfully.")
            st.session_state.stage = "analysis"
            st.rerun()

    with col_right:
        st.markdown('<div class="glass-card fade-in-delayed">', unsafe_allow_html=True)
        st.markdown("#### How this demo behaves")
        st.markdown(
            """
            - Uses your **resume and JD** to infer required skills.  
            - Generates **round-wise questions** similar to real hiring flows.  
            - Conducts a **voice-only interview** using your microphone.  
            - Produces an **HR-style recommendation** plus detailed feedback.
            """
        )
        st.markdown("---")
        st.markdown("#### Quick checklist")
        checklist_items = [
            ("Quiet environment", True),
            ("Stable internet connection", True),
            ("Working microphone", True),
        ]
        for label, _ in checklist_items:
            st.markdown(f"- ✅ {label}")
        st.markdown("</div>", unsafe_allow_html=True)

    # Developer debug info (kept but tucked away)
    with st.expander("Debug information (development only)"):
        st.write({
            "name": bool(st.session_state.get("name")),
            "company": bool(st.session_state.get("company")),
            "role": bool(st.session_state.get("role")),
            "experience": bool(st.session_state.get("experience")),
            "resume uploaded": st.session_state.get("resume_file") is not None,
            "jd pasted": bool(st.session_state.get("jd_text_input", "").strip()),
        })


# ======================================================================
#                           STAGE 2: ANALYSIS
# ======================================================================
def run_analysis():
    st.markdown(
        """
        <div class="glass-card fade-in">
            <div class="section-label">Step 2</div>
            <h2 style="margin-bottom:0.25rem;">Analysing profile & generating rounds</h2>
            <p style="margin-top:0;">
                The agent is reading your resume, understanding the job description, and assembling realistic interview rounds.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("")
    with st.spinner("Running resume and JD analysis..."):
        # Extract resume text
        resume_text = extract_text_from_pdf(st.session_state.resume_bytes)
        if not resume_text:
            st.error(
                "We could not extract text from your resume. "
                "Please upload a text-based PDF (not only a scanned image)."
            )
            if st.button("⬅ Back to start"):
                reset_everything()
                st.rerun()
            return

        # Analyze resume & JD
        resume_info = analyze_resume(resume_text)
        jd_info = analyze_job_description(st.session_state.jd_text)
        match_report = match_resume_to_jd(jd_info, resume_info)

        # Build interview plan & dynamic rounds based on JD (realistic company-style flow)
        plan = generate_interview_plan(jd_info)
        rounds = build_rounds(jd_info, plan)

        if not rounds:
            st.error(
                "The system was not able to design suitable interview rounds for this JD. "
                "Please try a different or slightly simplified job description."
            )
            if st.button("⬅ Back to start"):
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

    st.success("Interview rounds are ready. You can now begin the voice interview.")
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

    st.markdown(
        """
        <div class="glass-card fade-in">
            <div class="section-label">Step 3</div>
            <h2 style="margin-bottom:0.25rem;">Voice interview in progress</h2>
            <p style="margin-top:0;">
                The agent will read each question aloud. Respond using your microphone, then review and refine the transcript before submitting.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    total_rounds = len(rounds)

    # Intro screen before starting
    if not st.session_state.candidate_started:
        col_intro_left, col_intro_right = st.columns([2, 1])

        with col_intro_left:
            st.markdown(
                f"""
                **Welcome, {name}.**  
                We analysed your resume and the **{role}** role at **{company}** to design a focused interview.

                This session is optimised to:
                - Reflect the **skills and tools** mentioned in the JD  
                - Leverage your **projects and experience** from the resume  
                - Adapt expectations to your **experience level**  
                """
            )

            if match_report:
                scores = match_report.get("scores", {})
                st.markdown("### Pre-interview resume ↔ JD fit")
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric(
                        "Overall fit",
                        f"{scores.get('overall_fit_score', 0)} / 100",
                    )
                with col_b:
                    st.metric(
                        "Skill alignment",
                        f"{scores.get('skill_match_score', 0)} / 100",
                    )
                with col_c:
                    st.metric(
                        "Experience fit",
                        f"{scores.get('experience_fit_score', 0)} / 100",
                    )

        with col_intro_right:
            st.markdown("### Interview structure")
            for idx, rnd in enumerate(rounds, start=1):
                st.markdown(
                    f"- **Round {idx}:** {rnd['name']}  ·  {len(rnd['questions'])} questions"
                )

            threshold = get_round_pass_threshold()
            st.info(
                "Each round is scored on a 10-point scale.\n\n"
                f"For this profile, the pass threshold is approximately **{threshold:.1f}/10** on average."
            )

        st.markdown("---")
        if st.button("Start voice interview"):
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
            threshold = get_round_pass_threshold()

            if avg_score >= threshold and round_idx < total_rounds - 1:
                st.success(
                    f"You passed **{round_name}** with an average score of {avg_score:.2f}/10. "
                    "The next round will now begin."
                )
                st.session_state.current_round_index += 1
                st.session_state.question_index_in_round = 0
                st.rerun()
                return
            else:
                if round_idx < total_rounds - 1:
                    st.error(
                        f"You did not meet the threshold to progress beyond **{round_name}** "
                        f"(average score {avg_score:.2f}/10 vs threshold {threshold:.1f}). "
                        "The interview concludes here."
                    )
                else:
                    st.success(
                        f"You have completed the final round (**{round_name}**). "
                        "Thank you for participating in this interview simulation."
                    )
                st.session_state.interview_finished = True
                st.session_state.stage = "results"
                st.rerun()
                return

        # Ask next question (voice)
        current_question = questions[q_idx]

        top_col_1, top_col_2 = st.columns([3, 1])
        with top_col_1:
            st.markdown(
                f"#### {round_name} · Question {q_idx + 1} of {total_q_in_round}"
            )
        with top_col_2:
            round_progress = (q_idx) / float(total_q_in_round)
            st.progress(round_progress)

        st.markdown(
            f"""
            <div class="glass-card fade-in" style="margin-top:0.5rem;">
                <div class="section-label">Question</div>
                <p style="font-size:1.02rem; margin-bottom:0;">{current_question}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Make agent speak the question
        speak_text(
            current_question,
            key=f"round{round_idx}_q{q_idx}",
        )

        st.write("🎙️ Click to start recording your answer, then click again to stop:")

        col_mic, col_level = st.columns([1, 2])

        voice_level_key = f"voice_level_{round_idx}_{q_idx}"
        if voice_level_key not in st.session_state:
            st.session_state[voice_level_key] = 0.0

        with col_mic:
            st.markdown('<div class="mic-wrapper">', unsafe_allow_html=True)
            audio = mic_recorder(
                start_prompt="Start recording",
                stop_prompt="Stop recording",
                key=f"mic_{round_idx}_{q_idx}",
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with col_level:
            st.markdown("##### Voice level (last answer)")
            st.progress(st.session_state[voice_level_key])

        # Keep transcript in session
        transcript_key = f"transcript_{round_idx}_{q_idx}"
        if transcript_key not in st.session_state:
            st.session_state[transcript_key] = ""

        if audio and audio.get("bytes"):
            with st.spinner("Transcribing your answer..."):
                try:
                    audio_bytes = audio["bytes"]

                    # Compute voice level (0–1) from audio
                    level = compute_voice_level(audio_bytes)
                    st.session_state[voice_level_key] = level

                    text = transcribe_audio_bytes(audio_bytes)
                    if text:
                        st.session_state[transcript_key] = text
                        st.success(
                            "Transcription complete. You may refine the text below before submitting."
                        )
                    else:
                        st.error(
                            "We could not transcribe the audio. Please try recording again."
                        )
                except Exception as e:
                    st.error(f"Error during transcription: {e}")

        st.markdown("#### Review your answer")
        answer = st.text_area(
            "Transcribed answer (you can edit this before submission):",
            value=st.session_state[transcript_key],
            height=160,
            key=f"answer_box_{round_idx}_{q_idx}",
        )

        button_label = (
            "Submit answer & go to next question"
            if q_idx < total_q_in_round - 1
            else "Submit answer & complete this round"
        )

        st.markdown("")
        if st.button(
            button_label,
            key=f"candidate_submit_{round_idx}_{q_idx}",
        ):
            final_answer = answer.strip()
            if not final_answer:
                st.error(
                    "Please record and transcribe your answer before submitting."
                )
            else:
                with st.spinner("Storing your answer and running evaluation..."):
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
                        st.error(
                            f"Error while evaluating your answer: {e}"
                        )

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

    st.markdown(
        """
        <div class="glass-card fade-in">
            <div class="section-label">Step 4</div>
            <h2 style="margin-bottom:0.25rem;">Interview results & personalised feedback</h2>
            <p style="margin-top:0;">
                Below is a consolidated view of your performance, including strengths, improvement areas,
                and an HR-style recommendation.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not st.session_state.evaluations:
        st.info("No evaluations were recorded. It appears the interview did not run to completion.")
        if st.button("Start a new interview"):
            reset_everything()
            st.rerun()
        return

    st.success(
        f"The voice interview for **{role}** at **{company}** is complete. Well done, {name}."
    )

    # Candidate-facing feedback
    with st.spinner("Generating your candidate-facing feedback report..."):
        try:
            candidate_feedback = generate_candidate_feedback(
                role_title=st.session_state.jd_info.get("role_title", role),
                evaluations=st.session_state.evaluations,
            )
        except Exception as e:
            candidate_feedback = None
            st.error(f"Could not generate candidate feedback: {e}")

    if candidate_feedback:
        col_summary, col_side = st.columns([2.3, 1])
        with col_summary:
            st.markdown("### Overall summary")
            st.write(candidate_feedback.get("summary", ""))

        with col_side:
            st.markdown("### ✅ Suggested next steps")
            st.markdown(
                """
                <div class="glass-card fade-in-delayed">
                <ul style="font-size:0.9rem; padding-left:1.1rem; margin-bottom:0;">
                    <li>Revisit topics highlighted as weaker</li>
                    <li>Practise concise, structured responses</li>
                    <li>Run another simulation with a different JD</li>
                </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.markdown("### Key strengths")
        for s in candidate_feedback.get("strengths", []):
            st.write("- ", s)

        st.markdown("### Areas to improve")
        for w in candidate_feedback.get("improvement_areas", []):
            st.write("- ", w)

        st.markdown("### Concrete practice suggestions")
        for a in candidate_feedback.get("suggested_actions", []):
            st.write("- ", a)

    # HR-style report + score chart
    with st.expander("HR-style report (Hire / Hold / Reject signal)", expanded=False):
        with st.spinner("Generating HR-oriented summary..."):
            try:
                hr_report = generate_candidate_report(
                    role_title=st.session_state.jd_info.get("role_title", role),
                    evaluations=st.session_state.evaluations,
                )

                st.write("### Recommendation")
                st.write(f"**Verdict:** {hr_report.get('recommendation', 'unknown')}")
                st.write(hr_report.get("final_verdict_line", ""))

                st.write("### Rationale")
                for r in hr_report.get("recommendation_reasons", []):
                    st.write("- ", r)

                st.write("### Strengths (HR view)")
                for s in hr_report.get("strengths", []):
                    st.write("- ", s)

                st.write("### Concerns / risks")
                for w in hr_report.get("weaknesses", []):
                    st.write("- ", w)

                scores_dict = hr_report.get("aggregated_scores", {})
                if isinstance(scores_dict, dict) and scores_dict:
                    st.write("### Score overview")
                    df_scores = pd.DataFrame(
                        {"Dimension": list(scores_dict.keys()), "Score": list(scores_dict.values())}
                    ).set_index("Dimension")
                    st.bar_chart(df_scores)

                    st.write("Raw score data")
                    st.json(scores_dict)

            except Exception as e:
                st.error(f"Could not generate HR report: {e}")

    st.markdown("---")
    if st.button("🔁 Start a new interview"):
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
