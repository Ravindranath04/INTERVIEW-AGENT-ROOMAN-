
# 🎙️ AI  Interview Agent

An AI-powered **HR interviewer** that conducts company-style interviews using **voice**, evaluates candidate performance, and generates feedback and HR hiring reports.

Live interview flow is dynamically created based on:
- Resume content
- Job description (JD)
- Experience level
- Company expectations

---

## 🚀 Features

### Candidate Experience
- Upload resume (PDF)
- Paste job description (JD)
- AI interviews via **voice**
- One-question-at-a-time flow
- Resume-based, skill-based and project-based questions
- Auto speech transcription
- Human-style interview conversation
- Coaching-style feedback:
  - Strengths
  - Improvement areas
  - Learning suggestions

### HR Experience
- Dynamic interview rounds (no fixed flows)
- Hiring recommendation:
  - ✅ Hire
  - ⚠️ Hold
  - ❌ Reject
- HR-style justification
- Skill scoring across rounds

---

## 🧠 How it Works

```

Resume + JD
│
▼
AI Analysis Engine
│
▼
Dynamic Interview Plan (Rounds + Questions)
│
▼
Voice Interview (Speech ↔ Text)
│
▼
Evaluation Engine
│
▼
Candidate & HR Reports

```

---

## 🧩 Tech Stack

| Layer | Tools |
|--------|------|
| UI | Streamlit |
| Audio Recording | streamlit-mic-recorder |
| Speech Synthesis | Web Speech API |
| Speech to Text | Gemini AI |
| LLM | Google Gemini |
| Resume Parsing | PyPDF2 |
| Deployment | Streamlit Cloud |

---

## 📁 Project Structure

```

AI_INTERVIEW_AGENT/
├── app.py
├── audio_stt.py
├── file_utils.py
├── jd_analyzer.py
├── resume_matcher.py
├── question_generator.py
├── evaluator.py
├── report_generator.py
├── llm_client.py
├── requirements.txt
└── README.md

````

---

## ⚙️ Installation (Local Setup)

### 1) Clone repository
```bash
git clone https://github.com/<your-username>/INTERVIEW-AGENT-ROOMAN-.git
cd INTERVIEW-AGENT-ROOMAN-
````

### 2) Create virtual environment (optional)

```bash
python -m venv venv
venv\Scripts\activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Set up API key

Create `.env` file:

```env
GEMINI_API_KEY=your_api_key_here
```

### 5) Run app

```bash
streamlit run app.py
```

---

## 🌐 Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to [https://share.streamlit.io](https://share.streamlit.io)
3. Deploy your repo
4. Add Secret:

```toml
GEMINI_API_KEY = "your_api_key_here"
```

5. Reboot App

---

## 🎤 Voice Support Tips

* Use **Google Chrome / Edge**
* Allow mic permissions
* If voice doesn’t auto-play, click 🔊 “Play Question”
* HTTPS is required for voice (Streamlit Cloud provides it)

---

## ❗ Troubleshooting

### No AI Voice

Click the manual 🔊 play button and allow browser audio permissions.

### No transcription

Confirm Gemini API key is working and check Streamlit logs.

### API errors

Verify the secret key and restart the app.

---

## 📊 Outputs

### Candidate:

* Summary feedback
* Skill strengths
* Weakness analysis
* Improvement roadmap

### HR:

* Hire / Hold / Reject decision
* Score breakdown
* Interview justification

---

## 💡 Enhancements Planned

* Coding rounds
* Emotion analysis
* Multi-panel interviews
* ATS integration

---

## 👨‍💻 Author

**Ravindranath A**

