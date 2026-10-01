"""QuizCraft: an LLM-powered academic quiz generator."""

import os
import json
import sqlite3
from datetime import date

import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

load_dotenv()
DATABASE_PATH = "quiz_attempts.db"


def get_connection() -> sqlite3.Connection:
    connection = sqlite3.connect(DATABASE_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            submitted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            student_name TEXT NOT NULL,
            roll_number TEXT NOT NULL,
            quiz_title TEXT NOT NULL,
            total_questions INTEGER NOT NULL,
            correct_answers INTEGER NOT NULL,
            wrong_answers INTEGER NOT NULL,
            score_percent REAL NOT NULL,
            answers_json TEXT NOT NULL
        )
    """)
    connection.commit()
    return connection


def save_attempt(name: str, roll_number: str, quiz: "Quiz", selected_answers: list[str]) -> tuple[int, int, float]:
    correct = sum(answer == question.correct_answer for answer, question in zip(selected_answers, quiz.questions))
    total = len(quiz.questions)
    wrong = total - correct
    details = [
        {
            "question": question.question,
            "selected_answer": answer,
            "correct_answer": question.correct_answer,
            "is_correct": answer == question.correct_answer,
        }
        for answer, question in zip(selected_answers, quiz.questions)
    ]
    with get_connection() as connection:
        connection.execute(
            """INSERT INTO attempts
            (student_name, roll_number, quiz_title, total_questions, correct_answers, wrong_answers, score_percent, answers_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (name, roll_number, quiz.title, total, correct, wrong, round(correct / total * 100, 2), json.dumps(details)),
        )
    return correct, wrong, round(correct / total * 100, 2)


def recent_attempts() -> list[sqlite3.Row]:
    with get_connection() as connection:
        return connection.execute(
            "SELECT submitted_at, student_name, roll_number, quiz_title, correct_answers, wrong_answers, score_percent FROM attempts ORDER BY id DESC LIMIT 50"
        ).fetchall()


class Question(BaseModel):
    question: str = Field(description="Clear, self-contained question text")
    question_type: str = Field(description="multiple_choice, true_false, or short_answer")
    options: list[str] = Field(default_factory=list, description="Exactly 4 options for multiple choice")
    correct_answer: str = Field(description="Correct answer, or a concise marking guide")
    explanation: str = Field(description="Brief explanation to help the instructor")


class Quiz(BaseModel):
    title: str
    instructions: str
    questions: list[Question]


def build_llm() -> ChatGoogleGenerativeAI:
    """Create a Gemini Developer API chat model from environment settings."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not configured. Add it to your .env file.")
    return ChatGoogleGenerativeAI(
        model=os.getenv("QUIZ_MODEL", "gemini-3.6-flash"),
        api_key=api_key,
        temperature=0.45,
    )


def generate_quiz(subject: str, topic: str, level: str, count: int, time_limit: int, style: str, learning_goals: str) -> Quiz:
    system = """You are an experienced academic assessment designer. Create valid, fair, age-appropriate questions.
Avoid trick questions, ambiguous wording, and references that require unprovided context. Ensure every multiple-choice
question has exactly four plausible options and one unambiguously correct answer. Do not repeat a learning objective.
Return only the requested structured quiz."""
    request = f"""Create a {count}-question {style} quiz.
Subject: {subject}
Topic: {topic}
Learner level: {level}
Suggested completion time: {time_limit} minutes
Learning goals: {learning_goals or 'Assess core knowledge, conceptual understanding, and application.'}

Use question_type 'multiple_choice' for all questions if the style is 'Multiple choice'. For 'Mixed', use a balanced
mix of multiple_choice, true_false, and short_answer. For non-multiple-choice questions, leave options empty.
For every multiple-choice question, set correct_answer to the exact text of the one correct option (not its letter).
Give the quiz a specific title and short student-facing instructions."""
    model = build_llm().with_structured_output(Quiz, method="json_schema")
    return model.invoke([SystemMessage(content=system), HumanMessage(content=request)])


def quiz_as_markdown(quiz: Quiz, include_key: bool = True) -> str:
    lines = [f"# {quiz.title}", "", quiz.instructions, ""]
    for number, question in enumerate(quiz.questions, start=1):
        lines.extend([f"## {number}. {question.question}", ""])
        if question.options:
            lines.extend(f"{letter}. {option}" for letter, option in zip("ABCD", question.options))
            lines.append("")
        else:
            lines.extend(["Answer: _________________________________________________", ""])
    if include_key:
        lines.extend(["---", "", "# Answer key", ""])
        for number, question in enumerate(quiz.questions, start=1):
            lines.append(f"**{number}.** {question.correct_answer}  ")
            lines.append(f"_{question.explanation}_")
            lines.append("")
    return "\n".join(lines)


def render_student_quiz(quiz: Quiz) -> None:
    st.subheader("Student quiz")
    st.caption("Enter your details, select one answer for every question, then submit once.")
    with st.form("student_submission", clear_on_submit=False):
        name, roll_number = st.columns(2)
        with name:
            student_name = st.text_input("Student name", key="student_name")
        with roll_number:
            student_roll = st.text_input("Roll number", key="student_roll")
        selected_answers = []
        for number, question in enumerate(quiz.questions, start=1):
            st.markdown(f"**{number}. {question.question}**")
            selected_answers.append(
                st.radio(
                    "Choose an answer",
                    question.options,
                    index=None,
                    key=f"answer_{number}_{quiz.title}",
                    label_visibility="collapsed",
                )
            )
        submitted = st.form_submit_button("Submit answers", type="primary", use_container_width=True)

    if submitted:
        if not student_name.strip() or not student_roll.strip():
            st.error("Enter both your name and roll number.")
        elif any(answer is None for answer in selected_answers):
            st.error("Select an answer for every question before submitting.")
        else:
            correct, wrong, percentage = save_attempt(student_name.strip(), student_roll.strip(), quiz, selected_answers)
            st.session_state.latest_attempt = selected_answers
            st.success(f"Submitted successfully. Score: {correct}/{len(quiz.questions)} ({percentage}%)")
            st.metric("Correct answers", correct, f"{wrong} wrong")
            st.markdown("### Answer review")
            for number, (question, selected) in enumerate(zip(quiz.questions, selected_answers), start=1):
                if selected == question.correct_answer:
                    st.success(f"{number}. Correct — {selected}")
                else:
                    st.error(f"{number}. Your answer: {selected} | Correct answer: {question.correct_answer}")


st.set_page_config(page_title="QuizCraft", page_icon="✦", layout="wide")
st.markdown("""
<style>
  .stApp { background: #f5f3ec; color: #17261f; }
  h1, h2, h3 { font-family: Georgia, serif !important; color: #17261f; }
  .hero { padding: 1.5rem 0 .9rem; }
  .eyebrow { color:#647069; font-size:.78rem; font-weight:700; letter-spacing:.11em; text-transform:uppercase; }
  div[data-testid="stForm"] { background:#fffdf8; border:1px solid #dde2d7; padding:1.4rem; border-radius:16px; }
  .quiz-card { background:#fffefa; border:1px solid #dde2d7; border-radius:16px; padding:1.7rem 2rem; }
</style>
""", unsafe_allow_html=True)

if "quiz" not in st.session_state:
    st.session_state.quiz = None

st.markdown('<div class="hero"><div class="eyebrow">LLM-powered assessment studio</div><h1>QuizCraft</h1><p>Generate thoughtful, classroom-ready quizzes aligned to your teaching goals.</p></div>', unsafe_allow_html=True)

left, right = st.columns([0.95, 1.45], gap="large")
with left:
    with st.form("quiz_settings"):
        st.subheader("Build a quiz")
        subject = st.selectbox("Subject", ["B.Tech — Computer Science", "B.Tech — Information Technology", "B.Tech — Electronics & Communication", "B.Tech — Electrical Engineering", "B.Tech — Mechanical Engineering", "B.Tech — Civil Engineering", "Other"])
        topic = st.text_input("Topic or unit", placeholder="e.g. Data Structures, DBMS, Operating Systems, Digital Electronics")
        level = st.selectbox("Learner level", ["B.Tech 1st year", "B.Tech 2nd year", "B.Tech 3rd year", "B.Tech 4th year"])
        count, time_limit = st.columns(2)
        with count:
            count = st.slider("Questions", min_value=3, max_value=15, value=5)
        with time_limit:
            time_limit = st.slider("Time limit (minutes)", min_value=5, max_value=120, value=20, step=5)
        style = "Multiple choice"
        st.info("Online test mode: every question is multiple choice and scored automatically.")
        learning_goals = st.text_area("Learning goals (optional)", placeholder="e.g. Explain the role of chlorophyll and predict how light affects photosynthesis.")
        submitted = st.form_submit_button("Generate quiz ✦", use_container_width=True, type="primary")
    if submitted:
        if not topic.strip():
            st.warning("Add a topic or unit first.")
        else:
            try:
                with st.spinner("Designing your assessment…"):
                    st.session_state.quiz = generate_quiz(subject, topic.strip(), level, count, time_limit, style, learning_goals.strip())
                st.success("Your quiz is ready.")
            except Exception as exc:
                st.error(f"Could not generate the quiz: {exc}")
    st.caption("Your Gemini API key stays on the machine running this app. Review generated content before using it with learners.")

with right:
    quiz = st.session_state.quiz
    if not quiz:
        st.markdown("<div class='quiz-card'><h2>Your quiz will appear here</h2><p>Set the subject, topic, and learner level, then generate a tailored assessment.</p></div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="quiz-card">', unsafe_allow_html=True)
        st.markdown(f"## {quiz.title}\n\n{quiz.instructions}\n\n*Suggested time: {time_limit} minutes*")
        st.markdown("</div>", unsafe_allow_html=True)
        render_student_quiz(quiz)
        with st.expander("Instructor answer key"):
            for number, question in enumerate(quiz.questions, start=1):
                st.markdown(f"**{number}. {question.correct_answer}**  \n{question.explanation}")
        file_stem = "-".join(quiz.title.lower().split())[:50]
        download_col, student_col = st.columns(2)
        with download_col:
            st.download_button("Download with answer key", quiz_as_markdown(quiz), f"{file_stem}-quiz.md", "text/markdown", use_container_width=True)
        with student_col:
            st.download_button("Download student version", quiz_as_markdown(quiz, include_key=False), f"{file_stem}-student.md", "text/markdown", use_container_width=True)
        with st.expander("Teacher records (latest 50 submissions)"):
            attempts = recent_attempts()
            if attempts:
                st.dataframe([dict(attempt) for attempt in attempts], use_container_width=True, hide_index=True)
            else:
                st.caption("No student submissions yet.")

st.caption(f"QuizCraft · {date.today().year} · Built with Streamlit, LangChain, and Gemini")
