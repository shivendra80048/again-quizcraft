# QuizCraft

An LLM-powered B.Tech quiz generator built with Streamlit, LangChain, and the Gemini API. It produces automatically scored multiple-choice tests, an instructor answer key, downloadable versions, and a local record of student name, roll number, responses, and marks.

## Run locally

1. Create and activate a Python virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, then add your Gemini API key from Google AI Studio.
4. Start the app: `streamlit run app.py`

The default model is `gemini-3.6-flash`. Set `GEMINI_API_KEY` and, optionally, `QUIZ_MODEL` in `.env`.

## Design choices

- **Structured LLM output:** LangChain validates the response as a Pydantic `Quiz`, avoiding fragile free-form parsing.
- **Teacher-oriented controls:** Subject, level, learning goals, format, question count, and a suggested time limit shape the prompt.
- **Safe-by-default workflow:** The API key is server-side only; the UI reminds teachers to review generated material before classroom use.
- **Student submissions:** Answers, correct/wrong counts, marks, student name, and roll number are stored in the local `quiz_attempts.db` database.
