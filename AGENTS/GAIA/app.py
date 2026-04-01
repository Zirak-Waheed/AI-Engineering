import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import gradio as gr
import pandas as pd
import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.insert(0, str(Path(__file__).parent))

from agent import run_agent

API_BASE_URL = "https://agents-course-unit4-scoring.hf.space"
SUPPORTED_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
SUPPORTED_AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm"}
REQUEST_TIMEOUT = 30


def _download_task_file(task_id: str, file_name: str, tmp_dir: str) -> Optional[str]:
    """Download an attached file for a task and return the local path."""
    try:
        url = f"{API_BASE_URL}/files/{task_id}"
        response = requests.get(url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()

        safe_name = Path(file_name).name
        local_path = os.path.join(tmp_dir, safe_name)
        with open(local_path, "wb") as f:
            f.write(response.content)
        return local_path
    except Exception as e:
        print(f"Warning: Could not download file for task {task_id}: {e}")
        return None


def _fetch_questions() -> list[dict]:
    """Fetch all 20 evaluation questions from the course API."""
    response = requests.get(f"{API_BASE_URL}/questions", timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()


def _submit_answers(username: str, agent_code: str, answers: list[dict]) -> dict:
    """Submit answers to the course API and return the score response."""
    payload = {
        "username": username.strip(),
        "agent_code": agent_code,
        "answers": answers,
    }
    response = requests.post(f"{API_BASE_URL}/submit", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def run_and_submit_all(profile: gr.OAuthProfile | None) -> tuple[str, pd.DataFrame]:
    """Fetch questions, run agent on each, submit answers, and return results."""
    if not profile:
        return "Please log in to Hugging Face using the button above.", pd.DataFrame()

    username = profile.username
    print(f"Logged in as: {username}")

    space_id = os.getenv("SPACE_ID", "")
    agent_code = (
        f"https://huggingface.co/spaces/{space_id}/tree/main"
        if space_id
        else "https://github.com/placeholder"
    )

    try:
        questions_data = _fetch_questions()
    except Exception as e:
        return f"Error fetching questions: {e}", pd.DataFrame()

    if not questions_data:
        return "No questions returned from the API.", pd.DataFrame()

    print(f"Fetched {len(questions_data)} questions.")

    results_log: list[dict] = []
    answers_payload: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for item in questions_data:
            task_id: str = item.get("task_id", "")
            question_text: str = item.get("question", "")
            file_name: str = item.get("file_name", "") or ""

            if not task_id or not question_text:
                print(f"Skipping incomplete item: {item}")
                continue

            print(f"\n--- Task {task_id} ---")
            print(f"Q: {question_text[:120]}")

            file_path: Optional[str] = None
            if file_name:
                file_path = _download_task_file(task_id, file_name, tmp_dir)
                if file_path:
                    print(f"Downloaded file: {file_path}")

            try:
                submitted_answer = run_agent(question_text, file_path)
                print(f"Answer: {submitted_answer}")
            except Exception as e:
                submitted_answer = ""
                tb = traceback.format_exc()
                print(f"Agent error on task {task_id}: {e}\n{tb}")

            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append({
                "Task ID": task_id,
                "Question": question_text[:200],
                "File": file_name or "-",
                "Submitted Answer": submitted_answer,
            })

    if not answers_payload:
        return "Agent produced no answers.", pd.DataFrame(results_log)

    print(f"\nSubmitting {len(answers_payload)} answers...")
    try:
        result_data = _submit_answers(username, agent_code, answers_payload)
        score = result_data.get("score", "N/A")
        correct = result_data.get("correct_count", "?")
        total = result_data.get("total_attempted", "?")
        message = result_data.get("message", "")
        final_status = (
            f"Submission successful!\n"
            f"User: {username}\n"
            f"Score: {score}% ({correct}/{total} correct)\n"
            f"Message: {message}"
        )
        print(final_status)
    except requests.exceptions.HTTPError as e:
        detail = ""
        try:
            detail = e.response.json().get("detail", e.response.text[:300])
        except Exception:
            detail = e.response.text[:300]
        final_status = f"Submission failed: HTTP {e.response.status_code} — {detail}"
        print(final_status)
    except Exception as e:
        final_status = f"Submission failed: {e}"
        print(final_status)

    return final_status, pd.DataFrame(results_log)


with gr.Blocks(title="GAIA Agent — HF Agents Course Unit 4") as demo:
    gr.Markdown("# GAIA Benchmark Agent")
    gr.Markdown(
        """
        A LangGraph-powered agent with 6 tools (web search, page reading, Python execution,
        file parsing, image analysis, audio transcription) targeting **≥50%** on the
        HF Agents Course Unit 4 GAIA Level-1 evaluation set.

        **Steps:**
        1. Log in with your Hugging Face account below.
        2. Click **Run Evaluation & Submit** to run the agent on all 20 questions and submit.
        3. Results and score will appear below.

        _Note: This may take several minutes as the agent runs multi-step reasoning on each question._
        """
    )

    gr.LoginButton()

    run_button = gr.Button("Run Evaluation & Submit All Answers", variant="primary")
    status_output = gr.Textbox(label="Status / Score", lines=6, interactive=False)
    results_table = gr.DataFrame(label="Question Results", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )

if __name__ == "__main__":
    print("Starting GAIA Agent app...")
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    if space_host:
        print(f"Space URL: https://{space_host}.hf.space")
    if space_id:
        print(f"Space repo: https://huggingface.co/spaces/{space_id}/tree/main")
    demo.launch(debug=True, share=False)
