"""
Submit answers from dry_run_results.json to the GAIA course API.
Usage: python3 submit.py <your-hf-username> [agent_code_url]
"""
import json
import sys
from pathlib import Path

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

API_BASE_URL = "https://agents-course-unit4-scoring.hf.space"

_NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def _final_normalize(answer: str) -> str:
    """Apply last-mile normalization before submission."""
    text = answer.strip()
    if text.lower() in _NUMBER_WORDS:
        text = _NUMBER_WORDS[text.lower()]
    return text


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python3 submit.py <your-hf-username> [agent_code_url]")
        sys.exit(1)

    username = sys.argv[1].strip()
    agent_code = sys.argv[2] if len(sys.argv) > 2 else "https://github.com/placeholder"

    results_path = Path(__file__).parent / "dry_run_results.json"
    if not results_path.exists():
        print(f"Error: {results_path} not found. Run evaluate.py first.")
        sys.exit(1)

    with open(results_path) as f:
        results = json.load(f)

    answers = [
        {
            "task_id": item["task_id"],
            "submitted_answer": _final_normalize(item["answer"]),
        }
        for item in results
    ]

    print(f"\nSubmitting {len(answers)} answers as user '{username}'...")
    print("\nAnswers to be submitted:")
    for a in answers:
        print(f"  {a['task_id'][:8]}...  →  {a['submitted_answer']!r}")

    confirm = input("\nProceed with submission? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        sys.exit(0)

    payload = {
        "username": username,
        "agent_code": agent_code,
        "answers": answers,
    }

    try:
        response = requests.post(f"{API_BASE_URL}/submit", json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        print(f"\n{'='*50}")
        print(f"SUBMISSION SUCCESSFUL")
        print(f"{'='*50}")
        print(f"User:    {data.get('username')}")
        print(f"Score:   {data.get('score')}% ({data.get('correct_count')}/{data.get('total_attempted')} correct)")
        print(f"Message: {data.get('message')}")
        print(f"{'='*50}")
    except requests.exceptions.HTTPError as e:
        try:
            detail = e.response.json().get("detail", e.response.text[:300])
        except Exception:
            detail = e.response.text[:300]
        print(f"Submission failed: HTTP {e.response.status_code} — {detail}")
        sys.exit(1)
    except Exception as e:
        print(f"Submission failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
