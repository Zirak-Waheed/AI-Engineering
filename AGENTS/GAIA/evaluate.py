"""Local evaluation script — runs agent on all 20 GAIA questions and prints a dry-run table."""
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Optional

import requests
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

sys.path.insert(0, str(Path(__file__).parent))
from agent import run_agent

API_BASE_URL = "https://agents-course-unit4-scoring.hf.space"
REQUEST_TIMEOUT = 30


def _download_file(task_id: str, file_name: str, tmp_dir: str) -> Optional[str]:
    try:
        r = requests.get(f"{API_BASE_URL}/files/{task_id}", timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        path = os.path.join(tmp_dir, Path(file_name).name)
        with open(path, "wb") as f:
            f.write(r.content)
        return path
    except Exception as e:
        print(f"  [warn] Could not download file for {task_id}: {e}")
        return None


def main() -> None:
    print("Fetching questions...")
    r = requests.get(f"{API_BASE_URL}/questions", timeout=REQUEST_TIMEOUT)
    r.raise_for_status()
    questions = r.json()
    print(f"Got {len(questions)} questions.\n")

    results: list[dict] = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, item in enumerate(questions, 1):
            task_id = item.get("task_id", "")
            question = item.get("question", "")
            file_name = item.get("file_name", "") or ""
            level = item.get("Level", "?")

            print(f"[{i:02d}/{len(questions)}] Task: {task_id}  Level: {level}")
            print(f"  Q: {question[:120]}")

            file_path: Optional[str] = None
            if file_name:
                file_path = _download_file(task_id, file_name, tmp_dir)

            try:
                answer = run_agent(question, file_path)
            except Exception as e:
                answer = ""
                print(f"  [ERROR] {e}")
                traceback.print_exc()

            print(f"  A: {answer!r}\n")
            results.append({
                "task_id": task_id,
                "question": question[:150],
                "file": file_name or "-",
                "answer": answer,
            })

    print("\n" + "=" * 80)
    print("DRY-RUN RESULTS (not yet submitted)")
    print("=" * 80)
    for r_item in results:
        print(f"  [{r_item['task_id']}]  {r_item['question'][:80]}...")
        print(f"    Answer: {r_item['answer']!r}")
    print("=" * 80)

    out_path = Path(__file__).parent / "dry_run_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
