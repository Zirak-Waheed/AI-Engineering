import os
import sys
import json
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())
sys.path.insert(0, str(Path(__file__).parent))
from agent import run_agent

with open(Path(__file__).parent / "dry_run_results.json") as f:
    data = json.load(f)
results_by_id = {r["task_id"]: r for r in data}

questions = [
    (
        "8e867cd7-cff9-4e6c-867a-ff5ddc2550be",
        "How many studio albums were published by Mercedes Sosa between 2000 and 2009 (included)? You can use the latest 2022 version of english wikipedia.",
    ),
    (
        "3f57289b-8c60-48be-bd80-01f8099ca449",
        "How many at bats did the Yankee with the most walks in the 1977 regular season have that same season?",
    ),
    (
        "a0c07678-e491-4bbc-8f0b-07405144218f",
        "Who are the pitchers with the number before and after Taisho Tamai's number as of July 2023? Give them to me in the form Pitcher Before, Pitcher After, use their last names only, in Roman characters.",
    ),
]

for tid, question in questions:
    print(f"=== {tid[:8]} ===", flush=True)
    print(f"Current: {repr(results_by_id.get(tid, {}).get('answer', ''))}", flush=True)
    answer = run_agent(question, None)
    print(f"New answer: {repr(answer)}", flush=True)
    if answer:
        results_by_id[tid]["answer"] = answer
        with open(Path(__file__).parent / "dry_run_results.json", "w") as f:
            json.dump(list(results_by_id.values()), f, indent=2)
        print("Saved!", flush=True)
    print("", flush=True)

print("Done!", flush=True)
