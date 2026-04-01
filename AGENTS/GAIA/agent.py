import concurrent.futures
import os
import re
import unicodedata
from typing import Optional

from dotenv import find_dotenv, load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent  # noqa: F401 — still valid in LangGraph v1

from tools import ALL_TOOLS

load_dotenv(find_dotenv())

MAX_ITERATIONS = 100
QUESTION_TIMEOUT_SECONDS = 180

SYSTEM_PROMPT = """You are a precise research assistant solving questions from the GAIA benchmark.
You have access to the following tools:
- tavily_search: Search the web for factual information
- visit_webpage: Fetch and read a full web page (use URLs from search results)
- python_repl: Execute Python code for math, dates, data parsing, unit conversion
- read_file: Read a local file (PDF, CSV, Excel, text)
- analyze_image: Analyze an image file with GPT-4o vision
- transcribe_audio: Transcribe an audio file with Whisper
- get_youtube_transcript: Get the transcript/captions of a YouTube video

APPROACH:
1. Break the question into sub-questions if needed.
2. Use tools to gather facts — NEVER answer from memory alone.
3. For file-based questions, ALWAYS use read_file/analyze_image/transcribe_audio first.
4. For YouTube URL questions, ALWAYS use get_youtube_transcript first.
5. Cross-verify critical facts with a second search or source when unsure.
6. If a search result has a relevant URL, use visit_webpage to read the full content.

ANSWER FORMAT (critical — answers are graded by exact string match):
- Return ONLY the final answer value. No explanations, no reasoning.
- Numbers: integer if whole (e.g. "42" not "42.0"), decimal only if the question needs precision.
- Lists: comma-separated, in the order the question specifies.
- Names: use the most common official form as the question expects.
- Dates: match the format the question asks for.
- If asked "how many", answer with just the number.
- If asked for a name, answer with just the name.
- If asked for a yes/no, answer with just "Yes" or "No".
- Do NOT include "FINAL ANSWER" or any prefix in your response — just the value."""


VERIFICATION_PROMPT = """You are a quality checker for GAIA benchmark answers.

Question: {question}
Candidate answer: {candidate}

Check ALL of the following:
1. Does this answer directly and completely address what was asked?
2. Is the format correct? (number, name, list, date — matching what the question expects)
3. Is the answer free of extra words, explanations, or prefixes?
4. For numbers: is it an integer when it should be (no "42.0" when "42" is correct)?
5. For lists: are items in the order specified by the question?

If the answer is correct and properly formatted, respond with exactly:
VERIFIED: {candidate}

If the answer has issues, respond with:
RETRY: <one sentence explaining what is wrong or what to check>"""


def _build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        temperature=0,
    )


def normalize_answer(raw: str) -> str:
    """Normalize raw agent output to a clean, exact-match-safe string."""
    text = raw.strip()

    failure_patterns = [
        r"(?i)there (was|is) an? issue",
        r"(?i)could not download",
        r"(?i)could not find",
        r"(?i)please provide the",
        r"(?i)i was unable to find",
        r"(?i)i was unable to access",
        r"(?i)can'?t access or analyze youtube",
        r"(?i)unable to access the",
        r"(?i)please check the file path",
        r"(?i)please ensure the file path",
        r"(?i)no transcript found",
        r"(?i)transcripts are disabled",
    ]
    for fp in failure_patterns:
        if re.search(fp, text):
            return ""

    prefixes_to_strip = [
        r"^FINAL ANSWER:\s*",
        r"^Final answer:\s*",
        r"^Answer:\s*",
        r"^The answer is:?\s*",
        r"^Result:\s*",
        r"^Output:\s*",
        r"^VERIFIED:\s*",
        r"^\w[\w\s]+ played the character [\"'](\w+)[\"'] in [\"'].+[\"']\.\s*$",
        r"^Therefore,?\s*(the )?(\w+\s*)?(corrected )?final answer is:?\s*",
        r"^Therefore,?\s*the IOC country code is:?\s*",
    ]
    for pattern in prefixes_to_strip:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()

    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", "-")

    text = re.sub(r"\s+", " ", text).strip()

    text = text.strip(".")

    if re.match(r"^-?\d+\.0+$", text):
        text = str(int(float(text)))

    _number_words = {
        "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
        "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
        "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
        "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
        "eighteen": "18", "nineteen": "19", "twenty": "20",
    }
    if text.lower() in _number_words:
        text = _number_words[text.lower()]

    return text


def _extract_last_ai_text(messages: list) -> str:
    """Return the text content of the last AI message that has no pending tool calls."""
    for msg in reversed(messages):
        if not hasattr(msg, "content") or not msg.content:
            continue
        if getattr(msg, "tool_calls", []):
            continue
        from langchain_core.messages import AIMessage

        if not isinstance(msg, AIMessage):
            continue
        content = msg.content
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"].strip()
    return ""


EXTRACTION_PROMPT = """Extract ONLY the final answer value from the text below.

Question: {question}
Text: {text}

Rules:
- Return ONLY the answer value — no explanation, no labels, no full sentences.
- If the text contains a short final value (e.g. a name, number, or list), return just that.
- For names: just the name.
- For numbers: just the number.
- For lists: comma-separated values.
- For country codes or short identifiers: just the code/identifier.
- If you cannot extract a clear value, return an empty string.

Answer:"""


def _extract_final_value(llm: ChatOpenAI, question: str, text: str) -> str:
    """Use LLM to extract the short final value from a verbose answer."""
    prompt = EXTRACTION_PROMPT.format(question=question, text=text)
    response = llm.invoke([HumanMessage(content=prompt)])
    extracted = (response.content or "").strip()
    extracted = re.sub(r"^Answer:\s*", "", extracted, flags=re.IGNORECASE).strip()
    return extracted


def _verify_answer(llm: ChatOpenAI, question: str, candidate: str) -> tuple[bool, str]:
    """Run a verification pass. Returns (is_verified, retry_reason)."""
    prompt = VERIFICATION_PROMPT.format(question=question, candidate=candidate)
    response = llm.invoke([HumanMessage(content=prompt)])
    content = response.content.strip()

    if content.startswith("VERIFIED:"):
        return True, ""

    reason = content.replace("RETRY:", "").strip()
    return False, reason


def _run_agent_inner(question: str, file_path: Optional[str] = None) -> str:
    """Core agent logic — called with a timeout wrapper."""
    llm = _build_llm()
    agent = create_react_agent(
        llm,
        tools=ALL_TOOLS,
        prompt=SystemMessage(content=SYSTEM_PROMPT),
    )

    user_message = question
    if file_path and os.path.exists(file_path):
        user_message = (
            f"{question}\n\n"
            f"[An attached file is available at: {file_path}. "
            f"Use the appropriate tool to read/analyze it.]"
        )

    messages = [HumanMessage(content=user_message)]
    config = {"recursion_limit": MAX_ITERATIONS + 5}
    try:
        result = agent.invoke({"messages": messages}, config=config)
    except Exception as exc:
        exc_str = str(exc)
        if "Recursion limit" in exc_str or "recursion" in exc_str.lower():
            print(f"  [warn] Recursion limit hit; asking LLM to answer directly.")
            direct = llm.invoke([
                SystemMessage(content=SYSTEM_PROMPT),
                HumanMessage(content=user_message),
            ])
            return normalize_answer(direct.content or "")
        raise
    candidate = normalize_answer(_extract_last_ai_text(result.get("messages", [])))

    if len(candidate) > 80:
        extracted = _extract_final_value(llm, question, candidate)
        if extracted:
            candidate = normalize_answer(extracted)

    is_verified, retry_reason = _verify_answer(llm, question, candidate)
    if not is_verified and retry_reason:
        retry_message = (
            f"{user_message}\n\n"
            f"[Previous attempt gave: '{candidate}'. Issue: {retry_reason}. "
            f"Please reconsider and provide only the corrected final answer.]"
        )
        try:
            retry_result = agent.invoke(
                {"messages": [HumanMessage(content=retry_message)]},
                config=config,
            )
            retry_raw = _extract_last_ai_text(retry_result.get("messages", []))
            if retry_raw:
                candidate = normalize_answer(retry_raw)
                if len(candidate) > 80:
                    extracted = _extract_final_value(llm, question, candidate)
                    if extracted:
                        candidate = normalize_answer(extracted)
        except Exception:
            pass

    return candidate


def run_agent(question: str, file_path: Optional[str] = None) -> str:
    """Run the GAIA agent with a hard timeout. Returns empty string on timeout/error."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_run_agent_inner, question, file_path)
        try:
            return future.result(timeout=QUESTION_TIMEOUT_SECONDS)
        except concurrent.futures.TimeoutError:
            print(f"  [timeout] Question timed out after {QUESTION_TIMEOUT_SECONDS}s")
            return ""
        except Exception as e:
            print(f"  [error] Agent raised: {e}")
            return ""
