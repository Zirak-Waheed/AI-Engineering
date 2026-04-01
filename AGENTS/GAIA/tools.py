import base64
import io
import os
import subprocess
import sys
import tempfile
from typing import Optional

import requests
from langchain_core.tools import tool

MAX_WEBPAGE_CHARS = 10_000
MAX_FILE_CHARS = 20_000


@tool
def tavily_search(query: str) -> str:
    """Search the web for factual information. Returns top results with titles, URLs, and content snippets.
    Use this for factual lookups, entity queries, date verification, and general research.
    """
    from langchain_tavily import TavilySearch

    search = TavilySearch(max_results=5)
    results = search.invoke(query)

    if isinstance(results, list):
        output_parts: list[str] = []
        for r in results:
            title = r.get("title", "No title")
            url = r.get("url", "")
            content = r.get("content", "")
            output_parts.append(f"**{title}**\nURL: {url}\n{content}")
        return "\n\n---\n\n".join(output_parts) if output_parts else "No results found."

    return str(results)


@tool
def visit_webpage(url: str) -> str:
    """Fetch a web page and return its content as markdown text (truncated to ~10,000 chars).
    Use this to read full articles, Wikipedia pages, tables, or any page found via search.
    """
    try:
        from markdownify import markdownify as md

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
        }
        response = requests.get(url, headers=headers, timeout=15)
        response.raise_for_status()

        markdown = md(response.text, heading_style="ATX", strip=["script", "style", "nav", "footer"])
        markdown = "\n".join(line for line in markdown.splitlines() if line.strip())

        if len(markdown) > MAX_WEBPAGE_CHARS:
            markdown = markdown[:MAX_WEBPAGE_CHARS] + "\n\n[...content truncated...]"

        return markdown if markdown.strip() else "Page returned empty content."
    except requests.exceptions.Timeout:
        return "Error: Request timed out."
    except requests.exceptions.HTTPError as e:
        return f"Error: HTTP {e.response.status_code} when fetching {url}"
    except Exception as e:
        return f"Error fetching page: {e}"


@tool
def python_repl(code: str) -> str:
    """Execute Python code and return stdout output.
    Use this for: math calculations, date arithmetic, unit conversions,
    CSV/Excel data parsing, string manipulation, and any computation.
    The execution is isolated. Import libraries as needed (pandas, math, datetime, etc.).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=30,
        )
        output = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode != 0:
            return f"Error:\n{stderr}" if stderr else "Error: Script exited with non-zero code."

        if not output and stderr:
            return f"(no stdout)\nStderr: {stderr}"

        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."
    except Exception as e:
        return f"Error running code: {e}"


@tool
def read_file(file_path: str) -> str:
    """Read a local file and return its content as text.
    Supports: .txt, .py, .json, .md, .csv, .xlsx, .xls, .pdf
    For CSV/Excel, returns the first 100 rows as a string table.
    Use the exact file_path provided by the task context.
    """
    if not os.path.exists(file_path):
        return f"Error: File not found at path: {file_path}"

    ext = os.path.splitext(file_path)[1].lower()

    try:
        if ext == ".pdf":
            return _read_pdf(file_path)
        elif ext in (".csv",):
            return _read_csv(file_path)
        elif ext in (".xlsx", ".xls"):
            return _read_excel(file_path)
        else:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read()
            if len(content) > MAX_FILE_CHARS:
                content = content[:MAX_FILE_CHARS] + "\n\n[...content truncated...]"
            return content
    except Exception as e:
        return f"Error reading file: {e}"


def _read_pdf(file_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text)
    full_text = "\n\n".join(pages)
    if len(full_text) > MAX_FILE_CHARS:
        full_text = full_text[:MAX_FILE_CHARS] + "\n\n[...content truncated...]"
    return full_text if full_text.strip() else "PDF appears to have no extractable text."


def _read_csv(file_path: str) -> str:
    import pandas as pd

    df = pd.read_csv(file_path, nrows=100)
    return df.to_string(index=False)


def _read_excel(file_path: str) -> str:
    import pandas as pd

    df = pd.read_excel(file_path, nrows=100)
    return df.to_string(index=False)


@tool
def analyze_image(file_path: str, question: Optional[str] = None) -> str:
    """Analyze an image file using GPT-4o vision and return a detailed description or answer.
    Pass the question parameter to get a specific answer about the image content.
    Supports: .png, .jpg, .jpeg, .webp, .gif
    """
    from openai import OpenAI

    if not os.path.exists(file_path):
        return f"Error: Image file not found at path: {file_path}"

    ext = os.path.splitext(file_path)[1].lower().lstrip(".")
    mime_map = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png",
                "webp": "image/webp", "gif": "image/gif"}
    mime_type = mime_map.get(ext, "image/png")

    try:
        with open(file_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        prompt = question if question else (
            "Describe this image in detail. Include all text, numbers, labels, "
            "items, colors, and any other visible information."
        )

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{image_data}"},
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            max_tokens=1024,
        )
        return response.choices[0].message.content or "No description returned."
    except Exception as e:
        return f"Error analyzing image: {e}"


@tool
def transcribe_audio(file_path: str) -> str:
    """Transcribe an audio file to text using OpenAI Whisper.
    Supports: .mp3, .wav, .m4a, .ogg, .flac, .webm
    Returns the full transcript as text.
    """
    from openai import OpenAI

    if not os.path.exists(file_path):
        return f"Error: Audio file not found at path: {file_path}"

    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text",
            )
        return str(transcript)
    except Exception as e:
        return f"Error transcribing audio: {e}"


@tool
def get_youtube_transcript(video_url: str) -> str:
    """Fetch the transcript/captions of a YouTube video and return it as text.
    Use this whenever a question references a YouTube URL (youtube.com/watch?v=...).
    This is the primary way to analyze YouTube video content.
    """
    import re as _re
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

    video_id_match = _re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", video_url)
    if not video_id_match:
        return f"Error: Could not extract video ID from URL: {video_url}"

    video_id = video_id_match.group(1)
    try:
        transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
        lines = [entry["text"] for entry in transcript_list]
        full_text = " ".join(lines)
        if len(full_text) > MAX_WEBPAGE_CHARS:
            full_text = full_text[:MAX_WEBPAGE_CHARS] + "\n\n[...transcript truncated...]"
        return f"YouTube transcript for {video_url}:\n\n{full_text}"
    except TranscriptsDisabled:
        return f"Transcripts are disabled for video: {video_url}"
    except NoTranscriptFound:
        return f"No transcript found for video: {video_url}"
    except Exception as e:
        return f"Error fetching transcript: {e}"


ALL_TOOLS = [
    tavily_search,
    visit_webpage,
    python_repl,
    read_file,
    analyze_image,
    transcribe_audio,
    get_youtube_transcript,
]
