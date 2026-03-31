"""Run the RAG practice API: ``python -m RAG.practice --host 127.0.0.1 --port 8001 --reload``."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG practice FastAPI server (Uvicorn).")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Listen port (default: 8000).")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development).",
    )
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "RAG.practice.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
