"""ASGI entrypoint for Uvicorn. See ``README.md`` and ``api.py`` for routes."""

from __future__ import annotations

from .api import app

__all__ = ["app"]
